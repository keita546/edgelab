"""統計的優位性の検定。

素朴なバックテストが嘘をつく主因を個別に潰す:
  - 系列相関/重複窓 -> Newey-West HAC 標準誤差, ブロックブートストラップ
  - 多重検定(試した仮説の数) -> Benjamini-Hochberg FDR, Deflated Sharpe Ratio
  - 「ただ市場に乗っていただけ」 -> 非シグナル日を対照群にした Welch 検定
  - 取引コスト -> 往復コストを全トレードから控除
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EULER = 0.5772156649015329


def newey_west_t(x: np.ndarray, lag: int | None = None) -> tuple[float, float, float]:
    """平均がゼロという帰無仮説に対する HAC(Newey-West) t 値。

    保有期間が複数日のシグナルは前向きリターンの窓が重なるため、
    単純な t 検定は標準誤差を過小評価して有意に見せてしまう。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan, np.nan, np.nan
    mu = x.mean()
    e = x - mu
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 2))
    s = float(e @ e) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    s = max(s, 1e-18)
    se = np.sqrt(s / n)
    t = mu / se
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p), float(se)


def stationary_bootstrap_ci(x: np.ndarray, alpha: float = 0.05, n_boot: int = 4000,
                            mean_block: float = 5.0, seed: int = 7) -> tuple[float, float]:
    """定常ブートストラップによる平均の信頼区間(系列相関に頑健)。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)
    starts = rng.integers(0, n, size=(n_boot, n))
    cont = rng.random((n_boot, n)) >= p      # True なら直前の続きのバーを使う
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for k in range(1, n):                     # 時点方向のみループ(複製方向はベクトル化)
        idx[:, k] = np.where(cont[:, k], (idx[:, k - 1] + 1) % n, starts[:, k])
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH 法による FDR 調整 p 値。仮説を何十個も試した事実を補正する。"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """N 個の無意味な戦略を試したとき、偶然出てしまう最大シャープの期待値。"""
    if n_trials < 2 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(x: np.ndarray, n_trials: int, var_sr: float) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado)。

    「この戦略のシャープが本当にゼロより大きい確率」を、
    試行回数・歪度・尖度・標本数で割り引いて返す。0.95 以上が一応の目安。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 20 or x.std(ddof=1) == 0:
        return np.nan, np.nan
    sr = x.mean() / x.std(ddof=1)
    g3 = float(stats.skew(x))
    g4 = float(stats.kurtosis(x, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_sr)
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return sr, np.nan
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(sr), float(stats.norm.cdf(z))


def deflated_sharpe_m(sr: float, T: int, g3: float, g4: float, n_trials: int, var_sr: float) -> float:
    """deflated_sharpe と同じ値を、シャープ・標本数・歪度・尖度から計算する。"""
    if not (np.isfinite(sr) and T >= 20):
        return np.nan
    sr0 = expected_max_sharpe(n_trials, var_sr)
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)))


def welch(signal: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    """シグナル日 vs 非シグナル日の Welch t 検定。

    「上昇相場で買っていただけ」を除外するための対照群比較。
    """
    a = np.asarray(signal, float); a = a[np.isfinite(a)]
    b = np.asarray(control, float); b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def max_drawdown(rets: np.ndarray) -> float:
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def power_for_mean(x: np.ndarray, effect: float, alpha: float = 0.05) -> float:
    """この標本数で、effect の大きさの優位性を検出できる確率(検出力)。"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n, sd = len(x), (x.std(ddof=1) if len(x) > 2 else np.nan)
    if n < 5 or not np.isfinite(sd) or sd == 0:
        return np.nan
    ncp = effect / (sd / np.sqrt(n))
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(ncp - crit) + stats.norm.cdf(-ncp - crit))
