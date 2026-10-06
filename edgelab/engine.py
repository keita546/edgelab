"""全仮説を同じ土俵で評価する。

candidate_stats() で1件ずつの指標を出し、finalize() で多重検定補正(BH-FDR)と
DSR と判定をまとめて付ける。1銘柄なら evaluate()、全銘柄なら scan 側で
全銘柄ぶんの行を集めてから finalize() を1回だけ呼ぶ(補正は試した総数に対して効かせる)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import statlib as S
from .hypotheses import Candidate

BPS = 1e-4


def _naive(idx) -> pd.DatetimeIndex:
    """日足(tz無し)と分足(tz付き)の時刻を突き合わせ可能な形に揃える。"""
    di = pd.DatetimeIndex(pd.to_datetime(list(idx)))
    return di.tz_localize(None) if di.tz is not None else di


def candidate_stats(c: Candidate, cost_bps: float, cut, boot: int) -> dict:
    """1つの候補(1銘柄×1ルール)の指標。多重検定に依存しないものだけ。"""
    ts = _naive(c.rets.index)
    gross = c.rets.to_numpy(dtype=float)
    net = gross - cost_bps * BPS
    n = len(net)
    span_days = max((ts[-1] - ts[0]).days, 1) if n > 1 else 1
    per_year = n / (span_days / 365.25)
    t, p, _ = S.newey_west_t(net, lag=max(c.horizon - 1, 0) or None)
    lo, hi = S.stationary_bootstrap_ci(net, n_boot=boot, mean_block=max(c.horizon, 5)) if boot else (np.nan, np.nan)
    sd = net.std(ddof=1) if n > 2 else np.nan
    sr = net.mean() / sd if sd else np.nan
    if c.control is not None:
        wt, wp = S.welch(net, np.asarray(c.control, dtype=float) - cost_bps * BPS)
    else:
        wp = np.nan
    if cut is not None:
        is_r, oos_r = net[ts <= cut], net[ts > cut]
    else:
        is_r = oos_r = np.array([])
    return {
        "手法": c.name, "分類": c.family, "件数": n,
        "平均(bps)": net.mean() / BPS,
        "勝率%": float((net > 0).mean() * 100),
        "t値(HAC)": t, "p値": p,
        "CI下限(bps)": lo / BPS if np.isfinite(lo) else np.nan,
        "CI上限(bps)": hi / BPS if np.isfinite(hi) else np.nan,
        "対照群p": wp,
        "年率Sharpe": sr * np.sqrt(per_year) if np.isfinite(sr) else np.nan,
        "IS平均(bps)": is_r.mean() / BPS if len(is_r) > 5 else np.nan,
        "OOS平均(bps)": oos_r.mean() / BPS if len(oos_r) > 5 else np.nan,
        "最大DD%": S.max_drawdown(net) * 100,
        "損益分岐コスト(bps)": gross.mean() / BPS,
        "検出力(5bps)": S.power_for_mean(net, 5 * BPS),
        "年間取引数": per_year,
        "_sr": sr, "_T": n,
        "_g3": float(sps.skew(net)) if n > 3 else np.nan,
        "_g4": float(sps.kurtosis(net, fisher=False)) if n > 3 else np.nan,
        "_grp": "分足" if c.family == "分足" else "日足",
    }


def verdict(r) -> str:
    if not np.isfinite(r["p値"]):
        return "判定不能"
    pos = r["平均(bps)"] > 0
    if r["FDR補正p"] < 0.05 and not pos:
        return "有意にマイナス"          # 逆張りすれば勝てる、という意味ではない(コストは逆でもかかる)
    if r["FDR補正p"] < 0.05 and pos:
        if not (np.isfinite(r["IS平均(bps)"]) and np.isfinite(r["OOS平均(bps)"])):
            return "FDR通過・OOS不足"
        if np.sign(r["IS平均(bps)"]) != np.sign(r["OOS平均(bps)"]):
            return "IS/OOS不一致"
        if not (r["DSR"] > 0.95):
            return "FDR通過・DSR不足"
        return "有望(要追試)"
    if r["p値"] < 0.05:
        return "補正で消滅"
    return "優位性なし"


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """試した全候補に対して多重検定補正・DSR・判定を付ける。"""
    if df.empty:
        return df
    df = df.copy()
    df["FDR補正p"] = S.benjamini_hochberg(df["p値"].to_numpy())
    df["Bonferroni p"] = np.clip(df["p値"] * len(df), 0, 1)
    # DSR は同じ観測頻度の試行群(日足/分足)ごとに。尺度の違う試行を混ぜると全部 0 に潰れる
    df["DSR"] = np.nan
    for _, sub in df.groupby("_grp"):
        n_trials = len(sub)
        var_sr = float(np.nanvar(sub["_sr"].to_numpy(dtype=float), ddof=1)) if n_trials > 1 else 0.0
        df.loc[sub.index, "DSR"] = [S.deflated_sharpe_m(r["_sr"], int(r["_T"]), r["_g3"], r["_g4"], n_trials, var_sr)
                                    for _, r in sub.iterrows()]
    df["判定"] = df.apply(verdict, axis=1)
    return df


def evaluate(cands: list[Candidate], cost_bps: float = 10.0,
             split: float = 0.7, boot: int = 2000) -> pd.DataFrame:
    """1銘柄の候補を評価する(補正はこの銘柄の候補数に対して)。"""
    if not cands:
        return pd.DataFrame()
    # IS/OOS 分割日はグループ(日足/分足)ごと。混ぜると日足の OOS が数日しか残らない
    stamps = {c.name: _naive(c.rets.index) for c in cands}
    groups = {c.name: ("分足" if c.family == "分足" else "日足") for c in cands}
    split_at = {}
    for g in set(groups.values()):
        idx = pd.DatetimeIndex(sorted(set().union(*[set(stamps[c.name]) for c in cands if groups[c.name] == g])))
        split_at[g] = idx[int(len(idx) * split)] if len(idx) > 30 else None
    df = pd.DataFrame([candidate_stats(c, cost_bps, split_at[groups[c.name]], boot) for c in cands])
    df = finalize(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")]).sort_values("FDR補正p", na_position="last")
    return df.reset_index(drop=True)
