"""全仮説を同じ土俵で評価する。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import statlib as S
from .hypotheses import Candidate

BPS = 1e-4


def _naive(idx) -> pd.DatetimeIndex:
    """日足(tz無し)と分足(tz付き)の時刻を突き合わせ可能な形に揃える。"""
    di = pd.DatetimeIndex(pd.to_datetime(list(idx)))
    return di.tz_localize(None) if di.tz is not None else di


def evaluate(cands: list[Candidate], cost_bps: float = 10.0,
             split: float = 0.7, boot: int = 2000) -> pd.DataFrame:
    """候補ごとに指標を計算し、最後に多重検定補正をかける。"""
    if not cands:
        return pd.DataFrame()

    # IS/OOS 分割日はグループ(日足/分足)ごとに決める。
    # 日足15年分と分足2ヶ月分を混ぜて分割点を取ると、日足の OOS が数日しか残らない。
    stamps = {c.name: _naive(c.rets.index) for c in cands}
    groups = {c.name: ("分足" if c.family == "分足" else "日足") for c in cands}
    split_at: dict[str, pd.Timestamp | None] = {}
    for g in set(groups.values()):
        idx = pd.DatetimeIndex(sorted(set().union(
            *[set(stamps[c.name]) for c in cands if groups[c.name] == g])))
        split_at[g] = idx[int(len(idx) * split)] if len(idx) > 30 else None

    rows = []
    for c in cands:
        ts = stamps[c.name]
        grp = groups[c.name]
        cut = split_at[grp]
        gross = c.rets.to_numpy(dtype=float)
        net = gross - cost_bps * BPS          # 往復コストを毎トレード控除
        n = len(net)
        span_days = max((ts[-1] - ts[0]).days, 1) if n > 1 else 1
        per_year = n / (span_days / 365.25)

        t, p, _ = S.newey_west_t(net, lag=max(c.horizon - 1, 0) or None)
        lo, hi = S.stationary_bootstrap_ci(net, n_boot=boot, mean_block=max(c.horizon, 5))
        sd = net.std(ddof=1) if n > 2 else np.nan
        sr_trade = net.mean() / sd if sd else np.nan
        wt, wp = (S.welch(net, np.asarray(c.control) - cost_bps * BPS)
                  if c.control is not None else (np.nan, np.nan))

        if cut is not None:
            is_r = net[ts <= cut]
            oos_r = net[ts > cut]
        else:
            is_r = oos_r = np.array([])

        rows.append({
            "手法": c.name, "分類": c.family, "_grp": grp, "件数": n,
            "平均(bps)": net.mean() / BPS,
            "勝率%": float((net > 0).mean() * 100),
            "t値(HAC)": t, "p値": p,
            "CI下限(bps)": lo / BPS if np.isfinite(lo) else np.nan,
            "CI上限(bps)": hi / BPS if np.isfinite(hi) else np.nan,
            "対照群p": wp,
            "年率Sharpe": sr_trade * np.sqrt(per_year) if np.isfinite(sr_trade) else np.nan,
            "IS平均(bps)": is_r.mean() / BPS if len(is_r) > 5 else np.nan,
            "OOS平均(bps)": oos_r.mean() / BPS if len(oos_r) > 5 else np.nan,
            "最大DD%": S.max_drawdown(net) * 100,
            "損益分岐コスト(bps)": gross.mean() / BPS,
            "検出力(5bps)": S.power_for_mean(net, 5 * BPS),
            "年間取引数": per_year,
            "_sr": sr_trade, "_net": net,
        })

    df = pd.DataFrame(rows)

    # --- 多重検定補正: 「何十個も試した」という事実を織り込む ---
    df["FDR補正p"] = S.benjamini_hochberg(df["p値"].to_numpy())
    df["Bonferroni p"] = np.clip(df["p値"] * len(df), 0, 1)

    # DSR は「同じ観測頻度の試行群」の中で計算する。
    # 日足トレードと30分トレードは1件あたりのシャープの尺度が違うため、
    # 混ぜると試行分散が膨らんで全候補の DSR が 0 に潰れる。
    df["DSR"] = np.nan
    for g, sub in df.groupby("_grp"):
        n_trials = len(sub)
        var_sr = float(np.nanvar(sub["_sr"].to_numpy(dtype=float), ddof=1)) if n_trials > 1 else 0.0
        df.loc[sub.index, "DSR"] = [S.deflated_sharpe(r["_net"], n_trials, var_sr)[1]
                                    for _, r in sub.iterrows()]

    # 総合判定
    def verdict(r):
        if not np.isfinite(r["p値"]):
            return "判定不能"
        pos = r["平均(bps)"] > 0
        fdr_ok = r["FDR補正p"] < 0.05
        if fdr_ok and not pos:
            return "有意にマイナス"          # 逆張りすれば勝てる、という意味ではない(コスト対称でない)
        if fdr_ok and pos:
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

    df["判定"] = df.apply(verdict, axis=1)
    df = df.drop(columns=["_sr", "_net", "_grp"]).sort_values("FDR補正p", na_position="last")
    return df.reset_index(drop=True)
