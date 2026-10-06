"""銘柄集合(全銘柄・TOPIX100 など)をまとめて検定する。

2種類の答えを出す:
  1. 銘柄別: 全銘柄×全ルールを1つの検定群として多重検定補正する
     (3,700銘柄×28ルール ≈ 10万通り試した、という事実をそのまま織り込む)
  2. 市場全体: ルールごとに「同じ日にシグナルが出た銘柄の平均」を日次で並べて検定する。
     同じ日の銘柄どうしは連動するので、銘柄を独立に数えると有意に出すぎる。
     日ごとに1つの値へ畳んでから Newey-West で検定するのはそのため。
     さらに「同じ日にシグナルが出なかった銘柄」との差も検定し、相場全体の上げ下げを除く。
"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import engine, statlib as S
from .data import CACHE_DIR, back_adjust, sanitize
from .hypotheses import daily_features, daily_hypotheses

BPS = 1e-4


def load_cached(symbol: str, with_reason: bool = False):
    """キャッシュから読み、配当調整とデータの掃除をした日足を返す(使えなければ None)。"""
    key = CACHE_DIR / f"{symbol.replace('/', '_')}_1d.pkl"
    if not key.exists():
        return (None, "未取得") if with_reason else None
    with open(key, "rb") as f:
        d, reason = sanitize(back_adjust(pickle.load(f)))
    return (d, reason) if with_reason else d


def _scan_chunk(args):
    """ワーカー: 銘柄のかたまりを検定し、行と日次集計を返す。"""
    symbols, cal, cost, split, boot, rule_names = args
    frames = ((s, load_cached(s)) for s in symbols)
    return scan_frames(frames, cal, cost, split, boot, rule_names)


def scan_frames(frames, cal, cost, split, boot, rule_names):
    """(銘柄, 日足) の並びを検定する。較正テストでは乱数データをここに直接渡す。"""
    pos = pd.Index(pd.to_datetime(cal))
    N = len(cal)
    # 0-3: シグナル和, 件数, 対照和, 件数 / 4-7: 同じものを「1日遅れで入った場合」で
    agg = {nm: np.zeros((8, N)) for nm in rule_names}
    rows = []
    for sym, d in frames:
        if d is None or len(d) < 400:
            continue
        cands = daily_hypotheses(daily_features(d))
        cut = d.index[int(len(d) * split)]
        for c in cands:
            st = engine.candidate_stats(c, cost, cut, boot)
            st["symbol"] = sym
            rows.append(st)
            if c.name not in agg:
                continue
            ix = pos.get_indexer(engine._naive(c.rets.index))
            ok = ix >= 0
            np.add.at(agg[c.name][0], ix[ok], c.rets.to_numpy(float)[ok] - cost * BPS)
            np.add.at(agg[c.name][1], ix[ok], 1)
            if isinstance(c.control, pd.Series):
                jx = pos.get_indexer(engine._naive(c.control.index))
                ok2 = jx >= 0
                np.add.at(agg[c.name][2], jx[ok2], c.control.to_numpy(float)[ok2] - cost * BPS)
                np.add.at(agg[c.name][3], jx[ok2], 1)
            # 1日遅れで入る版(引けで入るルールのみ)。終値が気配の間を行き来するだけの
            # 見かけの戻りは、1日遅らせると消える。
            if c.meta.get("basis") == "c2c" and isinstance(c.control, pd.Series):
                close = d["close"].to_numpy(float)
                sign, h = c.meta["sign"], c.horizon
                for src, row_s, row_n in ((c.rets.index, 4, 5), (c.control.index, 6, 7)):
                    li = d.index.get_indexer(src)
                    li = li[(li >= 0) & (li + 1 + h < len(close))]
                    rr = sign * (close[li + 1 + h] / close[li + 1] - 1.0) - cost * BPS
                    gx = pos.get_indexer(d.index[li])
                    ok3 = gx >= 0
                    np.add.at(agg[c.name][row_s], gx[ok3], rr[ok3])
                    np.add.at(agg[c.name][row_n], gx[ok3], 1)
    return rows, agg


def run(symbols: list[str], calendar: list[str], cost: float = 10.0, split: float = 0.7,
        boot: int = 0, workers: int = 8, chunk: int = 40, log=print):
    # ルール名とその保有日数・測り方(先頭の銘柄から取得)
    probe = next((load_cached(s) for s in symbols if load_cached(s) is not None), None)
    proto = daily_hypotheses(daily_features(probe))
    meta = {c.name: (c.family, c.horizon, c.meta.get("basis"), c.meta.get("sign")) for c in proto}
    names = list(meta)

    tasks = [(symbols[i:i + chunk], calendar, cost, split, boot, names) for i in range(0, len(symbols), chunk)]
    rows, total = [], {nm: np.zeros((8, len(calendar))) for nm in names}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (r, agg) in enumerate(ex.map(_scan_chunk, tasks), 1):
            rows.extend(r)
            for nm in names:
                total[nm] += agg[nm]
            log(f"  検定 {min(k * chunk, len(symbols))}/{len(symbols)} 銘柄")

    df = engine.finalize(pd.DataFrame(rows))
    pooled = pool_rules(total, meta, calendar, split)
    return df, pooled, meta


def pool_rules(total: dict, meta: dict, calendar: list[str], split: float) -> pd.DataFrame:
    """ルールごとの市場全体の検定。"""
    cal = pd.to_datetime(calendar)
    cut = int(len(cal) * split)
    out = []
    for nm, a in total.items():
        fam, h, basis, sign = meta[nm]
        has = a[1] > 0
        m = np.where(has, a[0] / np.where(has, a[1], 1), np.nan)          # その日シグナルが出た銘柄の平均(コスト後)
        both = has & (a[3] > 0)
        ctl = np.where(a[3] > 0, a[2] / np.where(a[3] > 0, a[3], 1), np.nan)
        ex = np.where(both, m - ctl, np.nan)                               # 同じ日の非シグナル銘柄との差
        lag = max(h - 1, 0) or None
        # 1日遅れで入った場合の「同じ日の他銘柄との差」
        has2, ctl2_ok = a[5] > 0, a[7] > 0
        both2 = has2 & ctl2_ok
        ex2 = np.where(both2, a[4] / np.where(has2, a[5], 1) - a[6] / np.where(ctl2_ok, a[7], 1), np.nan)
        te2, pe2, _ = S.newey_west_t(ex2[both2], lag=lag) if both2.sum() > 20 else (np.nan, np.nan, np.nan)
        mv = m[has]
        t, p, _ = S.newey_west_t(mv, lag=lag)
        te, pe, _ = S.newey_west_t(ex[both], lag=lag) if both.sum() > 20 else (np.nan, np.nan, np.nan)
        sr = mv.mean() / mv.std(ddof=1) if len(mv) > 2 and mv.std(ddof=1) > 0 else np.nan
        is_m = m[:cut][has[:cut]]
        oos_m = m[cut:][has[cut:]]
        is_e = ex[:cut][both[:cut]]
        oos_e = ex[cut:][both[cut:]]
        # 概算の資金曲線: 毎日シグナル銘柄に均等配分、保有日数ぶんの資金を順番に回す
        curve_v = np.cumsum(np.where(has, m, 0.0)) / max(h, 1) * 100
        out.append({
            "name": nm, "family": fam, "h": h, "basis": basis,
            "days": int(has.sum()), "trades": int(a[1].sum()), "avgStocks": float(a[1][has].mean()) if has.any() else 0,
            "mean": float(mv.mean() / BPS) if len(mv) else np.nan, "t": t, "p": p,
            "ex": float(np.nanmean(ex[both]) / BPS) if both.any() else np.nan, "te": te, "pe": pe,
            "ex2": float(np.nanmean(ex2[both2]) / BPS) if both2.any() else np.nan, "pe2": pe2,
            "isM": float(is_m.mean() / BPS) if len(is_m) else np.nan, "oosM": float(oos_m.mean() / BPS) if len(oos_m) else np.nan,
            "isE": float(is_e.mean() / BPS) if len(is_e) else np.nan, "oosE": float(oos_e.mean() / BPS) if len(oos_e) else np.nan,
            "win": float((mv > 0).mean() * 100) if len(mv) else np.nan,
            "_sr": sr, "_T": len(mv),
            "_g3": float(pd.Series(mv).skew()) if len(mv) > 3 else np.nan,
            "_g4": float(pd.Series(mv).kurt() + 3) if len(mv) > 3 else np.nan,
            "curve": [[calendar[i], round(float(curve_v[i]), 2)] for i in np.linspace(0, len(cal) - 1, 300).astype(int)],
        })
    pdf = pd.DataFrame(out)
    pdf["fdr"] = S.benjamini_hochberg(pdf["p"].to_numpy())
    pdf["fdrE"] = S.benjamini_hochberg(pdf["pe"].to_numpy())
    pdf["fdrE2"] = S.benjamini_hochberg(pdf["pe2"].to_numpy())
    var_sr = float(np.nanvar(pdf["_sr"].to_numpy(float), ddof=1))
    pdf["dsr"] = [S.deflated_sharpe_m(r["_sr"], int(r["_T"]), r["_g3"], r["_g4"], len(pdf), var_sr) for _, r in pdf.iterrows()]

    def v(r):
        abs_ok = r["fdr"] < 0.05 and r["mean"] > 0
        ex_ok = np.isfinite(r["fdrE"]) and r["fdrE"] < 0.05 and r["ex"] > 0
        same = (np.isfinite(r["isE"]) and np.isfinite(r["oosE"]) and r["isE"] > 0 and r["oosE"] > 0) if np.isfinite(r["ex"]) \
            else (r["isM"] > 0 and r["oosM"] > 0)
        # 差が有意でも、1日遅らせると消えるなら約定できない見かけの効果の疑い
        delay_ok = (not np.isfinite(r["ex2"])) or (np.isfinite(r["fdrE2"]) and r["fdrE2"] < 0.05 and r["ex2"] > 0)
        if abs_ok and (ex_ok or not np.isfinite(r["ex"])) and same and r["dsr"] > 0.95 and (delay_ok or not ex_ok):
            return "市場全体で有効"
        if ex_ok and same and not delay_ok:
            return "1日遅らせると消える"
        if ex_ok and same:
            return "他銘柄より優位"
        if abs_ok or ex_ok:
            return "一部の関門のみ"
        if (r["fdr"] < 0.05 and r["mean"] < 0) or (np.isfinite(r["fdrE"]) and r["fdrE"] < 0.05 and r["ex"] < 0):
            return "有意にマイナス"
        return "優位性なし"

    pdf["verdict"] = pdf.apply(v, axis=1)
    return pdf
