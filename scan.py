#!/usr/bin/env python
"""銘柄集合をまとめて検定し、ダッシュボード用のファイルを書き出す。

  python scan.py --universe topix100          # 主要100銘柄(シミュレーター用データも出す)
  python scan.py --universe all               # 東証の内国株 全銘柄(約3,700)
  python scan.py --universe prime --workers 6

書き出し先: out/universe_<名前>/ (summary.json, detail_*.json, sim/*.json)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab import data, events, scan, universe
from edgelab.hypotheses import daily_features, daily_hypotheses
from edgelab.rules_doc import cond_label, describe

ROOT = Path(__file__).parent
VCODE = {"有望(要追試)": "P", "FDR通過・DSR不足": "N", "IS/OOS不一致": "N", "FDR通過・OOS不足": "N",
         "補正で消滅": "G", "優位性なし": "-", "有意にマイナス": "M", "判定不能": "-"}
DETAIL_COLS = ["件数", "平均(bps)", "CI下限(bps)", "CI上限(bps)", "勝率%", "t値(HAC)", "p値", "FDR補正p", "DSR",
               "対照群p", "年率Sharpe", "IS平均(bps)", "OOS平均(bps)", "最大DD%", "損益分岐コスト(bps)", "検出力(5bps)", "年間取引数"]


def num(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), nd)


def sig_round(v):
    """株価を有効数字6桁程度に丸める(低位株でも誤差が1bpsを超えないように)。"""
    if not math.isfinite(v) or v <= 0:
        return None
    return round(v, max(0, 5 - int(math.floor(math.log10(v)))))


def delta(ix):
    out, prev = [], -1
    for i in ix:
        out.append(int(i - prev)); prev = i
    return out


def write_sim(sym, cal_pos, outdir):
    d = scan.load_cached(sym)
    raw = data.fetch(sym, "1d", adjust=False) if False else None  # 下で pickle を直接読む
    with open(data.CACHE_DIR / f"{sym}_1d.pkl", "rb") as f:
        raw = pickle_load(f).reindex(d.index).ffill()
    cands = daily_hypotheses(daily_features(d))
    di = [cal_pos.get(x.strftime("%Y-%m-%d")) for x in d.index]
    keep = [k for k, v in enumerate(di) if v is not None]
    d, raw = d.iloc[keep], raw.iloc[keep]
    di = [di[k] for k in keep]
    local = {x: i for i, x in enumerate(d.index)}
    rules = {}
    for c in cands:
        ix = [local[t] for t in c.rets.index if t in local]
        full = len(ix) >= len(d) - 2
        rules[c.name] = {"b": c.meta["basis"], "h": int(c.horizon), "s": int(c.meta["sign"]),
                         "e": "all" if full else delta(ix)}
    obj = {"di": delta(di), "o": [sig_round(v) for v in d["open"]], "c": [sig_round(v) for v in d["close"]],
           "px": [sig_round(v) for v in raw["close"]], "rules": rules}
    (outdir / f"{sym.replace('.', '_')}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def pickle_load(f):
    import pickle
    return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="銘柄集合をまとめて検定する")
    ap.add_argument("--universe", default="topix100", choices=list(universe.UNIVERSES))
    ap.add_argument("--years", type=float, default=15)
    ap.add_argument("--cost", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=0, help="銘柄別の信頼区間のブートストラップ回数(0で省略、多いと遅い)")
    ap.add_argument("--sim", choices=["auto", "yes", "no"], default="auto", help="シミュレーター用データ(既定: 150銘柄以下なら出す)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-earnings", action="store_true", help="決算発表日を取得しない(速いがシナリオに決算が出ない)")
    a = ap.parse_args()
    t0 = time.time()

    u = universe.select(a.universe, refresh=a.refresh)
    print(f"■ {universe.UNIVERSES[a.universe]}: {len(u)} 銘柄（JPX 一覧 {u['asof'].iloc[0]} 時点）")
    got = data.fetch_many(list(u.symbol), years=a.years, refresh=a.refresh)
    ref = data.fetch("^N225", "1d", years=a.years)
    calendar = sorted(set().union(*[set(x.strftime("%Y-%m-%d") for x in df.index) for df in got.values()]))
    cal_start = ref.index[0].strftime("%Y-%m-%d")
    calendar = [x for x in calendar if x >= cal_start]
    syms, excluded, trimmed = [], {}, 0
    for s in u.symbol:
        d, why = scan.load_cached(s, with_reason=True)
        if d is None:
            excluded[why] = excluded.get(why, 0) + 1
            continue
        trimmed += bool(why)
        syms.append(s)
    print(f"  検定する銘柄 {len(syms)} / {len(u)}　営業日 {len(calendar)}（{calendar[0]}〜{calendar[-1]}）")
    print(f"  除外: {excluded}　不連続以降のみ使用: {trimmed} 銘柄")

    ext = events.market_series(years=a.years, refresh=a.refresh)
    earn = events.earnings_dates(syms) if not a.no_earnings else {}
    df, pooled, meta, fans = scan.run(syms, calendar, cost=a.cost, boot=a.boot, workers=a.workers, ext=ext, earn=earn)
    names = list(meta)
    print(f"  検定 {len(df):,} 通り　補正後 p<0.05 かつプラス: {int(((df['FDR補正p'] < .05) & (df['平均(bps)'] > 0)).sum())}　"
          f"全関門: {int((df['判定'] == '有望(要追試)').sum())}　({time.time() - t0:.0f}秒)")

    out = ROOT / "out" / f"universe_{a.universe}"
    if out.exists():
        shutil.rmtree(out)
    (out / "sim").mkdir(parents=True)

    # 銘柄ごとの行(ルール順)
    info = u.set_index("symbol")
    df["ri"] = df["手法"].map({n: i for i, n in enumerate(names)})
    by_sym = {s: g.sort_values("ri") for s, g in df.groupby("symbol")}
    want_sim = a.sim == "yes" or (a.sim == "auto" and len(syms) <= 150)
    cal_pos = {x: i for i, x in enumerate(calendar)}

    stocks, chunks, fchunks = [], {}, {}
    for s in syms:
        if s not in by_sym:
            continue
        g = by_sym[s]
        code = info.loc[s, "code"]
        k = code[0]                                   # 詳細ファイルは証券コードの先頭1文字で分割
        chunks.setdefault(k, {})[code] = [[num(v) for v in row] for row in g[DETAIL_COLS].to_numpy()]
        mean = [int(round(v)) if math.isfinite(v) else None for v in g["平均(bps)"]]
        fan = fans.get(s)
        stocks.append({"c": code, "n": info.loc[s, "name"], "s": info.loc[s, "sector"], "m": info.loc[s, "market"],
                       "z": info.loc[s, "size"], "v": "".join(VCODE.get(v, "-") for v in g["判定"]), "mu": mean,
                       "k": k, "k2": code[:2], "sim": bool(want_sim), "t": fan["t"] if fan else "",
                       "p": fan["p"] if fan else None, "ne": fan["ne"] if fan else None})
        if fan:
            fchunks.setdefault(code[:2], {})[code] = fan
    for k, obj in chunks.items():
        (out / f"detail_{k}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    for k, obj in fchunks.items():
        (out / f"fwd_{k}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    if want_sim:
        for i, s in enumerate(syms):
            write_sim(s, cal_pos, out / "sim")
        print(f"  シミュレーター用データ {len(syms)} 銘柄")

    # 銘柄ごとの集計(ルール別の「プラスの銘柄の割合」「全関門の銘柄数」)
    br = df.groupby("手法").agg(pos=("平均(bps)", lambda x: float((x > 0).mean() * 100)),
                               passN=("判定", lambda x: int((x == "有望(要追試)").sum())),
                               fdrN=("FDR補正p", lambda x: int((x < .05).sum())))
    rules = []
    for _, r in pooled.iterrows():
        nm = r["name"]
        rules.append({k: (num(r[k]) if isinstance(r[k], (float, np.floating)) else r[k]) for k in
                      ["name", "family", "h", "basis", "days", "trades", "avgStocks", "mean", "t", "p", "fdr", "ex", "te", "pe", "fdrE", "ex2", "fdrE2",
                       "isM", "oosM", "isE", "oosE", "win", "dsr", "verdict", "curve"]}
                     | {"doc": describe(nm), "pos": num(br.loc[nm, "pos"]), "passN": int(br.loc[nm, "passN"]),
                        "fdrPosN": int(((df["手法"] == nm) & (df["FDR補正p"] < .05) & (df["平均(bps)"] > 0)).sum())})
    order = {n: i for i, n in enumerate(names)}
    rules.sort(key=lambda r: order[r["name"]])
    summary = {
        "universe": a.universe, "label": universe.UNIVERSES[a.universe], "asof": u["asof"].iloc[0],
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "cost": a.cost, "nListed": len(u), "nStocks": len(stocks),
        "calendar": calendar, "split": calendar[int(len(calendar) * 0.7)],
        "excluded": excluded, "trimmed": trimmed,
        "tests": int(len(df)), "plus": int((df["平均(bps)"] > 0).sum()),
        "raw": int(((df["p値"] < .05) & (df["平均(bps)"] > 0)).sum()),
        "fdr": int(((df["FDR補正p"] < .05) & (df["平均(bps)"] > 0)).sum()),
        "pass": int((df["判定"] == "有望(要追試)").sum()),
        "rules": rules, "stocks": stocks, "detailCols": DETAIL_COLS,
        "fanH": scan.FAN_H, "fanQ": scan.FAN_Q, "lastDate": calendar[-1],
        "conds": [{"i": i, "label": cond_label(n)} for i, n in enumerate(names) if cond_label(n)],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    size = sum(p.stat().st_size for p in out.rglob("*.json")) / 1e6
    print(f"書き出し: {out.relative_to(ROOT)}  ({size:.1f} MB, {time.time() - t0:.0f}秒)")
    print("\n市場全体の判定:")
    for r in sorted(rules, key=lambda r: -(r["te"] if r["te"] is not None else -99))[:8]:
        fm = lambda v: "—" if v is None else f"{v:+.1f}"
        print(f"  {r['verdict']:<10} {r['name']:<28} 平均{fm(r['mean'])}bps  他銘柄差{fm(r['ex'])}bps  "
              f"1日遅れ差{fm(r['ex2'])}bps  FDR p={r['fdr']}/{r['fdrE']}/{r['fdrE2']}  プラス銘柄{r['pos']:.0f}%  全関門{r['passN']}銘柄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
