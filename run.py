#!/usr/bin/env python
"""統計的優位性スキャナ

  例:  python run.py --symbol 7203.T --intraday 5m --cost 10
       python run.py --symbol ^N225 --years 25 --cost 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from edgelab import data, engine, hypotheses as H, report

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="株価データから統計的優位性のある手法を探す")
    ap.add_argument("--symbol", default="7203.T", help="銘柄コード (例: 7203.T, ^N225, SPY)")
    ap.add_argument("--intraday", default="5m", choices=["1m", "5m", "15m", "30m", "none"])
    ap.add_argument("--years", type=float, default=15.0, help="日足の遡及年数")
    ap.add_argument("--cost", type=float, default=10.0, help="往復コスト(bps)。10 = 0.10%%")
    ap.add_argument("--split", type=float, default=0.7, help="IS/OOS 分割点")
    ap.add_argument("--boot", type=int, default=2000, help="ブートストラップ回数")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを無視して再取得")
    a = ap.parse_args()

    print(f"\n銘柄: {a.symbol} / 往復コスト: {a.cost}bps / IS:OOS = {a.split:.0%}:{1-a.split:.0%}")

    # ---- 日足 ----
    print("\n日足データ取得中 ...")
    d = data.fetch(a.symbol, "1d", years=a.years, refresh=a.refresh)
    print(f"  {len(d)} 本  {d.index.min().date()} 〜 {d.index.max().date()}")
    feats = H.daily_features(d)
    cands = H.daily_hypotheses(feats)
    print(f"  日足の仮説: {len(cands)} 件")

    # ---- 分足 ----
    intr = []
    bars = None
    if a.intraday != "none":
        print(f"\n{a.intraday} 足データ取得中 ...")
        try:
            bars = data.fetch(a.symbol, a.intraday, refresh=a.refresh)
            t = H.session_table(bars)
            print(f"  {len(bars)} 本 / {len(t)} セッション  "
                  f"{bars.index.min().date()} 〜 {bars.index.max().date()}")
            intr = H.intraday_hypotheses(bars, t)
            print(f"  分足の仮説: {len(intr)} 件")
        except Exception as e:
            print(f"  取得失敗({e}) — 日足のみで続行します")

    allc = cands + intr
    if not allc:
        print("検定できる仮説がありません"); return 1

    print(f"\n検定中 ... 仮説 {len(allc)} 件 × ブートストラップ {a.boot} 回")
    res = engine.evaluate(allc, cost_bps=a.cost, split=a.split, boot=a.boot)

    dly = res[res["分類"] != "分足"]
    itd = res[res["分類"] == "分足"]
    print(report.console(dly, f"{a.symbol} 日足 — FDR補正p の昇順"))
    if len(itd):
        print(report.console(itd, f"{a.symbol} {a.intraday}足 — FDR補正p の昇順"))
    print(report.summary(res, a.cost))

    if len(itd):
        med = itd["検出力(5bps)"].median()
        print(f"注) 分足の検出力(5bps の優位性を捉えられる確率)の中央値 = {med:.2f}")
        if med < 0.5:
            print("    yfinance の分足は遡及期間が短く(1m:30日/5m:60日)、標本が足りません。")
            print("    分足側の『優位性なし』は『無い』ではなく『判断できない』と読むべきです。\n")

    csv = OUT / f"{a.symbol.replace('^','').replace('.','_')}_{a.intraday}.csv"
    res.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"詳細CSV: {csv.relative_to(Path(__file__).parent)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
