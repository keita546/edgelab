#!/usr/bin/env python
"""ルールどおりに売買していたら資金がどうなったかを計算する。

  例:  python sim.py 6758.T "逆張り:RSI2<10→5日保有" --capital 3000000 --frac 50
       python sim.py 6758.T RSI2\\<10→5日 --no-lot          (ルール名は部分一致でよい)
"""
from __future__ import annotations

import argparse
import sys

from edgelab import data, hypotheses as H
from edgelab.simulate import simulate


def main() -> int:
    ap = argparse.ArgumentParser(description="ルールの資金シミュレーション")
    ap.add_argument("symbol")
    ap.add_argument("rule", help="ルール名(部分一致)")
    ap.add_argument("--capital", type=float, default=3_000_000, help="初期資金(円)")
    ap.add_argument("--frac", type=float, default=50, help="1回の投入比率(%%)")
    ap.add_argument("--cost", type=float, default=10, help="往復コスト(bps)")
    ap.add_argument("--simple", action="store_true", help="単利(毎回 初期資金×比率 を投入)")
    ap.add_argument("--no-lot", action="store_true", help="100株単位にしない(端数OK)")
    ap.add_argument("--years", type=float, default=15)
    a = ap.parse_args()

    d = data.fetch(a.symbol, "1d", years=a.years)
    raw = data.fetch(a.symbol, "1d", years=a.years, adjust=False).reindex(d.index).ffill()
    cands = H.daily_hypotheses(H.daily_features(d))
    hit = [c for c in cands if a.rule in c.name]
    if not hit:
        print("該当ルールなし。候補:\n  " + "\n  ".join(c.name for c in cands)); return 1
    c = hit[0]
    dates = [x.strftime("%Y-%m-%d") for x in d.index]
    pos = {x: i for i, x in enumerate(dates)}
    e = [pos[t.strftime("%Y-%m-%d")] for t in c.rets.index]
    r = simulate(e, list(c.rets.values), c.meta["basis"], c.horizon, dates, list(raw["close"].values),
                 capital=a.capital, frac=a.frac / 100, cost_bps=a.cost, compound=not a.simple, lot=not a.no_lot)

    yrs = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (r.final / a.capital) ** (1 / yrs) - 1 if r.final > 0 else -1
    bh = a.capital * d["close"].iloc[-1] / d["close"].iloc[0]
    pf = r.gross_win / r.gross_loss if r.gross_loss else float("inf")
    print(f"\n{a.symbol}  {c.name}")
    print(f"  初期資金 {a.capital:,.0f}円  投入比率 {a.frac:.0f}%  コスト {a.cost}bps  "
          f"{'単利' if a.simple else '複利'}  {'端数可' if a.no_lot else '100株単位'}")
    print(f"  最終資産   {r.final:,.0f}円  ({r.final / a.capital - 1:+.1%}, 年率 {cagr:+.1%})")
    print(f"  持ち続け   {bh:,.0f}円  ({bh / a.capital - 1:+.1%})")
    print(f"  最大下落   {r.max_dd:.1%}   最大連敗 {r.max_losing_streak}回")
    print(f"  取引       {r.executed}回 (資金不足で見送り {r.skipped}回)  勝率 {r.wins / max(r.executed, 1):.1%}  PF {pf:.2f}")
    print("  年別損益   " + "  ".join(f"{y}:{v / 1e4:+,.0f}万" for y, v in sorted(r.yearly.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
