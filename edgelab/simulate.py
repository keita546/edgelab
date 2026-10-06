"""ルールどおりに売買していたら資金がどう動いたかを再現する。

ダッシュボードのシミュレーター(JavaScript)と同じ計算をする。
前提:
  - ルールの各トレードを古い順に処理する。決済(出口)は同じ日の新規より先に行う
  - 1回の投入額 = 基準額 × 投入比率。基準額は複利なら「現金+建玉(取得額)」、単利なら初期資金
  - 現金が足りなければ手持ち現金まで。100株単位なら株数を切り捨て、0株なら見送り
  - 損益 = 投入額 × (トレードの騰落率 − 往復コスト)
  - 評価は決済時のみ(保有中の含み損益は資産推移に反映しない)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SimResult:
    final: float
    executed: int
    skipped: int
    wins: int
    gross_win: float
    gross_loss: float
    max_dd: float
    max_losing_streak: int
    curve: list = field(default_factory=list)      # [(日付index, 資産)]
    trades: list = field(default_factory=list)     # dict のリスト
    yearly: dict = field(default_factory=dict)     # 年 -> 確定損益


def exit_index(e: int, basis: str, h: int) -> int:
    if basis == "c2c":
        return e + h
    if basis == "c2o":
        return e + 1
    return e                                       # o2c と分足は同日決済


def simulate(entries, rets, basis, h, dates, px, *, capital=1_000_000, frac=0.5, cost_bps=10.0,
             compound=True, lot=True, start=0, end=None) -> SimResult:
    end = len(dates) - 1 if end is None else end
    cost = cost_bps * 1e-4
    cash, opened = float(capital), []
    res = SimResult(capital, 0, 0, 0, 0.0, 0.0, 0.0, 0, [(start, float(capital))])
    peak, streak = float(capital), 0

    def close_until(t):
        nonlocal cash, peak, streak
        opened.sort(key=lambda o: o["x"])
        while opened and opened[0]["x"] <= t:
            o = opened.pop(0)
            pnl = o["alloc"] * o["net"]
            cash += o["alloc"] + pnl
            eq = cash + sum(p["alloc"] for p in opened)
            res.curve.append((o["x"], eq))
            peak = max(peak, eq)
            res.max_dd = min(res.max_dd, eq / peak - 1)
            res.yearly[dates[o["x"]][:4]] = res.yearly.get(dates[o["x"]][:4], 0.0) + pnl
            if pnl > 0:
                res.wins += 1; res.gross_win += pnl; streak = 0
            else:
                res.gross_loss += -pnl; streak += 1; res.max_losing_streak = max(res.max_losing_streak, streak)
            res.trades.append({"e": o["e"], "x": o["x"], "alloc": o["alloc"], "shares": o["shares"],
                               "net": o["net"], "pnl": pnl})

    for e, r in sorted(zip(entries, rets), key=lambda t: t[0]):
        x = exit_index(e, basis, h)
        if e < start or x > end or x >= len(dates):
            continue
        close_until(e)
        base = (cash + sum(o["alloc"] for o in opened)) if compound else capital
        alloc = min(base * frac, cash)
        shares = 0
        if lot:
            unit = px[e] * 100
            shares = int(alloc // unit) * 100 if unit > 0 else 0
            alloc = shares * px[e]
        if alloc <= 0:
            res.skipped += 1
            continue
        cash -= alloc
        opened.append({"e": e, "x": x, "alloc": alloc, "shares": shares, "net": r - cost})
        res.executed += 1
    close_until(math.inf)
    res.final = cash
    return res
