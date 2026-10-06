#!/usr/bin/env python
"""複数銘柄を検定し、ダッシュボード用の JSON を書き出す。

  python export.py                       # 既定の銘柄セット
  python export.py 7203.T 6758.T ^N225   # 銘柄を指定
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab import data, engine, hypotheses as H
from edgelab.rules_doc import describe

ROOT = Path(__file__).parent
DEFAULT = [
    ("7203.T", "トヨタ自動車"), ("6758.T", "ソニーグループ"), ("9984.T", "ソフトバンクグループ"),
    ("8306.T", "三菱UFJ FG"), ("6861.T", "キーエンス"), ("^N225", "日経平均"),
]
COST = {"^N225": 5.0}          # 指数は先物想定でコストを低めに
YEARS = {"^N225": 25.0}


def _unused_curve(ts, net, pts: int = 260) -> list:
    """累積リターン(単利, %)を日付順に間引いて返す。"""
    s = pd.Series(net, index=ts).sort_index()
    s = s.groupby(level=0).sum().cumsum() * 100
    if len(s) > pts:
        s = s.iloc[np.linspace(0, len(s) - 1, pts).astype(int)]
    return [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in s.items()]


def _clean(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 5)
    return v


def run_one(sym: str, label: str) -> dict:
    cost = COST.get(sym, 10.0)
    years = YEARS.get(sym, 15.0)
    print(f"■ {sym} {label}  cost={cost}bps", flush=True)
    d = data.fetch(sym, "1d", years=years)
    cands = H.daily_hypotheses(H.daily_features(d))
    sessions = 0
    if not sym.startswith("^"):
        try:
            bars = data.fetch(sym, "5m")
            t = H.session_table(bars)
            sessions = len(t)
            cands += H.intraday_hypotheses(bars, t)
        except Exception as e:
            print("  分足スキップ:", e)
    res = engine.evaluate(cands, cost_bps=cost, boot=2000)

    by = {c.name: c for c in cands}
    raw = data.fetch(sym, "1d", years=years, adjust=False)      # 100株単位の必要資金は分割調整のみの株価で
    raw = raw.reindex(d.index).ffill()
    dates = [x.strftime("%Y-%m-%d") for x in d.index]
    pos = {x: i for i, x in enumerate(dates)}
    rows, trades = [], {}
    for _, r in res.iterrows():
        c = by[r["手法"]]
        intra = c.family == "分足"
        ts = engine._naive(c.rets.index)
        idx = [pos.get(t.strftime("%Y-%m-%d")) for t in ts]
        keep = [k for k, v in enumerate(idx) if v is not None]
        trades[c.name] = {
            "basis": "intra" if intra else c.meta.get("basis", "c2c"),
            "h": 0 if intra else int(c.horizon),
            "e": [idx[k] for k in keep],
            "r": [round(float(c.rets.iloc[k]), 5) for k in keep],
        }
        rows.append({
            "name": r["手法"], "family": r["分類"], "n": int(r["件数"]), "doc": describe(c.name),
            "mean": _clean(r["平均(bps)"]), "lo": _clean(r["CI下限(bps)"]), "hi": _clean(r["CI上限(bps)"]),
            "win": _clean(r["勝率%"]), "t": _clean(r["t値(HAC)"]), "p": _clean(r["p値"]),
            "fdr": _clean(r["FDR補正p"]), "dsr": _clean(r["DSR"]), "ctrl": _clean(r["対照群p"]),
            "sharpe": _clean(r["年率Sharpe"]), "is": _clean(r["IS平均(bps)"]), "oos": _clean(r["OOS平均(bps)"]),
            "dd": _clean(r["最大DD%"]), "gross": _clean(r["損益分岐コスト(bps)"]),
            "power": _clean(r["検出力(5bps)"]), "perYear": _clean(r["年間取引数"]),
            "verdict": r["判定"],
        })

    return {
        "symbol": sym, "label": label, "cost": cost,
        "start": dates[0], "end": dates[-1],
        "bars": len(d), "sessions5m": sessions,
        "split": dates[int(len(d) * 0.7)],
        "dates": dates,
        "px": [round(float(v), 1) for v in raw["close"]],     # 分割調整のみ(100株単位の計算用)
        "tr": [round(float(v), 2) for v in d["close"]],       # 配当込み(持ち続けた場合の比較用)
        "rows": rows, "trades": trades,
    }


def main() -> int:
    args = sys.argv[1:]
    syms = [(a, a) for a in args] if args else DEFAULT
    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "selftest": {"before": {"rate": 12.649, "fdr": 55, "promising": 48},
                     "after": {"rate": 5.804, "fdr": 1, "promising": 0}, "tests": 672, "sims": 24},
        "symbols": [run_one(s, l) for s, l in syms],
    }
    p = ROOT / "out" / "dashboard.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"書き出し: {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
