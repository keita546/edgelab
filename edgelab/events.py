"""予想画面のシナリオに使う外部の動き: 日経平均、ドル円、決算発表日。

決算発表日は1銘柄ずつ取るため時間がかかる(全銘柄で数分〜十数分)。
cache/earnings/ に1週間保存し、それより古いものだけ取り直す。
小型株は Yahoo に決算日がないことが多く、その場合は空のリストになる。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import yfinance as yf

from .data import CACHE_DIR, fetch

EARN_DIR = CACHE_DIR / "earnings"
EARN_TTL = 7 * 86400


def _one(sym: str) -> list[str]:
    for attempt in range(3):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=60)
            if ed is None or len(ed) == 0:
                return []
            return sorted({x.strftime("%Y-%m-%d") for x in ed.index})
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def earnings_dates(symbols: list[str], workers: int = 4, log=print) -> dict[str, list[str]]:
    """銘柄 → 決算発表日(過去と予定)のリスト。"""
    EARN_DIR.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for s in symbols:
        f = EARN_DIR / f"{s}.json"
        if f.exists() and time.time() - f.stat().st_mtime < EARN_TTL:
            out[s] = json.loads(f.read_text())
        else:
            todo.append(s)
    if todo:
        log(f"  決算発表日を取得: {len(todo)} 銘柄（キャッシュ済み {len(out)}）")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, (s, dates) in enumerate(zip(todo, ex.map(_one, todo)), 1):
                (EARN_DIR / f"{s}.json").write_text(json.dumps(dates))
                out[s] = dates
                if k % 500 == 0:
                    log(f"    {k}/{len(todo)}")
    have = sum(1 for v in out.values() if v)
    log(f"  決算発表日あり: {have} / {len(out)} 銘柄")
    return out


def market_series(years: float = 15.0, refresh: bool = False):
    """(日経平均の終値, ドル円) を返す。"""
    n225 = fetch("^N225", "1d", years=years, refresh=refresh)["close"]
    fx = fetch("JPY=X", "1d", years=years, refresh=refresh)["close"]
    return n225, fx
