"""株価データの取得とローカルキャッシュ。

yfinance の足種ごとの取得可能期間の制約をここで吸収する:
  1m  : 直近30日まで(1リクエストにつき7日) -> 7日窓に分割して結合
  5m  : 直近60日まで
  1d  : 数十年
"""
from __future__ import annotations

import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 足種 -> (yfinance が許す最大遡及日数, キャッシュ有効秒数)
LIMITS = {
    "1m": (30, 3600),
    "2m": (60, 3600),
    "5m": (60, 3600),
    "15m": (60, 3600),
    "30m": (60, 3600),
    "60m": (730, 3600),
    "1h": (730, 3600),
    "1d": (None, 43200),
}


def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance が返す MultiIndex 列を単層に落とす。"""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = lvl0
    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "adj close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def _download(symbol: str, interval: str, start=None, end=None, period=None) -> pd.DataFrame:
    kw = dict(interval=interval, progress=False, auto_adjust=False, actions=False)
    if period:
        kw["period"] = period
    else:
        kw["start"] = start
        kw["end"] = end
    df = yf.download(symbol, **kw)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return _flatten(df, symbol)


def back_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """配当落ちを遡及調整(トータルリターン化)。

    OHLC 全てに同じ日次係数 adj_close/close を掛けるため、日中の値位置関係
    (寄り/引け/高値安値の比率)は保たれたまま、配当落ちの下窓だけが消える。
    無調整のままだと権利落ち日が偽の「下落シグナル」として混入する。
    """
    if "adj close" not in df.columns:
        return df
    f = (df["adj close"] / df["close"]).replace([float("inf")], 1.0).fillna(1.0)
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * f
    return out


def fetch(symbol: str, interval: str = "1d", years: float = 15.0,
          refresh: bool = False, adjust: bool = True) -> pd.DataFrame:
    """OHLCV を取得する。intraday はチャンク分割して可能な限り遡る。"""
    if interval not in LIMITS:
        raise ValueError(f"未対応の足種: {interval}")
    key = CACHE_DIR / f"{symbol.replace('/', '_')}_{interval}.pkl"
    ttl = LIMITS[interval][1]
    if key.exists() and not refresh and (time.time() - key.stat().st_mtime) < ttl:
        with open(key, "rb") as f:
            return back_adjust(pickle.load(f)) if adjust else pickle.load(f)

    max_days = LIMITS[interval][0]
    if max_days is None:  # 日足
        df = _download(symbol, interval, period=f"{int(years)}y")
        if df.empty:
            df = _download(symbol, interval, period="max")
    else:
        # intraday: 7日窓に分割して max_days ぶん遡る
        chunk = 7 if interval in ("1m", "2m") else 30
        max_days = max_days - 2          # Yahoo は境界ちょうどを弾くので安全マージン
        end = datetime.now()
        parts = []
        cursor = end
        while (end - cursor).days < max_days:
            lo = cursor - timedelta(days=chunk)
            if (end - lo).days > max_days:
                lo = end - timedelta(days=max_days)
            p = _download(symbol, interval, start=lo.date(), end=(cursor + timedelta(days=1)).date())
            if not p.empty:
                parts.append(p)
            if lo >= cursor:
                break
            cursor = lo
        df = pd.concat(parts).sort_index() if parts else pd.DataFrame()
        if not df.empty:
            df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        raise RuntimeError(f"{symbol} ({interval}) のデータを取得できませんでした")
    with open(key, "wb") as f:
        pickle.dump(df, f)
    return back_adjust(df) if adjust else df


def to_sessions(intraday: pd.DataFrame) -> pd.DataFrame:
    """intraday バーに session(日付) と bar_of_day(その日の何本目) を付与。"""
    df = intraday.copy()
    idx = df.index
    df["session"] = idx.tz_convert(idx.tz).date if idx.tz is not None else idx.date
    df["bar_of_day"] = df.groupby("session").cumcount()
    df["time"] = idx.time
    return df
