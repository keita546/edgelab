"""検証する候補手法(仮説)の一覧。

重要: シグナルは必ず「エントリー時点で知り得る情報」だけで作る(先読み禁止)。
レジーム判定の分位点も expanding で取り、全期間の分布を覗かないようにしている。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Candidate:
    name: str
    family: str
    rets: pd.Series          # シグナル発生時の方向付きリターン(コスト控除前)
    control: pd.Series | np.ndarray | None  # 非シグナル時に同じ方向で賭けた場合のリターン(対照群)
    horizon: int             # 保有バー数(HAC のラグに使う)
    note: str = ""
    meta: dict = field(default_factory=dict)


# ---------- 日足の特徴量 ----------

def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret1"] = d["close"].pct_change()
    d["gap"] = d["open"] / d["close"].shift(1) - 1.0
    d["o2c"] = d["close"] / d["open"] - 1.0
    d["c2o"] = d["open"].shift(-1) / d["close"] - 1.0
    d["dow"] = d.index.dayofweek
    d["rv20"] = d["ret1"].rolling(20).std()
    d["vol_ma20"] = d["volume"].shift(1).rolling(20).mean() if "volume" in d else np.nan
    d["hi252"] = d["close"].rolling(252).max()

    # RSI(2) — Wilder
    delta = d["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    d["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # 月内の営業日番号(月初から / 月末から)
    g = d.groupby([d.index.year, d.index.month])
    d["tdom"] = g.cumcount() + 1
    d["tdom_rev"] = g["close"].transform("size") - g.cumcount()
    return d


def _fwd(d: pd.DataFrame, basis: str, h: int) -> pd.Series:
    """エントリー時点を起点とする前向きリターン。"""
    if basis == "c2c":      # 終値で入り、h 日後の終値で出る
        return d["close"].shift(-h) / d["close"] - 1.0
    if basis == "o2c":      # 寄りで入り、同日の引けで出る
        return d["o2c"]
    if basis == "c2o":      # 引けで入り、翌日の寄りで出る(オーバーナイト)
        return d["c2o"]
    raise ValueError(basis)


def _mk(name: str, family: str, d: pd.DataFrame, mask: pd.Series, sign: int,
        basis: str, h: int, note: str = "") -> Candidate | None:
    f = _fwd(d, basis, h)
    mask = mask.fillna(False).astype(bool) & f.notna()
    if mask.sum() < 20:
        return None
    rets = (sign * f[mask]).dropna()
    ctrl = (sign * f[~mask & f.notna()]).dropna()
    return Candidate(name, family, rets, ctrl if len(ctrl) >= 20 else None, h, note,
                     {"basis": basis, "sign": sign})


def daily_hypotheses(d: pd.DataFrame) -> list[Candidate]:
    out: list[Candidate] = []
    add = lambda c: out.append(c) if c is not None else None
    names = ["月", "火", "水", "木", "金"]

    # 1. 曜日効果
    for i, nm in enumerate(names):
        add(_mk(f"曜日:{nm}曜に買い(終→終1日)", "曜日", d, d["dow"] == i, +1, "c2c", 1,
                "その曜日だけ1日持つ"))

    # 2. モメンタム / 逆張り(過去 k 日リターンの符号)
    for k in (1, 3, 5, 10, 20):
        past = d["close"] / d["close"].shift(k) - 1.0
        add(_mk(f"順張り:過去{k}日上昇→翌日買い", "モメンタム", d, past > 0, +1, "c2c", 1))
        add(_mk(f"逆張り:過去{k}日下落→翌日買い", "逆張り", d, past < 0, +1, "c2c", 1))

    # 3. RSI(2) 逆張り
    add(_mk("逆張り:RSI2<10→翌日買い", "逆張り", d, d["rsi2"] < 10, +1, "c2c", 1))
    add(_mk("逆張り:RSI2<10→5日保有", "逆張り", d, d["rsi2"] < 10, +1, "c2c", 5))
    add(_mk("逆張り:RSI2>90→翌日売り", "逆張り", d, d["rsi2"] > 90, -1, "c2c", 1))

    # 4. 寄りギャップ
    # 【重要】「窓を寄りで仕掛けて引けで返す」はシグナル(gap=open/prev_close)と
    # 売買(o2c=close/open)が同じ open を共有するため、寄り値のノイズだけで
    # 巨大な見せかけの優位性が出る(ランダムウォークで t=+15 を確認済み)。
    # 価格を共有しない「引けエントリー・翌日引け決済」に置き換える。
    add(_mk("ギャップ:-1%超の下窓→引けで買い翌日引け", "ギャップ", d, d["gap"] < -0.01, +1, "c2c", 1))
    add(_mk("ギャップ:+1%超の上窓→引けで売り翌日引け", "ギャップ", d, d["gap"] > 0.01, -1, "c2c", 1))

    # 5. オーバーナイト vs ザラ場(無条件・対照群なし)
    add(_mk("時間帯:毎日オーバーナイト保有(引→翌寄)", "時間帯", d, d["c2o"].notna(), +1, "c2o", 1,
            "引けで買い翌日の寄りで売る"))
    add(_mk("時間帯:毎日ザラ場のみ保有(寄→引)", "時間帯", d, d["o2c"].notna(), +1, "o2c", 1,
            "寄りで買い同日の引けで売る"))

    # 6. 月末月初アノマリー
    add(_mk("暦:月末2日+月初3日を保有", "暦", d, (d["tdom"] <= 3) | (d["tdom_rev"] <= 2), +1, "c2c", 1))

    # 7. ボラティリティ・レジーム(分位点は expanding = 先読みなし)
    q33 = d["rv20"].expanding(min_periods=250).quantile(0.33)
    q67 = d["rv20"].expanding(min_periods=250).quantile(0.67)
    add(_mk("レジーム:低ボラ局面で買い", "レジーム", d, d["rv20"] < q33, +1, "c2c", 1))
    add(_mk("レジーム:高ボラ局面で買い", "レジーム", d, d["rv20"] > q67, +1, "c2c", 1))

    # 8. 出来高スパイク + 下落 → リバウンド
    if "vol_ma20" in d:
        add(_mk("出来高:平均2倍+陰線→翌日買い", "出来高", d,
                (d["volume"] > 2 * d["vol_ma20"]) & (d["ret1"] < 0), +1, "c2c", 1))

    # 9. 連続陰線
    down3 = (d["ret1"] < 0) & (d["ret1"].shift(1) < 0) & (d["ret1"].shift(2) < 0)
    add(_mk("逆張り:3日連続下落→翌日買い", "逆張り", d, down3, +1, "c2c", 1))

    # 10. 52週高値圏のブレイク継続
    add(_mk("順張り:52週高値の2%以内→5日保有", "モメンタム", d, d["close"] >= 0.98 * d["hi252"], +1, "c2c", 5))
    return out


# ---------- 分足 ----------

def session_table(bars: pd.DataFrame) -> pd.DataFrame:
    """分足から1日1行のセッション表を作る(寄り/最初の30分/引け前1時間/引け)。"""
    b = bars.copy()
    idx = b.index
    b["session"] = pd.Index(idx.date)
    rows = []
    for s, g in b.groupby("session"):
        if len(g) < 12:
            continue
        n = len(g)
        bm = _bar_minutes(g.index)
        n30 = max(1, min(n - 4, int(round(30 / bm))))
        n60 = max(1, min(n - 4, int(round(60 / bm))))
        o, c = g["open"].iloc[0], g["close"].iloc[-1]
        cl = g["close"]
        # シグナル終点(i)と建玉開始(i+1)を1バーずらす。同じ価格を共有すると
        # その価格のノイズだけで見せかけの継続/反転が生まれるため。
        rows.append({
            "session": s,
            "open": o, "close": c,
            "entry_early": cl.iloc[1],                       # 寄り直後(窓トレード用の建値)
            "first30_ret": cl.iloc[n30 - 1] / o - 1.0,       # シグナル: 寄り30分
            "after30_ret": c / cl.iloc[n30] - 1.0,           # 売買: 1バー後から引けまで
            "pre_last_ret": cl.iloc[-n60 - 2] / o - 1.0,     # シグナル: 引け前1時間の手前まで
            "last60_ret": c / cl.iloc[-n60 - 1] - 1.0,       # 売買: 1バー後から引けまで
            "first30_hi": g["high"].iloc[:n30].max(),
            "first30_lo": g["low"].iloc[:n30].min(),
            "day_hi": g["high"].max(), "day_lo": g["low"].min(),
        })
    t = pd.DataFrame(rows).set_index("session").sort_index()
    t["prev_close"] = t["close"].shift(1)
    t["gap"] = t["open"] / t["prev_close"] - 1.0
    return t


def _bar_minutes(idx) -> float:
    if len(idx) < 2:
        return 5.0
    return max(1.0, float(pd.Series(idx).diff().dt.total_seconds().median() / 60.0))


def intraday_hypotheses(bars: pd.DataFrame, t: pd.DataFrame) -> list[Candidate]:
    out: list[Candidate] = []

    def push(name, rets, ctrl, note=""):
        r = pd.Series(rets).dropna()
        if len(r) and not isinstance(r.index, pd.DatetimeIndex):
            r.index = pd.to_datetime(list(r.index))
        if len(r) >= 20:
            c = np.asarray(pd.Series(ctrl).dropna()) if ctrl is not None else None
            out.append(Candidate(name, "分足", r, c if (c is not None and len(c) >= 20) else None, 1, note))

    # 時間帯別リターン(30分バケット)
    # 30分を1トレードとして集約する。5分足1本ずつを別トレードとして数えると
    # 往復コストを6回課金することになり、結果が完全に壊れる。
    b = bars.copy()
    b["bucket"] = [f"{x.hour:02d}:{(x.minute // 30) * 30:02d}" for x in b.index]
    b["session"] = pd.Index(b.index.date)
    agg = b.groupby(["session", "bucket"]).agg(o=("open", "first"), c=("close", "last"))
    agg["ret"] = agg["c"] / agg["o"] - 1.0
    agg = agg.reset_index()
    agg["session"] = pd.to_datetime(agg["session"])
    for bk, g in agg.groupby("bucket"):
        # 東証の昼休みのように実質的に値が動かない時間帯は検定対象から外す
        if g["ret"].abs().median() < 1e-4:
            continue
        others = agg.loc[agg["bucket"] != bk, "ret"]
        push(f"時間帯:{bk}台の30分を買い", g.set_index("session")["ret"], others, "その30分だけ保有")

    # 寄り30分の方向がその後も続くか
    up = t["first30_ret"] > 0
    push("分足:寄り30分が上昇→引けまで買い", t.loc[up, "after30_ret"], t.loc[~up, "after30_ret"])
    push("分足:寄り30分が下落→引けまで売り", -t.loc[~up, "after30_ret"], -t.loc[up, "after30_ret"])

    # 日中の方向が引け前1時間も続くか
    u2 = t["pre_last_ret"] > 0
    push("分足:日中が上昇→引け前1時間を買い", t.loc[u2, "last60_ret"], t.loc[~u2, "last60_ret"])

    # 窓埋め: 建値は「寄り値」ではなく寄り直後のバー終値。
    # gap の分子が open なので、open で建てると同じ価格を共有してしまう。
    gdn = t["gap"] < -0.005
    r_early = t["close"] / t["entry_early"] - 1.0
    push("分足:0.5%超の下窓→寄り直後に買い→引け", r_early[gdn], r_early[~gdn])
    return out
