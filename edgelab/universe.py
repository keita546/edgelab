"""東証の上場銘柄一覧(JPX 公式)から検定対象の銘柄集合を作る。

JPX「東証上場銘柄一覧」(毎月更新) を取得してキャッシュする。
"""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "cache"
PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
XLSX = CACHE / "jpx_data_j.xlsx"

DOMESTIC = ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"]
UNIVERSES = {
    "core30": "TOPIX Core30（超大型30銘柄）",
    "topix100": "TOPIX100（Core30 + Large70）",
    "topix500": "TOPIX500（TOPIX100 + Mid400）",
    "prime": "プライム市場の全銘柄",
    "all": "東証の内国株 全銘柄（プライム・スタンダード・グロース）",
}


def _download() -> None:
    html = urllib.request.urlopen(PAGE, timeout=30).read().decode("utf-8", "ignore")
    m = re.search(r'href="([^"]*data_j\.xlsx?)"', html)
    if not m:
        raise RuntimeError("JPX の銘柄一覧ファイルのリンクが見つかりません")
    url = "https://www.jpx.co.jp" + m.group(1) if m.group(1).startswith("/") else m.group(1)
    XLSX.write_bytes(urllib.request.urlopen(url, timeout=60).read())


def listing(refresh: bool = False) -> pd.DataFrame:
    """内国株の一覧: code, symbol(Yahoo 形式), name, market, sector, size"""
    CACHE.mkdir(exist_ok=True)
    if refresh or not XLSX.exists() or time.time() - XLSX.stat().st_mtime > 7 * 86400:
        _download()
    df = pd.read_excel(XLSX, dtype=str)
    df = df[df["市場・商品区分"].isin(DOMESTIC)].copy()
    out = pd.DataFrame({
        "code": df["コード"].str.strip(),
        "name": df["銘柄名"].str.strip(),
        "market": df["市場・商品区分"].str.replace("（内国株式）", "", regex=False),
        "sector": df["33業種区分"].str.strip(),
        "size": df["規模区分"].str.strip(),
        "asof": df["日付"],
    })
    out["symbol"] = out["code"] + ".T"
    return out.reset_index(drop=True)


def select(name: str, refresh: bool = False) -> pd.DataFrame:
    df = listing(refresh)
    if name == "core30":
        return df[df["size"] == "TOPIX Core30"]
    if name == "topix100":
        return df[df["size"].isin(["TOPIX Core30", "TOPIX Large70"])]
    if name == "topix500":
        return df[df["size"].isin(["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"])]
    if name == "prime":
        return df[df["market"] == "プライム"]
    if name == "all":
        return df
    raise ValueError(f"未知の銘柄集合: {name}（{', '.join(UNIVERSES)}）")
