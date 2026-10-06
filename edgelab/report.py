"""結果の表示。"""
from __future__ import annotations
import pandas as pd

COLS = ["手法", "件数", "平均(bps)", "勝率%", "t値(HAC)", "p値", "FDR補正p", "DSR",
        "対照群p", "年率Sharpe", "OOS平均(bps)", "損益分岐コスト(bps)", "判定"]


def console(df: pd.DataFrame, title: str, top: int = 60) -> str:
    if df.empty:
        return f"\n[{title}] 検定可能な仮説がありませんでした\n"
    d = df.head(top).copy()
    for c in ["平均(bps)", "勝率%", "t値(HAC)", "年率Sharpe", "OOS平均(bps)", "損益分岐コスト(bps)"]:
        d[c] = d[c].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "-")
    for c in ["p値", "FDR補正p", "DSR", "対照群p"]:
        d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    with pd.option_context("display.max_columns", None, "display.width", 250,
                           "display.unicode.east_asian_width", True):
        body = d[COLS].to_string(index=False)
    return f"\n{'=' * 100}\n[{title}]\n{'=' * 100}\n{body}\n"


def summary(df: pd.DataFrame, cost_bps: float) -> str:
    if df.empty:
        return ""
    raw = int((df["p値"] < 0.05).sum())
    fdr = int((df["FDR補正p"] < 0.05).sum())
    win = df[df["判定"] == "有望(要追試)"]
    near = df[df["判定"].isin(["FDR通過・DSR不足", "IS/OOS不一致", "FDR通過・OOS不足"])]
    L = [
        "",
        "=" * 100,
        "サマリー",
        "=" * 100,
        f"検定した仮説数            : {len(df)}",
        f"往復コスト                : {cost_bps:.1f} bps を全トレードから控除済み",
        f"素の p<0.05              : {raw} 件  ← 多重検定を無視すればこれだけ「勝てる手法」に見える",
        f"FDR補正後 p<0.05         : {fdr} 件",
        f"DSRも0.95超 かつ IS/OOS整合 : {len(win)} 件",
        "",
    ]
    if len(win):
        L.append("▼ 統計的に生き残った候補(それでも実運用の保証ではない):")
        for _, r in win.iterrows():
            L.append(f"   ・{r['手法']}  平均{r['平均(bps)']:.1f}bps  "
                     f"n={int(r['件数'])}  FDR p={r['FDR補正p']:.4f}  DSR={r['DSR']:.3f}")
    else:
        L.append("▼ 補正後に生き残った手法: なし")
        L.append("   = 今回の銘柄・期間・仮説集合の範囲では、コスト控除後に")
        L.append("     統計的優位性を主張できるものは検出されませんでした。")
    rev = df[df["判定"] == "有意にマイナス"].copy()
    if len(rev):
        # 両側検定なので逆方向も同じ p 値でカバーされる。ただしコストは
        # 逆方向でも同じだけ掛かるので、必ずコスト控除後の値で判断する。
        rev["逆方向net(bps)"] = -rev["損益分岐コスト(bps)"] - cost_bps
        rev = rev[rev["逆方向net(bps)"] > 0].sort_values("逆方向net(bps)", ascending=False)
        if len(rev):
            L.append("▼ 検定方向では有意にマイナス = 逆方向が優位の可能性:")
            for _, r in rev.iterrows():
                L.append(f"   ・{r['手法']}  → 逆方向なら +{r['逆方向net(bps)']:.1f}bps/回  "
                         f"(FDR p={r['FDR補正p']:.4f})")
            L.append("     ※事後的に符号を選んでいるため、別銘柄・別期間での追試が必須")
            L.append("")
    if len(near):
        L.append("")
        L.append("▼ FDR は通ったが最終関門で落ちた候補(過剰適合の疑い):")
        for _, r in near.iterrows():
            L.append(f"   ・{r['手法']}  平均{r['平均(bps)']:.1f}bps  "
                     f"FDR p={r['FDR補正p']:.4f}  DSR={r['DSR']:.3f}  判定={r['判定']}")
    L.append("")
    return "\n".join(L)
