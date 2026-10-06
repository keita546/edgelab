#!/usr/bin/env python
"""較正テスト: 優位性が存在しないと分かっているデータで、
このツールが偽の「勝てる手法」を生み出さないことを確認する。

ランダムウォーク(ドリフト=0)を生成して全仮説を検定する。
理論上の期待値:
  - 素の p<0.05 の割合 ... 約 5% (これが有意水準の定義)
  - FDR 補正後の生存数 ... ほぼ 0
ここで FDR 生存が多発するなら、ツール側にバグか先読みがある。

  python selftest.py            # 銘柄別の検定
  python selftest.py --pooled   # 市場全体の検定(連動する銘柄群)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from edgelab import engine, hypotheses as H

N_SIM, N_DAYS = 24, 3700


def synth(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2011-09-12", periods=N_DAYS)
    r = rng.normal(0.0, 0.015, N_DAYS)          # ドリフト 0 = 優位性ゼロ
    close = 2000 * np.exp(np.cumsum(r))
    on = rng.normal(0.0, 0.007, N_DAYS)          # オーバーナイト分
    open_ = np.r_[close[0], close[:-1]] * np.exp(on)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, N_DAYS)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, N_DAYS)))
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "adj close": close,
                         "volume": rng.lognormal(15, 0.4, N_DAYS)}, index=idx)


def main() -> int:
    raw_hits, fdr_hits, promising, total = 0, 0, 0, 0
    for s in range(N_SIM):
        cands = H.daily_hypotheses(H.daily_features(synth(s)))
        # コスト 0 で検定する(コストを引くと全戦略が当然マイナスになり較正が測れない)
        res = engine.evaluate(cands, cost_bps=0.0, boot=0)
        raw_hits += int((res["p値"] < 0.05).sum())
        fdr_hits += int((res["FDR補正p"] < 0.05).sum())
        promising += int((res["判定"] == "有望(要追試)").sum())
        total += len(res)
        print(f"  seed {s:02d}: 仮説{len(res):3d}  素p<0.05={int((res['p値']<0.05).sum()):2d}  "
              f"FDR<0.05={int((res['FDR補正p']<0.05).sum()):2d}", flush=True)

    rate = raw_hits / total
    print("\n" + "=" * 70)
    print(f"検定総数           : {total}  ({N_SIM} 本のランダムウォーク)")
    print(f"素の p<0.05 の割合 : {rate:.3%}   (理論値 5% / 許容 2〜9%)")
    print(f"FDR 補正後の生存   : {fdr_hits} 件  ({fdr_hits/total:.3%})")
    print(f"『有望』判定       : {promising} 件  (理想は 0)")
    ok = 0.02 <= rate <= 0.09 and promising <= 1
    print("=" * 70)
    print("結果: 較正OK — 偽の優位性を量産していない" if ok
          else "結果: 較正NG — 先読みかバグの疑いあり")
    return 0 if ok else 1




def pooled_main(n_stocks: int = 60, n_sims: int = 30) -> int:
    """市場全体の検定(日ごとにまとめてから検定)の較正テスト。約2分。

    共通の市場要因で連動する乱数の銘柄群を作る。単純リターンの期待値がゼロになるよう
    対数リターンに -σ²/2 のずれを入れている(入れないと、平均がわずかに本当にプラスになり、
    それを正しく検出して p<0.05 が増える)。優位性はゼロなので、
    p<0.05 は約5%、「市場全体で有効」「他銘柄より優位」はほぼ0件になるはず。
    """
    from edgelab import scan
    A, E, eff = [], [], 0
    for k in range(n_sims):
        rng = np.random.default_rng(5000 + k)
        idx = pd.bdate_range("2011-09-12", periods=2000)
        sm, si, so = 0.011, 0.016, 0.006
        mkt = rng.normal(0, sm, len(idx))
        frames = []
        for i in range(n_stocks):
            b = rng.uniform(0.6, 1.4)
            r = b * mkt + rng.normal(0, si, len(idx)) - ((b * sm) ** 2 + si ** 2) / 2
            close = 1000 * np.exp(np.cumsum(r))
            open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, so, len(idx)) - so ** 2 / 2)
            frames.append((f"S{i}", pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.004,
                                                    "low": np.minimum(open_, close) * 0.996, "close": close,
                                                    "volume": rng.lognormal(12, .5, len(idx))}, index=idx)))
        cal = [x.strftime("%Y-%m-%d") for x in idx]
        cs = H.daily_hypotheses(H.daily_features(frames[0][1]))
        meta = {c.name: (c.family, c.horizon, c.meta.get("basis"), c.meta.get("sign")) for c in cs}
        _, agg, _ = scan.scan_frames(frames, cal, 0.0, 0.7, 0, list(meta))
        pdf = scan.pool_rules(agg, meta, cal, 0.7)
        A += list(pdf["p"].dropna()); E += list(pdf["pe"].dropna())
        eff += int(pdf["verdict"].isin(["市場全体で有効", "他銘柄より優位"]).sum())
        print(f"  sim {k:02d} 完了", flush=True)
    A, E = np.array(A), np.array(E)
    print(f"\n平均の検定 p<0.05: {(A < .05).mean():.1%}　差の検定 p<0.05: {(E < .05).mean():.1%}（理論値 5%）"
          f"　有効判定 {eff} 件 / {n_sims * len(meta)} ルール（理想は 0）")
    ok = (A < .05).mean() <= 0.08 and (E < .05).mean() <= 0.08 and eff <= 2
    print("結果: 較正OK" if ok else "結果: 較正NG")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(pooled_main() if "--pooled" in sys.argv else main())
