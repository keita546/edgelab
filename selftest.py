#!/usr/bin/env python
"""較正テスト: 優位性が存在しないと分かっているデータで、
このツールが偽の「勝てる手法」を生み出さないことを確認する。

ランダムウォーク(ドリフト=0)を生成して全仮説を検定する。
理論上の期待値:
  - 素の p<0.05 の割合 ... 約 5% (これが有意水準の定義)
  - FDR 補正後の生存数 ... ほぼ 0
ここで FDR 生存が多発するなら、ツール側にバグか先読みがある。
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


if __name__ == "__main__":
    sys.exit(main())
