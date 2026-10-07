# edgelab — 株価の統計的優位性スキャナ

日足と分足を取得し、40個前後の候補手法を**同じ土俵で**検定して、
「取引コストと多重検定を考慮してもなお優位性が残るか」を判定します。
結果はダッシュボード（検索・資金シミュレーター・説明書つき）で見られます。

**詳しい使い方は [docs/MANUAL.md](docs/MANUAL.md)。**

**ダッシュボード（毎朝6時ごろ自動更新）: https://keita546.github.io/edgelab/**

> 過去データの統計的検定であり、将来の成績や特定の売買を勧めるものではありません。

## クイックスタート

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python export.py                     # 6銘柄を5分足も含めて詳しく検定 → out/dashboard.json
.venv/bin/python scan.py --universe topix100   # 主要100銘柄（シミュレーター用データも）
.venv/bin/python scan.py --universe all        # 東証の内国株 全銘柄（約3,600、数分）
.venv/bin/python build_site.py                 # → site/dashboard.html と site/u/
.venv/bin/python -m http.server 8000 -d site   # http://localhost:8000/dashboard.html で見る
```

## 自動更新

`.github/workflows/update.yml` が火〜土の 6:00（日本時間）に GitHub のサーバーで `update.sh` を実行し、前の営業日の引けまでの株価で
GitHub Pages に公開します。パソコンの電源は不要です。手動で動かすときは Actions タブ →「毎日の更新」→ Run workflow。

手元で更新するときは `./update.sh`（約6分、ログは `out/update.log`）。

## 個別のコマンド

```bash
.venv/bin/python run.py --symbol 7203.T --intraday 5m --cost 10
.venv/bin/python run.py --symbol ^N225 --years 25 --cost 5 --intraday none
.venv/bin/python sim.py 6758.T "RSI2<10→5日" --capital 3000000 --frac 50
.venv/bin/python selftest.py          # ツール自体の較正テスト（銘柄別）
.venv/bin/python selftest.py --pooled # 市場全体の検定の較正テスト
```

| 引数 | 意味 |
|---|---|
| `--symbol` | 銘柄 (`7203.T` 日本株 / `^N225` 指数 / `SPY` 米株) |
| `--intraday` | `1m` `5m` `15m` `30m` `none` |
| `--years` | 日足の遡及年数 (既定15年) |
| `--cost` | 往復コスト bps。10 = 0.10%。**全トレードから毎回控除** |
| `--split` | IS/OOS 分割点 (既定 0.7) |

## 出力の読み方

| 列 | 意味 |
|---|---|
| `平均(bps)` | コスト控除後の1トレードあたり平均リターン |
| `t値(HAC)` | Newey-West 標準誤差による t 値。保有期間が重なる影響を補正済み |
| `p値` | 素の p 値。**これだけを見てはいけない** |
| `FDR補正p` | 何十個も試した事実を補正した p 値 (Benjamini-Hochberg) |
| `DSR` | Deflated Sharpe Ratio。試行回数・歪度・尖度・標本数で割り引いた「シャープが正である確率」。0.95 超が目安 |
| `対照群p` | 非シグナル日に同じ方向で賭けた場合との差 (Welch)。「ただ上昇相場に乗っていただけ」を除外する |
| `OOS平均` | 後方30%期間だけでの平均。IS と符号が違えば過剰適合 |
| `損益分岐コスト` | コスト控除**前**の平均。これを下回るコストで初めて黒字 |
| `検出力` | 5bps の優位性を検出できる確率。低ければ「無い」ではなく「判断できない」 |

判定は `有望(要追試)` → `FDR通過・DSR不足` → `IS/OOS不一致` → `補正で消滅` → `優位性なし`
→ `有意にマイナス` の順に厳しさが下がります。

## このツールが特に気をつけていること

1. **多重検定** — 40個の無意味な手法を試せば、平均2個は偶然 p<0.05 になる。
   BH-FDR と Deflated Sharpe Ratio でこれを補正する。
2. **取引コスト** — 往復コストを全トレードから控除。1トレードあたりの
   優位性がコストを下回る手法は、統計的に有意でも実運用では負ける。
3. **価格の共有による偽の優位性** — シグナルと売買が同じ価格を参照すると、
   その価格のノイズだけで巨大な見せかけの優位性が出る
   (例:「窓を寄りで仕掛けて引けで返す」は `gap=open/prev_close` と
   `o2c=close/open` が `open` を共有し、ランダムウォークでも t=+15 が出る)。
   全仮説でシグナルと建玉の間に必ず価格の隔たりを置いている。
4. **先読み** — ボラティリティ・レジームの分位点は expanding で取り、
   将来の分布を覗かない。配当落ちは遡及調整済み、株式分割も調整済み。
5. **系列相関** — 重複窓には HAC 標準誤差、信頼区間は定常ブートストラップ。
6. **較正の自己検証** — `selftest.py` がランダムウォーク24本で全仮説を回し、
   素の p<0.05 が約5%に収まり「有望」判定が0件になることを確認する。
   ここが崩れていればツール側のバグ。

## 既知の制約

- yfinance の分足は遡及期間が短い (1m: 30日 / 5m: 60日)。分足の
  「優位性なし」は**「無い」ではなく「標本が足りず判断できない」**。
  出力末尾に検出力の中央値を表示するので必ず確認すること。
- 単一銘柄の検定。実用には複数銘柄・複数期間での追試が必須。
- 売買インパクト、流動性制約、空売り規制・逆日歩は考慮していない。
- 生存者バイアス: 上場廃止銘柄は yfinance から取得できない。

## 構成

```
run.py                1銘柄の検定（表とCSV）
export.py             複数銘柄の検定 → out/dashboard.json
build_site.py         ダッシュボード HTML の生成
sim.py                資金シミュレーション（CLI）
selftest.py           較正テスト（ランダムウォークで偽陽性率を確認）
edgelab/data.py       取得・キャッシュ・配当遡及調整
edgelab/hypotheses.py 候補手法の定義
edgelab/rules_doc.py  ルールの日本語説明
edgelab/statlib.py    HAC t検定 / ブートストラップ / BH-FDR / DSR / 検出力
edgelab/engine.py     全仮説の一括評価と多重検定補正
edgelab/simulate.py   資金シミュレーション（ダッシュボードと同じ計算）
edgelab/report.py     表示
site/dashboard.src.html ダッシュボードのテンプレート
docs/MANUAL.md        説明書
```
