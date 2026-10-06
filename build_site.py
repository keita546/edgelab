#!/usr/bin/env python
"""ダッシュボードを組み立てる。

  out/dashboard.json            (export.py の6銘柄)       → ページに埋め込み
  out/universe_*/summary.json   (scan.py の銘柄集合)      → ページに埋め込み
  out/universe_*/detail_*.json, sim/*.json                 → site/u/<名前>/ にコピー(ページが必要時に読む)

出力: site/dashboard.html と site/u/
"""
import json
import shutil
from pathlib import Path

root = Path(__file__).parent
site = root / "site"
src = (site / "dashboard.src.html").read_text(encoding="utf-8")
data = (root / "out" / "dashboard.json").read_text(encoding="utf-8").replace("</", "<\\/")

udata, files = {}, []
if (site / "u").exists():
    shutil.rmtree(site / "u")
for name in ("all", "topix100"):
    d = root / "out" / f"universe_{name}"
    if not (d / "summary.json").exists():
        continue
    udata[name] = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    # 予想画面の値動きデータは、いちばん広い範囲(全銘柄があれば全銘柄)のものだけ使う
    use_fwd = name == "all" or not (root / "out" / "universe_all" / "summary.json").exists()
    fwd = list(d.glob("fwd_*.json")) if use_fwd else []
    for f in list(d.glob("detail_*.json")) + fwd + list((d / "sim").glob("*.json")):
        dst = site / "u" / name / f.relative_to(d)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        files.append(dst.relative_to(site).as_posix())

ud = json.dumps(udata, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
out = site / "dashboard.html"
out.write_text(src.replace("__DATA__", data).replace("__UDATA__", ud), encoding="utf-8")
(site / "files.json").write_text(json.dumps(sorted(files), ensure_ascii=False, indent=0))
total = out.stat().st_size + sum((site / f).stat().st_size for f in files)
print(f"{out.relative_to(root)}  {out.stat().st_size / 1e6:.1f} MB ＋ 付属ファイル {len(files)} 個（合計 {total / 1e6:.1f} MB）")
print("ローカルで見るとき:  .venv/bin/python -m http.server 8000 -d site   →  http://localhost:8000/dashboard.html")
