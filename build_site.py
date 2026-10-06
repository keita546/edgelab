#!/usr/bin/env python
"""out/dashboard.json をページに埋め込んで site/dashboard.html を作る。"""
from pathlib import Path
root = Path(__file__).parent
src = (root / "site" / "dashboard.src.html").read_text(encoding="utf-8")
data = (root / "out" / "dashboard.json").read_text(encoding="utf-8").replace("</", "<\\/")
out = root / "site" / "dashboard.html"
out.write_text(src.replace("__DATA__", data), encoding="utf-8")
print(f"{out.relative_to(root)}  {out.stat().st_size/1024:.0f} KB")
