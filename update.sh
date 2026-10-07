#!/bin/bash
# 株価を取り直して、ダッシュボードを作り直す（約5〜8分）。
#   ./update.sh            すべて更新
# ログ: out/update.log
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
LOG=out/update.log
LOCK=out/.update.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date '+%F %T') 別の更新が実行中のため中止" >> "$LOG"; exit 0
fi
trap 'rmdir "$LOCK"' EXIT
PY=.venv/bin/python
{
  echo "===== $(date '+%F %T') 更新開始 ====="
  $PY export.py --refresh
  $PY scan.py --universe topix100 --refresh
  $PY scan.py --universe all --refresh --workers 8
  $PY build_site.py
  echo "===== $(date '+%F %T') 更新完了 ====="
} >> "$LOG" 2>&1 || { echo "===== $(date '+%F %T') 失敗（上のエラーを参照） =====" >> "$LOG"; exit 1; }
