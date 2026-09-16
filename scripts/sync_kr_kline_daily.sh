#!/usr/bin/env bash
# Weekday 18:00 KST: Kiwoom ka10081 daily bars -> kr_kline_processed.parquet
# Backfills any missing session days after the last parquet date through today.
set -euo pipefail
ROOT="/mnt/data/projects/kr_stock"
LOG_DIR="$ROOT/data/logs"
LOCK="/tmp/kr_stock_kline_daily.lock"
mkdir -p "$LOG_DIR"

export PATH="/home/mingyu/.local/bin:/usr/bin:/bin:$PATH"
export PYTHONPATH="$ROOT/src:$ROOT"
cd "$ROOT"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date '+%F %T') already running, skip"
  exit 0
fi

echo "========================================"
echo "$(date '+%F %T') kr_kline daily sync start"
echo "========================================"
uv run python scripts/refresh_daily_from_kiwoom.py --workers 2 --min-ok 2500
echo "$(date '+%F %T') kr_kline daily sync done"
