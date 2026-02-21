#!/bin/bash
# AlphaGPT daily pipeline - accumulate tokens and refresh OHLCV
set -e

PROJECT_DIR="/Users/sky/python_demo/AlphaGPT"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

echo "=== Pipeline started at $(date) ===" >> "$LOG_FILE"
cd "$PROJECT_DIR"
"$PYTHON" main.py pipeline >> "$LOG_FILE" 2>&1
echo "=== Pipeline finished at $(date) ===" >> "$LOG_FILE"

# keep only last 30 days of logs
find "$LOG_DIR" -name "pipeline_*.log" -mtime +30 -delete 2>/dev/null || true
