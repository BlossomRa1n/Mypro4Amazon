#!/usr/bin/env bash
set -u

ROOT=/root/MyPro-Amazon
PY=/root/miniconda3/bin/python
RUN=/root/autodl-tmp/future_window_token_suite_20260920
LOG=$RUN/suite.log

mkdir -p "$RUN"
"$PY" -B "$ROOT/code/run_token_experiments.py" \
  --base-run /root/autodl-tmp/future_window_baseline_20260919 \
  --cf-run /root/autodl-tmp/future_window_cf300_dev_20260919 \
  --run-dir "$RUN" \
  --screen-users 100000 \
  --test-users 100000 \
  --epochs 3 \
  --token-dim 256 \
  --negatives 4 \
  --candidates 75 \
  --fusion-mode rrf \
  --fusion-weights 2.0 1.0 0.7 0.05 \
  --itemcf-half-life-days 180 \
  >> "$LOG" 2>&1
status=$?

if [ "$status" -eq 0 ] && [ -f "$RUN/COMPLETED.json" ]; then
  date -u +%FT%TZ > "$RUN/shutdown_requested_utc.txt"
  /sbin/shutdown -h now
fi
exit "$status"
