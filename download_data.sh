#!/bin/bash
# Amazon Reviews 2023 下载脚本 (5 品类) — AutoDL 数据盘
# 用法: bash download_data.sh [并发数]
set -u
CONCURRENCY="${1:-6}"

# 启用 AutoDL 学术加速 (huggingface 走内网代理)
source /etc/network_turbo >/dev/null 2>&1

BASE='https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main'
DATA_DIR='/root/autodl-tmp/amazon_data'
CATS='Musical_Instruments Office_Products CDs_and_Vinyl Video_Games Toys_and_Games'
JOBS='/tmp/dl_jobs.txt'
LOG='/tmp/dl_progress.log'

mkdir -p "$DATA_DIR/raw/review_categories" "$DATA_DIR/raw/meta_categories"
: > "$JOBS"; : > "$LOG"

for cat in $CATS; do
  printf '%s|%s\n' "$BASE/benchmark/5core/rating_only/$cat.csv" "$DATA_DIR/$cat.csv" >> "$JOBS"
  printf '%s|%s\n' "$BASE/raw/review_categories/$cat.jsonl" "$DATA_DIR/raw/review_categories/$cat.jsonl" >> "$JOBS"
  printf '%s|%s\n' "$BASE/raw/meta_categories/meta_$cat.jsonl" "$DATA_DIR/raw/meta_categories/meta_$cat.jsonl" >> "$JOBS"
done

dl() {
  url="${1%%|*}"; dest="${1#*|}"
  mkdir -p "$(dirname "$dest")"
  remote=$(curl -sIL "$url" 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{print $2}' | tail -1)
  if [ -n "$remote" ] && [ -f "$dest" ]; then
    lsize=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$lsize" = "$remote" ]; then echo "SKIP $dest" >> "$LOG"; return 0; fi
  fi
  curl -sL -C - --retry 10 --retry-delay 3 -o "$dest" "$url"
  if [ -s "$dest" ]; then echo "DONE $dest $(stat -c%s "$dest")" >> "$LOG"; else echo "FAIL $dest" >> "$LOG"; fi
}
export -f dl

cat "$JOBS" | xargs -P "$CONCURRENCY" -I '{}' bash -c 'dl "$1"' _ '{}'

echo "=== ALL_DONE ==="
echo "完成文件数: $(grep -c '^DONE\|^SKIP' "$LOG")"
grep '^FAIL' "$LOG" || echo "(无失败)"
