#!/usr/bin/env bash
set -euo pipefail

run_dir=$1
target=$2
epoch=$3
log_path="$run_dir/train.log"

while ! grep -a -q "\"stage\": \"din\", \"epoch\": $epoch" "$log_path"; do
    sleep 20
done
sleep 5

checkpoint="$run_dir/din_latest.pth"
if [ -f "$checkpoint" ] && [ ! -L "$checkpoint" ]; then
    rm -f "$target"
    mv "$checkpoint" "$target"
    ln -s "$target" "$checkpoint"
    printf 'moved din_latest after epoch %s to %s\n' "$epoch" "$target" \
        > "$run_dir/checkpoint_relocation.log"
fi
