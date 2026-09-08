#!/bin/bash
# ============================================================
# 全流程编排器: train_v2 → 清理三件套 → train_din_ext → inference_full
# 完全运行在服务器上, 脱离 SSH / Claude / 用户本地电脑
# 启动: nohup bash /root/MyPro-Amazon/code/orchestrator.sh > /tmp/orchestrator.log 2>&1 < /dev/null &
# 日志: /tmp/orchestrator.log
# ============================================================

LOCK=/tmp/orchestrator.lock
if [ -f "$LOCK" ]; then
    OLDPID=$(cat "$LOCK" 2>/dev/null)
    if kill -0 "$OLDPID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] 已有 orchestrator 在运行 (PID $OLDPID), 退出"
        exit 0
    fi
    echo "[$(date +%H:%M:%S)] 清理陈旧 lock (PID $OLDPID 已不存在)"
    rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "===== orchestrator 启动 $(date) ====="

wait_done() {
    local name="$1"
    while pgrep -f "$name" >/dev/null 2>&1; do
        echo "[$(date +%H:%M:%S)] 等待 $name 结束 ..."
        sleep 60
    done
    echo "[$(date +%H:%M:%S)] $name 已退出"
}

check_ok() {
    local log="$1"; local marker="$2"; local stage="$3"
    if grep -q 'Traceback (most recent call last)' "$log"; then
        echo "!!! [$stage] $log 发现 Traceback (crash), 编排终止, 需人工介入"
        return 1
    fi
    if grep -q "$marker" "$log"; then
        echo ">>> [$stage] 检测到完成标志 [$marker]"
        return 0
    fi
    echo "!!! [$stage] $log 既无完成标志也无 Traceback, 状态不明, 终止"
    return 1
}

# ---- Stage 1: 等 train_v2 正常结束 ----
echo "[$(date +%H:%M:%S)] Stage 1/4: 等待 train_v2 结束 ..."
wait_done 'train_v2.py'
check_ok /tmp/train_v2.log 'All done!' 'train_v2' || exit 1

# ---- Stage 2: 清理三件套 + 启动 train_din ----
echo "[$(date +%H:%M:%S)] Stage 2/4: 清理 + 启动 train_din_ext ..."
bash /root/MyPro-Amazon/code/cleanup_and_launch_din.sh
sleep 10

# ---- Stage 3: 等 train_din 正常结束 ----
echo "[$(date +%H:%M:%S)] Stage 3/4: 等待 train_din_ext 结束 ..."
wait_done 'train_din_ext.py'
check_ok /tmp/train_din.log 'DIN-Ext training complete!' 'train_din' || exit 1

# ---- Stage 4: 启动 inference 并等它结束 ----
echo "[$(date +%H:%M:%S)] Stage 4/4: 启动 inference_full ..."
cd /root/MyPro-Amazon/code
nohup /root/miniconda3/bin/python -u inference_full.py > /tmp/inference.log 2>&1 < /dev/null &
echo "[$(date +%H:%M:%S)] inference_full 已启动 PID=$!"
sleep 10

wait_done 'inference_full.py'
check_ok /tmp/inference.log 'Full pipeline complete!' 'inference' || exit 1

echo "===== 全流程完成 $(date) ====="
echo ">>> 提交文件目录: /root/MyPro-Amazon/prediction_result/"
