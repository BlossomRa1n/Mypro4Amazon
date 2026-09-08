#!/bin/bash
# train_v2 结束后: 清理三件套 + 启动 train_din_ext
# 自带双重保护: train_v2 还在跑 → 拒绝清理; train_din 已在跑 → 跳过启动

UD=/root/MyPro-Amazon/user_data
ENC="$UD/tmp_data/id_encoders_ext.pkl"
LATEST="$UD/model_data/two_tower_v2_latest.pth"
EPOCH_DIR="$UD/model_data/checkpoints"

# 保护 1: train_v2 (含其 DataLoader worker) 仍在运行就拒绝清理
if pgrep -f 'train_v2.py' >/dev/null; then
    echo "[cleanup] train_v2 仍在运行, 拒绝清理, 退出"
    exit 1
fi

# 保护 2: train_din 已在运行则不重复启动 (幂等)
if pgrep -f 'train_din_ext.py' >/dev/null; then
    echo "[cleanup] train_din_ext 已在运行, 跳过启动"
    exit 0
fi

echo "[cleanup] 开始清理三件套 ..."
rm -f "$ENC" && echo "  ✓ 删除 $ENC (15G 膨胀编码器)"
rm -f "$LATEST" && echo "  ✓ 删除 $LATEST (latest checkpoint)"
rm -f "$EPOCH_DIR"/two_tower_v2_epoch*.pth && echo "  ✓ 删除 $EPOCH_DIR/two_tower_v2_epoch*.pth (epoch 存档)"

echo "[cleanup] 清理后磁盘:"
df -h /root/autodl-tmp | tail -n 1

cd /root/MyPro-Amazon/code
nohup /root/miniconda3/bin/python -u train_din_ext.py > /tmp/train_din.log 2>&1 < /dev/null &
echo "[cleanup] train_din_ext 已启动 PID=$!, 日志 /tmp/train_din.log"
