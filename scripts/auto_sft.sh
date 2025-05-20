#!/bin/bash

OUTPUT_DIR="./output/sft_stage2"
TRAIN_SCRIPT="./scripts/finetune_video.sh"

# 检查一个 checkpoint 是否有效
is_valid_checkpoint() {
    ckpt_dir="$1"
    [[ -f "$ckpt_dir/scheduler.pt" ]]
}

get_latest_valid_checkpoint() {
    for ckpt in $(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -Vr); do
        if is_valid_checkpoint "$ckpt"; then
            echo "$ckpt"  # ✅ 只输出这个
            return
        else
            >&2 echo "⚠️ 无效 checkpoint：$ckpt（缺少 scheduler.pt），删除中..."
            rm -rf "$ckpt"
        fi
    done
    echo ""  # 没有有效 checkpoint
}

# 主训练循环
while true; do
    latest_ckpt=$(get_latest_valid_checkpoint)

    if [[ -n "$latest_ckpt" ]]; then
        echo "✅ 检测到有效 checkpoint：$latest_ckpt，开始断点续训..."
        bash "$TRAIN_SCRIPT" --resume_from_checkpoint "$latest_ckpt"
    else
        echo "🟡 未检测到任何有效 checkpoint，开始首次训练..."
        bash "$TRAIN_SCRIPT"
    fi

    exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ 训练正常结束。"
        break
    else
        echo "❌ 训练异常退出，10 秒后自动重试..."
        sleep 10
    fi
done
