#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="../Video-R1/Qwen2.5-VL-7B-Instruct-local"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=2
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

RESUME_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume_from_checkpoint)
            RESUME_ARG="--resume_from_checkpoint $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# If your dataset is mixed with images and videos, you need to use zero2.
deepspeed src/training/train.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /cephfs/shared/yicheng/4D-LLM/Video-R1/Video-R1-COT-Final.json \
    --image_folder /cephfs/shared/yicheng/4D-LLM/Video-R1 \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_depth_encoder False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/sft_stage2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((128 * 28 * 28)) \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --fps 1.0 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 2 \
    --wandb_project 4d-llm \
    --wandb_run_name sft_stage2 \
    --report_to wandb \
    $RESUME_ARG