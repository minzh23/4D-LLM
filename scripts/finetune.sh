#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --use_liger True \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /cephfs/shared/yicheng/4D-LLM/Video-R1/Video-R1-COT-Converted-Fixed.json \
    --image_folder /cephfs/shared/yicheng/4D-LLM/Video-R1 \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_depth_encoder True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/video-r1_stage1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
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
    --dataloader_num_workers 0 \
    --wandb_project 4d-llm \
    --wandb_run_name video-r1_stage1 \
    --report_to wandb \