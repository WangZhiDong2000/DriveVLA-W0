#!/usr/bin/env bash

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29513
NGPUS=1

# Kill stale processes
STALE_PIDS=$(pgrep -f "torchrun.*train_grpo_stage2a\|python.*train_grpo_stage2a" 2>/dev/null)
if [ -n "$STALE_PIDS" ]; then
    echo "[startup] Found stale training processes (PIDs: $STALE_PIDS), killing them..."
    kill -9 $STALE_PIDS 2>/dev/null
    sleep 2
fi

find_free_port() {
    local port=$1
    while lsof -i TCP:"${port}" -sTCP:LISTEN &>/dev/null 2>&1; do
        echo "[startup] Port ${port} is occupied, trying $((port + 1))..." >&2
        port=$((port + 1))
    done
    echo "$port"
}
MASTER_PORT=$(find_free_port ${MASTER_PORT})
echo "[startup] Using MASTER_PORT=${MASTER_PORT}"

DATAPATH='/home/wang/Dataset/navsim/download/processed_data/meta/navsim_emu_vla_256_144_mini_pre_1s.pkl'
EXP_NAME=train_grpo_stage2a_mini

MODEL_NAME_OR_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
MODEL_CONFIG_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"

export PYTHONPATH=$(pwd)
export PATH="/home/wang/anaconda3/envs/drivevla/bin:$PATH"
export VLA_NORM_STATS="$(pwd)/configs/normalizer_navsim_trainval/norm_stats.json"

export WANDB_API_KEY="a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f"
export WANDB_PROJECT="drivevla-navsim-grpo-stage2a"
export WANDB_RUN_NAME="stage2a_mini_smoke"

# ── Step 1: Smoke test (1 forward+backward, no ckpt written) ─────────────────
echo "[startup] Running smoke test first..."
python utils/train_grpo_stage2a.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --actions_format fast \
  --action_tokenizer_path configs/fast \
  --output_dir logs/${EXP_NAME} \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --freeze_vlm True \
  --driving True \
  --use_previous_actions True \
  --action_dim 3 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --normalizer_path configs/normalizer_navsim_mini \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --anchor_warmup_steps 50 \
  --bce_strategy none \
  --smoke_test True \
  --seed 0

SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[startup] Smoke test FAILED (exit code ${SMOKE_EXIT}). Aborting training."
    exit 1
fi
echo "[startup] Smoke test PASSED. Starting 300-step mini training..."

# ── Step 2: 300-step mini training ───────────────────────────────────────────
torchrun \
  --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  utils/train_grpo_stage2a.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --actions_format fast \
  --action_tokenizer_path configs/fast \
  --output_dir logs/${EXP_NAME} \
  --learning_rate 2e-4 \
  --weight_decay 1e-4 \
  --min_learning_rate 1e-6 \
  --max_grad_norm 1.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-6 \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --freeze_vlm True \
  --max_steps 300 \
  --dataloader_num_workers 0 \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps 30 \
  --per_device_train_batch_size 4 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 4 \
  --save_strategy steps \
  --save_steps 100 \
  --eval_strategy no \
  --driving True \
  --use_previous_actions True \
  --action_dim 3 \
  --normalizer_path configs/normalizer_navsim_mini \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --anchor_warmup_steps 50 \
  --sigma_anchor_max 0.04 \
  --bce_strategy none \
  --action_loss_weight 1.0 \
  --action_sample_steps 10 \
  --report_to wandb
