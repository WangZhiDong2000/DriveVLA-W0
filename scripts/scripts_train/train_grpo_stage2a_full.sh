#!/usr/bin/env bash
# Stage 2-A full IL adaptation training — server entry (Task 3.4 will launch).
# Differs from train_grpo_stage2a_mini.sh:
#   - Full navtrain pkl (not mini)
#   - 2000 steps (vs 300 mini)
#   - anchor_warmup_steps=500 (vs 50 mini)
#   - bce_strategy=full + bce_loss_weight=0.5 + bce_chunk_size=5 (Task 3.3)
#   - per_device_train_batch_size=2, gradient_accumulation_steps=8 (effective batch 16,
#     keeps BCE-pass effective batch B*chunk = 2*5 = 10 → fits H100 80GB with grad ckpt)
# Server paths assume conda env + dataset locations match prod; adjust if needed.

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

# ── Server paths (adjust if dataset / env differ) ────────────────────────────
DATAPATH='/home/wang/Dataset/navsim/download/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
EXP_NAME=train_grpo_stage2a_full

MODEL_NAME_OR_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
MODEL_CONFIG_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"

export PYTHONPATH=$(pwd)
export PATH="/home/wang/anaconda3/envs/drivevla/bin:$PATH"
export VLA_NORM_STATS="$(pwd)/configs/normalizer_navsim_trainval/norm_stats.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f"
export WANDB_PROJECT="drivevla-navsim-grpo-stage2a"
export WANDB_RUN_NAME="stage2a_full_run1"

# ── Step 1: Smoke test (1 forward+backward, full BCE chunked path) ───────────
echo "[startup] Running smoke test first (with full BCE)..."
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
  --normalizer_path configs/normalizer_navsim_trainval \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --anchor_warmup_steps 500 \
  --bce_strategy full \
  --bce_loss_weight 0.5 \
  --bce_chunk_size 5 \
  --smoke_test True \
  --seed 0

SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[startup] Smoke test FAILED (exit code ${SMOKE_EXIT}). Aborting training."
    exit 1
fi
echo "[startup] Smoke test PASSED. Starting 2000-step full Stage 2-A training..."

# ── Step 2: 2000-step full training ──────────────────────────────────────────
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
  --max_steps 2000 \
  --dataloader_num_workers 2 \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps 100 \
  --per_device_train_batch_size 2 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 8 \
  --save_strategy steps \
  --save_steps 500 \
  --eval_strategy no \
  --driving True \
  --use_previous_actions True \
  --action_dim 3 \
  --normalizer_path configs/normalizer_navsim_trainval \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --anchor_warmup_steps 500 \
  --sigma_anchor_max 0.04 \
  --bce_strategy full \
  --bce_loss_weight 0.5 \
  --bce_chunk_size 5 \
  --action_loss_weight 1.0 \
  --action_sample_steps 10 \
  --report_to wandb
