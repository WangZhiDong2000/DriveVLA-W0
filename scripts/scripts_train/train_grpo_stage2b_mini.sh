#!/usr/bin/env bash
# Stage 2-B GRPO training — local mini smoke (Task 4.1).
#
# Loads Stage 2-A checkpoint-500, runs 4 steps with MockPDMRewardWrapper,
# verifies RolloutBatch shapes and diagnostics (zero-loss placeholder).
# Task 4.3 will add real REINFORCE + IL loss before the full run.
#
# Differences from train_grpo_stage2a_mini_full_bce.sh:
#   - Entrypoint: utils/train_grpo_stage2b.py
#   - --init_ckpt_path: points to Stage 2-A checkpoint-500
#   - --use_mock_scorer True (no navsim / no metric_cache needed)
#   - --num_groups 2, --anchor_chunk 5 (mini OOM guard on single 5090)
#   - --num_truncated_steps 4 (quick smoke)
#   - --max_steps 4

set -euo pipefail

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29600
NGPUS=1

# Kill stale processes
STALE_PIDS=$(pgrep -f "torchrun.*train_grpo_stage2b\|python.*train_grpo_stage2b" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[startup] Found stale training processes (PIDs: $STALE_PIDS), killing them..."
    kill -9 $STALE_PIDS 2>/dev/null || true
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
EXP_NAME=train_grpo_stage2b_mini

MODEL_NAME_OR_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
MODEL_CONFIG_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
INIT_CKPT_PATH="$(pwd)/logs/train_grpo_stage2a_mini_full_bce/checkpoint-500"

export PYTHONPATH=$(pwd)
export PATH="/home/wang/anaconda3/envs/drivevla/bin:$PATH"
export VLA_NORM_STATS="$(pwd)/configs/normalizer_navsim_trainval/norm_stats.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f"
export WANDB_PROJECT="drivevla-navsim-grpo-stage2b"
export WANDB_RUN_NAME="stage2b_mini_mock_smoke"

# ── Step 1: Rollout smoke test (no training loop) ─────────────────────────────
echo "[startup] Running Stage 2-B rollout smoke test..."
python utils/train_grpo_stage2b.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --init_ckpt_path ${INIT_CKPT_PATH} \
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
  --num_groups 2 \
  --num_truncated_steps 4 \
  --sigma_init 0.5 \
  --sigma_step 0.04 \
  --sigma_logprob_min 0.10 \
  --discount 0.8 \
  --anchor_chunk 1 \
  --use_mock_scorer True \
  --smoke_test True \
  --seed 0

SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[startup] Smoke test FAILED (exit code ${SMOKE_EXIT}). Aborting."
    exit 1
fi
echo "[startup] Smoke test PASSED. Starting 4-step Stage 2-B mini training..."

# ── Step 2: 4-step mini training (REINFORCE + IL loss, verify training loop) ───
# Use plain python (not torchrun) for the mini smoke: NCCL process-group init
# reserves ~100-200 MB of CUDA memory which eliminates the thin headroom left
# after loading the 7B model on a single 32 GB GPU. The full run (train_grpo_
# stage2b_full.sh) uses torchrun + multi-GPU where the budget is ample.
python utils/train_grpo_stage2b.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --init_ckpt_path ${INIT_CKPT_PATH} \
  --actions_format fast \
  --action_tokenizer_path configs/fast \
  --output_dir logs/${EXP_NAME} \
  --learning_rate 1e-5 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --freeze_vlm True \
  --max_steps 4 \
  --dataloader_num_workers 0 \
  --lr_scheduler_type cosine \
  --warmup_steps 0 \
  --per_device_train_batch_size 1 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 1 \
  --save_strategy no \
  --eval_strategy no \
  --driving True \
  --use_previous_actions True \
  --action_dim 3 \
  --normalizer_path configs/normalizer_navsim_mini \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --num_groups 2 \
  --num_truncated_steps 4 \
  --sigma_init 0.5 \
  --sigma_step 0.04 \
  --sigma_logprob_min 0.10 \
  --discount 0.8 \
  --anchor_chunk 1 \
  --use_mock_scorer True \
  --report_to wandb
