#!/usr/bin/env bash
# Stage 2-B GRPO training — server full run (Task 4.5 entry point, PLACEHOLDER).
#
# Prerequisites before enabling:
#   1. Task 4.3 fills compute_loss with REINFORCE IS-ratio + IL vs GT L1.
#   2. Task 4.2 async PDM scorer is wired in (use_mock_scorer=False).
#   3. METRIC_CACHE_ROOT points to the navtrain metric_cache .lzma directory.
#   4. Stage 2-A full ckpt (train_grpo_stage2a_full) is available at INIT_CKPT_PATH.
#
# TODO: update INIT_CKPT_PATH once Stage 2-A full training completes.
# TODO: update METRIC_CACHE_ROOT to actual metric_cache directory on server.

set -euo pipefail

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29601
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

# ── Server paths (adjust before running) ─────────────────────────────────────
DATAPATH='/home/wang/Dataset/navsim/download/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
EXP_NAME=train_grpo_stage2b_full

MODEL_NAME_OR_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
MODEL_CONFIG_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
# TODO: replace with Stage 2-A full ckpt once available
INIT_CKPT_PATH="$(pwd)/logs/train_grpo_stage2a_full/checkpoint-2000"
# TODO: replace with actual metric_cache root
METRIC_CACHE_ROOT="/TODO/navsim/metric_cache/trainval"

export PYTHONPATH=$(pwd)
export PATH="/home/wang/anaconda3/envs/drivevla/bin:$PATH"
export VLA_NORM_STATS="$(pwd)/configs/normalizer_navsim_trainval/norm_stats.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f"
export WANDB_PROJECT="drivevla-navsim-grpo-stage2b"
export WANDB_RUN_NAME="stage2b_full_run1"

# ── Step 1: Smoke test with real PDM ─────────────────────────────────────────
echo "[startup] Running Stage 2-B smoke test (real PDM scorer)..."
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
  --normalizer_path configs/normalizer_navsim_trainval \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --num_groups 8 \
  --num_truncated_steps 10 \
  --sigma_init 0.5 \
  --sigma_step 0.04 \
  --sigma_logprob_min 0.10 \
  --discount 0.8 \
  --anchor_chunk 20 \
  --use_mock_scorer False \
  --metric_cache_root ${METRIC_CACHE_ROOT} \
  --scorer_num_workers 8 \
  --smoke_test True \
  --seed 0

SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "[startup] Smoke test FAILED (exit code ${SMOKE_EXIT}). Aborting."
    exit 1
fi
echo "[startup] Smoke test PASSED. Starting full Stage 2-B training..."

# ── Step 2: Full Stage 2-B training (Task 4.5: 10 epochs) ────────────────────
# NOTE: max_steps should be set to cover 10 epochs of navtrain.
# Rough estimate: navtrain ~220k samples / effective_batch 16 ≈ 13750 steps/epoch
# → 10 epochs ≈ 137500 steps. Adjust per actual dataset size.
torchrun \
  --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  utils/train_grpo_stage2b.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --init_ckpt_path ${INIT_CKPT_PATH} \
  --actions_format fast \
  --action_tokenizer_path configs/fast \
  --output_dir logs/${EXP_NAME} \
  --learning_rate 5e-6 \
  --weight_decay 1e-4 \
  --min_learning_rate 5e-7 \
  --max_grad_norm 1.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-6 \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --freeze_vlm True \
  --max_steps 137500 \
  --dataloader_num_workers 2 \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps 500 \
  --per_device_train_batch_size 2 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 10 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 8 \
  --save_strategy steps \
  --save_steps 1000 \
  --eval_strategy no \
  --driving True \
  --use_previous_actions True \
  --action_dim 3 \
  --normalizer_path configs/normalizer_navsim_trainval \
  --anchor_cache_path cache/anchor_centers_N20.npy \
  --num_groups 8 \
  --num_truncated_steps 10 \
  --sigma_init 0.5 \
  --sigma_step 0.04 \
  --sigma_logprob_min 0.10 \
  --discount 0.8 \
  --anchor_chunk 20 \
  --use_mock_scorer False \
  --metric_cache_root ${METRIC_CACHE_ROOT} \
  --scorer_num_workers 8 \
  --lambda_il 0.1 \
  --report_to wandb
