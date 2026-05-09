#!/usr/bin/env bash

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29512
NGPUS=1

# Kill any lingering torchrun/python processes from a previous interrupted run
# (they hold the port and GPU memory, causing silent hangs on restart)
STALE_PIDS=$(pgrep -f "torchrun.*train_pi0\|python.*train_pi0" 2>/dev/null)
if [ -n "$STALE_PIDS" ]; then
    echo "[startup] Found stale training processes (PIDs: $STALE_PIDS), killing them..."
    kill -9 $STALE_PIDS 2>/dev/null
    sleep 2
fi

# Auto-find a free port starting from MASTER_PORT to avoid conflicts
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
EXP_NAME=train_navsim_flow_matching_mini

# Load trained world model (VLM) from the full Pi0 checkpoint, fresh action expert.
# Triggers init_fresh_expert branch: VLM weights extracted shard-by-shard (no double loading),
# action expert randomly initialized.
MODEL_NAME_OR_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
MODEL_CONFIG_PATH="$(pwd)/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"

export PYTHONPATH=$(pwd)
export PATH="/home/wang/anaconda3/envs/drivevla/bin:$PATH"

export WANDB_API_KEY="a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f"
export WANDB_PROJECT="drivevla-navsim-flow-matching"
export WANDB_RUN_NAME="mini_fresh_expert"

torchrun \
  --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  utils/train_pi0.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --actions_format fast \
  --action_tokenizer_path configs/fast \
  --output_dir logs/${EXP_NAME} \
  --learning_rate 5e-5 \
  --null_prompt_prob 0.15 \
  --weight_decay 0.1 \
  --min_learning_rate 1e-6 \
  --max_grad_norm 5.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-6 \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --init_fresh_expert True \
  --freeze_vlm True \
  --num_train_epochs 1 \
  --dataloader_num_workers 2 \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps 150 \
  --per_device_train_batch_size 4 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 4 \
  --save_strategy steps \
  --save_steps 600 \
  --eval_strategy steps \
  --eval_steps 300 \
  --per_device_eval_batch_size 4 \
  --apply_loss_on_only_vision True \
  --apply_loss_on_only_action False \
  --actions True \
  --use_gripper False \
  --driving True \
  --use_previous_actions True \
  --action_dim 3 \
  --train_action_only False \
  --action_loss_weight 1.0 \
  --action_sample_steps 10 \
  --normalizer_path configs/normalizer_navsim_mini \
  --report_to wandb
