#!/bin/bash
# VLA QFormer 推理脚本
# 
# 使用方法：
# 1. 直接运行：bash infer_navsim_qformer.sh
# 2. 或者覆盖环境变量后运行：
#    export EMU_HUB="/path/to/your/model"
#    export OUTPUT_DIR="/path/to/your/output"
#    bash infer_navsim_qformer.sh

# ============================================================================
# 配置区域：在这里设置所有路径和参数
# ============================================================================

# 项目根目录（自动检测，通常不需要修改）
if [ -z "$DRIVEVLA_ROOT" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export DRIVEVLA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# 模型和配置路径
export VLA_ACTION_TOKENIZER="${VLA_ACTION_TOKENIZER:-${DRIVEVLA_ROOT}/pretrained_models/fast}"
export VLA_VLM_MODEL="${VLA_VLM_MODEL:-${DRIVEVLA_ROOT}/pretrained_models/Emu3-Stage1}"
export VLA_NORM_STATS="${VLA_NORM_STATS:-${DRIVEVLA_ROOT}/configs/normalizer_navsim_trainval/norm_stats.json}"
export VLA_TOKEN_YAML="${VLA_TOKEN_YAML:-${DRIVEVLA_ROOT}/data/navsim/processed_data/scene_files/scene_filter/navtest.yaml}"

# 推理参数（可通过环境变量覆盖）
export EMU_HUB="${EMU_HUB:-${DRIVEVLA_ROOT}/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DRIVEVLA_ROOT}/outputs/infer_navsim_flow_matching_PDMS_87.2}"
export TEST_DATA_PKL="${TEST_DATA_PKL:-/data2/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl}"

# 可选参数
export VLA_NUM_WORKERS="${VLA_NUM_WORKERS:-12}"
export VLA_BATCH_SIZE="${VLA_BATCH_SIZE:-1}"

# Anchor 相关路径（用于模型内部，可通过环境变量覆盖）
export VLA_ANCHOR_CLUSTER_PATH="${VLA_ANCHOR_CLUSTER_PATH:-${DRIVEVLA_ROOT}/reference/Emu3/cluster_centers_8192.npy}"
# VLA_ANCHOR_METRIC_SCORE_PATH not needed for flow matching inference

# ============================================================================
# 执行推理
# ============================================================================

# 设置 PYTHONPATH
export PYTHONPATH="${DRIVEVLA_ROOT}/inference/navsim/navsim:${DRIVEVLA_ROOT}:${PYTHONPATH}"

# 切换到项目根目录
cd "$DRIVEVLA_ROOT"

# 运行推理脚本
torchrun --nproc_per_node=8 inference/vla/inference_action_navsim_flow_matching_vava.py \
    --emu_hub "$EMU_HUB" \
    --output_dir "$OUTPUT_DIR" \
    --test_data_pkl "$TEST_DATA_PKL"

