#!/bin/bash
# ==============================================================================
# Run attention modulation experiments on CounterCount benchmark
# ==============================================================================
# Usage:
#   bash scripts/run_experiment.sh <MODEL> <CATEGORY> <IMAGE_TYPE> <PROMPT_TYPE>
#
# Examples:
#   bash scripts/run_experiment.sh Qwen/Qwen3-VL-8B-Instruct Birds anomaly open_ended
#   bash scripts/run_experiment.sh google/gemma-3-4b-it Mammals anomaly mcq
#   bash scripts/run_experiment.sh Qwen/Qwen3-VL-32B-Instruct Birds ordinary open_ended
# ==============================================================================

set -e

MODEL=${1:?"Usage: $0 <MODEL> <CATEGORY> <IMAGE_TYPE> <PROMPT_TYPE>"}
CATEGORY=${2:?"Specify category (e.g., Birds, Mammals, Housing)"}
IMAGE_TYPE=${3:-anomaly}       # anomaly | ordinary
PROMPT_TYPE=${4:-open_ended}   # open_ended | mcq

# --- Paths (update these to match your data layout) ---
DATA_ROOT="./data/CounterCount"
RESULTS_ROOT="./results"
CACHE_DIR="./model_cache"

# Determine image subdirectory
if [ "$IMAGE_TYPE" = "ordinary" ]; then
    IMAGES_DIR="${DATA_ROOT}/${CATEGORY}/Original"
else
    IMAGES_DIR="${DATA_ROOT}/${CATEGORY}/Anomaly"
fi

MASKS_DIR="${DATA_ROOT}/${CATEGORY}/masks"
BBOX_JSON="${DATA_ROOT}/${CATEGORY}/masks/${CATEGORY,,}_bbox.json"
PROMPTS="${DATA_ROOT}/${CATEGORY}/${CATEGORY,,}_prompts.json"
METADATA="${DATA_ROOT}/${CATEGORY}/${CATEGORY,,}_metadata.json"

echo "=============================================="
echo "CounterCount Attention Modulation Experiment"
echo "=============================================="
echo "Model:       ${MODEL}"
echo "Category:    ${CATEGORY}"
echo "Image type:  ${IMAGE_TYPE}"
echo "Prompt type: ${PROMPT_TYPE}"
echo "Images dir:  ${IMAGES_DIR}"
echo "Results dir: ${RESULTS_ROOT}/${CATEGORY}"
echo "=============================================="

python src/attention_modulation.py \
    --image_type "${IMAGE_TYPE}" \
    --images_dir "${IMAGES_DIR}" \
    --results_output_dir "${RESULTS_ROOT}/${CATEGORY}" \
    --masks_dir "${MASKS_DIR}" \
    --bbox_json_path "${BBOX_JSON}" \
    --prompts_path "${PROMPTS}" \
    --metadata_path "${METADATA}" \
    --model_name_or_path "${MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --prompt_type "${PROMPT_TYPE}" \
    --configs_path "configs/image_configurations.json"
