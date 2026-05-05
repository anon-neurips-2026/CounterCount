#!/bin/bash
# ==============================================================================
# Run experiments across ALL categories for a given model
# ==============================================================================
# Usage:
#   bash scripts/run_all_categories.sh <MODEL> <IMAGE_TYPE> <PROMPT_TYPE>
#
# Example:
#   bash scripts/run_all_categories.sh Qwen/Qwen3-VL-8B-Instruct anomaly mcq
# ==============================================================================

set -e

MODEL=${1:?"Usage: $0 <MODEL> <IMAGE_TYPE> <PROMPT_TYPE>"}
IMAGE_TYPE=${2:-anomaly}
PROMPT_TYPE=${3:-open_ended}

CATEGORIES=(
    "Birds"
    "Bugs"
    "Currency"
    "Functional"
    "Housing"
    "Mammals"
    "Landmarks"
    "Transportation"
    "Sea"
    "Food"
)

for CATEGORY in "${CATEGORIES[@]}"; do
    echo ""
    echo "=============================="
    echo "  Category: ${CATEGORY}"
    echo "=============================="
    bash scripts/run_experiment.sh "${MODEL}" "${CATEGORY}" "${IMAGE_TYPE}" "${PROMPT_TYPE}"
done

echo ""
echo "All categories complete."
