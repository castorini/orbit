#!/usr/bin/env bash
# =============================================================================
# Round-4 External Verification via gpt-oss-120b or Qwen
# =============================================================================
#
# STEP 1 — Start the vLLM server (run once, in a separate terminal / tmux pane)
# -----------------------------------------------------------------------------
#
#   export OMP_NUM_THREADS=6
#   export HF_HOME=/u3/n3thakur/projects/cache
#   export VLLM_SKIP_P2P_CHECK=1
#   export NCCL_P2P_DISABLE=1
#   vllm serve openai/gpt-oss-120b \
#       --tensor-parallel-size 1 \
#       --async-scheduling \
#       --port 6050
#
#   # For multi-GPU (e.g. 4 A6000 GPUs hosting with gpt-oss-20b):
#   vllm serve openai/gpt-oss-20b \
#       --tensor-parallel-size 4 \
#       --async-scheduling \
#       --port 6050 \
#       --gpu-memory-utilization 0.8
#
#   # For multi-GPU (e.g. 4 A6000 GPUs hosting with Qwen3-4B-Instruct-2507):
#   vllm serve Qwen/Qwen3-4B-Instruct-2507 \
#       --max-model-len 262144 \
#       --tensor-parallel-size 4 \
#       --async-scheduling \
#       --port 6080 \
#       --gpu-memory-utilization 0.8
#
# =============================================================================
# STEP 2 — Run the evaluation
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS=1
export HF_HOME=/u3/n3thakur/projects/cache
export DATASETS_HF_HOME=/u3/n3thakur/projects/cache

MODEL_NAME="openai/gpt-oss-120b" # or Qwen/Qwen3-4B-Instruct-2507
BASE_URL="http://localhost:6050/v1" # or http://localhost:6080/v1
API_KEY="EMPTY"
DATASET_NAME="orbit-ai/orbit-stage-3-27k"
DATASET_SPLIT="train"
OUTPUT_PATH="${SCRIPT_DIR}/results/gpt_oss_120b.jsonl" # or ${SCRIPT_DIR}/results/qwen3-4b-instruct-2507.jsonl
NUM_THREADS=4

python "${SCRIPT_DIR}/external_verification.py" \
    --model-name "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-split "${DATASET_SPLIT}" \
    --output-path "${OUTPUT_PATH}" \
    --num-threads "${NUM_THREADS}"
