#!/bin/bash
# Usage: bash model_merge_upload_hf.sh <local_ckpt_dir> <target_hf_dir> [hf_upload_path]

local_dir=$1
target_dir=$2
hf_upload_path=${3:-}

# export HF_HOME=/home/n3thakur/scratch/cache (set if needed)
# export DATASETS_HF_HOME=/home/n3thakur/scratch/cache (set if needed)

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$local_dir" \
    --target_dir "$target_dir" \
    ${hf_upload_path:+--hf_upload_path "$hf_upload_path"}
