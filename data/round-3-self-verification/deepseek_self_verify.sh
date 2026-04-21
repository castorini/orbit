#!/bin/bash
# Local file (default):
#   bash run.sh
#
# HuggingFace dataset:
#   bash run.sh --hf_dataset orbit-ai/orbit-stage-1-44k
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/deepseek_self_verify.py" \
    --input_file "${SCRIPT_DIR}/../outputs/round-2/responses_test.jsonl" \
    --output_file "${SCRIPT_DIR}/outputs/round-3/verified.jsonl" \
    --max_chat_length 30 \
    "$@"
