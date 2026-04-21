#!/bin/bash
# Local file (default):
#   bash deepseek_generate_qa.sh
#
# HuggingFace dataset:
#   bash deepseek_generate_qa.sh --hf_dataset orbit-ai/orbit-seeds --domain mathematics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/deepseek_generate_qa.py" \
    --entity_file "${SCRIPT_DIR}/../outputs/round-1-seeds/test.jsonl" \
    --output_file "${SCRIPT_DIR}/../outputs/round-2-qa-pairs/responses.jsonl" \
    --max_chat_length 30 \
    "$@"
