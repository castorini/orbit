#!/bin/bash
# Required packages:
# conda install -c pytorch -c nvidia faiss-gpu=1.8.0
# uv pip install torch sentence-transformers datasets fastapi uvicorn pydantic tqdm numpy

# We run this on 4xA6000 GPUs (if you GPUs with more VRAM, you can decrease the number of GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

index_file=retriever_index/bge_m3_Flat.index
corpus_file=retriever_index/wiki-24.jsonl
retriever_path="BAAI/bge-m3"

# Run the retrieval server on port 8280
echo "Running retrieval server on host: $(hostname)"
echo "Host IPs: $(hostname -I)"

python retrieval_server_bge.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 5 \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --port 8280
