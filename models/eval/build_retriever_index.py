"""
Build a sharded FAISS flat index and JSONL corpus from a HuggingFace dataset
that contains pre-computed BGE-M3 embeddings.

Usage:
    export OMP_NUM_THREADS=4
    export HF_HOME=/u3/n3thakur/projects/cache
    export DATASETS_HF_HOME=/u3/n3thakur/projects/cache

    python build_retriever_index.py \
        --dataset Upstash/wikipedia-2024-06-bge-m3 \
        --lang en \
        --corpus_path wiki-24.jsonl \
        --output_dir retriever_index \
        --shard_size 5000000 \
        --batch_size 8192 \
        --metric IP
"""

import argparse
import json
import os
import string

import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def shard_filename(shard_id: int) -> str:
    """Map integer shard_id to a filename suffix: 0→part_a, 1→part_b, ..., 25→part_z, 26→part_ba, ..."""
    letters = string.ascii_lowercase
    name = ""
    n = shard_id
    while True:
        name = letters[n % 26] + name
        n = n // 26
        if n == 0:
            break
    return f"part_{name}.index"


def build_corpus_and_index(
    dataset_name: str,
    lang: str,
    batch_size: int,
    metric: str,
    corpus_path: str,
    shard_size: int,
    output_dir: str,
    max_samples: int = None,
):
    os.makedirs(output_dir, exist_ok=True)
    corpus_full_path = os.path.join(output_dir, corpus_path)

    print(f"Streaming dataset: {dataset_name} (lang={lang})")
    dataset = load_dataset(dataset_name, lang, split="train", streaming=True)

    first_item = next(iter(dataset))
    dim = len(first_item["embedding"])
    print(f"Embedding dimension: {dim}")

    # Reload after consuming first item
    dataset = load_dataset(dataset_name, lang, split="train", streaming=True)

    def new_index():
        return faiss.IndexFlatIP(dim) if metric == "IP" else faiss.IndexFlatL2(dim)

    shard_id = 0
    idx_in_shard = 0
    embeddings_buffer = []
    index = new_index()

    with open(corpus_full_path, "w", encoding="utf-8") as corpus_file:
        for global_idx, item in enumerate(tqdm(dataset, desc="Building corpus + index")):
            # Write corpus record
            contents = f'"{item["title"]}"\n{item["text"]}'
            corpus_file.write(json.dumps({"id": global_idx, "contents": contents}, ensure_ascii=False) + "\n")

            embeddings_buffer.append(np.array(item["embedding"], dtype="float32"))
            idx_in_shard += 1

            if len(embeddings_buffer) >= batch_size:
                index.add(np.vstack(embeddings_buffer))
                embeddings_buffer = []

            if idx_in_shard >= shard_size:
                if embeddings_buffer:
                    index.add(np.vstack(embeddings_buffer))
                    embeddings_buffer = []

                shard_path = os.path.join(output_dir, shard_filename(shard_id))
                faiss.write_index(index, shard_path)
                print(f"Saved shard {shard_path} ({index.ntotal} vectors)")

                shard_id += 1
                idx_in_shard = 0
                index = new_index()

            if max_samples and global_idx + 1 >= max_samples:
                print(f"Stopping at max_samples={max_samples}")
                break

    # Flush remaining vectors
    if embeddings_buffer:
        index.add(np.vstack(embeddings_buffer))
    if index.ntotal > 0:
        shard_path = os.path.join(output_dir, shard_filename(shard_id))
        faiss.write_index(index, shard_path)
        print(f"Saved final shard {shard_path} ({index.ntotal} vectors)")

    print(f"Done. Corpus written to {corpus_full_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build sharded FAISS index from HuggingFace embedding dataset")
    parser.add_argument("--dataset", type=str, default="Upstash/wikipedia-2024-06-bge-m3")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--corpus_path", type=str, default="wiki-24.jsonl")
    parser.add_argument("--output_dir", type=str, default="retriever_index", help="Directory to write shards and corpus")
    parser.add_argument("--shard_size", type=int, default=5_000_000, help="Vectors per shard")
    parser.add_argument("--batch_size", type=int, default=8192, help="Embeddings added to FAISS per batch")
    parser.add_argument("--metric", type=str, default="IP", choices=["IP", "L2"])
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total samples (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_corpus_and_index(
        dataset_name=args.dataset,
        lang=args.lang,
        batch_size=args.batch_size,
        metric=args.metric,
        corpus_path=args.corpus_path,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
