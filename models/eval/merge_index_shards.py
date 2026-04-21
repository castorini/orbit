"""
Concatenate multiple FAISS flat index shards into a single index file.
Shards are discovered automatically from the input directory (part_*.index),
sorted lexicographically to preserve the construction order.

Usage:
    python merge_index_shards.py \
        --shard_dir retriever_index \
        --output retriever_index/bge_m3_Flat.index

    # Or with explicit shard list:
    python merge_index_shards.py \
        --shards retriever_index/part_a.index retriever_index/part_b.index \
        --output retriever_index/bge_m3_Flat.index
"""

import argparse
import glob
import os

import faiss
import numpy as np


def concatenate_shards(shard_paths: list[str], output_path: str):
    if not shard_paths:
        raise ValueError("No shard files found.")

    print(f"Found {len(shard_paths)} shards to concatenate:")
    for p in shard_paths:
        print(f"  {p}")

    # Read first shard to get dimension and metric type
    first = faiss.read_index(shard_paths[0])
    d = first.d
    print(f"Embedding dimension: {d}")
    final_index = faiss.IndexFlatIP(d)

    for shard_path in shard_paths:
        print(f"Loading {shard_path} ...", end=" ", flush=True)
        shard = faiss.read_index(shard_path)
        n = shard.ntotal
        print(f"{n} vectors", end=" ", flush=True)
        vectors = shard.reconstruct_n(0, n)
        final_index.add(vectors)
        print(f"(running total: {final_index.ntotal})")

    print(f"\nFinal index: {final_index.ntotal} vectors")
    faiss.write_index(final_index, output_path)
    print(f"Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate FAISS index shards into a single flat index")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--shard_dir",
        type=str,
        default="retriever_index",
        help="Directory containing part_*.index shard files (auto-discovered, sorted)",
    )
    group.add_argument(
        "--shards",
        type=str,
        nargs="+",
        help="Explicit list of shard files in order",
    )
    parser.add_argument("--output", type=str, default="retriever_index/bge_m3_Flat.index", help="Output index filename")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.shards:
        shard_paths = args.shards
    else:
        pattern = os.path.join(args.shard_dir, "part_*.index")
        shard_paths = sorted(glob.glob(pattern))
        if not shard_paths:
            raise FileNotFoundError(f"No shard files matching '{pattern}' found in {args.shard_dir}")

    concatenate_shards(shard_paths, args.output)
