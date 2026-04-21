"""
Prepare training parquet files for orbit GRPO training.

When --ratio is omitted, all samples from every dataset are used (no subsampling).
When --ratio is provided (e.g. 1:1:1), datasets are balanced proportionally.

Usage:
    export HF_HOME=/home/n3thakur/scratch/cache

    # NQ + HotpotQA — all samples, no subsampling
    python prepare_train_data.py \
        --datasets 'nq,hotpotqa' \
        --local_dir ../train/data/nq-hotpotqa

    # ORBIT + NQ + HotpotQA (1:1:1)
    python prepare_train_data.py \
        --datasets 'nq,hotpotqa,orbit-ai/orbit-20k' \
        --local_dir ../train/data/mix-nq-hotpotqa-orbit-ratio-1-1-1 \
        --ratio 1:1:1

    # InfoSeek + NQ + HotpotQA (1:1:1)
    python prepare_train_data.py \
        --datasets 'nq,hotpotqa,Lk123/InfoSeek' \
        --local_dir ../train/data/mix-nq-hotpotqa-infoseek-ratio-1-1-1 \
        --ratio 1:1:1
"""

import hashlib
import json
import os
import argparse
import datasets
from huggingface_hub import hf_hub_download

FLASHRAG_NAMES = {"nq", "hotpotqa"}

PROMPT_TEMPLATE = (
    "Answer the given question. Please break down the question, using it to plan a potential solution trajectory. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want, but make you sure to avoid duplicate searches. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. "
    "For example, a short answer can be <answer> Beijing </answer> or longer such as <answer> Sydney and Athens </answer>. "
    "Question: {question}\n"
)


def make_prompt(question: str) -> list:
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        question = example.get("question") or example["root"]["question"]
        question = question.strip()
        prompt = make_prompt(question)

        if data_source == "orbit":
            return {
                "id": example["_id"],
                "question": question,
                "golden_answers": [example["answer"]],
                "data_source": data_source,
                "prompt": prompt,
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": {"target": [example["answer"]]}},
                "extra_info": {"split": split, "index": idx},
            }

        if data_source == "infoseek":
            entity = example["root"]["entity"]
            return {
                "id": md5(question),
                "question": question,
                "golden_answers": [entity],
                "data_source": data_source,
                "prompt": prompt,
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": {"target": [entity]}},
                "extra_info": {"split": split, "index": idx},
            }

        # FlashRAG datasets (nq, hotpotqa, …)
        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule", "ground_truth": {"target": example["golden_answers"]}},
            "extra_info": {"split": split, "index": idx},
        }

    return process_fn


KEEP_COLS = {"id", "question", "golden_answers", "data_source", "prompt", "ability", "reward_model", "extra_info"}


def load_and_process(ds_name: str) -> datasets.Dataset:
    print(f"Loading: {ds_name}")

    if ds_name in FLASHRAG_NAMES:
        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", ds_name)["train"]
        ds = ds.map(make_map_fn("train", ds_name), with_indices=True)

    elif "orbit" in ds_name.lower():
        ds = datasets.load_dataset(ds_name, split="train")
        ds = ds.map(make_map_fn("train", "orbit"), with_indices=True)

    elif "infoseek" in ds_name.lower():
        local_path = hf_hub_download(
            repo_id="Lk123/InfoSeek",
            filename="data/InfoSeek-Hard-18K.jsonl",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        with open(local_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        ds = datasets.Dataset.from_list(rows)
        ds = ds.map(make_map_fn("train", "infoseek"), with_indices=True)

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.remove_columns(drop) if drop else ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated dataset names, e.g. 'nq,hotpotqa,Lk123/InfoSeek'")
    parser.add_argument("--ratio", type=str, default=None,
                        help="Colon-separated sampling ratios, e.g. '1:1:1'. Omit to use all samples.")
    DATA_ROOT = os.path.join(os.path.dirname(__file__), "../train/data")
    parser.add_argument("--local_dir", type=str,
                        default=os.path.join(DATA_ROOT, "train"),
                        help="Output directory for train.parquet")
    args = parser.parse_args()

    ds_names = [n.strip() for n in args.datasets.split(",")]
    processed = [load_and_process(n) for n in ds_names]

    if args.ratio is None:
        # No ratio — use all samples from every dataset
        sampled = []
        for name, ds in zip(ds_names, processed):
            print(f"  {name}: {len(ds)} samples (all)")
            sampled.append(ds)
    else:
        ratios = list(map(int, args.ratio.split(":")))
        if len(ds_names) != len(ratios):
            raise ValueError("--datasets and --ratio must have the same number of entries")
        lengths = [len(ds) for ds in processed]
        min_unit = min(lengths[i] // ratios[i] for i in range(len(ds_names)))
        sampled = []
        for i, ds in enumerate(processed):
            n = min_unit * ratios[i]
            ds = ds.shuffle(seed=42).select(range(n))
            print(f"  {ds_names[i]}: {n} samples")
            sampled.append(ds)

    final = datasets.concatenate_datasets(sampled).shuffle(seed=42)
    os.makedirs(args.local_dir, exist_ok=True)
    out = os.path.join(args.local_dir, "train.parquet")
    final.to_parquet(out)
    print(f"Saved {len(final)} records → {out}")
