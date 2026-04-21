"""
Prepare evaluation parquet files for orbit.

Usage:
    export HF_HOME=/home/n3thakur/scratch/cache
    export HF_TOKEN=hf_xxxxx
    export DIR=/home/n3thakur/scratch/browsecomp

    # All 12 benchmarks
    python prepare_eval_data.py \
        --data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle,frames,gaia,monaco,webwalkerqa,webshaper \
        --local_dir ${DIR}/data/all-12-val-datasets-modified

"""

import json
import os
import argparse
import datasets
from huggingface_hub import hf_hub_download

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


def load_dataset(data_source: str, max_samples: int) -> datasets.Dataset:
    print(f"Loading: {data_source}")

    if data_source == "webshaper":
        raw = datasets.load_dataset("Alibaba-NLP/WebShaper", split="main")
        ds = raw.map(lambda x, i: {
            "id": f"test_{i}",
            "question": x["question"],
            "golden_answers": [x["answer"]],
        }, with_indices=True)

    elif data_source == "webwalkerqa":
        raw = datasets.load_dataset("callanwu/WebWalkerQA", split="main")
        ds = raw.filter(lambda x: isinstance(x["info"].get("lang"), str) and x["info"]["lang"].lower() == "en")
        ds = ds.map(lambda x, i: {
            "id": f"test_{i}",
            "question": x["question"],
            "golden_answers": [x["answer"]],
        }, with_indices=True)
        print(f"  WebWalkerQA: {len(ds)} English samples from {len(raw)}")

    elif data_source == "monaco":
        local_path = hf_hub_download(
            repo_id="allenai/MoNaCo_Benchmark",
            filename="monaco_version_1_release.jsonl",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        rows = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("validated_answer", [])
                if isinstance(ans, list) and all(isinstance(a, str) for a in ans):
                    rows.append({"ex_num": d["ex_num"], "question": d["question"], "golden_answers": ans})
                elif ans == [True]:
                    rows.append({"ex_num": d["ex_num"], "question": d["question"], "golden_answers": ["Yes"]})
                elif ans == [False]:
                    rows.append({"ex_num": d["ex_num"], "question": d["question"], "golden_answers": ["No"]})
        ds = datasets.Dataset.from_list(rows)
        ds = ds.map(lambda x: {"id": f"test_{x['ex_num']}"})
        ds = ds.remove_columns(["ex_num"])

    elif data_source == "frames":
        raw = datasets.load_dataset("google/frames-benchmark")
        orig_cols = raw["test"].column_names
        ds = raw.map(lambda x: {
            "id": f"test_{x['Unnamed: 0']}",
            "question": x["Prompt"],
            "golden_answers": [x["Answer"]],
        })
        ds = ds.remove_columns(orig_cols)["test"]

    elif data_source == "gaia":
        raw = datasets.load_dataset("VerlTool/deepsearch")["test_gaia"]
        orig_cols = raw.column_names
        ds = raw.map(lambda x, i: {
            "id": f"test_{i}",
            "question": x["prompt"][1]["content"],
            "golden_answers": x["reward_model"]["ground_truth"],
        }, with_indices=True)
        ds = ds.remove_columns(orig_cols)

    else:
        # FlashRAG datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, …
        raw = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
        split = "test" if "test" in raw else ("dev" if "dev" in raw else "train")
        print(f"  Using split: {split}")
        ds = raw[split]

    # Normalise question punctuation and attach prompt
    def process_fn(example, idx):
        q = example["question"].strip()
        if not q.endswith("?"):
            q += "?"
        return {
            "data_source": data_source,
            "prompt": make_prompt(q),
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule", "ground_truth": {"target": example["golden_answers"]}},
            "extra_info": {"split": "test", "index": idx},
        }

    ds = ds.map(process_fn, with_indices=True)

    n = min(len(ds), max_samples)
    ds = ds.shuffle(seed=42).select(range(n))
    print(f"  {data_source}: {n} samples")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_sources", type=str, required=True,
                        help="Comma-separated benchmark names")
    DATA_ROOT = os.path.join(os.path.dirname(__file__), "../train/data")
    parser.add_argument("--local_dir", type=str,
                        default=os.path.join(DATA_ROOT, "eval"),
                        help="Output directory for test.parquet")
    parser.add_argument("--max_samples", type=int, default=125,
                        help="Max samples per dataset (default: 125)")
    args = parser.parse_args()

    sources = [s.strip() for s in args.data_sources.split(",")]
    all_ds = [load_dataset(src, args.max_samples) for src in sources]

    final = datasets.concatenate_datasets(all_ds)
    os.makedirs(args.local_dir, exist_ok=True)
    out = os.path.join(args.local_dir, "test.parquet")
    final.to_parquet(out)
    print(f"Saved {len(final)} records → {out}")
