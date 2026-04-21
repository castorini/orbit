# models/data_process

Scripts to prepare training and evaluation datasets as parquet files for ORBIT and baseline GRPO training.

---

## Setup

```bash
export HF_HOME=/home/n3thakur/scratch/cache
export HF_TOKEN=hf_xxxxx
```

Datasets are saved under `../train/data/` by default (relative to this directory).

---

## 1. Training data — `prepare_train_data.py`

Downloads and merges datasets into a single `train.parquet` with ratio-based sampling.

**Supported datasets:** `nq`, `hotpotqa` (FlashRAG), any HuggingFace QA dataset, `Lk123/InfoSeek` (local JSONL)

```bash

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

# NQ + HotpotQA
python prepare_train_data.py \
    --datasets 'nq,hotpotqa' \
    --local_dir ../train/data/nq-hotpotqa 
```

---

## 2. Evaluation data — `prepare_eval_data.py`

Downloads and merges evaluation benchmarks into a single `test.parquet` with up to `--max_samples` per dataset.

**Supported datasets:** `nq`, `triviaqa`, `popqa`, `hotpotqa`, `2wikimultihopqa`, `musique`, `bamboogle`, `frames`, `gaia`, `monaco`, `webwalkerqa`, `webshaper`

```bash
# All 12 benchmarks (125 samples each)
python prepare_eval_data.py \
    --data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle,frames,gaia,monaco,webwalkerqa,webshaper \
    --local_dir ../train/data/eval-all-12-datasets

# Hard benchmarks only
python prepare_eval_data.py \
    --data_sources frames,gaia,monaco \
    --local_dir ../train/data/eval-hard-3-datasets
```

**Note:** `monaco` is downloaded automatically from [`allenai/MoNaCo_Benchmark`](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark). `Lk123/InfoSeek` is downloaded automatically from [`Lk123/InfoSeek`](https://huggingface.co/datasets/Lk123/InfoSeek).
