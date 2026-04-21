# Orbit Evaluation

This directory contains all scripts and utilities for evaluating Orbit models using a dense retrieval pipeline. Evaluation runs a single validation pass (no training) with the verl-tool framework, scoring answers with exact-match against ground truth.

## Directory Structure

```
eval/
├── README.md                   # This file
├── build_retriever_index.py    # Step 1: build sharded FAISS index from HF embedding dataset
├── merge_index_shards.py       # Step 2: merge shards into a single flat index
├── retrieval_server_bge.py     # Step 3: serve the FAISS index over HTTP (FastAPI)
├── retrieval_server_bge.sh     # Helper script to launch the retrieval server
└── run_eval.sh                 # Step 4: run full evaluation with verl-tool
```

---

## Prerequisites

Install dependencies into your conda environment:

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
uv pip install torch sentence-transformers datasets fastapi uvicorn pydantic tqdm numpy
```

---

## Step-by-Step Usage

### Step 1 — Build the FAISS Index

`build_retriever_index.py` streams a HuggingFace dataset that contains pre-computed BGE-M3 embeddings (e.g. [Upstash/wikipedia-2024-06-bge-m3](https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3)), writes a JSONL corpus file, and splits the FAISS index into shards of a fixed size.

```bash
export OMP_NUM_THREADS=4
export HF_HOME=/u3/n3thakur/projects/cache

python build_retriever_index.py \
    --dataset Upstash/wikipedia-2024-06-bge-m3 \
    --lang en \
    --corpus_path wiki-24.jsonl \
    --output_dir retriever_index \
    --shard_size 5000000 \
    --batch_size 8192 \
    --metric IP
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `Upstash/wikipedia-2024-06-bge-m3` | HuggingFace dataset with `embedding`, `title`, `text` fields |
| `--lang` | `en` | Language config to load |
| `--corpus_path` | `wiki-24.jsonl` | Output corpus filename (written inside `--output_dir`) |
| `--output_dir` | `retriever_index` | Directory for shards and corpus |
| `--shard_size` | `5000000` | Max vectors per shard file |
| `--batch_size` | `8192` | Vectors added to FAISS per batch |
| `--metric` | `IP` | Distance metric: `IP` (inner product) or `L2` |
| `--max_samples` | `None` | Cap total samples (useful for testing) |

**Output:** `retriever_index/wiki-24.jsonl` + `retriever_index/part_a.index`, `part_b.index`, ...

---

### Step 2 — Merge Index Shards

`merge_index_shards.py` concatenates the per-shard `.index` files produced above into a single flat FAISS index. Shards are discovered automatically by scanning for `part_*.index` files in sorted order.

```bash
python merge_index_shards.py \
    --shard_dir retriever_index \
    --output retriever_index/bge_m3_Flat.index
```

Or with an explicit ordered shard list:

```bash
python merge_index_shards.py \
    --shards retriever_index/part_a.index retriever_index/part_b.index \
    --output retriever_index/bge_m3_Flat.index
```

| Argument | Default | Description |
|---|---|---|
| `--shard_dir` | `retriever_index` | Directory to auto-discover `part_*.index` files |
| `--shards` | — | Explicit shard list (mutually exclusive with `--shard_dir`) |
| `--output` | `retriever_index/bge_m3_Flat.index` | Path for the merged index |

**Output:** `retriever_index/bge_m3_Flat.index` — a single `IndexFlatIP` ready to serve.

---

### Step 3 — Launch the Retrieval Server

`retrieval_server_bge.py` loads the merged FAISS index and the JSONL corpus into memory, then starts a FastAPI HTTP server. The evaluation script queries this server at runtime to fetch relevant passages.

#### Using the helper script

```bash
bash retrieval_server_bge.sh
```

The helper script defaults to:
- Index: `retriever_index/bge_m3_Flat.index`
- Corpus: `retriever_index/wiki-24.jsonl`
- Model: `BAAI/bge-m3`
- Port: `8000`
- GPUs: `0,1,2,3`

#### Running directly

```bash
python retrieval_server_bge.py \
    --index_path retriever_index/bge_m3_Flat.index \
    --corpus_path retriever_index/wiki-24.jsonl \
    --retriever_model BAAI/bge-m3 \
    --topk 5 \
    --faiss_gpu \
    --port 8000
```

| Argument | Default | Description |
|---|---|---|
| `--index_path` | _(required)_ | Path to the merged FAISS index |
| `--corpus_path` | _(required)_ | Path to the JSONL corpus |
| `--retriever_model` | `BAAI/bge-m3` | HuggingFace model name or local path |
| `--topk` | `3` | Default number of passages to retrieve |
| `--faiss_gpu` | `False` | Move FAISS index to GPU(s) for faster search |
| `--port` | `8000` | Port to serve on |

#### API

`POST /retrieve`

```json
{
  "queries": ["What is the capital of France?"],
  "topk": 5,
  "return_scores": false
}
```

Response:

```json
{
  "result": [
    [
      {"id": 123, "contents": "\"Paris\"\nParis is the capital of France..."},
      ...
    ]
  ]
}
```

When `return_scores=true`, each entry is `{"document": {...}, "score": 0.92}`.

---

### Step 4 — Run Evaluation

`run_eval.sh` starts the verl-tool tool server, then runs a single validation pass of the model using `verl_tool.trainer.main_ppo` with `trainer.val_before_train=True` and `total_epochs=0` / `total_training_steps=0` — meaning no training occurs.

> **Note:** The retrieval server (Step 3) must already be running before launching this script. Set `RETRIEVER_PORT` to match its port.

```bash
bash run_eval.sh <model_name> [dataset_name]

# Examples:
bash run_eval.sh orbit-ai/orbit-4b-v0.1
bash run_eval.sh /path/to/local/checkpoint all-8-wikipedia-test-datasets
```

| Argument | Default | Description |
|---|---|---|
| `$1` model_name | `orbit-ai/orbit-4b-v0.1` | HuggingFace model ID or local checkpoint path |
| `$2` dataset_name | `all-8-wikipedia-test-datasets` | Folder name under `eval/data/` containing `test.parquet` |

#### Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `rl_alg` | `gae` | Advantage estimator (PPO/GAE) |
| `batch_size` | `64` | Validation batch size |
| `max_response_length` | `4096` | Max tokens the model can generate per turn |
| `max_obs_length` | `1024` | Max tokens for retrieved passages fed back to model |
| `max_turns` | `5` | Max search-and-answer turns per question |
| `retriever_topk` | `5` | Passages retrieved per search query |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.8` | Nucleus sampling |
| `top_k` | `20` | Top-k sampling |
| `gpu_memory_utilization` | `0.4` | vLLM GPU memory fraction |
| `n_gpus_per_node` | `1` | Number of GPUs for the eval job |
| `reward_manager` | `search_r1_qa_em` | Exact-match scoring reward |

#### What the script does

1. Writes action stop tokens (`</search>`, `</answer>`) to a temp file.
2. Starts the `verl_tool` search tool server on a random port (30000–31000) with 4 workers and waits until it is healthy.
3. Runs `verl_tool.trainer.main_ppo` in evaluation-only mode.
4. Saves rollout records to `eval/verl_step_records/<run_name>-val/`.
5. Kills the tool server on exit.

#### Output

Rollout trajectories and scores are written to:

```
eval/verl_step_records/<run_name>-val/
```

The run name is automatically derived from the model and dataset names, e.g.:

```
orbit-4b-v0-all-8-wikipedia-eval-t0.7-search_r1-5turns-5docs
```
