# models/train

Scripts for training search-augmented LLMs with GRPO and serving the web search retriever.

---

## 1. DDGS Web Search

A FastAPI server that handles search queries during training. Each query is fanned out to multiple backends (Google, Brave, Bing, Wikipedia, Grokipedia) in parallel, results are deduplicated, and the top-k are returned.

**Start the server:**
```bash
bash ddgs_web_search.sh          # default port 8280
bash ddgs_web_search.sh 8050     # custom port
```

**Test it:**
```bash
python test_retriever.py
```

**Files:**
- `ddgs_web_search.py` — FastAPI retrieval server with async LRU cache and parallel fan-out
- `ddgs_web_search.sh` — one-line launcher
- `test_retriever.py` — quick sanity check against a running server

---

## 2. Training Search Agents with GRPO and verl-tool

Trains a search-augmented LLM using GRPO via [verl-tool](https://github.com/RLHF-V/VeRL-Tool). The script starts the web search server, launches the verl-tool server, then runs the training loop.

**Activate the environment first:**
```bash
source /home/n3thakur/scratch/2025/verl-tool/.venv/bin/activate
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
```

**Run interactively:**
```bash
bash run_grpo.sh                                                       # defaults
bash run_grpo.sh Qwen/Qwen3-4B data/mix-nq-hotpotqa-orbit-ratio-1-1-1  # custom model + dataset
# bash run_grpo.sh Qwen/Qwen3-4B data/mix-nq-hotpotqa-infoseek-ratio-1-1-1
```

**Submit to SLURM (Compute Canada):**
```bash
sbatch run_grpo.sh
```

Edit the variables at the top of `run_grpo.sh` to change the model, dataset, or any hyperparameter.

**Files:**
- `run_grpo.sh` — main training script with all hyperparameters hardcoded at the top

---

## 3. Uploading a Model Once Trained

After training, checkpoints are saved in FSDP sharded format. This script merges the shards into a standard HuggingFace model and optionally uploads it to the Hub.

**Usage:**
```bash
bash model_merge_upload_hf.sh \
    /home/n3thakur/scratch/browsecomp/models/<run>/global_step_75/actor \
    /home/n3thakur/scratch/browsecomp/hf_models/orbit-4b-step-75 \
    nthakur/orbit-4b-step-75    # optional: omit to skip Hub upload
```

**Files:**
- `model_merge_upload_hf.sh` — merges FSDP shards and optionally pushes to HuggingFace Hub

---

## Hyperparameters

| Module | Parameter | Value |
|---|---|---|
| **Data** | Training batch size | 256 |
| | Val batch size | 256 |
| | Max prompt length | 2,048 |
| | Max response length | 8,192 |
| | Max action length | 2,048 |
| | Max observation length | 1,024 |
| | Retriever top-k | 5 |
| | Total training steps | 1,005 |
| | Total epochs | 10 |
| **Actor** | RL algorithm | GRPO |
| | PPO mini-batch size | 32 |
| | PPO micro-batch size per GPU | 1 |
| | Learning rate | 1e-6 |
| | LR warmup steps | 10 |
| | LR warmup ratio | 0.285 |
| | KL loss coefficient | 0.0 |
| | KL loss type | low_var_kl |
| | Entropy coefficient | 0.0 |
| | Parallelism strategy | FSDP (full sharding) |
| | Parameter / optimizer offload | Yes |
| **Rollout** | Max turns per trajectory | 5 |
| | Group size G (rollouts per sample) | 8 |
| | Temperature | 1.0 |
| | Top-p | 1.0 |
| | Top-k | disabled (-1) |
| | Max concurrent trajectories | 32 |
| | vLLM GPU memory utilisation | 0.6 |
| | vLLM max model length | 8,192 |
| | Rollout mode | Async (vLLM v1) |
| **Reward** | Reward function | Exact Match (EM) |
| | Mask observations in loss | Yes |
| **Infrastructure** | GPUs | 4× H100 SXM5 80GB |
| | System memory | 256GB |
| | CPUs (allocated) | 48 |
| | Checkpoint frequency | Every 5 steps |
