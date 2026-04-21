# ORBIT (Open Web-Reasoning for Information Retrieval Tasks)

Training code for orbit search agents, built on top of [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool).

## Structure

```
orbit/
├── models/
│   ├── train/
│   │   ├── run_grpo.sh                # GRPO training script
│   │   ├── ddgs_web_search.py         # FastAPI web search retrieval server
│   │   ├── ddgs_web_search.sh         # retrieval server launcher
│   │   ├── model_merge_upload_hf.sh   # merge FSDP checkpoints → HuggingFace
│   │   └── test_retriever.py          # sanity-check the retrieval server
│   └── data_process/
│       ├── prepare_train_data.py      # build train.parquet from HF datasets
│       └── prepare_eval_data.py       # build test.parquet from eval benchmarks
└── README.md
```

See [`models/train/README.md`](models/train/README.md) and [`models/data_process/README.md`](models/data_process/README.md) for detailed usage.

## Quick start

```bash
# 1. Activate environment
source /home/n3thakur/scratch/2025/verl-tool/.venv/bin/activate
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."

# 2. Run training
bash models/train/run_grpo.sh
```
