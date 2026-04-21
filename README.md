# ORBIT (Open Web-Reasoning for Information Retrieval Tasks)

Training code and data pipeline for ORBIT search agents, built on top of [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool).

## Structure

```
orbit/
├── data/                              # 4-round dataset construction pipeline
│   ├── round-1-seed-creation/         # Wikipedia pages → seed JSONL
│   ├── round-2-qa-generation/         # Seeds → inverted QA pairs (DeepSeek browser)
│   ├── round-3-self-verification/     # QA pairs → self-verified answers (DeepSeek browser)
│   ├── round-4-external-verification/ # Verified pairs → externally judged (vLLM)
│   └── outputs/                       # Intermediate + final JSONL files
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

## Data pipeline

ORBIT QA pairs are constructed through four sequential rounds. See [`data/README.md`](data/README.md) for the full overview.

| Round | What it does | Key script |
|-------|-------------|------------|
| 1 | Wikipedia category pages → seed titles | `create_seeds.py` |
| 2 | Seeds → multi-hop inverted questions + answers | `deepseek_generate_qa.py` |
| 3 | Self-verification via DeepSeek web search | `deepseek_self_verify.py` |
| 4 | External judge via vLLM (gpt-oss-120b) | `external_verification.py` |

```bash
# Run the full pipeline
python data/round-1-seed-creation/create_seeds.py --all
bash data/round-2-qa-generation/deepseek_generate_qa.sh
bash data/round-3-self-verification/deepseek_self_verify.sh
bash data/round-4-external-verification/external_verification.sh
```

## Model training quick start

```bash
# 1. Activate environment
source /home/n3thakur/scratch/2025/verl-tool/.venv/bin/activate
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."

# 2. Run training
bash models/train/run_grpo.sh
```

See [`models/train/README.md`](models/train/README.md) and [`models/data_process/README.md`](models/data_process/README.md) for detailed usage.
