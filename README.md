# ORBIT

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/orbit-with-name-logo.png">
    <img alt="ORBIT" src="images/orbit-with-name-logo.png" width=30%>
  </picture>
</p>

<h3 align="center">
ORBIT: Open Web-Reasoning for Information Retrieval Tasks
</h3>

<p align="center">
|
<a href="https://arxiv.org/abs/2604.01195"><b>Paper</b></a> |
<a href="https://huggingface.co/orbit-ai/orbit-4b-v0.1"><b>Model</b></a> |
<a href="https://huggingface.co/datasets/orbit-ai/orbit-20k"><b>Dataset</b></a> |
<a href="data/README.md"><b>Data Pipeline</b></a> |
<a href="models/train/README.md"><b>Training</b></a> |
<a href="models/eval/README.md"><b>Evaluation</b></a> |
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.01195">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.01195-b31b1b.svg">
  </a>
  <a href="https://github.com/castorini/orbit/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/castorini/orbit?style=flat&logo=github&color=yellow">
  </a>
  <a href="https://github.com/castorini/orbit/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
  <a href="https://huggingface.co/orbit-ai/orbit-4b-v0.1">
    <img alt="Model" src="https://img.shields.io/badge/🤗-orbit--4b--v0.1-yellow">
  </a>
  <a href="https://huggingface.co/datasets/orbit-ai/orbit-20k">
    <img alt="Dataset" src="https://img.shields.io/badge/🤗-orbit--20k-blue">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg">
</p>

---

## Features

- 🌐 **4-round data pipeline** — turns raw Wikipedia seeds into a verified multi-hop QA dataset through generation, self-verification, and external judging
- 🤖 **GRPO training with web search** — trains search-augmented LLMs via [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) with a parallel multi-backend DDGS retrieval server
- 📊 **12-benchmark evaluation suite** — covers single-hop (NQ, TriviaQA, PopQA), multi-hop (HotpotQA, 2WikiMHQA, MuSiQue), and hard reasoning (FRAMES, GAIA, MoNaCo) benchmarks
- 🔍 **BGE retrieval index** — dense retrieval with FAISS for closed-domain evaluation

## 📚 Contents

- 🗂️ [Data Pipeline](data/README.md)
- 🌱 [Round 1 — Seed Creation](data/round-1-seed-creation/README.md)
- ❓ [Round 2 — QA Generation](data/round-2-qa-generation/README.md)
- ✅ [Round 3 — Self-Verification](data/round-3-self-verification/README.md)
- 🏅 [Round 4 — External Verification](data/round-4-external-verification/README.md)
- 🏋️ [Model Training](models/train/README.md)
- 📐 [Data Processing](models/data_process/README.md)
- 🔎 [Evaluation](models/eval/README.md)

## Data Pipeline

ORBIT QA pairs are built through four sequential rounds. See [`data/README.md`](data/README.md) for the full overview.

| Round | What it does | Key script |
|-------|-------------|------------|
| 1 | Wikipedia category pages → seed titles | `create_seeds.py` |
| 2 | Seeds → multi-hop inverted questions + answers | `deepseek_generate_qa.py` |
| 3 | Self-verification via DeepSeek web search | `deepseek_self_verify.py` |
| 4 | External judge via vLLM (gpt-oss-120b) | `external_verification.py` |

```bash
python data/round-1-seed-creation/create_seeds.py --all
bash data/round-2-qa-generation/deepseek_generate_qa.sh
bash data/round-3-self-verification/deepseek_self_verify.sh
bash data/round-4-external-verification/external_verification.sh
```

## Quick Start

**Install:**
```bash
uv sync
```

**Train:**
```bash
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
bash models/train/run_grpo.sh
```

**Evaluate:**
```bash
bash models/eval/run_eval.sh
```

## Repository Structure

```
orbit/
├── data/
│   ├── round-1-seed-creation/         # Wikipedia pages → seed JSONL
│   ├── round-2-qa-generation/         # Seeds → inverted QA pairs (DeepSeek browser)
│   ├── round-3-self-verification/     # QA pairs → self-verified answers (DeepSeek browser)
│   ├── round-4-external-verification/ # Verified pairs → externally judged (vLLM)
│   └── outputs/                       # Intermediate + final JSONL files
├── models/
│   ├── train/                         # GRPO training + DDGS retrieval server
│   ├── eval/                          # BGE retrieval index + evaluation
│   └── data_process/                  # Prepare train/eval parquet files
├── images/
├── pyproject.toml
└── README.md
```

## Acknowledgements

We thank the following open-source projects:
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) for the GRPO training framework for training search agents.
- [vLLM](https://github.com/vllm-project/vllm) for LLM inference.
- [DDGS](https://github.com/deedy5/ddgs) for a web search aggregator tool (used for search agent training with GRPO).
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for retrieval server design inspiration and initial experiments.


## Citation

If you find this repository helpful, feel free to cite our preprint [ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget](https://arxiv.org/abs/2604.01195):

```
@misc{thakur2026orbit,
      title={ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget}, 
      author={Nandan Thakur and Zijian Chen and Xueguang Ma and Jimmy Lin},
      year={2026},
      eprint={2604.01195},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.01195}, 
}
```
