# Contributing to ORBIT

Thank you for your interest in contributing! This document covers the essentials.

## Getting started

```bash
git clone https://github.com/castorini/orbit.git
cd orbit
uv sync
```

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproducer
- **New topics** — add a `wikipedia-categories/<topic>.txt` file and submit a PR
- **New data rounds** — add a script under `data/round-N-*/` following the existing structure
- **Model improvements** — update training hyperparameters or reward functions in `models/train/`
- **Evaluation benchmarks** — extend `models/data_process/prepare_eval_data.py` with new datasets

## Pull request checklist

- [ ] Code follows the existing style (no unnecessary comments, type hints where practical)
- [ ] New scripts include a docstring describing usage
- [ ] Any new dependencies are added to `pyproject.toml`
- [ ] Relevant README files are updated

## Project layout

See the root [`README.md`](README.md) for the full structure and [`data/README.md`](data/README.md) for the data pipeline.

## Questions

Open a GitHub issue or discussion — we are happy to help.
