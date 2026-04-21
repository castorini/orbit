# ORBIT Data Pipeline

Four sequential rounds that turn raw Wikipedia seeds into a verified, high-quality QA dataset.

```
data/
├── round-1-seed-creation/       # Wikipedia page titles → seed JSONL
├── round-2-qa-generation/       # Seeds → inverted QA pairs (DeepSeek browser)
├── round-3-self-verification/   # QA pairs → self-verified answers (DeepSeek browser)
├── round-4-external-verification/  # Verified pairs → externally judged (gpt-oss via vLLM)
└── outputs/
    ├── round-1-seeds/           # <topic>.jsonl  {_id, seed, seed_url, category}
    ├── round-2-qa-pairs/        # responses.jsonl  {_id, seed, inverted_question, answer, ...}
    ├── round-3-verified/        # verified.jsonl  {_id, question, answer, verification, ...}
    └── round-4-judged/          # judged.jsonl    {_id, question, answer, judge_correctness, ...}
```

## Round overview

| Round | Input | Script | Output |
|-------|-------|--------|--------|
| 1 — Seed creation | Wikipedia category `.txt` files | `create_seeds.py` | `{_id, seed, seed_url, category}` |
| 2 — QA generation | Round-1 seeds | `deepseek_generate_qa.py` | `{_id, seed, inverted_question, answer, response, reasoning}` |
| 3 — Self-verification | Round-2 QA pairs | `deepseek_self_verify.py` | `{_id, question, answer, verification, revised_answer, ...}` |
| 4 — External verification | Round-3 output or HF dataset | `external_verification.py` | `{_id, question, answer, predicted_answer, judge_correctness, ...}` |

## Topics

15 domains, each with its own category file:

`art` · `code` · `finance` · `geography` · `history` · `law` · `mathematics` · `medicine` · `music` · `politics` · `puzzles` · `science_and_technology` · `sports` · `tv_shows_and_movies` · `video_games`

## Running the full pipeline

```bash
# Round 1 — create seeds for all topics
python round-1-seed-creation/create_seeds.py --all

# Round 2 — generate QA pairs
bash round-2-qa-generation/deepseek_generate_qa.sh

# Round 3 — self-verify with DeepSeek
bash round-3-self-verification/deepseek_self_verify.sh

# Round 4 — external judge via vLLM (start server first — see round-4 README)
bash round-4-external-verification/external_verification.sh
```
