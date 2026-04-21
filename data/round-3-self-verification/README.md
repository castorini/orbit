# Round 3 — Self-Verification

For each (question, answer) pair from Round 2, DeepSeek performs an independent web search, produces a verification checklist with citations, and optionally revises the answer. Only pairs that pass self-verification proceed to Round 4.

## Structure

```
round-3-self-verification/
├── deepseek_self_verify.py   # main script
└── deepseek_self_verify.sh   # launcher
```

Output lands in `../outputs/round-3-verified/verified.jsonl`.

## Input format

Records must contain `_id` and either `inverted_question` or `question`, plus `answer`.  
Reads from a local JSONL file or directly from a HuggingFace dataset.

## Output format

```json
{
  "_id":           "d2d1c26f78ddc08c03f9fb4819793dda",
  "question":      "Which painting technique ...",
  "answer":        "Tenebrism",
  "verification":  "<DeepSeek verification markdown>",
  "reasoning":     "<chain-of-thought markdown>"
}
```

## Usage

```bash
# from local file
bash deepseek_self_verify.sh

# from HuggingFace dataset
bash deepseek_self_verify.sh --hf_dataset orbit-ai/orbit-stage-1-44k

# filter by topic
python deepseek_self_verify.py \
    --hf_dataset orbit-ai/orbit-stage-1-44k \
    --topic mathematics \
    --output_file ../outputs/round-3-verified/mathematics.jsonl
```

Same resume-safety and browser-restart behaviour as Round 2.

## Dependencies

Same as Round 2: `selenium`, `undetected-chromedriver`, `beautifulsoup4`, `markdownify`, `tqdm`.
