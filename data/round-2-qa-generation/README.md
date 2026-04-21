# Round 2 — QA Generation

Uses a Selenium-controlled DeepSeek browser session (Expert mode + DeepThink + Smart Search) to generate complex, multi-hop inverted questions from the Round-1 seeds.

## Structure

```
round-2-qa-generation/
├── deepseek_generate_qa.py   # main script
└── deepseek_generate_qa.sh   # launcher
```

Output lands in `../outputs/round-2-qa-pairs/responses.jsonl`.

## Output format

Each line is a JSON object:

```json
{
  "_id":               "d2d1c26f78ddc08c03f9fb4819793dda",
  "seed":              "Tenebrism",
  "inverted_question": "Which painting technique ...",
  "answer":            "Tenebrism",
  "response":          "<full markdown from DeepSeek>",
  "reasoning":         "<chain-of-thought markdown>"
}
```

`inverted_question` and `answer` are parsed inline from the XML block in `response` — no separate post-processing step is needed.

## Usage

```bash
# from local seed file
bash deepseek_generate_qa.sh

# from HuggingFace dataset
bash deepseek_generate_qa.sh --hf_dataset orbit-ai/orbit-seeds --domain mathematics

# directly
python deepseek_generate_qa.py \
    --entity_file ../outputs/round-1-seeds/mathematics.jsonl \
    --output_file ../outputs/round-2-qa-pairs/mathematics.jsonl \
    --max_chat_length 30
```

`--max_chat_length` controls how many seeds are processed before the browser is restarted (avoids context bloat). The script is **resume-safe**: already-completed `_id`s are skipped on restart.

## Dependencies

```bash
pip install selenium undetected-chromedriver beautifulsoup4 markdownify tqdm
```

A Chrome installation and a pre-logged-in Chrome profile are required. On first run the browser opens DeepSeek and waits up to 10 minutes for manual login, then saves the session to the profile directory.
