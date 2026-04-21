# Round 1 — Seed Creation

Queries the Wikipedia API for page members of curated categories and writes one seed JSONL file per topic.

## Structure

```
round-1-seed-creation/
├── create_seeds.py          # main script
└── wikipedia-categories/    # one .txt file per topic
    ├── art.txt
    ├── mathematics.txt
    └── ...                  # 15 topics total
```

Output lands in `../outputs/round-1-seeds/<topic>.jsonl`.

## Output format

Each line is a JSON object:

```json
{
  "_id":      "d2d1c26f78ddc08c03f9fb4819793dda",
  "seed":     "Tenebrism",
  "seed_url": "https://en.wikipedia.org/wiki/Tenebrism",
  "category": "Category:Painting"
}
```

`_id` is the MD5 hash of the seed title, used as a stable deduplication key.

## Usage

```bash
# single topic
python create_seeds.py --topic mathematics

# several topics
python create_seeds.py --topic mathematics science_and_technology history

# all 15 topics
python create_seeds.py --all

# custom sample size or output directory
python create_seeds.py --all --max_per_category 20 --output_dir ./seeds
```

The script is **resume-safe**: if an output file already exists, already-written `_id`s are skipped so a crashed run can be restarted without duplicates.

## Adding a new topic

1. Create `wikipedia-categories/<topic>.txt` with one `Category:Foo` per line.
2. Run `python create_seeds.py --topic <topic>`.
