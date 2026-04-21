"""
Create seed JSONL files from Wikipedia category lists.

Reads one or more category `.txt` files from the wikipedia-categories folder,
queries the Wikipedia API for page members of each category, and writes one
JSONL file per topic to the output directory.

Each output record:
    {"_id": "<md5>", "seed": "<page title>", "seed_url": "<wiki url>", "category": "<Category:Foo>"}

Usage — single topic:
    python create_seeds.py --topic mathematics

Usage — multiple topics:
    python create_seeds.py --topic mathematics science_and_technology

Usage — all topics:
    python create_seeds.py --all

Optional flags:
    --categories_dir   Path to category txt files  (default: ./wikipedia-categories)
    --output_dir       Output directory            (default: ../outputs/round-1-seeds)
    --max_per_category Max pages to sample per category (default: 50)
    --seed             Random seed for reproducibility
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import category_members
from tqdm import tqdm


CATEGORIES_DIR = Path(__file__).parent / "wikipedia-categories"
OUTPUT_DIR     = Path(__file__).parent.parent / "outputs" / "round-1-seeds"


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_completed(output_file: Path) -> set[str]:
    """Return set of _ids already written to output_file (for resuming)."""
    seen: set[str] = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(json.loads(line).get("_id", ""))
    return seen


def process_topic(
    topic: str,
    categories_dir: Path,
    output_dir: Path,
    max_per_category: int,
    rng: random.Random,
) -> int:
    cat_file = categories_dir / f"{topic}.txt"
    if not cat_file.exists():
        print(f"[{topic}] category file not found: {cat_file}", file=sys.stderr)
        return 0

    categories = [c for c in cat_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    print(f"[{topic}] {len(categories)} categories found")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{topic}.jsonl"
    completed = load_completed(output_file)
    if completed:
        print(f"[{topic}] resuming — {len(completed)} seeds already written")

    written = 0
    with open(output_file, "a", encoding="utf-8") as out:
        for category in tqdm(categories, desc=topic, unit="cat"):
            try:
                results = category_members.retrieve(category, types=["page"])
            except Exception as exc:
                print(f"[{topic}] {category}: {exc}", file=sys.stderr)
                continue

            if not results:
                continue

            if len(results) > max_per_category:
                results = rng.sample(results, max_per_category)

            for r in results:
                _id = md5(r["name"])
                if _id in completed:
                    continue
                record = {
                    "_id":      _id,
                    "seed":     r["name"],
                    "seed_url": r["link"],
                    "category": category,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed.add(_id)
                written += 1

    print(f"[{topic}] wrote {written} new seeds → {output_file}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Create seed JSONL files from Wikipedia categories")

    topic_group = parser.add_mutually_exclusive_group(required=True)
    topic_group.add_argument("--topic", nargs="+", metavar="TOPIC", help="One or more topic names (without .txt)")
    topic_group.add_argument("--all", action="store_true", help="Process every .txt file in the categories directory")

    parser.add_argument("--categories_dir", type=Path, default=CATEGORIES_DIR)
    parser.add_argument("--output_dir",     type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max_per_category", type=int, default=50, help="Max pages sampled per category (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.all:
        topics = sorted(p.stem for p in args.categories_dir.glob("*.txt"))
        if not topics:
            print(f"No .txt files found in {args.categories_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Processing all {len(topics)} topics: {', '.join(topics)}")
    else:
        topics = args.topic

    total = 0
    for topic in topics:
        total += process_topic(topic, args.categories_dir, args.output_dir, args.max_per_category, rng)

    print(f"\nDone. Total seeds written: {total}")


if __name__ == "__main__":
    main()
