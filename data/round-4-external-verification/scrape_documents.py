"""Async URL scraper using trafilatura.

Reads a list of URLs (plain text, one per line, or JSONL with {"url": ...}),
fetches each page, extracts clean text with trafilatura, and writes results to
two JSONL files:

  success.jsonl  — {"url": "...", "content": "..."}
  failed.jsonl   — {"url": "...", "error": "..."}

The script is resume-safe: URLs already present in either output file are
skipped on re-run, so it can be interrupted and restarted freely.

Output format for successes is intentionally compatible with the
scraped_documents field consumed by external_verification.py.
"""

import argparse
import asyncio
import json
import os
from typing import List, Set, Tuple

import aiofiles
import aiohttp
import trafilatura
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_urls(input_path: str) -> List[str]:
    """Return a deduplicated list of URLs from a plain-text or JSONL file.

    Auto-detection: if the first non-empty line starts with '{', the file is
    treated as JSONL with {"url": ...} objects; otherwise every non-empty line
    is used as a raw URL.
    """
    with open(input_path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if not lines:
        return []

    urls: Set[str] = set()
    is_jsonl = lines[0].startswith("{")

    for line in lines:
        if is_jsonl:
            try:
                obj = json.loads(line)
                url = obj.get("url", "")
                if url:
                    urls.add(url)
                else:
                    print(f"[warn] No 'url' key in: {line[:120]}")
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed JSON: {line[:120]}")
        else:
            urls.add(line)

    return list(urls)


def read_already_processed(success_path: str, failed_path: str) -> Set[str]:
    """Return the set of URLs already written to either output file."""
    seen: Set[str] = set()
    for path in (success_path, failed_path):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        url = obj.get("url")
                        if isinstance(url, str) and url:
                            seen.add(url)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    return seen


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

async def fetch_html(
    session: aiohttp.ClientSession, url: str, timeout_s: int
) -> Tuple[bool, str]:
    """Fetch a URL. Returns (ok, html_or_error_message)."""
    try:
        async with session.get(url, timeout=timeout_s) as resp:
            text = await resp.text(errors="ignore")
            if resp.status == 200:
                return True, text
            return False, f"HTTP {resp.status}: {text[:2000]}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def extract_text(html: str) -> str:
    extracted = trafilatura.extract(html, favor_recall=True)
    return extracted or ""


async def process_url(
    url: str,
    session: aiohttp.ClientSession,
    success_f,
    failed_f,
    write_lock: asyncio.Lock,
    timeout_s: int,
) -> bool:
    ok, payload = await fetch_html(session, url, timeout_s)

    if not ok:
        async with write_lock:
            await failed_f.write(
                json.dumps({"url": url, "error": payload}, ensure_ascii=False) + "\n"
            )
            await failed_f.flush()
        return False

    content = extract_text(payload)
    if content.strip():
        async with write_lock:
            await success_f.write(
                json.dumps({"url": url, "content": content}, ensure_ascii=False) + "\n"
            )
            await success_f.flush()
        return True
    else:
        async with write_lock:
            await failed_f.write(
                json.dumps(
                    {"url": url, "error": "trafilatura returned empty content"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            await failed_f.flush()
        return False


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def run(
    input_path: str,
    success_path: str,
    failed_path: str,
    concurrency: int,
    timeout_s: int,
) -> None:
    already_done = read_already_processed(success_path, failed_path)
    all_urls = load_urls(input_path)
    urls = [u for u in all_urls if u not in already_done]

    print(f"Total unique URLs : {len(all_urls)}")
    print(f"Already processed : {len(already_done)}")
    print(f"To scrape now     : {len(urls)}")

    if not urls:
        print("Nothing to do.")
        return

    os.makedirs(os.path.dirname(success_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(failed_path) or ".", exist_ok=True)

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        async with (
            aiofiles.open(success_path, "a", encoding="utf-8") as success_f,
            aiofiles.open(failed_path, "a", encoding="utf-8") as failed_f,
        ):
            async def bounded(u: str) -> bool:
                async with sem:
                    return await process_url(
                        u, session, success_f, failed_f, write_lock, timeout_s
                    )

            tasks = [asyncio.create_task(bounded(u)) for u in urls]

            completed = successes = 0
            pbar = tqdm(total=len(urls), desc="Scraping", unit="url")
            try:
                for fut in asyncio.as_completed(tasks):
                    res = await fut
                    completed += 1
                    if res:
                        successes += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=successes, fail=completed - successes)
            finally:
                pbar.close()

    print(
        f"\nDone. scraped={completed}, success={successes}, failed={completed - successes}"
    )
    print(f"  Results  → {success_path}")
    print(f"  Failures → {failed_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape and extract text from a list of URLs using trafilatura. "
            "Accepts a plain-text file (one URL per line) or a JSONL file "
            "({\"url\": ...} per line). Resume-safe."
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to URL list (plain text or JSONL with {\"url\": ...})",
    )
    parser.add_argument(
        "--success", default="results/scraped_success.jsonl",
        help="Output JSONL for successfully scraped pages  (default: results/scraped_success.jsonl)",
    )
    parser.add_argument(
        "--failed", default="results/scraped_failed.jsonl",
        help="Output JSONL for failed / empty pages  (default: results/scraped_failed.jsonl)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=40,
        help="Max concurrent HTTP requests  (default: 40)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Per-request timeout in seconds  (default: 30)",
    )
    args = parser.parse_args()

    print(f"Input       : {args.input}")
    print(f"Success out : {args.success}")
    print(f"Failed out  : {args.failed}")
    print(f"Concurrency : {args.concurrency}  |  Timeout : {args.timeout}s")
    print()

    asyncio.run(
        run(
            input_path=args.input,
            success_path=args.success,
            failed_path=args.failed,
            concurrency=max(1, args.concurrency),
            timeout_s=max(1, args.timeout),
        )
    )


if __name__ == "__main__":
    main()
