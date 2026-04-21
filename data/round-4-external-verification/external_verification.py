import json
import os
import asyncio
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "http://localhost:6050/v1"


def make_async_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=0,
    )


def normalize_scraped_docs(scraped_docs: Optional[object]) -> Dict[str, str]:
    """Normalize scraped_documents to a dict of url->content."""
    if isinstance(scraped_docs, dict):
        return scraped_docs
    if isinstance(scraped_docs, list):
        if scraped_docs and isinstance(scraped_docs[0], str):
            return {f"doc_{i}": s for i, s in enumerate(scraped_docs)}
        if scraped_docs and isinstance(scraped_docs[0], dict):
            out = {}
            for i, d in enumerate(scraped_docs):
                url = d.get("url") or f"doc_{i}"
                content = d.get("content") or d.get("text") or ""
                out[url] = content
            return out
    return {}


async def generate_answer(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    scraped_docs: Dict[str, str],
) -> str:
    doc_texts = [f"URL: {url}\nContent: {content}" for url, content in scraped_docs.items()]
    evidence_text = "\n\n".join(doc_texts) if doc_texts else "No evidence available."

    prompt = f"""You are given a question and some evidence from scraped web documents. Please answer the question based on the evidence provided.

Question: {question}

Evidence:
{evidence_text}

Please provide your answer based on the evidence above."""

    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
        temperature=0.1,
        reasoning_effort="medium",
    )
    return response.choices[0].message.content.strip()


async def judge_correctness(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str,
) -> Tuple[bool, str]:
    def extract_verdict(judge_output: str) -> Optional[bool]:
        if not judge_output:
            return None
        upper = judge_output.upper()
        correct_idx = upper.rfind("JUDGE: CORRECT")
        incorrect_idx = upper.rfind("JUDGE: INCORRECT")
        if correct_idx != -1 and incorrect_idx != -1:
            return correct_idx > incorrect_idx
        return None

    prompt = f"""You are a judge evaluating whether a candidate response correctly answers a question, compared to a ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth_answer}

Candidate Response: {predicted_answer}

Please judge whether the candidate response is correct or incorrect with respect to the ground truth answer. You should provide a brief explanation for your judgment, then output Judge: CORRECT or Judge: INCORRECT."""

    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        extra_body={"top_k": 20},
    )

    content = response.choices[0].message.content.strip()
    verdict = extract_verdict(content)

    if verdict is not None:
        is_correct = verdict
    else:
        upper = content.upper()
        if "JUDGE: CORRECT" in upper:
            is_correct = True
        elif "JUDGE: INCORRECT" in upper:
            is_correct = False
        elif "CORRECT" in upper:
            is_correct = True
        elif "INCORRECT" in upper:
            is_correct = False
        else:
            print(f"Unparseable judge response: {content}")
            is_correct = False

    return is_correct, content


def load_processed_ids(output_path: str) -> Tuple[set, int]:
    """Load already-processed IDs from output file and tally correct count."""
    processed_ids: set = set()
    correctness_count = 0
    if not os.path.exists(output_path):
        return processed_ids, correctness_count
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            if (pid := result.get("id")) is not None:
                processed_ids.add(pid)
            if result.get("judge_correctness", False):
                correctness_count += 1
    return processed_ids, correctness_count


file_lock = asyncio.Lock()


async def save_result(output_path: str, result: Dict):
    """Append a single result to the JSONL output file (async-safe)."""
    async with file_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()


async def process_example(
    example: Dict,
    output_path: str,
    pbar: tqdm,
    correctness_count: List[int],
    semaphore: asyncio.Semaphore,
    pbar_lock: asyncio.Lock,
    client: AsyncOpenAI,
    model_name: str,
):
    async with semaphore:
        _id = example.get("_id", "")
        question = example.get("question", "")
        answer = example.get("answer", "")
        scraped_docs = normalize_scraped_docs(example.get("scraped_documents") or {})

        try:
            predicted_answer = await generate_answer(client, model_name, question, scraped_docs)
            is_correct, judge_output = await judge_correctness(
                client, model_name, question, predicted_answer, answer
            )

            if is_correct:
                correctness_count[0] += 1

            await save_result(
                output_path,
                {
                    "id": _id,
                    "question": question,
                    "answer": answer,
                    "scraped_documents": scraped_docs,
                    "predicted_answer": predicted_answer,
                    "judge_correctness": is_correct,
                    "judge_output": judge_output,
                },
            )
        except Exception as e:
            print(f"Error processing {_id}: {e}")

        async with pbar_lock:
            pbar.set_postfix(correct=correctness_count[0], total=pbar.n + 1)
            pbar.update(1)


async def run_async(
    examples_to_process: List[Dict],
    output_path: str,
    pbar: tqdm,
    correctness_count: List[int],
    num_concurrent: int,
    client: AsyncOpenAI,
    model_name: str,
):
    semaphore = asyncio.Semaphore(num_concurrent)
    pbar_lock = asyncio.Lock()

    batch_size = num_concurrent * 2
    for i in range(0, len(examples_to_process), batch_size):
        batch = examples_to_process[i : i + batch_size]
        tasks = [
            process_example(
                example, output_path, pbar, correctness_count, semaphore, pbar_lock, client, model_name
            )
            for example in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a dataset with gpt-oss-120b: generate answers then judge correctness."
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Number of concurrent operations")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--dataset-name", type=str, default="s42chen/odyssey-verified-27K-oracled")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--output-path", type=str, default="results/gpt_oss_120b.jsonl")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    client = make_async_client(args.base_url, args.api_key)

    processed_ids, existing_correct = load_processed_ids(args.output_path)
    print(f"Found {len(processed_ids)} already processed. Existing correct: {existing_correct}")

    print("Loading dataset...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    ds = ds.filter(lambda x: x.get("judge_correctness") is False)

    examples_to_process = [ex for ex in ds if ex.get("_id", "") not in processed_ids]

    print(f"Processing {len(examples_to_process)} new examples (skipping {len(processed_ids)} already done)")
    print(f"Concurrency: {args.num_threads}")

    correctness_count = [existing_correct]
    pbar = tqdm(total=len(ds), initial=len(processed_ids), desc="Processing", unit="ex")
    pbar.set_postfix(correct=correctness_count[0], total=len(processed_ids))

    asyncio.run(
        run_async(
            examples_to_process,
            args.output_path,
            pbar,
            correctness_count,
            args.num_threads,
            client,
            args.model_name,
        )
    )

    pbar.close()
    print(f"\nDone! Processed {len(examples_to_process)} examples. Total correct: {correctness_count[0]}")


if __name__ == "__main__":
    main()
