# Round 4 — External Verification

Uses a locally hosted reasoning model (via vLLM) to independently answer each question from the self-verified dataset, then judges the predicted answer against the ground truth. Records where `judge_correctness == true` form the final ORBIT dataset.

## Structure

```
round-4-external-verification/
├── external_verification.py   # main script
└── external_verification.sh   # launcher (includes vLLM setup instructions)
```

Output lands in `results/`.

## Output format

```json
{
  "_id":               "d2d1c26f78ddc08c03f9fb4819793dda",
  "question":          "Which painting technique ...",
  "answer":            "Tenebrism",
  "scraped_documents": {"https://...": "..."},
  "predicted_answer":  "Tenebrism",
  "judge_correctness": true,
  "judge_output":      "<judge reasoning> Judge: CORRECT"
}
```

## Step 1 — Start the vLLM server

Run once in a separate terminal or tmux pane before launching the evaluation.

```bash
export OMP_NUM_THREADS=6
export HF_HOME=/u3/n3thakur/projects/cache
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_DISABLE=1

# single GPU
vllm serve openai/gpt-oss-120b --tensor-parallel-size 1 --async-scheduling --port 6050

# multi-GPU
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --async-scheduling --port 6050
```

Wait for `Application startup complete.` before proceeding. Verify with:

```bash
curl http://localhost:6050/v1/models
```

## Step 2 — Run the evaluation

```bash
bash external_verification.sh
```

Or directly:

```bash
python external_verification.py \
    --dataset-name  orbit-ai/orbit-stage-3-27k \
    --output-path   results/gpt_oss_120b.jsonl \
    --num-threads   4
```

`--num-threads` controls async concurrency against the vLLM server. The script is **resume-safe**: already-written `_id`s are skipped on restart.

## Dependencies

```bash
pip install openai datasets tqdm
```
