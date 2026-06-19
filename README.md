# Pre-built pipeline — drop into code/

These files are tested (dry-run, no live API calls) against your actual
cloned `dataset/` — image loading, prompt construction, JSON validation/
repair, and evaluation scoring all verified to work on real rows. Copy
them into your repo's `code/` folder and they should run as-is.

## Files

- `pipeline.py` → `code/pipeline.py` (shared core: CSV loading, prompt
  building, API calls with retry, response validation/repair)
- `main.py` → `code/main.py` (entry point for the full claims.csv run)
- `evaluation/main.py` → `code/evaluation/main.py` (runs against labeled
  sample_claims.csv, scores accuracy, compares two model configs)

## What's already verified

- Image paths resolve correctly relative to `dataset/` (this was a real
  bug I caught and fixed — `image_paths` in the CSVs are relative to
  `dataset/`, not repo root)
- `load_user_history` / `load_evidence_requirements` default paths
  resolve at call time, not import time (caught a stale-default bug)
- Prompt construction produces correct interleaved text+image content
  blocks with the right image_ids, tested against `claims.csv` row 1
  (3 images, all load with correct media types and base64 lengths)
- `validate_and_repair` correctly clamps invalid enum values and strips
  hallucinated image IDs that don't exist in the claim's actual
  `image_paths` (tested with a synthetic bad response)
- Evaluation scoring correctly computes per-field and claim_status
  accuracy against `sample_claims.csv` (tested with a synthetic
  near-perfect prediction set)

## What's NOT yet verified (needs your API key — do this in Claude Code)

- An actual live VLM call has not been made. The prompt is designed
  carefully against the real evidence_requirements.csv and several real
  sample_claims.csv rows (including a deliberately tricky one —
  case_005, where the claim says "pretty bad damage" but the image
  shows only a minor scratch, correctly labeled `contradicted` with both
  `claim_mismatch` and `user_history_risk` flags — use this as your
  first smoke-test case to sanity check the model's judgment).
- Rate limit / retry behavior under real API conditions.
- Actual token usage and cost (needed for evaluation_report.md).

## How to run

```bash
cd code
pip install anthropic

export ANTHROPIC_API_KEY=sk-ant-...     # macOS/Linux
$env:ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell

# 1. Smoke test first — 2 claims only, cheap, fast
python main.py --limit 2

# 2. Run the evaluation against labeled sample_claims.csv
#    (compares Sonnet vs Haiku by default — your "two strategies" requirement)
python evaluation/main.py

# Inspect evaluation/evaluation_results.json and the two
# sample_predictions_*.csv files it produces, decide which model/config
# you're shipping as final, then write evaluation/evaluation_report.md
# summarizing accuracy, mismatches, and the operational analysis
# (calls, tokens, images, cost, latency, rate-limit strategy) using the
# real numbers from this run.

# 3. Once you're confident, run the full test set
python main.py
# writes output.csv in repo root — move/rename as needed for submission
```

## Model name check

I used `claude-sonnet-4-6` and `claude-haiku-4-5-20251001` as the model
strings (per current Anthropic model names). Verify these are still
correct against https://docs.claude.com when you actually run this —
model strings occasionally change and I can't guarantee this matches
what's live at the moment you run it.

## Cost/rate-limit notes for your evaluation_report.md

- 1 VLM call per claim (not per image) — all images for a claim go in
  one multimodal message. This is the main lever keeping cost down: 44
  test claims = 44 calls, not ~85 (one per image).
- Stage A (history/requirements lookup, image loading) is pure Python,
  zero model calls.
- Retry logic: up to 3 retries on rate-limit/API errors with exponential
  backoff (2s, 4s, 8s); up to 2 attempts at getting valid JSON before
  falling back to a safe `manual_review_required` row.
- No prompt caching implemented yet — the system prompt + allowed-values
  block is sent fresh every call. Given the tiny dataset size (44+20
  claims) this is a deliberate simplicity-over-optimization choice; worth
  noting in the report as something you'd add at larger scale (Anthropic
  prompt caching on the system prompt would cut repeated input tokens
  significantly if this scaled to thousands of claims).
- To get real token counts: `resp.usage.input_tokens` /
  `resp.usage.output_tokens` are available on every API response object
  but currently not captured in `pipeline.py` — add that and log it per
  call if you want exact (not estimated) cost figures for the report.
