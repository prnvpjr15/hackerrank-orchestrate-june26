"""
main.py — entry point. Runs the evidence-review pipeline over
dataset/claims.csv and writes output.csv.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python code/main.py
    python code/main.py --model claude-haiku-4-5-20251001   # cheaper run
    python code/main.py --limit 5                           # smoke test
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import anthropic

from pipeline import (
    DATASET_DIR,
    REPO_ROOT,
    load_csv,
    load_evidence_requirements,
    load_user_history,
    process_claim,
    summarize_usage,
    write_output_csv,
)

DEFAULT_MODEL = "claude-opus-4-8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--claims-csv", default=str(DATASET_DIR / "claims.csv"))
    parser.add_argument("--output-csv", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument("--limit", type=int, default=None, help="process only first N rows (smoke test)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Set it as an environment variable first.")

    client = anthropic.Anthropic(api_key=api_key)

    claims = load_csv(Path(args.claims_csv))
    if args.limit:
        claims = claims[: args.limit]

    history_by_user = load_user_history()
    all_reqs = load_evidence_requirements()

    print(f"Processing {len(claims)} claims with model={args.model} ...")
    rows = []
    t0 = time.time()
    for i, claim in enumerate(claims, 1):
        print(f"[{i}/{len(claims)}] {claim['user_id']} ({claim['claim_object']}) {claim['image_paths']}")
        row = process_claim(client, args.model, claim, history_by_user, all_reqs, DATASET_DIR)
        rows.append(row)
    elapsed = time.time() - t0

    write_output_csv(rows, Path(args.output_csv))
    usage = summarize_usage(rows)
    print(f"\nWrote {len(rows)} rows to {args.output_csv} in {elapsed:.1f}s "
          f"({elapsed / max(len(rows),1):.1f}s/claim)")
    print(f"Token usage: {usage}")


if __name__ == "__main__":
    main()
