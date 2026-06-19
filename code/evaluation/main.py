"""
evaluation/main.py — runs the pipeline against the LABELED
dataset/sample_claims.csv, scores predictions against ground truth,
and compares three model configurations (Opus, Sonnet, Haiku) as required
by the evaluation report.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python code/evaluation/main.py
    python code/evaluation/main.py --models claude-sonnet-4-6 claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `pipeline` imports

from pipeline import (
    DATASET_DIR,
    OUTPUT_COLUMNS,
    load_csv,
    load_evidence_requirements,
    load_user_history,
    process_claim,
)

EVAL_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = DATASET_DIR / "sample_claims.csv"

# Columns we actually score against ground truth (the predicted output cols)
SCORED_COLUMNS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]

# Exact-match is too strict for free-text/semicolon-set fields in some cases;
# define per-field comparison.
def fields_match(field: str, expected: str, actual) -> bool:
    actual_str = str(actual)
    if field == "risk_flags" or field == "supporting_image_ids":
        exp_set = {x.strip() for x in str(expected).split(";") if x.strip()}
        act_set = {x.strip() for x in actual_str.split(";") if x.strip()}
        return exp_set == act_set
    if field in ("evidence_standard_met", "valid_image"):
        return str(expected).strip().lower() == actual_str.strip().lower()
    return str(expected).strip().lower() == actual_str.strip().lower()


def run_model_on_sample(client: anthropic.Anthropic, model: str, sample_rows: list[dict]) -> list[dict]:
    history_by_user = load_user_history()
    all_reqs = load_evidence_requirements()
    predictions = []
    for i, row in enumerate(sample_rows, 1):
        print(f"  [{i}/{len(sample_rows)}] {row['user_id']} ({row['claim_object']})")
        pred = process_claim(client, model, row, history_by_user, all_reqs, DATASET_DIR)
        predictions.append(pred)
    return predictions


def score(predictions: list[dict], ground_truth: list[dict]) -> dict:
    per_field_correct = {f: 0 for f in SCORED_COLUMNS}
    n = len(ground_truth)
    exact_match_status = 0
    mismatches = []

    for pred, truth in zip(predictions, ground_truth):
        row_mismatches = []
        for f in SCORED_COLUMNS:
            if fields_match(f, truth[f], pred[f]):
                per_field_correct[f] += 1
            else:
                row_mismatches.append((f, truth[f], pred[f]))
        if fields_match("claim_status", truth["claim_status"], pred["claim_status"]):
            exact_match_status += 1
        if row_mismatches:
            mismatches.append({
                "user_id": truth["user_id"],
                "image_paths": truth["image_paths"],
                "diffs": row_mismatches,
            })

    accuracy = {f: round(per_field_correct[f] / n, 3) for f in SCORED_COLUMNS}
    return {
        "n": n,
        "claim_status_accuracy": round(exact_match_status / n, 3),
        "per_field_accuracy": accuracy,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        help="model configs to compare",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=api_key)

    ground_truth = load_csv(SAMPLE_CSV)
    if args.limit:
        ground_truth = ground_truth[: args.limit]

    results = {}
    for model in args.models:
        print(f"\n=== Evaluating model: {model} ===")
        t0 = time.time()
        preds = run_model_on_sample(client, model, ground_truth)
        elapsed = time.time() - t0
        scored = score(preds, ground_truth)
        scored["elapsed_seconds"] = round(elapsed, 1)
        scored["seconds_per_claim"] = round(elapsed / max(len(preds), 1), 2)
        results[model] = scored

        # persist predictions for this model
        pred_path = EVAL_DIR / f"sample_predictions_{model.replace('/', '_')}.csv"
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in preds:
                writer.writerow({k: row[k] for k in OUTPUT_COLUMNS})
        print(f"  predictions written to {pred_path}")

        from pipeline import summarize_usage
        scored["usage"] = summarize_usage(preds)

    # summary
    summary_path = EVAL_DIR / "evaluation_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, r in results.items():
        print(f"\n{model}")
        print(f"  claim_status accuracy: {r['claim_status_accuracy']}")
        print(f"  per-field accuracy: {r['per_field_accuracy']}")
        print(f"  {r['seconds_per_claim']}s/claim, {r['elapsed_seconds']}s total")
        print(f"  mismatched rows: {len(r['mismatches'])}/{r['n']}")

    print(f"\nFull results: {summary_path}")
    print("Use this output plus token/cost figures (see README) to fill in evaluation_report.md")


if __name__ == "__main__":
    main()
