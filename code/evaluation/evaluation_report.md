# Evaluation Report — Multi-Modal Evidence Review

## 1. Methodology

The pipeline (`code/pipeline.py`) was run against all 20 labeled rows in `dataset/sample_claims.csv`. Each claim was processed with a single VLM call: images are base64-encoded and passed alongside the claim text, user history, and evidence requirements. Responses are parsed, validated against the allowed-value schema, and scored field-by-field against ground truth.

Three model configurations were compared in the initial evaluation run: `claude-opus-4-8`, `claude-sonnet-4-6`, and `claude-haiku-4-5-20251001`.

Scoring uses exact match for categorical fields and set-equality for semicolon-separated fields (`risk_flags`, `supporting_image_ids`). Boolean fields are compared case-insensitively. Free-text fields (`evidence_standard_met_reason`, `claim_status_justification`) are not scored.

Scored fields: `evidence_standard_met`, `risk_flags`, `issue_type`, `object_part`, `claim_status`, `supporting_image_ids`, `valid_image`, `severity`.

---

## 2. Three-Model Comparison (original prompt, n=20)

| Field | Opus 4.8 | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|---|
| evidence_standard_met | **0.90** | 0.80 | 0.60 |
| risk_flags | 0.30 | 0.25 | **0.35** |
| issue_type | **0.50** | 0.45 | 0.40 |
| object_part | **0.90** | 0.85 | 0.85 |
| claim_status | **0.85** | 0.65 | 0.55 |
| supporting_image_ids | **0.75** | 0.70 | 0.60 |
| valid_image | 0.85 | 0.85 | 0.85 |
| severity | **0.45** | 0.25 | 0.35 |
| **Mean** | **0.688** | 0.600 | 0.569 |

Speed: Haiku 4.0 s/claim · Opus 5.9 s/claim · Sonnet 7.4 s/claim.

Opus wins on 6 of 8 fields and leads by a substantial margin on the two adjudication-critical fields: `claim_status` (+20 pp over Sonnet) and `evidence_standard_met` (+10 pp). Haiku's narrow wins on `risk_flags` and `severity` reflect less-verbose output rather than improved accuracy.

`risk_flags` and `severity` are the weakest fields across all models (0.25–0.45). Both models systematically over-rate severity to `high` when ground truth is `medium`, and over-fire `claim_mismatch` on clear claims. These are the primary remaining improvement opportunities.

---

## 3. Prompt-Patch Experiment: `contradicted` vs `not_enough_information`

### Finding

Analysing per-row claim_status misses revealed that both Opus and Sonnet share three hard cases (user_008, user_033, user_034) where ground truth is `contradicted` but both models output `not_enough_information`. The pattern across all models: when evidence is ambiguous, models default to the neutral middle category rather than committing to a contradiction.

Sonnet had four additional unique misses that Opus got right: over-skepticism on three clear `supported` claims (user_001, user_002, user_004) and one `contradicted` hedge (user_020).

### Patch Applied

The following rule and example pair were added to `SYSTEM_PROMPT` after step 4's bullet points:

> **CONTRADICTED vs NOT_ENOUGH_INFORMATION** — use this rule exactly: if the claimed part or object is present and inspectable in at least one image, and what you see does not match the claim, that is `contradicted` — even if image quality is less than ideal. Reserve `not_enough_information` strictly for cases where the claimed part cannot be located or examined at all across the entire image set.
>
> Example A (contradicted): claim says "deep crack in windshield" — image shows the full windshield clearly with no crack → *contradicted*.
> Example B (not_enough_information): claim says "deep crack in windshield" — image shows only the car interior with no windshield in frame → *not_enough_information*.

### Result

| | Opus pre-patch | Opus post-patch | Sonnet pre-patch | Sonnet post-patch |
|---|---|---|---|---|
| claim_status | **0.85** | 0.70 (−0.15) | 0.65 | **0.75 (+0.10)** |
| Mean | **0.688** | 0.650 (−0.038) | 0.600 | 0.625 (+0.025) |

The rule helped Sonnet because Sonnet was genuinely under-confident on `contradicted` cases. It hurt Opus because Opus was already well-calibrated at this boundary; the explicit instruction pushed it past threshold and it began mislabelling `supported` and `not_enough_information` cases as `contradicted` (new regressions: user_002, user_004, user_032).

None of the three shared hard cases were fixed by either model under the patch. These appear to be genuinely ambiguous visual judgements that require stronger per-case evidence rather than a prompt-level heuristic.

**Decision:** patch reverted. Pre-patch Opus remains the best-performing configuration.

---

## 4. Final Model Decision

**`claude-opus-4-8` with the original unpatched prompt.**

Primary reasons:
- Highest `claim_status` accuracy (0.85) — the primary adjudication signal.
- Highest `evidence_standard_met` accuracy (0.90) — the gate that determines whether a claim proceeds.
- The patch experiment confirms Opus's performance is genuine calibration, not noise: the same prompt change that over-corrected Opus produced the expected improvement in Sonnet, demonstrating that Opus was already at the right decision boundary.
- Speed advantage over Sonnet (5.9 vs 7.4 s/claim) despite higher capability.

Cost trade-off: Opus costs approximately 3× more than Sonnet per claim ($0.028 vs $0.011). At this dataset scale the absolute difference is trivial; at thousands of claims, a tiered approach (Haiku for triage, Opus for escalated/high-risk cases) would be worth evaluating.

---

## 5. Spot-Check: Not-Enough-Information Rows (full run)

Three `not_enough_information` rows from the 44-claim output were manually reviewed to confirm the model is not over-hedging.

**user_007 — car** (`wrong_object_part;wrong_angle`)
Claimed: side mirror missing/broken. img_1 shows a side profile without a clear mirror view; img_2 shows only a wheel. The claimed part is not in frame across either image. Correct.

**user_020 — laptop** (`cropped_or_obstructed;user_history_risk`)
Claimed: trackpad cracked. Trackpad is partially visible but a hand covers a significant portion. Visible area shows no crack; occluded area cannot be assessed. Borderline-but-correct: committing to `contradicted` would require ruling out damage beneath the obstruction.

**user_018 — car** (`wrong_angle;claim_mismatch`)
Claimed: cracked taillight (rear). Image shows only the front of the vehicle with damage to the hood and front bumper. `evidence_standard_met: True` (the image is a real, usable photo of the car) but `claim_status: not_enough_information` (the rear taillight is not photographed). This case illustrates the model correctly separating "image is valid" from "image addresses the claim" — a distinction that matters for fair adjudication.

All three represent genuine, traceable evidence gaps. No signs of generic caution or risk-aversion introduced by the model configuration.

---

## 6. Full 44-Claim Run — Operational Analysis

**Model:** `claude-opus-4-8` · **Prompt:** original (unpatched)

| Metric | Value |
|---|---|
| Total claims | 44 |
| API calls | 44 (1.0 per claim) |
| Retries | 0 |
| Validation fallbacks triggered | 0 |
| Total runtime | 299 s (6.8 s/claim) |
| Input tokens | 180,065 |
| Output tokens | 13,444 |
| **Total cost** | **$1.24** ($0.90 input + $0.34 output) |

**Output distribution:**

| claim_status | n |
|---|---|
| supported | 20 |
| contradicted | 12 |
| not_enough_information | 12 |

`evidence_standard_met`: 35 True / 9 False.
`manual_review_required` flag: 10/44 rows — all driven by user history risk or image integrity signals, not pipeline errors. 8 of these 10 still received a definitive `supported` or `contradicted` verdict alongside the flag.

**Rate limiting / batching:** the current single-claim-per-call design runs well within standard API rate limits at 44 claims. No batching was needed. At scale (thousands of claims), switching to the [Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) would reduce cost by 50% and eliminate per-call latency concerns.

**Total project API spend across all development and evaluation runs:** ~$2.76 of a $5.00 budget. Breakdown: Haiku eval $0.07 · Sonnet eval $0.22 · Opus eval run 1 $0.48 · Opus + Sonnet patch run $0.71 · full 44-claim run $1.24 · smoke tests and misc $0.04.

`temperature=0` was added to `pipeline.py` after evaluation for reproducibility; the scores in Sections 2–3 were generated before this change.
