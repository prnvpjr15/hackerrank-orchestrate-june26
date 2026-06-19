# Multi-Modal Evidence Review — Code

This system verifies damage claims by comparing a user's written claim against submitted photos. For each claim it produces a structured verdict: whether the visual evidence supports, contradicts, or is insufficient to assess the claim, along with risk flags, severity, and supporting image IDs.

## Setup

**Python 3.10+** required.

```bash
pip install -r code/requirements.txt
```

`pillow-heif` is required to handle images stored in AVIF/HEIC format (common in the dataset despite `.jpg` extensions). Without it, those images silently fail to decode.

Set your Anthropic API key:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Running the pipeline

### Full dataset run

Reads `dataset/claims.csv`, writes `output.csv` at the repo root:

```bash
python code/main.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `claude-opus-4-8` | Model to use |
| `--claims-csv` | `dataset/claims.csv` | Input CSV path |
| `--output-csv` | `output.csv` | Output CSV path |
| `--limit N` | off | Process only first N rows (smoke test) |

Examples:

```bash
# Smoke test — first 2 rows
python code/main.py --limit 2

# Cheaper run with Sonnet
python code/main.py --model claude-sonnet-4-6

# Custom paths
python code/main.py --claims-csv dataset/claims.csv --output-csv results/output.csv
```

### Evaluation

Runs the pipeline against the 20 labeled rows in `dataset/sample_claims.csv` and scores predictions against ground truth, comparing one or more model configurations:

```bash
python code/evaluation/main.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--models` | `claude-opus-4-8 claude-sonnet-4-6 claude-haiku-4-5-20251001` | Space-separated model list |
| `--limit N` | off | Evaluate on first N sample rows only |

Examples:

```bash
# Compare all three default models
python code/evaluation/main.py

# Single model
python code/evaluation/main.py --models claude-opus-4-8

# Quick 5-row smoke check
python code/evaluation/main.py --models claude-sonnet-4-6 --limit 5
```

Outputs written to `code/evaluation/`:
- `evaluation_results.json` — per-model accuracy and mismatch details
- `sample_predictions_<model>.csv` — raw predictions for each model

## Layout

```
code/
├── README.md                  # this file
├── main.py                    # CLI entry point for full dataset run
├── pipeline.py                # shared core: image loading, prompt, API call, validation
└── evaluation/
    ├── main.py                # evaluation entry point
    └── evaluation_report.md   # model selection rationale and results
```

## Model selection and results

See `evaluation/evaluation_report.md` for the full evaluation methodology and results. In brief: three models were compared on 20 labeled claims (`claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`). Opus 4.8 was selected for the final submission — it scored highest on the two adjudication-critical fields (`claim_status` 0.85, `evidence_standard_met` 0.90) and led on mean per-field accuracy (0.688 vs 0.600 for Sonnet). A prompt-patch experiment, the per-row miss analysis that motivated it, and its mixed results are documented in the report alongside the final 44-claim operational metrics.
