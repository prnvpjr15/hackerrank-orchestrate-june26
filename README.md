# Multi-Modal Evidence Review

A pipeline that verifies damage claims by comparing a user's written description against submitted photos. For each claim it produces a structured verdict: whether the visual evidence supports, contradicts, or is insufficient to assess the claim — along with risk flags, severity, issue type, and supporting image IDs.

Built for the HackerRank Orchestrate hackathon (June 2026).

## How it works

Each claim is processed with a single multimodal API call to `claude-opus-4-8`. All images for a claim are base64-encoded and sent alongside the claim text, user history, and evidence requirements in one message. The model returns a structured JSON response that is validated and repaired against the allowed-value schema before being written to `output.csv`.

## Setup

**Python 3.10+** required.

```bash
pip install -r code/requirements.txt
```

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

| Flag | Default | Description |
|---|---|---|
| `--model` | `claude-opus-4-8` | Model to use |
| `--claims-csv` | `dataset/claims.csv` | Input CSV path |
| `--output-csv` | `output.csv` | Output CSV path |
| `--limit N` | off | Process only first N rows |

### Evaluation

Runs the pipeline against the 20 labeled rows in `dataset/sample_claims.csv` and scores predictions field-by-field against ground truth:

```bash
python code/evaluation/main.py
```

| Flag | Default | Description |
|---|---|---|
| `--models` | `claude-opus-4-8 claude-sonnet-4-6 claude-haiku-4-5-20251001` | Models to compare |
| `--limit N` | off | Evaluate on first N rows only |

Results are written to `code/evaluation/evaluation_results.json`.

## Results

Three models were evaluated on 20 labeled claims. `claude-opus-4-8` was selected for the final run.

| Field | Opus 4.8 | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|---|
| claim_status | **0.85** | 0.65 | 0.55 |
| evidence_standard_met | **0.90** | 0.80 | 0.60 |
| Mean (all fields) | **0.688** | 0.600 | 0.569 |

Full methodology, per-row miss analysis, and a prompt-patch experiment are documented in [`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md).

## Repo layout

```
.
├── code/
│   ├── main.py                    # Entry point for full dataset run
│   ├── pipeline.py                # Image loading, prompt, API call, validation
│   ├── requirements.txt           # Pinned dependencies
│   └── evaluation/
│       ├── main.py                # Evaluation entry point
│       ├── evaluation_results.json
│       └── evaluation_report.md
├── dataset/
│   ├── claims.csv                 # Test inputs
│   ├── sample_claims.csv          # Labeled development set
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
├── output.csv                     # Final predictions (44 claims, claude-opus-4-8)
└── problem_statement.md
```
