# Receipty evaluations

## Purpose

Receipty extracts structured receipt fields from image inputs. The evaluation suite compares model predictions with human-owned labels in `evals/labels.jsonl` on the **same** `ExtractionService` path the API uses. A passing extraction is schema-valid and label-matching for the fields being scored; it is not proof that every value is visually evidence-grounded.

## Dataset

- Fixtures: 59 labeled images in `evals/fixtures/` (54 receipts, 5 non-receipts)
- Labels: `evals/labels.jsonl` (human-curated, not model-generated)
- Scored fields: receipt classification, total, transaction date
- Logged, not scored: merchant (printed for inspection; name variance is noisy)
- Intentional nulls: one receipt with a missing total; three receipts with missing dates
- Current limitation: this set is still a curated English-heavy snapshot. It does not establish production-wide accuracy, Arabic/English mixed receipts, refunds, prompt-injection resistance, or poor-image robustness.

## Scoring contract

Implemented in `evals/scoring.py`. Combined `ok` requires all three hits.

| Field | Comparison |
|---|---|
| Receipt | Predicted receipt vs gold. `extraction_failed` is always a miss — the system made no classification. |
| Total | Decimal equality, including both-null |
| Date | Canonical `YYYY-MM-DD` equality, including both-null |
| Overall | Receipt, total, and date all hit |

`null` is a valid outcome only when the matching ground-truth label is also `null`. A non-null predicted total or date for a gold-null field fails. A null prediction for a readable non-null gold field also fails. Merchant string mismatches do **not** fail the row.

## Current configuration

| Item | Value |
|---|---|
| Default model | `google/gemini-3.1-flash-lite` (override with `MODEL`) |
| Temperature | `0.0` |
| Seed | `42` (provider support varies) |
| Fixtures | 59 labeled images |
| Labels | `evals/labels.jsonl` |
| Command | `python -m evals.run` |

Live LLM evaluation can vary by model and OpenRouter route. Record `MODEL`, temperature, seed, and commit SHA when comparing runs.

## Run locally

```bash
source .venv/bin/activate
python -m evals.run

# Optional: save a local report. `reports/` is gitignored.
python -m evals.run --json reports/eval.json
```

Requires `OPENROUTER_API_KEY`. Unit tests mock the provider and do not call the live model.

## Known limits

- A schema-valid model prediction is not evidence that every field is visually grounded in the input.
- The dataset is curated; do not claim production-wide accuracy from these scores.
- Low-confidence, unreadable, or ambiguous financial fields must remain reviewable rather than guessed.
- Merchant is inspected in the runner output but is not part of combined `ok`.
