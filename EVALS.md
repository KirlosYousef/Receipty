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

## Baseline 2026-09-15

Local report: `reports/baseline-2026-09-15-3.json` (gitignored).

| Item | Value |
|---|---|
| Commit | `a985bb683b0818ebc131239beb92139f2f932a72` |
| Model | `google/gemini-3.1-flash-lite` |
| Temperature / seed | `0.0` / `42` |
| N | 59 |
| `is_receipt` | 54/59 |
| `total` | 58/59 |
| Combined `ok` (from rows) | 57/59 |
| Gold-null total case (`1164`) | predicted `null` (correct; not an invented total) |

Live OpenRouter routes can vary between runs. An earlier same-day run (`baseline-2026-09-15-2.json`) had more date misses; treat these numbers as a dated snapshot, not a permanent SLA.

### Three failures worth explaining

1. **`1024-receipt.jpg` — missing date**  
   Pred total `33.92` matched gold, but pred date was `null` while gold is `2006-11-15`.  
   Likely cause: date is hard to read or in an unusual layout; the model correctly refused to invent a total but also failed to extract a readable date.  
   Lesson: date misses are usually vision/prompt coverage, not money-parser bugs.

2. **`1027-receipt.jpg` — wrong total**  
   Pred `44.31` vs gold `41.31` (same date).  
   Likely cause: the model latched onto a nearby amount (subtotal/tax-inclusive line) instead of the payable total.  
   Lesson: field accuracy can look “almost right” while still being accounting-unsafe; evals must catch near-miss totals.

3. **`1009-receipt.jpg` (from `baseline-2026-09-15-2.json`) — wrong date**  
   Pred date `2019-06-09` vs gold `2019-08-09`; total matched.  
   Likely cause: digit confusion on the month (`06` vs `08`), not a serialization bug.  
   Lesson: before blaming the model, confirm the report fields themselves are trustworthy — Session 1 also fixed a `pred_date` report bug that used `pred.total` when deciding how to serialize the date.

## Known limits

- A schema-valid model prediction is not evidence that every field is visually grounded in the input.
- The dataset is curated; do not claim production-wide accuracy from these scores.
- Low-confidence, unreadable, or ambiguous financial fields must remain reviewable rather than guessed.
- Merchant is inspected in the runner output but is not part of combined `ok`.
- This baseline does **not** yet publish hallucination-rate denominators, `needs_review` precision/recall, or a CI eval gate (Week 2 later sessions).
