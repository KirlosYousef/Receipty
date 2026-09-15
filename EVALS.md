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

## Baseline 2026-09-15 (authoritative)

Local report: `reports/baseline-2026-09-15.json` (gitignored).  
This is the **best dated run** and the one to cite. Same-day files like `baseline-2026-09-15-2.json` / `-3.json` were intermediate experiments while changing prompts/settings — **do not treat them as the baseline**.

| Item | Value |
|---|---|
| Commit | `a985bb683b0818ebc131239beb92139f2f932a72` |
| Model | `google/gemini-3.1-flash-lite` |
| Temperature / seed | `0.0` / `42` |
| N | 59 |
| Combined `ok` (row-level) | **59/59** |
| `receipt_ok` / `total_ok` / `date_ok` | 59/59 each |
| Runner summary `is_receipt` printout | `54/59` — this counts receipt-positive hits only (54 labeled receipts). All 5 non-receipts also scored `receipt_ok`; it is **not** five classification failures. |
| Runner summary `total` printout | `59/59` |
| Gold-null total (`1164`) | predicted total absent (correct; not invented) |

Live OpenRouter routes can still vary between future runs. Re-record commit SHA, model, temperature, and seed whenever you claim a new baseline.

### Notable cases from this run

There were **no combined `ok` failures** on the authoritative baseline. These cases document the scoring contract:

1. **`1164-receipt.jpg` — unreadable total**  
   Gold total `null`, gold date `2015-08-06`. Scoring reported `total_ok` and `date_ok` (no invented total; date matched).  
   The JSON report previously showed `pred_total: "None"` (string) and `pred_date: null` because `score_row` gated date serialization on `isinstance(pred.total, Decimal)`. Report fields are now serialized from the matching prediction attributes.

2. **`2200-receipt.png` — non-receipt**  
   Gold is not a receipt; prediction cleared money/date fields and still received combined `ok`. Correct refusal is scored separately from receipt-field extraction quality.

3. **`1000-receipt.jpg` — clean receipt**  
   Pred total `56.58` and date `2016-05-26` matched gold. Scores on this curated English-heavy set do not establish production-wide grounding.

## Known limits

- A schema-valid model prediction is not evidence that every field is visually grounded in the input.
- The dataset is curated; do not claim production-wide accuracy from these scores.
- Low-confidence, unreadable, or ambiguous financial fields must remain reviewable rather than guessed.
- Merchant is inspected in the runner output but is not part of combined `ok`.
- This baseline does **not** yet publish hallucination-rate denominators, `needs_review` precision/recall, or a CI eval gate.
