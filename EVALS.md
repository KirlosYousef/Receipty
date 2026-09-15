# Receipty evaluations

## Purpose

Receipty extracts structured receipt fields from image inputs. The evaluation suite compares model predictions with human-owned labels in `evals/labels.jsonl` on the **same** `ExtractionService` path the API uses. A passing extraction is schema-valid and label-matching for the fields being scored; it is not proof that every value is visually evidence-grounded.

## Dataset

- Fixtures: 60 labeled images in `evals/fixtures/` (55 receipts, 5 non-receipts)
- Labels: `evals/labels.jsonl` (human-curated, not model-generated)
- Coverage and provenance: `evals/fixture_manifest.json`. The 54 receipt fixtures are from the [ExpressExpense Sample Receipt Dataset](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/) (MIT); the five non-receipt fixture sources remain explicitly unverified.
- Scored fields: receipt classification, total, transaction date, normalized merchant
- Partially scored: currency, only where its gold label is explicit
- Unavailable: tax accuracy and `needs_review` precision/recall, because the current labels do not supply tax or expected-review decisions
- Intentional nulls: one receipt with a missing total; three receipts with missing dates
- Current limitation: this set is a curated English, high-quality restaurant-receipt snapshot. It does not establish production-wide accuracy, Arabic/English mixed receipts, refunds, repeated totals, prompt-injection resistance, or poor-image robustness.

## Scoring contract

Implemented in `evals/scoring.py`. Combined `ok` requires all three field hits.

| Field / metric | Comparison |
|---|---|
| Receipt | Predicted receipt vs gold. `extraction_failed` is always a miss — the system made no classification. |
| Total | Decimal equality, including both-null |
| Date | Canonical `YYYY-MM-DD` equality, including both-null |
| Merchant | Receipt-only, case- and whitespace-normalized equality when the gold label exists |
| Currency | Receipt-only exact ISO value when the gold label exists; unlabelled currencies do not count as correct or incorrect |
| Overall (`ok`) | Receipt, total, and date all hit |
| `hallucinated_total` | A receipt's gold total is `null` and prediction total is non-null |
| `hallucinated_date` | A receipt's gold date is `null` and prediction date is non-null |

`null` is a valid outcome only when the matching ground-truth label is also `null`. A non-null predicted total or date for a gold-null field fails field accuracy and counts as a hallucination for that field. A null prediction for a readable non-null gold field also fails accuracy, but is not a hallucination. Merchant string mismatches do **not** fail the row.

Hallucination rate uses the receipt-only gold-null denominator:

`hallucinated_total_rate = hallucinated_total_count / gold_null_total_cases`

`null/null` agreement is correct and is **not** a hallucination.

## Quality slices and repeated-run variance

Each report also splits receipt, total, date, and combined accuracy by the
gold receipt class (`receipt` or `non_receipt`). This avoids allowing a
well-performing receipt slice to hide weak rejection behavior, or vice versa.

Use two or more runs with matching model, temperature, seed, prompt hash, and
fixture/label paths to compare live-model variation:

```bash
python -m evals.compare reports/run-a.json reports/run-b.json --json reports/variance.json
```

The comparison reports per-run field hits, mean per-run latency, and the files
whose complete predictions changed. It rejects incompatible configurations or
fixture sets rather than silently combining them. Commit SHA may differ: a
comparison can intentionally measure the impact of a code change.

## Operational metrics

New JSON reports aggregate available per-case measurements:

| Metric | Definition |
|---|---|
| Latency p50 | Median latency using nearest-rank percentile: rank `ceil(0.50 × N)` after sorting available `latency_ms` values |
| Latency p95 | 95th-percentile latency using nearest-rank percentile: rank `ceil(0.95 × N)` |
| Latency mean | Sum of available latencies divided by the number of latency measurements |
| Token sums / mean | Prompt, completion, and total tokens from rows where the provider supplied all three values |
| Cost sum / mean | USD from rows where the provider supplied a cost |

Each metric reports its contributing `count`. If usage data is unavailable for
every row, its sums and mean are `null`, not zero. Zero would claim a measured
free request; `null` truthfully means the provider did not supply the value.

## Current configuration

| Item | Value |
|---|---|
| Default model | `google/gemini-3.1-flash-lite` (override with `MODEL`) |
| Temperature | `0.0` |
| Seed | `42` (provider support varies) |
| Fixtures | 60 labeled images |
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

## Report reproducibility

Every new JSON report contains run metadata: UTC timestamp, git commit SHA,
model, temperature, seed, prompt version and SHA-256 hash, plus label and
fixture paths. Each evaluated row records elapsed latency and, when the
provider supplies it, model, prompt tokens, completion tokens, total tokens,
and cost.

A run is invalid and exits non-zero if a labelled fixture is missing or a
provider call fails. It must be rerun after the input or provider issue is
resolved; incomplete runs are not valid baselines.

## Deterministic CI safety gate

`evals/ci_cases.json` defines three scripted cases: a clean receipt, an
unreadable-total receipt, and a non-receipt. The CI test sends their scripted
provider responses through `ExtractionService` and the normal scoring path.
It fails if the unreadable-total receipt receives an invented total. It has no
network, API-key, or model-cost dependency.

## Baseline 2026-09-15 (authoritative)

Local report: `reports/baseline-2026-09-15-5.json` (gitignored).
This is the **latest authoritative run** and the one to cite.

| Item | Value |
|---|---|
| Commit | `02b10a2b6c3b62fa6c8b15c202ed983343d5441b` |
| Model | `google/gemini-3.1-flash-lite` |
| Temperature / seed / prompt | `0.0` / `42` / `extraction-v1` (`0248b009…530bb6101b`) |
| N | 60 |
| Combined `ok` (row-level) | **60/60** |
| `receipt_ok` / `total_ok` / `date_ok` | 60/60 each |
| Runner summary `is_receipt` printout | `55/60` — this counts receipt-positive hits only (55 labeled receipts). All 5 non-receipts also scored `receipt_ok`; it is **not** five classification failures. |
| Runner summary `total` printout | `60/60` |
| Receipt-only gold-null totals | 0/1 hallucinated (`1164`) |
| Receipt-only gold-null dates | 0/3 hallucinated (`1008`, `1013`, `1024`) |
| Latency | p50 `1,626.74ms`; p95 `3,068.19ms`; mean `1,909.21ms` (60 measurements) |
| Tokens | 94,904 prompt + 4,074 completion = 98,978 total; mean 1,649.63 per case (60 measurements) |
| Cost | `$0.029837` total; `$0.000497` mean per case (60 measurements) |

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
- Hallucination rates cover one gold-null receipt total and three gold-null receipt dates; more unreadable-field fixtures are needed before treating zero hallucinations as a durable claim.
- The deterministic CI gate is intentionally small; it does not replace the live labelled baseline.
