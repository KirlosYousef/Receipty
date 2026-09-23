# Receipty evaluations

## Purpose

Receipty extracts structured receipt fields from image inputs. The evaluation suite compares model predictions with human-owned labels in `evals/labels.jsonl` on the **same** `ExtractionService` path the API uses. A passing extraction is schema-valid and label-matching for the fields being scored; it is not proof that every value is visually evidence-grounded.

## Dataset

- Fixtures: 60 labeled images in `evals/fixtures/` (55 receipts, 5 non-receipts)
- Labels: `evals/labels.jsonl` (human-curated, not model-generated)
- Coverage and provenance: `evals/fixture_manifest.json`. The 55 receipt fixtures are from the [ExpressExpense Sample Receipt Dataset](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/) (MIT); the five non-receipt fixture sources remain explicitly unverified.
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

The comparison reports per-run field hits, mean per-run latency, total tokens,
total cost, receipt-only hallucination rates, and the files whose complete
predictions changed. It rejects incompatible configurations or fixture sets
rather than silently combining them. Commit SHA may differ: a comparison can
intentionally measure the impact of a code change.

A prediction is “changed” when any of `pred_outcome`, `pred_merchant`,
`pred_total`, `pred_currency`, `pred_date`, or `pred_tax` differs between
runs (`evals/compare.py` `_prediction_signature`). Latency, tokens, and cost
are reported separately and do not count as prediction changes.

### Repeat-run 2026-09-17 (v1 / v1)

Local reports: `reports/baseline-2026-09-17.json` and
`reports/baseline-2026-09-17-b.json` (gitignored). Comparison:
`reports/variance.json`.

| Item | Value |
|---|---|
| Commit | `60133b838f720e97e1b0e2eb17666f6f08b7244b` |
| Model / temperature / seed | `google/gemini-3.1-flash-lite` / `0.0` / `42` |
| Prompt | `extraction-v1` (`0248b009…530bb6101b`) |
| Labels / fixtures | `evals/labels.jsonl` / `evals/fixtures` |
| Compared fields | `pred_outcome`, `pred_merchant`, `pred_total`, `pred_currency`, `pred_date`, `pred_tax` |
| Prediction changes | **0/60** |
| Combined `ok` / receipt / total / date | 60/60 each, both runs |
| Merchant / currency (scored) | 51/55 and 30/31, both runs |
| Hallucinated total / date | 0/1 and 0/3, both runs |
| Tokens / cost | 98,978 / `$0.029837`, both runs |
| Mean latency | 2,046ms then 1,744ms (route variance; not a prediction change) |

This is a same-configuration live-model repeat, not a prompt experiment. The
v1/v2 12-file delta below is a different comparison.

## Prompt and model experiments

The runner selects a versioned prompt, records its version and SHA-256 hash,
and accepts the model through the existing `MODEL` environment variable. Run
each candidate against the same fixture set, then compare them explicitly:

```bash
python -m evals.run --prompt-version extraction-v1 --json reports/v1.json
python -m evals.run --prompt-version extraction-v2-evidence --json reports/v2.json
python -m evals.compare --allow-config-differences reports/v1.json reports/v2.json
```

Use `--allow-config-differences` only for intentional prompt/model experiments.
It still requires identical fixture and label paths and emits every run's model,
temperature, seed, prompt version, and prompt hash alongside the results.

### Prompt experiment 2026-09-16

Local reports: `reports/v1.json` and `reports/v2.json` (gitignored).
Same model (`google/gemini-3.1-flash-lite`), temperature `0.0`, seed `42`,
labels, and fixtures. The only intended difference is the prompt.

| Metric | `extraction-v1` | `extraction-v2-evidence` |
|---|---|---|
| Combined `ok` | 60/60 | 60/60 |
| Receipt / total / date | 60/60 each | 60/60 each |
| Hallucinated total / date | 0/1, 0/3 | 0/1, 0/3 |
| Merchant (scored when labelled) | 48/55 | 47/55 |
| Currency (scored when labelled) | 2/31 | 1/31 |
| Latency p50 / p95 / mean | 1,767ms / 2,784ms / 1,953ms | 1,798ms / 2,638ms / 1,835ms |
| Tokens | 98,978 | 95,217 |
| Cost | `$0.029837` | `$0.028771` |
| Changed predictions | — | 12/60 files |

A second independent v1/v2 comparison reproduced the same 12-file set and the
same merchant/currency deltas. That is a prompt effect, not one-run noise.

`extraction-v2-evidence` remains a documented experiment, not the production
prompt. It did not improve the extraction contract. It often copied `$` or
left currency `null`, which moved correct totals into `needs_review`, and it
shortened some merchant names (`McDonald's Restaurant #3100` → `McDonald's`).
Keep `extraction-v1` until a later experiment beats it on `ok`, hallucination
rate, or a labelled review metric.

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
| Default prompt | `extraction-v1` (override with `--prompt-version`) |
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

## Failures and measured changes

Combined `ok` is 60/60 on the current labelled set. These three failures are still
real: they hid in report serialization, gold labels, or un-scored fields.

### 1. Null-total reports dropped `pred_date` (fixed)

**Evidence:** `1164-receipt.jpg` has gold total `null` and gold date `2015-08-06`.
The model returned a matching date and no total, so scoring was `total_ok` and
`date_ok`. The JSON report still wrote `pred_total: "None"` (the string) and
`pred_date: null`, because `score_row` serialized the date only when
`pred.total` was a `Decimal`.

**Change:** serialize `pred_total` from `pred.total` and `pred_date` from
`pred.date`. A later 60-case run (`baseline-2026-09-15-5.json` and `reports/v1.json`)
shows `1164` as `pred_total: null`, `pred_date: "2015-08-06"`.

### 2. Three gold merchants contradicted the fixture images (fixed)

**Evidence:** on `reports/v1.json`, merchant accuracy was 48/55. Three of the
seven misses were label errors, confirmed against the images:

| File | Wrong gold | Printed merchant | Total / date still matched |
|---|---|---|---|
| `1015-receipt.jpg` | Deccan Spice | HAMMOCKS TRADING COMPANY | 50.29 / 2017-03-10 |
| `1016-receipt.jpg` | Hammocks Trading Company | Chef Wang | 35.52 / 2019-02-02 |
| `1017-receipt.jpg` | Grotto Pizzeria & Tavern | UMIX | 11.04 / 2016-04-24 |

Combined `ok` stayed green because merchant is not part of that metric.

**Change:** replace those three gold merchants with the printed names. Replaying
the same `v1.json` predictions against the corrected labels yields **51/55**
merchant hits. Remaining misses are extra store numbers or location words
(`Taco Bell 017314`, `Thai Gusto` vs `Thai GUSTO Restaurant`, `ESQUIRE GRILLE`
vs the airport-qualified gold, `DEL FRISCO'S` vs `#8620`).

### 3. Copied `$` failed ISO currency scoring (fixed)

**Evidence:** 31 receipts have an explicit gold currency of `USD`. On
`reports/v1.json`, currency accuracy was **2/31**. The model copied a visible
`$` on 29 of those cases; `1145-receipt.jpg` returned `null` and
`needs_review`. Post-processing only inferred currency when the field was
already `null`, and image evals pass the MIME type as that hint, so `$` was
stored as `$`.

**Change:** `apply_postprocess` now maps copied aliases (`$` → `USD`, `€` →
`EUR`, case-insensitive `usd` / `egp` / `eur`) to ISO codes. Replaying that
normalization on `v1.json` would score **30/31**. The remaining miss is `1145`,
which still has no predicted currency. Re-run the live suite before citing a
new authoritative currency number.

## Known limits

- A schema-valid model prediction is not evidence that every field is visually grounded in the input.
- The dataset is curated; do not claim production-wide accuracy from these scores.
- Low-confidence, unreadable, or ambiguous financial fields must remain reviewable rather than guessed.
- Merchant is scored with normalized matching but is not part of combined `ok`. Extra store numbers and location words still fail that field.
- Hallucination rates cover one gold-null receipt total and three gold-null receipt dates; more unreadable-field fixtures are needed before treating zero hallucinations as a durable claim.
- The deterministic CI gate is intentionally small; it does not replace the live labelled baseline.

## Retrieval questions and ablation

Labelled questions live in `evals/retrieval_questions.jsonl` (**69** rows: 64
expected-found, 5 not-found). Each row has a question, gold
`relevant_source_ids` on the receipt corpus built from `evals/labels.jsonl`,
and an expected answer check (`expected_found`, `expected_answer_contains`).

Production `POST /v1/ask` does not reject citations outside the retrieved set.
Faithfulness below is an eval metric, not an API guarantee.

Metrics (implemented in `evals/retrieval_scoring.py`):

| Metric | Meaning |
|---|---|
| `recall@k` | Fraction of gold source_ids that appear in the top-k hits. Queries with an empty gold set (not-found) are excluded. |
| MRR | `1 / rank` of the first gold hit, else `0`. Same exclusion. |
| Answer correctness | `found` matches the label; if found, the answer contains the expected snippet. |
| Faithfulness | Every citation is a retrieved `source_id` (no invented IDs). |

Ablation (hash embeddings in CI/tests; live embeddings optional):

```bash
python -m evals.retrieval_run --json reports/retrieval-ablation.json
python -m evals.retrieval_run --live-embeddings --json reports/retrieval-live.json
```

Hash embeddings are not semantic. Use `--live-embeddings` before citing dense or
hybrid numbers. Keyword and `hybrid_rerank` are meaningful with either provider
because they use term overlap.

## Scripted agent tool checks

`evals/agent_cases.jsonl` has five cases. A scripted provider proposes the tool
calls, and `AgentService` runs them. The check scores the first tool name, its
status, the stop reason, and whether the seeded receipt's `used` flag changed.

```bash
python -m evals.agent_run
```

This is a safety check of the production agent path. It does not measure how
often a live model picks the right tool. A write case must stop at
`needs_approval` with `used` still false. The SQL case must end in a tool
error. The loop case must stop at `max_steps`.
