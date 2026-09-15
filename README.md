# Receipty

**Constrained multimodal receipt extraction.** A vision-capable LLM reads a receipt photo or pasted text and returns schema-valid `{merchant, total, currency, date, tax}` — or an explicit outcome when it cannot. Totals are never invented.

Receipty is a production-minded FastAPI service: OpenRouter chat completions with **strict JSON schema**, Pydantic validation, locale-aware money parsing, deterministic post-rules, a SQLite ledger, cost logging, and a labeled eval harness on the **same extraction path** the API uses.

| Stack | |
| --- | --- |
| Runtime | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| Model | OpenAI SDK → [OpenRouter](https://openrouter.ai) (default `google/gemini-3.1-flash-lite`) |
| Data | SQLite ledger + JSONL cost log |
| UI | Static scan-deck dashboard (multi-upload, review board, usage pulse) |
| Quality | Ruff, Pyright, Pytest (≥80% branch coverage), GitHub Actions |

## Why this exists

Expense and “OCR” products fail quietly when a model hallucinates a total. Receipty treats the LLM as an **untrusted extractor**, not a source of truth:

1. Ask the model only for fields that appear on the receipt.
2. Force a **strict structured-output schema** so extra keys and free-form prose cannot slip through.
3. Parse money and dates with **deterministic rules** that refuse ambiguous tokens.
4. Emit an **outcome** (`success` / `needs_review` / `not_receipt` / `extraction_failed`) so downstream software can route, not guess.
5. Measure the pipeline on labeled images — and document what those scores do *not* prove.

That combination — constrained generation, fail-closed validation, human-in-the-loop outcomes, and honest evaluation — is the core of the project.

## What this project demonstrates

Aimed at AI / applied-ML engineering work: shipping an extraction system rather than a chatbot.

| Theme | In this repo |
| --- | --- |
| **Structured generation** | OpenRouter `response_format` = `json_schema` named `receipt_extraction`, `strict: true`, schema from `ReceiptLLMOutput.model_json_schema()`. `extra_body.provider.require_parameters` so the gateway must honor the schema. Wire model uses `extra="forbid"`. |
| **Hallucination control** | Prompt: copy visible fields, never invent, null if unreadable. Dates must be visibly present — no inferred calendar math. Invalid or fenced JSON → `outcome=extraction_failed` with no invented fields. |
| **Document / vision LLM** | Image ingest as a `data:{mime};base64,...` `image_url` (JPEG / PNG / WebP, max ~8 MB). Text paste uses the same prompt and post-rules. No classical OCR stack. |
| **Locale-aware parsing** | `_money()` handles `1,234.56` vs `1.234,56`; a single separator plus three fractional digits is treated as ambiguous and becomes `null`. Dates accept several common formats; unparseable dates become `null`. |
| **Post-LLM rules** | Non-receipts scrub merchant/money/date. Missing total forces `needs_review`. Currency may be inferred from text hints (EGP / USD / EUR) only when unique. Negative totals need review. |
| **Evaluation** | 60 labeled fixtures (`evals/labels.jsonl`). Scoring runs the **production** `ExtractionService`. Metrics include class-conditional receipt, total, date, merchant, and available currency accuracy; tax is explicitly unavailable without labels. Combined `ok` requires receipt, total, and date. `extraction_failed` is never a correct receipt classification. See [EVALS.md](EVALS.md). |
| **Reproducible decoding** | Completions use `temperature=0.0` and `seed=42` by default so eval runs are comparable. Both are configurable. |
| **Provider reliability** | App-owned retries (SDK retries disabled): full-jitter backoff, per-attempt timeout, total deadline. Typed errors for credits (402), free-tier daily cap (429), deadline (504), other upstream (502). |
| **Cost observability** | Per-call prompt/completion tokens and USD → `logs/cost.jsonl`; aggregated on `GET /v1/usage` and the dashboard. |
| **Testability** | `LLMProvider` protocol + `create_app(provider_factory=...)`. CI runs lint, types, and coverage **without** a live API key. |

## Truthful extraction contract

Receipty guarantees a **schema-valid prediction**, not visual verification of every extracted field. A schema-valid result has the required JSON shape, permitted fields and types, and has passed Receipty’s deterministic parsing and post-processing rules. Software can consume it safely as a typed object; it can still be wrong about what the source image or text visibly contained.

An **evidence-grounded** result would require proof that each returned value is supported by the receipt itself. Receipty does not make that claim: model output is a prediction, not a citation. Use labeled fixtures to measure accuracy, and require human review before accounting, payment, reimbursement, or tax decisions.

| Outcome | What Receipty has established | What it does not establish |
| --- | --- | --- |
| `success` | The model identified a receipt, and the final total and currency passed validation rules. | That every returned field is visibly supported by the source or factually correct. |
| `needs_review` | The input looks like a receipt, but a required money value is missing, ambiguous, invalid, or conservatively flagged. | That absent values should be guessed without review. |
| `not_receipt` | The model or post-rules concluded that the input is not a receipt; receipt fields are cleared. | That the input has no value for any other workflow. |
| `extraction_failed` | No usable structured model response was available (empty content, invalid JSON, or schema violation). | Why the provider failed, or that retrying will succeed. |

## Architecture

```
Dashboard / curl
        │
        ▼
 FastAPI  (X-Request-ID, MIME + size gates)
        │
        ▼
 ExtractionService
        │  system prompt + text | image_url
        ▼
 OpenRouterProvider
   • strict json_schema (receipt_extraction)
   • require_parameters
   • temperature + seed
   • retries, jitter, deadline
        │
        ▼
 ReceiptLLMOutput  →  ReceiptExtract
   Pydantic v2, extra=forbid, money/date parsers
        │
        ▼
 apply_postprocess  (scrub / infer currency / outcomes)
        │
        ├──► ReceiptRepository (SQLite)
        └──► UsageLogger (JSONL cost)
```

```
app/
  api/            HTTP routes, DI, provider-error → HTTP mapping
  core/           settings, typed provider exceptions
  domain/         Outcome, ReceiptLLMOutput, ReceiptExtract, parsers
  llm/            OpenRouter client + extraction prompt
  services/       ExtractionService, postprocess
  repository/     SQLite ledger
  observability/  per-call cost JSONL
  static/         scan-deck dashboard
evals/            fixtures, gold labels, runner, scoring
EVALS.md          eval contract, dataset, how to run
tests/            unit + API (mocked provider)
```

## Extraction pipeline

**Prompt** (`app/llm/prompts.py`): extract a purchase receipt; if it is not a receipt, set `is_receipt=false` and null the rest; copy visible merchant / total / date / tax exactly; never invent; unreadable fields are `null`. Transaction dates are extracted when visibly readable — including `5/26/2016`, `05/26/2016`, `2016-05-26`, `06Aug'16`, and month-name forms — then returned as `YYYY-MM-DD`. Absent, ambiguous, or unreadable dates stay `null`; the model must not infer or fabricate them. JSON only.

**Structured output** (`app/llm/provider.py`): every completion requests

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "receipt_extraction",
    "strict": true,
    "schema": { "...ReceiptLLMOutput..." }
  }
}
```

plus OpenRouter `provider.require_parameters: true`. Completions also send `temperature` (default `0.0`) and `seed` (default `42`) so extraction is as deterministic as the provider allows. The SDK’s own retries are off (`max_retries=0`) so backoff, timeouts, and billing/rate-limit mapping live in one place.

**Fail closed** (`app/services/extraction.py`): missing message content or `ValidationError` becomes `is_receipt=false`, `outcome=extraction_failed`. Markdown fences are stripped as a defensive fallback; they are not a license to invent fields.

**Post-rules** (`app/services/postprocess.py` + `ReceiptExtract._resolve_outcome`):

- Not a receipt → clear merchant, total, currency, date, tax; outcome `not_receipt` (unless already `extraction_failed`).
- Receipt with `total is None`, missing currency, or `total < 0` → `needs_review`.
- Missing currency on text ingest may be filled from unique hints (`EGP` / `LE` / `ج.م`, `USD` / `$`, `EUR` / `€`). Ambiguous mixed hints stay `null`.

## Dashboard

Open [http://localhost:8000/](http://localhost:8000/) after startup.

- **Scan deck** — drag-and-drop or file picker; multiple JPEG / PNG / WebP at once (client concurrency = 3).
- **Scan queue** — in-flight uploads.
- **Extraction board** — tickets filterable by all / receipts / needs review / not receipts.
- **Live pulse** — processed count, receipt rate, review rate, spend by currency.
- **Cost pulse** — aggregated tokens and USD from `GET /v1/usage`.
- **Paste text** — same extraction path without an image.
- Health pill (`GET /health`) and a link to FastAPI `/docs`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # set OPENROUTER_API_KEY
python -m uvicorn app.main:app --reload
```

Get an OpenRouter key at [openrouter.ai](https://openrouter.ai). The default model is multimodal; if you change `MODEL`, pick one that accepts `image_url`.

### Docker

```bash
docker compose up --build
# stop while keeping local receipt data
docker compose down
# reset the local database completely
docker compose down -v
```

Compose stores SQLite in a Docker-managed `receipty_data` volume at `/app/data/receipts.db` (a missing host file is never mistaken for a directory). Cost logs bind-mount to `./logs`.

## API

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| GET | `/` | Dashboard HTML |
| POST | `/v1/ingest` | JSON `{"text": "..."}` |
| POST | `/v1/ingest/image` | multipart file (`jpeg` / `png` / `webp`, max ~8 MB) |
| GET | `/v1/receipts` | SQLite ledger (newest first) |
| GET | `/v1/usage` | Aggregated cost / tokens (last 50 call rows) |

Interactive OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs).

Optional `X-Request-ID` (1–64 of `A-Za-z0-9._-`) is echoed on the response and included in provider retry logs; otherwise a UUID is generated.

### Text ingest

```bash
curl -s localhost:8000/v1/ingest \
  -H 'content-type: application/json' \
  -d '{"text":"Carrefour\nTOTAL 186.50 EGP"}'
```

Example success payload:

```json
{
  "extract": {
    "is_receipt": true,
    "merchant": "Carrefour",
    "total": "186.50",
    "currency": "EGP",
    "date": null,
    "tax": null,
    "outcome": "success"
  }
}
```

### Image ingest

```bash
curl -s localhost:8000/v1/ingest/image \
  -F 'file=@evals/fixtures/1131-receipt.jpg'
```

### Provider errors

| Condition | HTTP |
| --- | --- |
| Total provider deadline exhausted | 504 |
| OpenRouter credits exhausted | 402 |
| Free-tier daily request cap | 429 |
| Other upstream / retry exhaustion | 502 |
| Unsupported MIME | 400 |
| Image larger than `MAX_IMAGE_BYTES` | 413 |

## Evaluation

59 labeled images under `evals/fixtures/` with gold labels in `evals/labels.jsonl` (54 receipts, 5 non-receipts). The set includes a receipt with a null total and three receipts with null dates so missing values are scored, not guessed. Labels are the source of truth for the fields the harness scores. Dataset contract and run notes: [EVALS.md](EVALS.md).

```bash
python -m evals.run
# optional local report (gitignored)
python -m evals.run --json reports/eval.json
```

The runner builds the same `ExtractionService` + `OpenRouterProvider` + `UsageLogger` as the API (live key required). Per-file output covers receipt, total, and date; merchant is printed for inspection but is not a scored field.

| Metric | Rule |
| --- | --- |
| `is_receipt` | Predicted receipt vs gold. `extraction_failed` is always a miss — the system made no classification. |
| `total` | Exact numeric match, including both-null. |
| `date` | Exact `YYYY-MM-DD` match against the gold label, including both-null. |
| Combined `ok` | All three hits. |

Merchant is measured with case- and whitespace-normalized matching, but does not
change combined `ok`. Currency is measured only when the gold label is explicit;
tax remains unavailable because labels do not contain tax values. Scores are
evidence for this fixture set, not evidence-grounding or production traffic.

## Tests and CI

```bash
ruff format --check .
ruff check .
pyright
pytest --cov
```

Coverage fails under 80% (branch coverage on `app/`). GitHub Actions runs the same four gates on every pull request. No API key is required; the provider is mocked.

Covered behavior includes:

- Strict structured-output request shape (`json_schema` + `require_parameters`)
- Temperature and seed forwarded to the provider
- Money parsing (US / EU separators, ambiguous three-digit fractions, negatives)
- Outcomes for success, review, non-receipt scrub, malformed JSON, extra model fields
- Provider retries, jitter, deadline, non-retryable errors, SDK retries disabled
- HTTP mapping, request-id propagate/generate, MIME rejection, lifespan close

## Config

| Env | Default |
|-----|---------|
| `OPENROUTER_API_KEY` | (required for live calls) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `MODEL` | `google/gemini-3.1-flash-lite` |
| `TEMPERATURE` | `0.0` |
| `SEED` | `42` |
| `DB_PATH` | `receipts.db` |
| `COST_LOG_PATH` | `logs/cost.jsonl` |
| `MAX_IMAGE_BYTES` | `8388608` |
| `MAX_ATTEMPTS` | `3` |
| `REQUEST_TIMEOUT_SECONDS` | `60` |
| `TOTAL_DEADLINE_SECONDS` | `120` |
| `RETRY_BASE_DELAY_SECONDS` | `1` |

## Out of scope

Auth, multi-tenant isolation, bank sync, line-item extraction, RAG / embeddings, classical OCR, fine-tuning, and guessing missing totals.

Apache License 2.0 — see [LICENSE.md](LICENSE.md).
