# Receipty

Receipt photo or pasted text → structured `{merchant, total, currency, date}`.
If the total is missing or unreadable, the API returns `total: null` and `needs_review: true`. It does not invent amounts.

## Why

Expense tools fail quietly when a model hallucinates a total. Receipty treats extraction as a constrained pipeline: model JSON → Pydantic validation → deterministic post-rules → SQLite ledger, with a small labeled eval set on the same code path.

## Truthful extraction contract

Receipty guarantees a **schema-valid prediction**, not visual verification of every extracted field. A schema-valid result has the required JSON shape, permitted fields and types, and has passed Receipty’s deterministic parsing and post-processing rules. It can be safely consumed by software, but it can still be wrong about what the source image or text visibly contained.

An **evidence-grounded** result would require proof that each returned value is supported by the receipt itself. Receipty does not make that claim today: model output is a prediction, not a citation or a human verification step. Use labeled evaluation fixtures to measure accuracy, and require human review before high-stakes use such as accounting, payment, reimbursement, or tax decisions.

| Outcome | What Receipty has established | What it does not establish |
| --- | --- | --- |
| `success` | The model identified a receipt, and the final total and currency passed validation rules. | That every returned field is visibly supported by the source or factually correct. |
| `needs_review` | The input looks like a receipt, but a required money value is missing, ambiguous, invalid, or conservatively flagged. | That absent values should be guessed or inferred without review. |
| `not_receipt` | The model or post-rules concluded that the input is not a receipt; receipt fields are cleared. | That the input has no business value for any other workflow. |
| `extraction_failed` | No usable structured model response was available. | Why the provider failed, or that retrying will definitely succeed. |

## Layout

```
app/
  api/            HTTP routes + DI
  core/           settings, domain errors
  domain/         ReceiptExtract schema
  llm/            OpenRouter client + prompt
  services/       extraction + postprocess
  repository/     SQLite
  observability/  per-call cost JSONL
  static/         scan-deck dashboard (multi-upload board)
evals/            fixtures, labels, scoring
tests/            unit + API (mocked provider)
```

Request flow: `routes` → `ExtractionService` → OpenRouter → validate → postprocess → `ReceiptRepository`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # set OPENROUTER_API_KEY
python -m uvicorn app.main:app --reload
```

Open [http://localhost:8000/](http://localhost:8000/) for the dashboard: drop multiple receipt images, watch the scan queue, and read the extraction board + live pulse analytics.
Docker / OrbStack:

```bash
docker compose up --build
# stop while keeping local receipt data
docker compose down
# reset the local database completely
docker compose down -v
```

Compose stores SQLite in a Docker-managed `receipty_data` volume at `/app/data/receipts.db`. This makes a fresh clone safe: Docker never has to guess whether a missing host path is a file or directory.

## API

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| POST | `/v1/ingest` | JSON `{"text": "..."}` |
| POST | `/v1/ingest/image` | multipart file (`jpeg` / `png` / `webp`, max ~8MB) |
| GET | `/v1/receipts` | SQLite ledger |

Example:

```bash
curl -s localhost:8000/v1/ingest \
  -H 'content-type: application/json' \
  -d '{"text":"Carrefour\nTOTAL 186.50 EGP"}'
```

## Design notes

- **Never invent totals.** Bad or non-JSON model output becomes `is_receipt=false`, `needs_review=true`.
- **Post-rules** clear money fields when `is_receipt` is false, force review when `total` is null, and optionally infer currency from text hints (EGP/USD/EUR).
- **Provider errors** stay out of the service layer; routes map credits / daily-limit / upstream failures / deadline expiry to HTTP 402 / 429 / 502 / 504.
- **Cost log** at `logs/cost.jsonl` (OpenRouter `usage.cost` when present).

## Evals

20 labeled images under `evals/fixtures/` (`evals/labels.jsonl`). These human labels are the source of truth for the fields the evaluation measures.

```bash
python -m evals.run
# optional: python -m evals.run --json /tmp/eval-report.json
```

Scored fields: `is_receipt` and `total` only. Merchant string mismatches still count as OK because OCR/name variance is noisy compared to money. This is evidence for those two fields on this fixture set; it is not evidence-grounding for merchant, date, tax, or all production receipts.

## Tests

```bash
ruff format --check .
ruff check .
pyright
pytest --cov
```

GitHub Actions runs the same four quality gates for every pull request. No API key is required; the provider is mocked in API tests.

## Config

| Env | Default |
|-----|---------|
| `OPENROUTER_API_KEY` | (required for live calls) |
| `MODEL` | `z-ai/glm-5.3-flash` |
| `DB_PATH` | `receipts.db` |
| `COST_LOG_PATH` | `logs/cost.jsonl` |
| `MAX_IMAGE_BYTES` | `8388608` |

## Out of scope

Auth, bank sync, guessing missing totals.
