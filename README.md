# Receipty

**Constrained multimodal receipt extraction, grounded retrieval, and a bounded ledger agent.** A vision-capable LLM reads a receipt photo or pasted text and returns schema-valid `{merchant, total, currency, date, tax}` — or an explicit outcome when it cannot. Totals are never invented. Indexed receipts can then be searched (keyword / dense / hybrid / hybrid+rerank) and answered with **receipt-ID citations**, or an explicit not-found response. A separate agent can call allowlisted ledger tools over several steps. Writes wait for approval on the HTTP agent. The same tools are also available over MCP stdio.

Receipty is a production-minded FastAPI service: OpenRouter chat completions with **strict JSON schema**, Pydantic validation, locale-aware money parsing, deterministic post-rules, a SQLite or Postgres+pgvector ledger, embedding-backed retrieval, a LangGraph tool loop, an MCP stdio server for those same tools, cost logging, and labeled eval harnesses on the **same extraction and retrieval paths** the API uses.

| Stack | |
| --- | --- |
| Runtime | Python 3.12, FastAPI, Uvicorn, Pydantic v2, LangGraph, MCP |
| Models | OpenAI SDK → [OpenRouter](https://openrouter.ai): chat `google/gemini-3.1-flash-lite`, embeddings `openai/text-embedding-3-small` |
| Data | SQLite by default; Postgres + pgvector when `DATABASE_URL` is set. JSONL cost log. |
| UI | Static scan-deck dashboard (multi-upload, review board, usage pulse) |
| Quality | Ruff, Pyright, Pytest (≥80% branch coverage), deterministic eval gate, `pip-audit`, GitHub Actions |

## Why this exists

Expense and “OCR” products fail quietly when a model hallucinates a total. Chat-over-receipts products fail the same way when an answer is not tied to a retrieved document. Receipty treats the LLM as an **untrusted extractor and answerer**, not a source of truth:

1. Ask the model only for fields that appear on the receipt.
2. Force a **strict structured-output schema** so extra keys and free-form prose cannot slip through.
3. Parse money and dates with **deterministic rules** that refuse ambiguous tokens.
4. Emit an **outcome** (`success` / `needs_review` / `not_receipt` / `extraction_failed`) so downstream software can route, not guess.
5. Index successful extracts and answer questions only from retrieved context, with citations — or say not found.
6. For multi-step ledger questions, let the model call allowlisted tools. Reads run; writes pause for approval. Cap the number of rounds.
7. Measure both pipelines on labeled sets — and document what those scores do *not* prove.

That combination — constrained generation, fail-closed validation, human-in-the-loop outcomes, grounded retrieval, and honest evaluation — is the core of the project.

## What this project demonstrates

Aimed at AI / applied-ML engineering work: shipping extraction and retrieval systems rather than a chatbot.

| Theme | In this repo |
| --- | --- |
| **Structured generation** | OpenRouter `response_format` = `json_schema` named `receipt_extraction`, `strict: true`, schema from `ReceiptLLMOutput.model_json_schema()`. `extra_body.provider.require_parameters` so the gateway must honor the schema. Wire model uses `extra="forbid"`. |
| **Hallucination control** | Prompt: copy visible fields, never invent, null if unreadable. Dates must be visibly present — no inferred calendar math. Invalid JSON → `outcome=extraction_failed`. Ask path: empty retrieval / bad JSON → `found=false`, no citations. |
| **Document / vision LLM** | Image ingest as a `data:{mime};base64,...` `image_url` (JPEG / PNG / WebP, max ~8 MB). Text paste uses the same prompt and post-rules. No classical OCR stack. |
| **Locale-aware parsing** | `_money()` handles `1,234.56` vs `1.234,56`; a single separator plus three fractional digits is treated as ambiguous and becomes `null`. Dates accept several common formats; unparseable dates become `null`. Copied `$` / `€` normalize to ISO codes. |
| **Post-LLM rules** | Non-receipts scrub merchant/money/date. Missing total forces `needs_review`. Unique text hints or copied symbols map to ISO currency. Negative totals need review. |
| **Hybrid retrieval** | Keyword, dense (cosine / pgvector), hybrid merge, and hybrid + lexical rerank over receipt docs, merchant aliases, and policy notes. |
| **Grounded answering** | `POST /v1/ask` retrieves receipt context, then a strict `AnswerResponse` schema (`answer`, `citations`, `found`). Citation *faithfulness* is scored in evals, not enforced at request time. |
| **Bounded agent** | `POST /v1/agent` is a LangGraph tool loop over four application-owned tools. Reads run immediately. Writes call `interrupt()` and wait for `POST /v1/agent/resume`. `MAX_AGENT_STEPS` (default 8) stops a tool loop. The in-memory checkpointer is process-local. The same four tools are also registered on an MCP stdio server (`python -m app.mcp_server`); an MCP client call runs the Python function, including writes. |
| **Evaluation** | 60 labeled extraction fixtures and 69 retrieval questions. Extraction scoring runs the production `ExtractionService`. Retrieval ablation runs the production search/ask path. See [EVALS.md](EVALS.md). |
| **Reproducible decoding** | Completions use `temperature=0.0` and `seed=42` by default. Prompt versions (`extraction-v1` production; `extraction-v2-evidence` experiment) are hashed in eval reports. |
| **Provider reliability** | App-owned retries (SDK retries disabled): full-jitter backoff, per-attempt timeout, total deadline. Typed errors for credits (402), free-tier daily cap (429), deadline (504), other upstream (502). |
| **Cost observability** | Per-call prompt/completion tokens and USD → `logs/cost.jsonl`; aggregated on `GET /v1/usage` and the dashboard. |
| **Step traces** | Each chat completion, embedding call, retrieval, and tool call appends one JSON line to `logs/traces.jsonl`: request id, duration, model, token counts, provider USD cost when present, and `error.type` when the step fails. Tool arguments are not stored. |
| **Embedding cache** | The same text reuses its vector for the life of the process, up to `EMBEDDING_CACHE_SIZE` entries. A cache hit skips the embedding provider, so it writes no embedding span and no cost line. Restarting the process drops the cache. |
| **PII redaction** | Payment cards (13–19 digits), email addresses, phone numbers, and credentials (`Bearer`, `sk-*`) are redacted via `RedactingFilter` and `redact_pii` before reaching logs or trace attributes. |
| **Testability** | `LLMProvider` / `EmbeddingProvider` protocols + `create_app(provider_factory=...)`. CI runs lint, types, a deterministic eval safety gate, coverage, and `pip-audit` **without** a live API key. |

## Truthful extraction contract

Receipty guarantees a **schema-valid prediction**, not visual verification of every extracted field. A schema-valid result has the required JSON shape, permitted fields and types, and has passed Receipty’s deterministic parsing and post-processing rules. Software can consume it safely as a typed object; it can still be wrong about what the source image or text visibly contained.

An **evidence-grounded** result would require proof that each returned value is supported by the receipt itself. Receipty does not make that claim for extraction: model output is a prediction, not a citation. The ask path *does* attach retrieved `source_id`s, but it does not prove those IDs are visually grounded in the original image. Use labeled evaluation fixtures to measure accuracy, and require human review before accounting, payment, reimbursement, or tax decisions.

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
        ├─ ingest ──► ExtractionService ──► OpenRouter (strict json_schema)
        │                    │
        │                    ▼
        │              Pydantic + postprocess (outcomes, ISO currency)
        │                    │
        │                    ├──► ReceiptRepository (SQLite | Postgres)
        │                    ├──► IndexingService (embed + documents table)
        │                    └──► UsageLogger (JSONL cost)
        │                         SpanRecorder (JSONL traces)
        │
        ├─ /v1/search ──► RetrievalService
        │                   keyword | dense | hybrid | hybrid_rerank
        │
        ├─ /v1/ask ──► AnsweringService
        │                retrieve receipts → context → JSON answer
        │                empty hits / bad JSON → found=false
        │
        └─ /v1/agent ──► AgentService (LangGraph + InMemorySaver)
                         model ↔ allowlisted tools, max N rounds
                         reads run; writes interrupt until /v1/agent/resume
                         time, token budget, or provider failure → stopped_reason=fallback
           /v1/agent/stream ──► the same run
                         SSE: step events, then done

MCP stdio (`python -m app.mcp_server`) calls the same four Python tools.
Writes run immediately there, because the MCP client invoked the tool.
```

```
app/
  api/            HTTP routes, DI, provider-error → HTTP mapping
  core/           settings, typed provider exceptions
  domain/         Outcome, ReceiptLLMOutput, ReceiptExtract, Ask/Answer schemas
  llm/            OpenRouter chat + embeddings + versioned prompts
  services/       extraction, postprocess, indexing, retrieval, answering, tools, agent
  mcp_server.py   MCP stdio entry; same AgentTools
  repository/     SQLite or Postgres+pgvector ledger and document store
  seed/           merchant aliases + policy notes
  observability/  per-call cost JSONL, step traces, and PII redactor
  static/         scan-deck dashboard
evals/            extraction fixtures, retrieval questions, runners, scoring
EVALS.md          eval contract, baselines, how to run
tests/            unit + API (mocked provider / hash embeddings)
```

Local default is SQLite (`DB_PATH=receipts.db`). Set `DATABASE_URL` for Postgres with a `vector(1536)` column and HNSW cosine index. Docker Compose always starts pgvector and points the API at it.

## Extraction pipeline

**Prompt** (`app/llm/prompts.py`, production `extraction-v1`): extract a purchase receipt; if it is not a receipt, set `is_receipt=false` and null the rest; copy visible merchant / total / date / tax exactly; never invent; unreadable fields are `null`. Transaction dates are extracted when visibly readable — including `5/26/2016`, `05/26/2016`, `2016-05-26`, `06Aug'16`, and month-name forms — then returned as `YYYY-MM-DD`. Absent, ambiguous, or unreadable dates stay `null`; the model must not infer or fabricate them. JSON only.

`extraction-v2-evidence` is a documented experiment (stricter TOTAL/AMOUNT DUE wording). It did not beat v1 on the labelled contract; v1 remains production. See [EVALS.md](EVALS.md).

**Structured output** (`app/llm/provider.py`): extraction completions request

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

plus OpenRouter `provider.require_parameters: true`. `POST /v1/ask` sends the `AnswerResponse` schema instead. Agent completions send `tools` and skip the extraction schema. Completions also send `temperature` (default `0.0`) and `seed` (default `42`) so extraction is as deterministic as the provider allows. The SDK’s own retries are off (`max_retries=0`) so backoff, timeouts, and billing/rate-limit mapping live in one place.

**Fail closed** (`app/services/extraction.py`): missing message content or `ValidationError` becomes `is_receipt=false`, `outcome=extraction_failed`. Markdown fences are stripped as a defensive fallback; they are not a license to invent fields.

**Escalation:** `MODEL` handles every extraction. When `ESCALATION_MODEL` is set to a different model, a result of `needs_review` or `extraction_failed` is sent once to that model. `success` and `not_receipt` stay on `MODEL`. The second call is logged as `text_escalation` or `image_escalation`. An empty `ESCALATION_MODEL` makes one call.

**Post-rules** (`app/services/postprocess.py` + `ReceiptExtract._resolve_outcome`):

- Not a receipt → clear merchant, total, currency, date, tax; outcome `not_receipt` (unless already `extraction_failed`).
- Receipt with `total is None`, missing currency, or `total < 0` → `needs_review`.
- Missing currency may be filled from unique hints (`EGP` / `LE` / `ج.م`, `USD` / `$`, `EUR` / `€`). Copied `$` / `€` / case-insensitive ISO aliases normalize to ISO codes. Ambiguous mixed hints stay `null`.

Successful extracts (not `not_receipt` / `extraction_failed`) are written to the document index as `receipt:{id}` text plus an embedding.

## Retrieval and grounded answers

After ingest, `IndexingService` embeds a compact receipt string and upserts it into `documents`. Startup also seeds merchant aliases and policy notes from `app/seed/`.

`GET /v1/search` strategies (`app/services/retrieval.py`):

| Strategy | Behavior |
| --- | --- |
| `keyword` | SQLite `LIKE` AND-terms, or Postgres `tsvector` / `ts_rank_cd` |
| `dense` | Query embedding vs stored vectors (SQLite cosine scan, Postgres `<=>`) |
| `hybrid` | Rank fusion of keyword + dense (equal weights) |
| `hybrid_rerank` | Larger hybrid pool, then lexical-overlap rerank |

Optional `kind` filter: `receipt` \| `merchant_alias` \| `policy_note`.

`POST /v1/ask` always searches `kind=receipt`, builds a `[source_id: …]` context block, and asks the chat model for a strict `AnswerResponse`. No hits, invalid JSON, or provider failure return the fixed not-found message with `found=false` and empty citations. The API does **not** currently reject citations that were not in the retrieved set; `evals/retrieval_scoring.py` measures that faithfulness separately.

`POST /v1/agent` is a LangGraph tool-calling loop over the same ledger. The model may call `search_receipts` and `query_ledger` (named aggregates only — no SQL). `flag_for_review` and `mark_used` pause the graph with `interrupt()`; the response includes a `thread_id` and `stopped_reason=needs_approval`. `POST /v1/agent/resume` with `{thread_id, approved}` either runs the write or returns a tool error, then continues the loop. A hard `MAX_AGENT_STEPS` cap (default 8) stops a model that keeps requesting tools. `MAX_AGENT_SECONDS` (default 60) stops the run before another model call when that budget is spent. `MAX_AGENT_TOKENS` (default 16000) does the same once the provider has reported that many prompt plus completion tokens on this run. Approving a write does not reset that token count. If the provider fails during a call, the run stops with `stopped_reason=fallback` and keeps any tool steps already finished.

The checkpointer is `InMemorySaver`: pending approvals live in the API process and are lost on restart.

`POST /v1/agent/stream` is the same run as server-sent events. Each finished tool call is a `step` event. The last event is `done`, with the same body as `POST /v1/agent`, including `thread_id` when a write is waiting and `stopped_reason=fallback` when the provider fails or the time budget is spent. The model’s tokens are not streamed one by one.

`python -m app.mcp_server` is a second doorbell for the same `AgentTools` class, over MCP stdio. `query_ledger` still only accepts `sum_total`, `count`, and `totals_by_merchant`. Calling `mark_used` or `flag_for_review` from an MCP client runs the write. `POST /v1/agent` still pauses those writes, because the caller there is the model.

Search, ask, and agent are API-only; the dashboard does not expose them yet.

## Dashboard

Open [http://localhost:8000/](http://localhost:8000/) after startup.

- **Scan deck** — drag-and-drop or file picker; multiple JPEG / PNG / WebP at once (client concurrency = 3).
- **Scan queue** — in-flight uploads.
- **Extraction board** — tickets filterable by all / receipts / needs review / not receipts. A ticket is “needs review” only when `outcome === "needs_review"`, not merely because it is a receipt.
- **Live pulse** — processed count, receipt count, needs-review count (same `outcome` rule), spend by currency.
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
# optional: MCP stdio server for the same four ledger tools
python -m app.mcp_server
```

Get an OpenRouter key at [openrouter.ai](https://openrouter.ai). The default chat model must accept `image_url`. Leave `DATABASE_URL` empty to use local SQLite. Run commands with the project virtualenv (`source .venv/bin/activate`), not a system Python that does not have these dependencies. `python -m app.mcp_server` waits on stdin for an MCP client. It does not print a URL.

### Docker

```bash
docker compose up --build
# stop while keeping Postgres data
docker compose down
# reset local volumes completely
docker compose down -v
```

Compose runs `pgvector/pgvector:pg16` and sets `DATABASE_URL` on the API. Cost and trace logs bind-mount to `./logs`. Named volumes: `receipty_data` (unused SQLite path kept for compatibility) and `receipty_pg`. Start, stop, and a failed-agent incident are in [RUNBOOK.md](RUNBOOK.md).

## API

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| GET | `/` | Dashboard HTML |
| POST | `/v1/ingest` | JSON `{"text": "..."}` |
| POST | `/v1/ingest/image` | multipart file (`jpeg` / `png` / `webp`, max ~8 MB) |
| GET | `/v1/receipts` | Ledger (newest first) |
| GET | `/v1/search` | Query `?q=...&strategy=keyword\|dense\|hybrid\|hybrid_rerank&limit=5&kind=receipt\|merchant_alias\|policy_note` |
| POST | `/v1/ask` | JSON `{"question": "...", "strategy": "hybrid", "limit": 5}` → `{answer, citations, found}` |
| POST | `/v1/agent` | JSON `{"question": "..."}` → `{answer, stopped_reason, steps, pending_mutation, thread_id}` |
| POST | `/v1/agent/stream` | Same body as `/v1/agent`, as SSE events `step`, then `done` |
| POST | `/v1/agent/resume` | JSON `{"thread_id": "...", "approved": true}` → same shape; 404 unknown thread, 409 if not paused |
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

### Search and ask

```bash
curl -s 'localhost:8000/v1/search?q=Carrefour&strategy=hybrid&limit=5'
curl -s localhost:8000/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question":"What did I spend at Carrefour?","strategy":"hybrid","limit":5}'
curl -s localhost:8000/v1/agent \
  -H 'content-type: application/json' \
  -d '{"question":"How much did I spend at Taco Bell?"}'
curl -N localhost:8000/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{"question":"How much did I spend at Taco Bell?"}'
curl -s localhost:8000/v1/agent/resume \
  -H 'content-type: application/json' \
  -d '{"thread_id":"THREAD_ID","approved":true}'
```

### Provider errors

| Condition | HTTP |
| --- | --- |
| Total provider deadline exhausted | 504 |
| OpenRouter credits exhausted | 402 |
| Free-tier daily request cap | 429 |
| Too many ingest, search, ask, or agent calls from one client address in `RATE_LIMIT_WINDOW_SECONDS` | 429 |
| Other upstream / retry exhaustion | 502 |
| Unsupported MIME | 400 |
| Image larger than `MAX_IMAGE_BYTES` | 413 |
| Empty search query / bad strategy or limit | 400 |
| Unknown agent thread | 404 |
| Resume when the agent is not waiting | 409 |

A provider failure inside `POST /v1/agent` or `POST /v1/agent/stream` is a finished run: HTTP 200, `stopped_reason=fallback`, and any tool steps already completed. Extraction, search, and ask still use the table above.

## Evaluation

60 labeled images under `evals/fixtures/` with gold labels in `evals/labels.jsonl` (55 receipts, 5 non-receipts). The set includes a receipt with a null total and three receipts with null dates so missing values are scored, not guessed. Dataset contract, prompt experiments, and baselines: [EVALS.md](EVALS.md).

```bash
python -m evals.run
python -m evals.run --prompt-version extraction-v1 --json reports/eval.json
python -m evals.compare reports/run-a.json reports/run-b.json
python -m evals.retrieval_run --json reports/retrieval-ablation.json
python -m evals.retrieval_run --live-embeddings --json reports/retrieval-live.json
python -m evals.agent_run
python -m evals.agent_demo
python -m evals.trace_report
```

The extraction runner builds the same `ExtractionService` + `OpenRouterProvider` + `UsageLogger` as the API (live key required). Combined `ok` requires receipt, total, and date. Merchant (normalized) and labelled currency are reported but do not change combined `ok`. Tax is stored, not scored.

Retrieval evals use 69 labelled questions (64 expected-found, 5 not-found) against a corpus built from the extraction labels. Default ablation uses hash embeddings (deterministic, not semantic). Pass `--live-embeddings` before citing dense or hybrid numbers.

| Extraction metric | Rule |
| --- | --- |
| `is_receipt` | Predicted receipt vs gold. `extraction_failed` is always a miss. |
| `total` | Exact numeric match, including both-null. |
| `date` | Exact `YYYY-MM-DD` match, including both-null. |
| Combined `ok` | Those three hits. |

Scores are evidence for **this fixture set**, not evidence-grounding or production traffic.

`python -m evals.agent_run` scores five scripted agent cases on the production `AgentService`: tool name, argument outcome, stop reason, and whether a write changed `used`. It does not measure a live model's tool choices.

`python -m evals.agent_demo` prints two scripted runs on that same service: a finished count, then a provider failure that keeps the search step. It is a walkthrough, not a score.

`python -m evals.trace_report` reads `logs/traces.jsonl` and prints span count, error rate, p50 and p95 of `duration_ms`, and USD cost per 100 spans. Spans with no cost count as zero. A missing file prints an empty summary. These numbers describe the steps in that file, not a labelled eval set.

## Tests and CI

```bash
ruff format --check .
ruff check .
pyright
pytest tests/test_deterministic_eval_gate.py -q
pytest --cov
pip-audit --strict
```

Coverage fails under 80% (branch coverage on `app/`). GitHub Actions runs lint, types, the eval safety gate, coverage, and `pip-audit` on every pull request. No API key is required; the chat provider is mocked and retrieval tests use hash embeddings.

Covered behavior includes:

- Strict structured-output request shape (`json_schema` + `require_parameters`)
- Temperature and seed forwarded to the provider
- Money parsing, ISO currency normalization, outcomes, malformed JSON
- Provider retries, jitter, deadline, non-retryable errors
- Indexing, keyword/dense/hybrid/rerank retrieval, grounded ask / not-found
- Agent tool allowlist, write pause until resume, SSE `step` / `done`, fallback stop
- MCP server lists and calls the same four tools
- Deterministic eval safety gate (invented totals must not pass)
- Scripted agent cases: valid tool calls, rejected SQL, write pause, step cap
- PII redaction: card numbers, email, phone, and tokens scrubbed from logs and traces
- HTTP mapping, request-id, MIME rejection, lifespan close

## Config

| Env | Default |
|-----|---------|
| `OPENROUTER_API_KEY` | (required for live calls) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `MODEL` | `google/gemini-3.1-flash-lite` |
| `ESCALATION_MODEL` | empty (one model). A different model receives one retry after `needs_review` or `extraction_failed` |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` |
| `EMBEDDING_CACHE_SIZE` | `256` (set `0` to call the embedding provider every time) |
| `TEMPERATURE` | `0.0` |
| `SEED` | `42` |
| `DB_PATH` | `receipts.db` (SQLite when `DATABASE_URL` is unset) |
| `DATABASE_URL` | unset (optional Postgres, e.g. `postgres://receipty:receipty@localhost:5432/receipty`) |
| `COST_LOG_PATH` | `logs/cost.jsonl` |
| `TRACE_LOG_PATH` | `logs/traces.jsonl` |
| `MAX_IMAGE_BYTES` | `8388608` |
| `MAX_ATTEMPTS` | `3` |
| `REQUEST_TIMEOUT_SECONDS` | `60` |
| `TOTAL_DEADLINE_SECONDS` | `120` |
| `RETRY_BASE_DELAY_SECONDS` | `1` |
| `MAX_AGENT_STEPS` | `8` |
| `MAX_AGENT_SECONDS` | `60` |
| `MAX_AGENT_TOKENS` | `16000` |
| `RATE_LIMIT_REQUESTS` | `60` (set `0` to allow every call) |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` |

## Out of scope

Auth, multi-tenant isolation, bank sync, line-item extraction, classical OCR, fine-tuning, dashboard search/ask UI, runtime citation-set enforcement, and guessing missing totals.

Apache License 2.0 — see [LICENSE.md](LICENSE.md).
