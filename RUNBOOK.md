# Runbook

Local API on port 8000. There is no public URL in this repo.

## Start and stop

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --reload
```

Docker Compose starts the API and Postgres with pgvector:

```bash
docker compose up --build
docker compose down
```

`./logs` holds `cost.jsonl` and `traces.jsonl` for both local and Compose runs. Compose sets `TRACE_LOG_PATH` and `COST_LOG_PATH` under `/app/logs`, which is that folder.

`GET /health` returns `{"status":"ok"}` when the process is up. It does not check OpenRouter or Postgres.

Pending agent approvals live in process memory. A restart drops them. The client must start a new `POST /v1/agent` call.

## Read a trace

Each line in `logs/traces.jsonl` is one finished step: a chat call, an embedding call, a search, or a tool call. Lines from one HTTP call share `request_id`.

```bash
python -m evals.trace_report
python -m evals.trace_report --path logs/traces.jsonl --json reports/traces.json
```

The summary prints span count, error rate, p50 and p95 of `duration_ms`, and USD per 100 spans. A step with no price counts as $0. A missing file prints an empty summary.

## Incident: the model provider failed during an agent call

**What you see.** `POST /v1/agent` or `POST /v1/agent/stream` returns HTTP 200. `stopped_reason` is `fallback`. The answer is `Stopped because the model provider failed.` Tool steps that finished before the failure are still in `steps`.

**What the trace shows.** Find the lines with that response's `X-Request-ID`. The chat line that failed has `error.type` set to the provider error, such as `ProviderError`. Earlier tool lines for the same request id have no `error.type`.

**What to do.** Read the API log for `agent_provider_failed` and the same request id. Check the OpenRouter key, credits, and model name. Ask the question again after the provider is healthy. Do not resume that `thread_id` for this failure: the run already finished, it is not waiting for approval. A write that was approved earlier in the same run is already saved.

A write that is waiting is a different stop: `stopped_reason` is `needs_approval`, and `POST /v1/agent/resume` continues it.
