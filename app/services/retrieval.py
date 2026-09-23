from __future__ import annotations

import json
import math
from typing import Any, Protocol

from app.llm.embeddings import EmbeddingProvider
from app.observability.tracing import SpanRecorder

SEARCH_STRATEGIES = ("keyword", "dense", "hybrid", "hybrid_rerank")
_CANDIDATE_MULTIPLIER = 4
_CANDIDATE_CAP = 50


class RetrievalRepository(Protocol):
    def search_keyword(
        self, query: str, *, limit: int, kind: str | None
    ) -> list[dict]: ...

    def search_dense(
        self, query_embedding: list[float], *, limit: int, kind: str | None
    ) -> list[dict]: ...

    def close(self) -> None: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "how",
        "much",
        "did",
        "do",
        "does",
        "was",
        "were",
        "is",
        "are",
        "i",
        "my",
        "me",
        "we",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "spend",
        "spent",
        "have",
        "had",
        "any",
        "from",
        "with",
        "about",
        "receipt",
        "receipts",
        "total",
        "totals",
    }
)


def keyword_terms(query: str) -> list[str]:
    terms: list[str] = []
    for raw in query.split():
        term = raw.strip("?,.!'\"")
        if term and term.lower() not in _STOPWORDS:
            terms.append(term)
    return terms or [part for part in query.split() if part]


def hybrid_merge(
    keyword_results: list[dict],
    dense_results: list[dict],
    *,
    keyword_weight: float = 0.5,
    dense_weight: float = 0.5,
    limit: int,
) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, row in enumerate(keyword_results):
        source_id = row["source_id"]
        score = keyword_weight * (1.0 - rank / max(len(keyword_results), 1))
        scores[source_id] = scores.get(source_id, 0.0) + score
        docs[source_id] = row

    for rank, row in enumerate(dense_results):
        source_id = row["source_id"]
        score = dense_weight * (1.0 - rank / max(len(dense_results), 1))
        scores[source_id] = scores.get(source_id, 0.0) + score
        docs[source_id] = docs.get(source_id, row)

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[:limit]
    return [{**docs[source_id], "score": scores[source_id]} for source_id in ranked]


def candidate_limit_for(limit: int) -> int:
    return min(max(limit * _CANDIDATE_MULTIPLIER, limit), _CANDIDATE_CAP)


def lexical_rerank(query: str, documents: list[dict], *, limit: int) -> list[dict]:
    terms = [term.lower() for term in keyword_terms(query)]
    scored: list[dict] = []
    pool = max(len(documents), 1)
    for rank, doc in enumerate(documents):
        content = str(doc.get("content") or "").lower()
        hits = sum(1 for term in terms if term in content)
        overlap = hits / max(len(terms), 1)
        prior = 1.0 - rank / pool
        score = 0.7 * overlap + 0.3 * prior
        scored.append({**doc, "score": score})
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:limit]


def postgres_keyword_query(
    query: str, *, limit: int, kind: str | None
) -> tuple[str, list[Any]]:
    terms = " ".join(keyword_terms(query))
    sql = (
        "SELECT id, kind, source_id, content, "
        "ts_rank_cd(to_tsvector('english', content), "
        "plainto_tsquery('english', %s)) AS score "
        "FROM documents WHERE to_tsvector('english', content) "
        "@@ plainto_tsquery('english', %s)"
    )
    params: list[Any] = [terms, terms]
    if kind:
        sql += " AND kind = %s"
        params.append(kind)
    sql += " ORDER BY score DESC LIMIT %s"
    params.append(limit)
    return sql, params


class SqliteRetrievalRepository:
    def __init__(self, db_path: Any):
        import sqlite3

        self._db_path = db_path
        self._sqlite = sqlite3

    def _connect(self) -> Any:
        conn = self._sqlite.connect(self._db_path)
        conn.row_factory = self._sqlite.Row
        return conn

    def search_keyword(self, query: str, *, limit: int, kind: str | None) -> list[dict]:
        conn = self._connect()
        try:
            terms = keyword_terms(query)
            if not terms:
                return []
            clauses = " AND ".join(["content LIKE ?" for _ in terms])
            sql = f"SELECT id, kind, source_id, content FROM documents WHERE {clauses}"
            params: list[Any] = [f"%{term}%" for term in terms]
            if kind:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY id LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search_dense(
        self, query_embedding: list[float], *, limit: int, kind: str | None
    ) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT id, kind, source_id, content, embedding FROM documents"
            params: list[Any] = []
            if kind:
                sql += " WHERE kind = ?"
                params.append(kind)
            sql += " ORDER BY id"
            rows = conn.execute(sql, params).fetchall()
            scored: list[dict] = []
            for row in rows:
                embedding = json.loads(row["embedding"])
                score = cosine_similarity(query_embedding, embedding)
                scored.append({**dict(row), "score": score})
            scored.sort(key=lambda r: r["score"], reverse=True)
            return scored[:limit]
        finally:
            conn.close()

    def close(self) -> None:
        return


class PostgresRetrievalRepository:  # pragma: no cover
    """Postgres-backed retrieval. Excluded from CI coverage."""

    def __init__(self, database_url: str):
        import psycopg

        self._psycopg = psycopg
        self._database_url = database_url
        self._conn: Any = None

    def _connect(self) -> Any:
        if self._conn is None:
            from pgvector.psycopg import register_vector

            self._conn = self._psycopg.connect(self._database_url, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def search_keyword(self, query: str, *, limit: int, kind: str | None) -> list[dict]:
        conn = self._connect()
        with conn.cursor() as cur:
            sql, params = postgres_keyword_query(query, limit=limit, kind=kind)
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def search_dense(
        self, query_embedding: list[float], *, limit: int, kind: str | None
    ) -> list[dict]:
        conn = self._connect()
        with conn.cursor() as cur:
            sql = (
                "SELECT id, kind, source_id, content, 1 - (embedding <=> %s::vector) AS score "
                "FROM documents"
            )
            params: list[Any] = [query_embedding]
            if kind:
                sql += " WHERE kind = %s"
                params.append(kind)
            sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.append(query_embedding)
            params.append(limit)
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def build_retrieval_repository(
    *,
    db_path: Any | None = None,
    database_url: str | None = None,
) -> RetrievalRepository:
    if database_url:
        return PostgresRetrievalRepository(database_url)
    return SqliteRetrievalRepository(db_path)


class RetrievalService:
    def __init__(
        self,
        repo: RetrievalRepository,
        embeddings: EmbeddingProvider,
        *,
        spans: SpanRecorder | None = None,
    ):
        self._repo = repo
        self._embeddings = embeddings
        self._spans = spans or SpanRecorder(None)

    def search(
        self,
        query: str,
        *,
        strategy: str = "hybrid",
        limit: int = 5,
        kind: str | None = None,
    ) -> list[dict]:
        with self._spans.span(
            "retrieval",
            {
                "gen_ai.operation.name": "retrieval",
                "receipty.retrieval.strategy": strategy,
            },
        ):
            return self._search(query, strategy=strategy, limit=limit, kind=kind)

    def _search(
        self,
        query: str,
        *,
        strategy: str,
        limit: int,
        kind: str | None,
    ) -> list[dict]:
        if strategy == "keyword":
            return self._repo.search_keyword(query, limit=limit, kind=kind)
        if strategy == "dense":
            embedding = self._embeddings.embed(query)
            return self._repo.search_dense(embedding, limit=limit, kind=kind)
        if strategy == "hybrid":
            keyword = self._repo.search_keyword(query, limit=limit, kind=kind)
            embedding = self._embeddings.embed(query)
            dense = self._repo.search_dense(embedding, limit=limit, kind=kind)
            return hybrid_merge(keyword, dense, limit=limit)
        if strategy == "hybrid_rerank":
            pool = candidate_limit_for(limit)
            keyword = self._repo.search_keyword(query, limit=pool, kind=kind)
            embedding = self._embeddings.embed(query)
            dense = self._repo.search_dense(embedding, limit=pool, kind=kind)
            merged = hybrid_merge(keyword, dense, limit=pool)
            return lexical_rerank(query, merged, limit=limit)
        raise ValueError(f"unknown strategy: {strategy}")
