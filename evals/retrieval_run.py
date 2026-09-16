from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.llm.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenRouterEmbeddings,
)
from app.repository.receipts import SqliteReceiptRepository
from app.services.retrieval import (
    SEARCH_STRATEGIES,
    RetrievalService,
    SqliteRetrievalRepository,
)
from evals.retrieval_corpus import QUESTIONS, index_receipt_corpus, load_jsonl
from evals.retrieval_scoring import summarize_retrieval


def build_retrieval(
    db_path: Path, embeddings: EmbeddingProvider
) -> tuple[RetrievalService, int]:
    repo = SqliteReceiptRepository(db_path)
    repo.init_db()
    count = index_receipt_corpus(repo, embeddings)
    service = RetrievalService(SqliteRetrievalRepository(db_path), embeddings)
    return service, count


def run_ablation(
    *,
    embeddings: EmbeddingProvider,
    db_path: Path,
    k: int = 5,
    strategies: tuple[str, ...] = SEARCH_STRATEGIES,
) -> dict[str, Any]:
    questions = load_jsonl(QUESTIONS)
    service, corpus_size = build_retrieval(db_path, embeddings)
    report: dict[str, Any] = {
        "k": k,
        "questions": len(questions),
        "corpus_size": corpus_size,
        "strategies": {},
    }
    for strategy in strategies:
        rows = []
        for question in questions:
            hits = service.search(
                question["question"],
                strategy=strategy,
                limit=k,
                kind="receipt",
            )
            rows.append(
                {
                    "id": question["id"],
                    "relevant_source_ids": question["relevant_source_ids"],
                    "retrieved_ids": [hit["source_id"] for hit in hits],
                }
            )
        report["strategies"][strategy] = summarize_retrieval(rows, k=k)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval strategy ablation")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--live-embeddings",
        action="store_true",
        help="Use OpenRouter embeddings instead of the hash provider",
    )
    parser.add_argument("--db", type=Path, default=Path("reports/retrieval-eval.db"))
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()

    if args.live_embeddings:
        from app.core.config import get_settings

        embeddings: EmbeddingProvider = OpenRouterEmbeddings(get_settings())
    else:
        embeddings = HashEmbeddingProvider()

    report = run_ablation(embeddings=embeddings, db_path=args.db, k=args.k)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
