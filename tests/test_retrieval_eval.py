from __future__ import annotations

from pathlib import Path

from app.llm.embeddings import HashEmbeddingProvider
from app.repository.receipts import SqliteReceiptRepository
from app.services.answering import AnsweringService
from app.services.retrieval import RetrievalService, SqliteRetrievalRepository
from evals.retrieval_corpus import QUESTIONS, index_receipt_corpus, load_jsonl
from evals.retrieval_scoring import (
    answer_correct,
    citation_faithfulness,
    mean_reciprocal_rank,
    recall_at_k,
    summarize_answers,
    summarize_retrieval,
)
from tests.test_answering import FakeProvider


def test_question_set_meets_minimum_size():
    questions = load_jsonl(QUESTIONS)
    assert len(questions) >= 40
    assert any(not q["expected_found"] for q in questions)
    assert any(len(q["relevant_source_ids"]) > 1 for q in questions)


def test_recall_at_k_and_mrr():
    retrieved = ["receipt:b", "receipt:a", "receipt:c"]
    relevant = ["receipt:a"]
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5


def test_answer_correctness_and_faithfulness():
    assert answer_correct(
        found=True,
        answer="Taco Bell total was 7.61 USD.",
        citations=["receipt:1002-receipt.jpg"],
        expected_found=True,
        expected_answer_contains="7.61",
    )
    assert not answer_correct(
        found=True,
        answer="something else",
        citations=[],
        expected_found=False,
        expected_answer_contains=None,
    )
    assert citation_faithfulness(["receipt:1"], ["receipt:1", "receipt:2"])
    assert not citation_faithfulness(["receipt:9"], ["receipt:1"])


def test_keyword_ablation_finds_labelled_merchants(tmp_path: Path):
    db_path = tmp_path / "retrieval.db"
    repo = SqliteReceiptRepository(db_path)
    repo.init_db()
    embeddings = HashEmbeddingProvider()
    count = index_receipt_corpus(repo, embeddings)
    assert count >= 50
    retrieval = RetrievalService(SqliteRetrievalRepository(db_path), embeddings)
    questions = [q for q in load_jsonl(QUESTIONS) if q["relevant_source_ids"]]
    rows = []
    for question in questions:
        hits = retrieval.search(
            question["question"],
            strategy="keyword",
            limit=5,
            kind="receipt",
        )
        rows.append(
            {
                "relevant_source_ids": question["relevant_source_ids"],
                "retrieved_ids": [hit["source_id"] for hit in hits],
            }
        )
    summary = summarize_retrieval(rows, k=5)
    assert summary["recall_at_k"] >= 0.9
    assert summary["mrr"] >= 0.9


def test_scripted_answers_are_faithful(tmp_path: Path):
    db_path = tmp_path / "retrieval.db"
    repo = SqliteReceiptRepository(db_path)
    repo.init_db()
    embeddings = HashEmbeddingProvider()
    index_receipt_corpus(repo, embeddings)
    retrieval = RetrievalService(SqliteRetrievalRepository(db_path), embeddings)
    hits = retrieval.search("Taco Bell", strategy="keyword", limit=5, kind="receipt")
    retrieved_ids = [hit["source_id"] for hit in hits]
    assert "receipt:1002-receipt.jpg" in retrieved_ids
    llm_json = (
        '{"answer": "Taco Bell total was 7.61.", '
        '"citations": ["receipt:1002-receipt.jpg"], "found": true}'
    )
    service = AnsweringService(FakeProvider(llm_json), retrieval)
    result = service.answer("What was the total at Taco Bell?", strategy="keyword")
    assert citation_faithfulness(result.citations, retrieved_ids)
    assert answer_correct(
        found=result.found,
        answer=result.answer,
        citations=result.citations,
        expected_found=True,
        expected_answer_contains="7.61",
    )
    summary = summarize_answers(
        [
            {
                "correct": True,
                "faithful": citation_faithfulness(result.citations, retrieved_ids),
            }
        ]
    )
    assert summary["correct_rate"] == 1.0
    assert summary["faithful_rate"] == 1.0
