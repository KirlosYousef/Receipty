from __future__ import annotations

from typing import Any


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("recall_at_k requires a non-empty relevant set")
    hit = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hit) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        raise ValueError("MRR requires a non-empty relevant set")
    relevant = set(relevant_ids)
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in relevant:
            return 1.0 / rank
    return 0.0


def citation_faithfulness(citations: list[str], retrieved_ids: list[str]) -> bool:
    retrieved = set(retrieved_ids)
    return all(citation in retrieved for citation in citations)


def answer_correct(
    *,
    found: bool,
    answer: str,
    citations: list[str],
    expected_found: bool,
    expected_answer_contains: str | None,
) -> bool:
    if found != expected_found:
        return False
    if not expected_found:
        return citations == []
    if expected_answer_contains is None:
        return True
    return expected_answer_contains.lower() in answer.lower()


def summarize_retrieval(rows: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    ranked = [row for row in rows if row["relevant_source_ids"]]
    recall_values = [
        recall_at_k(row["retrieved_ids"], row["relevant_source_ids"], k)
        for row in ranked
    ]
    mrr_values = [
        mean_reciprocal_rank(row["retrieved_ids"], row["relevant_source_ids"])
        for row in ranked
    ]
    return {
        "k": k,
        "n": len(ranked),
        "recall_at_k": _mean(recall_values),
        "mrr": _mean(mrr_values),
    }


def summarize_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "faithful": 0,
            "correct_rate": 0.0,
            "faithful_rate": 0.0,
        }
    correct = sum(1 for row in rows if row["correct"])
    faithful = sum(1 for row in rows if row["faithful"])
    n = len(rows)
    return {
        "n": n,
        "correct": correct,
        "faithful": faithful,
        "correct_rate": correct / n,
        "faithful_rate": faithful / n,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
