"""Retrieval metric implementations used by the evaluation CLI."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QueryMetrics:
    query_id: str
    hit: float
    reciprocal_rank: float
    recall: float
    ndcg: float


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> dict:
    """Compute macro retrieval metrics for ranked filenames."""
    per_query = []
    for query_id, relevant in qrels.items():
        ranked = rankings.get(query_id, [])[:k]
        per_query.append(_evaluate_one(query_id, ranked, relevant, k))

    if not per_query:
        return {
            "queries": 0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "per_query": [],
        }

    n = len(per_query)
    return {
        "queries": n,
        "hit_rate": sum(m.hit for m in per_query) / n,
        "mrr": sum(m.reciprocal_rank for m in per_query) / n,
        "recall": sum(m.recall for m in per_query) / n,
        "ndcg": sum(m.ndcg for m in per_query) / n,
        "per_query": [m.__dict__ for m in per_query],
    }


def _evaluate_one(
    query_id: str,
    ranked: list[str],
    relevant: dict[str, int],
    k: int,
) -> QueryMetrics:
    positive = {fname for fname, rel in relevant.items() if rel > 0}
    hit = 0.0
    reciprocal_rank = 0.0
    retrieved_positive = 0
    dcg = 0.0

    for rank, fname in enumerate(ranked, 1):
        rel = relevant.get(fname, 0)
        if rel > 0:
            retrieved_positive += 1
            if not hit:
                hit = 1.0
                reciprocal_rank = 1.0 / rank
        dcg += _dcg_gain(rel, rank)

    recall = retrieved_positive / len(positive) if positive else 0.0
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(_dcg_gain(rel, rank) for rank, rel in enumerate(ideal_rels, 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return QueryMetrics(
        query_id=query_id,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        recall=recall,
        ndcg=ndcg,
    )


def _dcg_gain(relevance: int, rank: int) -> float:
    return (2**relevance - 1) / math.log2(rank + 1)
