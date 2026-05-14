"""Tests for retrieval metrics."""

import pytest

from src.evaluation import evaluate_rankings


def test_evaluate_rankings_basic_metrics():
    rankings = {
        "q1": ["bad.png", "good.png", "ok.png"],
        "q2": ["miss.png", "other.png"],
    }
    qrels = {
        "q1": {"good.png": 2, "ok.png": 1},
        "q2": {"target.png": 1},
    }

    metrics = evaluate_rankings(rankings, qrels, k=3)

    assert metrics["queries"] == 2
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.25)
    assert metrics["recall"] == pytest.approx(0.5)
    assert 0.0 < metrics["ndcg"] < 1.0
