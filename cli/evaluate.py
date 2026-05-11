"""CLI: Evaluate retrieval quality on a small JSONL relevance set."""

import argparse
import json
import logging
import statistics
from pathlib import Path

from src.config import Config
from src.evaluation import evaluate_rankings
from src.retrieval.searcher import StickerSearcher


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate sticker retrieval")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries", type=str, default="eval/queries.jsonl")
    parser.add_argument("--qrels", type=str, default="eval/qrels.jsonl")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/metrics.json",
        help="Where to save the metrics JSON",
    )
    args = parser.parse_args()

    cfg = Config(clip_device=args.device, top_k=args.top_k)
    cfg.ensure_dirs()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(Path(args.queries))
    qrels = _load_qrels(Path(args.qrels))
    searcher = StickerSearcher(cfg)

    rankings = {}
    timings = []
    result_rows = []
    for row in queries:
        results, timing = searcher.search_with_timing(row["query"], top_k=args.top_k)
        timings.append(timing)
        filenames = [r.filename for r in results]
        rankings[row["query_id"]] = filenames
        result_rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "category": row.get("category", ""),
                "top_results": filenames,
                "total_ms": timing.total_ms,
                "scoring_ms": timing.scoring_ms,
            }
        )

    metrics = evaluate_rankings(rankings, qrels, args.top_k)
    metrics["top_k"] = args.top_k
    metrics["latency"] = _latency_summary(timings)
    metrics["results"] = result_rows

    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _print_summary(metrics, output_path)


def _load_queries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qrels.setdefault(row["query_id"], {})[row["filename"]] = int(row["relevance"])
    return qrels


def _latency_summary(timings) -> dict[str, float]:
    totals = sorted(t.total_ms for t in timings)
    scoring = sorted(t.scoring_ms for t in timings)
    return {
        "total_p50_ms": statistics.median(totals) if totals else 0.0,
        "total_p95_ms": _percentile(totals, 0.95),
        "scoring_p50_ms": statistics.median(scoring) if scoring else 0.0,
        "scoring_p95_ms": _percentile(scoring, 0.95),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def _print_summary(metrics: dict, output_path: Path):
    print("\n=== Evaluation ===")
    print(f"queries={metrics['queries']} top_k={metrics['top_k']}")
    print(f"HitRate@K={metrics['hit_rate']:.3f}")
    print(f"MRR@K={metrics['mrr']:.3f}")
    print(f"Recall@K={metrics['recall']:.3f}")
    print(f"nDCG@K={metrics['ndcg']:.3f}")
    print(
        "Latency: "
        f"total p50={metrics['latency']['total_p50_ms']:.1f} ms, "
        f"p95={metrics['latency']['total_p95_ms']:.1f} ms; "
        f"scoring p50={metrics['latency']['scoring_p50_ms']:.1f} ms, "
        f"p95={metrics['latency']['scoring_p95_ms']:.1f} ms"
    )
    print(f"Saved: {output_path}\n")


if __name__ == "__main__":
    main()
