"""CLI: Build a reproducible OCR-focused retrieval benchmark from local metadata."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from src.config import Config


TOKEN_RE = re.compile(r"[a-z0-9]+")
BLOCKLIST = {
    "ass",
    "bashqueer",
    "bitch",
    "damn",
    "fuck",
    "gay",
    "hell",
    "queer",
    "retarded",
    "shit",
    "stupid",
}


def main():
    parser = argparse.ArgumentParser(description="Build an OCR-based eval set")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-relevant", type=int, default=12)
    parser.add_argument("--queries-output", type=str, default="eval/queries.jsonl")
    parser.add_argument("--qrels-output", type=str, default="eval/qrels.jsonl")
    args = parser.parse_args()

    cfg = Config()
    meta_path = cfg.embeddings_dir / "ocr_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"OCR metadata not found at {meta_path}. Run preprocessing first."
        )

    rows = _build_rows(
        meta_path=meta_path,
        size=args.size,
        seed=args.seed,
        min_confidence=args.min_confidence,
        max_relevant=args.max_relevant,
    )
    queries, qrels = rows

    queries_path = Path(args.queries_output)
    qrels_path = Path(args.qrels_output)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    _write_jsonl(queries_path, queries)
    _write_jsonl(qrels_path, qrels)

    print(
        f"Built eval set: {len(queries)} queries, {len(qrels)} qrels; "
        f"saved to {queries_path} and {qrels_path}"
    )


def _build_rows(
    meta_path: Path,
    size: int,
    seed: int,
    min_confidence: float,
    max_relevant: int,
) -> tuple[list[dict], list[dict]]:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    by_query: dict[str, list[tuple[str, float, str]]] = defaultdict(list)

    for filename, item in metadata.items():
        text = str(item.get("text", "")).strip()
        confidence = float(item.get("confidence", 0.0) or 0.0)
        query = normalize_text(text)
        tokens = query.split()
        token_count = len(tokens)
        if (
            confidence >= min_confidence
            and 2 <= token_count <= 5
            and 5 <= len(query) <= 40
            and _is_clean_query(query, tokens)
        ):
            by_query[query].append((filename, confidence, text))

    candidates = [
        (query, items)
        for query, items in by_query.items()
        if 1 <= len(items) <= max_relevant
    ]
    candidates.sort(key=lambda row: (-len(row[1]), row[0]))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = sorted(candidates[:size], key=lambda row: row[0])

    queries = []
    qrels = []
    for idx, (query, items) in enumerate(selected, 1):
        query_id = f"ocr_{idx:03d}"
        best_confidence = max(conf for _, conf, _ in items)
        queries.append(
            {
                "query_id": query_id,
                "query": query,
                "category": "ocr_auto",
                "source": "ocr_metadata",
                "max_ocr_confidence": round(best_confidence, 4),
            }
        )
        for filename, confidence, raw_text in sorted(items):
            relevance = 2 if normalize_text(raw_text) == query else 1
            qrels.append(
                {
                    "query_id": query_id,
                    "filename": filename,
                    "relevance": relevance,
                    "ocr_confidence": round(confidence, 4),
                }
            )

    return queries, qrels


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _is_clean_query(query: str, tokens: list[str]) -> bool:
    if any(token in BLOCKLIST for token in tokens):
        return False

    letters = sum(ch.isalpha() for ch in query)
    digits = sum(ch.isdigit() for ch in query)
    if letters == 0 or letters < digits:
        return False

    return any(len(token) >= 3 and any(ch in "aeiouy" for ch in token) for token in tokens)


def _write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
