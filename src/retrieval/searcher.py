"""
Sticker search engine with late fusion of visual + OCR scores.

Score fusion (from notebook 4):
    image_scores = z_norm(query @ image_embeddings)
    ocr_scores   = z_norm(query @ ocr_embeddings)
    final = image_scores + alpha * (confidence²) * ocr_scores
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import Config
from src.features.clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    rank: int
    filename: str
    score: float
    visual_score: float
    ocr_score: float
    ocr_text: str
    image_path: Path


@dataclass
class SearchTiming:
    query_ms: float
    scoring_ms: float
    total_ms: float


class StickerSearcher:
    """Loads precomputed embeddings and performs fused search."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.clip_device)

        # Load filenames
        with open(cfg.embeddings_dir / "filenames.json") as f:
            self.filenames: list[str] = json.load(f)

        # Load all embeddings as tensors for z-score fusion
        self.image_embeddings = torch.load(
            cfg.embeddings_dir / "image_embeddings.pt", weights_only=True
        ).to(self.device)
        self.ocr_embeddings = torch.load(
            cfg.embeddings_dir / "ocr_embeddings.pt", weights_only=True
        ).to(self.device)
        self.ocr_confidences = torch.load(
            cfg.embeddings_dir / "ocr_confidences.pt", weights_only=True
        ).to(self.device)

        # Keep exact search, but make scoring a single matrix-vector operation.
        # OCR embeddings can contain zero rows, and F.normalize preserves them.
        self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
        self.ocr_embeddings = F.normalize(self.ocr_embeddings, dim=-1)

        # Load OCR metadata for display
        ocr_meta_path = cfg.embeddings_dir / "ocr_metadata.json"
        self.ocr_metadata = {}
        if ocr_meta_path.exists():
            with open(ocr_meta_path) as f:
                self.ocr_metadata = json.load(f)

        # CLIP embedder for queries
        self.embedder = CLIPEmbedder(cfg)

        logger.info("Searcher ready: %d stickers", len(self.filenames))

    @torch.no_grad()
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Search using z-score normalized fusion (matching notebook logic).

        final_score = z(image_score) + alpha * confidence² * z(ocr_score)
        """
        q_emb = self.embedder.embed_query(query).to(self.device)
        return self.search_by_embedding(q_emb, top_k=top_k)

    @torch.no_grad()
    def search_with_timing(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[SearchResult], SearchTiming]:
        """Search and return coarse latency measurements for reporting/demo."""
        start = time.perf_counter()
        q_start = time.perf_counter()
        q_emb = self.embedder.embed_query(query).to(self.device)
        q_end = time.perf_counter()
        results = self.search_by_embedding(q_emb, top_k=top_k)
        end = time.perf_counter()
        return results, SearchTiming(
            query_ms=(q_end - q_start) * 1000,
            scoring_ms=(end - q_end) * 1000,
            total_ms=(end - start) * 1000,
        )

    @torch.no_grad()
    def search_by_embedding(
        self, q_emb: torch.Tensor, top_k: int | None = None
    ) -> list[SearchResult]:
        """Search using an already computed query embedding."""
        top_k = min(top_k or self.cfg.top_k, len(self.filenames))
        q = F.normalize(q_emb.to(self.device), dim=-1).squeeze(0)

        image_scores = self.image_embeddings @ q
        ocr_scores = self.ocr_embeddings @ q

        # Z-score normalization
        image_scores = (image_scores - image_scores.mean()) / (
            image_scores.std() + 1e-8
        )
        ocr_scores = (ocr_scores - ocr_scores.mean()) / (ocr_scores.std() + 1e-8)

        # Fused score: image + alpha * conf² * ocr
        scores = (
            image_scores + self.cfg.ocr_alpha * (self.ocr_confidences**2) * ocr_scores
        )

        top_indices = scores.topk(top_k).indices

        results = []
        for rank, idx in enumerate(top_indices, 1):
            i = idx.item()
            fname = self.filenames[i]
            ocr_text = self.ocr_metadata.get(fname, {}).get("text", "")

            results.append(
                SearchResult(
                    rank=rank,
                    filename=fname,
                    score=scores[i].item(),
                    visual_score=image_scores[i].item(),
                    ocr_score=ocr_scores[i].item(),
                    ocr_text=ocr_text,
                    image_path=self.cfg.sticker_dir / fname,
                )
            )

        return results
