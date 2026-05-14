"""Tests for exact fused search without loading CLIP."""

import torch

from src.config import Config
from src.retrieval.searcher import StickerSearcher


def test_search_by_embedding_uses_fused_matrix_scores(tmp_path):
    searcher = StickerSearcher.__new__(StickerSearcher)
    searcher.cfg = Config(
        data_dir=tmp_path / "data",
        sticker_dir=tmp_path,
        embeddings_dir=tmp_path / "data" / "embeddings",
        clip_device="cpu",
        ocr_alpha=2.0,
    )
    searcher.device = torch.device("cpu")
    searcher.filenames = ["image_hit.png", "ocr_hit.png", "bad.png"]
    searcher.ocr_metadata = {"ocr_hit.png": {"text": "hello"}}
    searcher.image_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    searcher.ocr_embeddings = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    )
    searcher.ocr_confidences = torch.tensor([0.0, 1.0, 0.0])

    results = searcher.search_by_embedding(torch.tensor([[1.0, 0.0]]), top_k=2)

    assert [r.filename for r in results] == ["ocr_hit.png", "image_hit.png"]
    assert results[0].rank == 1
    assert results[0].ocr_text == "hello"
