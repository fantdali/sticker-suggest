"""Tests for the sticker-suggest pipeline."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from src.config import Config
from src.preprocessing.deduplicator import deduplicate
from src.preprocessing.image_processor import collect_image_paths, load_image


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_images(tmp_dir):
    """Create a handful of visually distinct test images."""
    colors = ["red", "blue", "green", "yellow", "purple"]
    paths = []
    for i, color in enumerate(colors):
        img = Image.new("RGB", (100, 100), color)
        draw = ImageDraw.Draw(img)
        # Draw distinct patterns so phash doesn't collapse them
        draw.rectangle([i * 15, i * 15, i * 15 + 40, i * 15 + 40], fill="white")
        draw.ellipse([50 - i * 10, 50 - i * 10, 80 + i * 5, 80 + i * 5], fill="black")
        p = tmp_dir / f"test_{i}.png"
        img.save(p)
        paths.append(p)
    return paths


@pytest.fixture
def sample_config(tmp_dir):
    return Config(
        data_dir=tmp_dir / "data",
        sticker_dir=tmp_dir,
        embeddings_dir=tmp_dir / "data" / "embeddings",
        clip_device="cpu",
        ocr_gpu=False,
        batch_size=4,
    )


# ── Unit tests (fast, no model loading) ──


class TestImageProcessor:
    def test_load_png(self, sample_images):
        img = load_image(sample_images[0])
        assert img is not None
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_load_gif(self, tmp_dir):
        img = Image.new("RGB", (50, 50), "red")
        gif_path = tmp_dir / "anim.gif"
        img.save(gif_path, format="GIF")
        loaded = load_image(gif_path)
        assert loaded is not None
        assert loaded.mode == "RGB"

    def test_load_rgba(self, tmp_dir):
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        p = tmp_dir / "alpha.png"
        img.save(p)
        loaded = load_image(p)
        assert loaded is not None
        assert loaded.mode == "RGB"

    def test_load_nonexistent(self, tmp_dir):
        assert load_image(tmp_dir / "nope.png") is None

    def test_collect_paths(self, tmp_dir, sample_images):
        # Also add a non-image file that should be ignored
        (tmp_dir / "readme.txt").write_text("not an image")
        paths = collect_image_paths(tmp_dir)
        assert len(paths) == 5
        assert all(p.suffix == ".png" for p in paths)


class TestDeduplicator:
    def test_no_duplicates(self, sample_images, sample_config):
        unique = deduplicate(sample_images, sample_config)
        assert len(unique) == 5

    def test_removes_exact_duplicates(self, tmp_dir, sample_config):
        img = Image.new("RGB", (100, 100), "red")
        paths = []
        for i in range(3):
            p = tmp_dir / f"dup_{i}.png"
            img.save(p)
            paths.append(p)
        unique = deduplicate(paths, sample_config)
        assert len(unique) == 1

    def test_empty_list(self, sample_config):
        assert deduplicate([], sample_config) == []


# ── Integration tests (loads CLIP + OCR models, slower) ──


@pytest.mark.slow
class TestEndToEnd:
    """Full pipeline: preprocess → build embeddings → search."""

    def test_full_pipeline(self, sample_images, sample_config):
        from src.features.feature_store import build_feature_store
        from src.preprocessing.pipeline import run_preprocessing
        from src.retrieval.searcher import StickerSearcher

        cfg = sample_config

        # Step 1: preprocessing
        result = run_preprocessing(cfg)
        paths = result["processed_paths"]
        ocr_meta = result["ocr_metadata"]
        assert len(paths) > 0

        # Step 2: build embeddings
        store = build_feature_store(paths, ocr_meta, cfg)
        assert store["image_embeddings"].shape == (len(paths), 768)
        assert store["ocr_embeddings"].shape == (len(paths), 768)
        assert store["ocr_confidences"].shape == (len(paths),)

        # Verify files on disk
        assert (cfg.embeddings_dir / "image_embeddings.pt").exists()
        assert (cfg.embeddings_dir / "ocr_embeddings.pt").exists()
        assert (cfg.embeddings_dir / "filenames.json").exists()

        # Step 3: search
        searcher = StickerSearcher(cfg)
        results = searcher.search("hello", top_k=3)
        assert len(results) == 3
        assert all(r.score != 0.0 for r in results)
        assert all(r.filename.endswith(".png") for r in results)
