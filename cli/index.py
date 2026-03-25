"""CLI: Build CLIP embeddings from preprocessed data."""

import argparse
import json
import logging
from pathlib import Path

from src.config import Config
from src.features.feature_store import build_feature_store


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build CLIP + OCR embeddings")
    parser.add_argument(
        "--device", type=str, default="mps", help="Torch device (mps/cuda/cpu)"
    )
    args = parser.parse_args()

    cfg = Config(clip_device=args.device)

    # Load manifest
    manifest_path = cfg.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run preprocessing first: python -m cli.preprocess"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    image_paths = [cfg.sticker_dir / fname for fname in manifest["image_files"]]

    # Load OCR metadata
    ocr_meta_path = cfg.embeddings_dir / "ocr_metadata.json"
    ocr_metadata = {}
    if ocr_meta_path.exists():
        with open(ocr_meta_path) as f:
            ocr_metadata = json.load(f)

    build_feature_store(image_paths, ocr_metadata, cfg)

    print(f"Done! Embeddings saved to {cfg.embeddings_dir}")


if __name__ == "__main__":
    main()
