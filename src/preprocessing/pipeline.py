"""Full preprocessing pipeline: collect images → deduplicate → OCR."""

import json
import logging
from pathlib import Path

from src.config import Config
from src.preprocessing.deduplicator import deduplicate
from src.preprocessing.image_processor import collect_image_paths
from src.preprocessing.ocr import OCRExtractor

logger = logging.getLogger(__name__)


def run_preprocessing(cfg: Config) -> dict:
    """
    Run the full preprocessing pipeline.

    Returns:
        processed_paths: list of unique image paths
        ocr_metadata: dict of filename -> {text, confidence}
        manifest_path: path to saved manifest
    """
    cfg.ensure_dirs()
    manifest_path = cfg.data_dir / "manifest.json"
    ocr_meta_path = cfg.embeddings_dir / "ocr_metadata.json"

    # Check if preprocessing was already completed
    if manifest_path.exists() and ocr_meta_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(ocr_meta_path) as f:
            ocr_metadata = json.load(f)
        if len(manifest["image_files"]) == len(ocr_metadata):
            logger.info(
                "Preprocessing already complete: %d images, skipping",
                len(manifest["image_files"]),
            )
            unique_paths = [
                cfg.sticker_dir / fname for fname in manifest["image_files"]
            ]
            return {
                "processed_paths": unique_paths,
                "ocr_metadata": ocr_metadata,
                "manifest_path": manifest_path,
            }

    logger.info("=== Step 1: Collect images from %s ===", cfg.sticker_dir)
    all_paths = collect_image_paths(cfg.sticker_dir)

    logger.info("=== Step 2: Deduplicate ===")
    unique_paths = deduplicate(all_paths, cfg)

    logger.info("=== Step 3: OCR ===")
    ocr = OCRExtractor(cfg)
    ocr_metadata = ocr.extract_batch(unique_paths)

    # Save manifest
    manifest = {
        "image_files": [p.name for p in unique_paths],
        "total_raw": len(all_paths),
        "total_unique": len(unique_paths),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Pipeline done: %d raw → %d unique",
        len(all_paths),
        len(unique_paths),
    )

    return {
        "processed_paths": unique_paths,
        "ocr_metadata": ocr_metadata,
        "manifest_path": manifest_path,
    }
