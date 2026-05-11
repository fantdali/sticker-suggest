"""
Migrate existing numpy embeddings (.npy) to the new torch (.pt) format.
Run once after setting up the new codebase.

Usage: python -m cli.migrate [--device mps]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import Config

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Migrate numpy embeddings → torch + FAISS"
    )
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    cfg = Config(clip_device=args.device)
    cfg.ensure_dirs()

    data = cfg.data_dir

    # --- Load existing numpy files ---
    logger.info("Loading existing numpy embeddings...")
    image_embeddings_np = np.load(data / "image_embeddings.npy")
    ocr_embeddings_np = np.load(data / "ocr_embeddings.npy")
    ocr_confidences_np = np.load(data / "ocr_confidences.npy")
    image_paths_np = np.load(data / "image_paths.npy", allow_pickle=True)

    logger.info(
        "Loaded: images=%s, ocr=%s, confidences=%s, paths=%d",
        image_embeddings_np.shape,
        ocr_embeddings_np.shape,
        ocr_confidences_np.shape,
        len(image_paths_np),
    )

    # --- Fix paths: convert "../data/sticker_suggest/xxx.png" → just "xxx.png" ---
    filenames = [Path(p).name for p in image_paths_np]

    # --- Convert to torch and save ---
    image_embeddings = torch.from_numpy(image_embeddings_np)
    ocr_embeddings = torch.from_numpy(ocr_embeddings_np)
    ocr_confidences = torch.from_numpy(ocr_confidences_np.astype(np.float32))

    emb_dir = cfg.embeddings_dir

    torch.save(image_embeddings, emb_dir / "image_embeddings.pt")
    torch.save(ocr_embeddings, emb_dir / "ocr_embeddings.pt")
    torch.save(ocr_confidences, emb_dir / "ocr_confidences.pt")

    with open(emb_dir / "filenames.json", "w") as f:
        json.dump(filenames, f)

    logger.info("Saved torch embeddings to %s", emb_dir)

    # --- Save manifest ---
    manifest = {
        "image_files": filenames,
        "total_raw": len(filenames),
        "total_unique": len(filenames),
    }
    with open(data / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest.json")
    logger.info("Migration complete! You can now run: python -m cli.search")


if __name__ == "__main__":
    main()
