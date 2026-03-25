"""Perceptual-hash based deduplication of images."""

import logging
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from src.config import Config

logger = logging.getLogger(__name__)

HASH_FUNCS = {
    "phash": imagehash.phash,
    "ahash": imagehash.average_hash,
    "dhash": imagehash.dhash,
    "whash": imagehash.whash,
}


def deduplicate(image_paths: list[Path], cfg: Config) -> list[Path]:
    """Remove duplicate images using perceptual hashing. Returns unique paths."""
    hash_fn = HASH_FUNCS[cfg.hash_func]

    seen: set[str] = set()
    unique: list[Path] = []
    dup_count = 0

    for p in tqdm(image_paths, desc="Deduplicating"):
        try:
            with Image.open(p) as img:
                h = str(hash_fn(img.convert("RGB"), hash_size=cfg.hash_size))
        except Exception as e:
            logger.warning("Cannot hash %s: %s", p, e)
            continue

        if h not in seen:
            seen.add(h)
            unique.append(p)
        else:
            dup_count += 1

    logger.info("Dedup: %d unique, %d duplicates removed", len(unique), dup_count)
    return unique
