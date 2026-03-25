"""Image loading utilities."""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def load_image(path: Path) -> Image.Image | None:
    """Load an image, take first frame if GIF, return as RGB."""
    try:
        img = Image.open(path)

        # GIF: grab first frame
        if getattr(img, "is_animated", False) or path.suffix.lower() == ".gif":
            img.seek(0)

        return img.convert("RGB")

    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def collect_image_paths(sticker_dir: Path) -> list[Path]:
    """Collect all .png and .gif image paths from a directory."""
    paths = sorted(
        p
        for p in sticker_dir.iterdir()
        if p.suffix.lower() in (".png", ".gif") and p.is_file()
    )
    logger.info("Found %d images in %s", len(paths), sticker_dir)
    return paths
