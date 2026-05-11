"""CLI: Generate a new sticker from a text query."""

import argparse
import logging
from pathlib import Path

from src.config import Config
from src.generation import StickerGenerator, resolve_generation_device


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate a sticker by text query")
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--generation-device",
        type=str,
        default="auto",
        help="Generation device: auto, mps, cuda, or cpu",
    )
    parser.add_argument("--sam-device", type=str, default="cpu")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of using only the local cache",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip SAM segmentation and save the raw generated image as RGBA",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated sticker on a checkerboard background",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generation_device = resolve_generation_device(args.generation_device)
    cfg = Config(
        generation_device=generation_device,
        generation_local_files_only=not args.allow_model_download,
        sam_device=args.sam_device,
    )
    generator = StickerGenerator(cfg, device=generation_device)
    result = generator.generate(
        args.query,
        output_path=output_path,
        seed=args.seed,
        postprocess=not args.no_postprocess,
    )

    print(f"Generated: {result.output_path}")
    print(f"Prompt: {result.prompt}")
    print(f"Alpha coverage: {result.mask_coverage:.3f}")
    if args.show:
        _show_generated(result.output_path, result.mask_coverage)


def _show_generated(path: Path, mask_coverage: float):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    w, h = img.size
    y, x = np.indices((h, w))
    pattern = ((x // 24 + y // 24) % 2).astype(np.uint8)
    bg = np.where(pattern[..., None] == 0, 235, 255).astype(np.uint8)
    bg = np.repeat(bg, 3, axis=2)

    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(bg)
    ax.imshow(img)
    ax.set_title(f"generated sticker alpha={mask_coverage:.2f}", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
