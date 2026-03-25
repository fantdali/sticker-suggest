"""CLI: Run the full preprocessing pipeline (dedup + OCR)."""

import argparse
import logging

from src.config import Config
from src.preprocessing.pipeline import run_preprocessing


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Preprocess sticker dataset")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Override data directory"
    )
    parser.add_argument("--ocr-gpu", action="store_true", help="Use GPU for EasyOCR")
    args = parser.parse_args()

    kwargs = {}
    if args.data_dir:
        from pathlib import Path

        kwargs["data_dir"] = Path(args.data_dir)
    kwargs["ocr_gpu"] = args.ocr_gpu

    cfg = Config(**kwargs)
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
