"""OCR extraction using EasyOCR."""

import json
import logging
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import Config

logger = logging.getLogger(__name__)


class OCRExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reader = easyocr.Reader(cfg.ocr_languages, gpu=cfg.ocr_gpu)

    def extract_single(self, image_path: Path) -> dict:
        """Extract text from a single image. Returns {text, confidence}."""
        try:
            img = Image.open(image_path)
            if image_path.suffix.lower() == ".gif":
                img.seek(0)
            img_array = np.array(img.convert("RGB"))

            result = self.reader.readtext(img_array)

            if not result:
                return {"text": "", "confidence": 0.0}

            texts = [r[1] for r in result]
            confs = [r[2] for r in result]

            full_text = " ".join(texts).strip()[: self.cfg.ocr_max_text_len]
            avg_conf = sum(confs) / len(confs)

            return {"text": full_text, "confidence": avg_conf}

        except Exception as e:
            logger.warning("OCR failed for %s: %s", image_path, e)
            return {"text": "", "confidence": 0.0}

    def extract_batch(self, image_paths: list[Path]) -> dict[str, dict]:
        """
        Run OCR on all images. Returns dict: filename -> {text, confidence}.
        Saves periodically and supports resuming.
        """
        output_path = self.cfg.embeddings_dir / "ocr_metadata.json"
        self.cfg.ensure_dirs()

        # Load existing results for resume support
        existing: dict[str, dict] = {}
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)

        results = dict(existing)
        to_process = [p for p in image_paths if p.name not in results]
        logger.info("OCR: %d cached, %d to process", len(existing), len(to_process))

        for i, p in enumerate(tqdm(to_process, desc="Running OCR")):
            results[p.name] = self.extract_single(p)

            # Periodic save every 500 images
            if (i + 1) % 500 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False)

        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("OCR done: %d images", len(results))
        return results
