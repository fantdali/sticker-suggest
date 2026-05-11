"""Compute and persist CLIP image + OCR text embeddings."""

import json
import logging
from pathlib import Path

import torch

from src.config import Config
from src.features.clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


def build_feature_store(
    image_paths: list[Path],
    ocr_metadata: dict[str, dict],
    cfg: Config,
) -> dict:
    """
    Compute and cache:
        - image_embeddings.pt  (N, 768)
        - ocr_embeddings.pt    (N, 768)
        - ocr_confidences.pt   (N,)
        - filenames.json

    Returns dict with all tensors + filenames.
    """
    cfg.ensure_dirs()
    filenames = [p.name for p in image_paths]
    n = len(filenames)

    img_emb_path = cfg.embeddings_dir / "image_embeddings.pt"
    ocr_emb_path = cfg.embeddings_dir / "ocr_embeddings.pt"
    ocr_conf_path = cfg.embeddings_dir / "ocr_confidences.pt"
    filenames_path = cfg.embeddings_dir / "filenames.json"

    # Check if everything is already cached with correct size
    if all(
        p.exists() for p in [img_emb_path, ocr_emb_path, ocr_conf_path, filenames_path]
    ):
        image_embeddings = torch.load(img_emb_path, weights_only=True)
        if image_embeddings.shape[0] == n:
            ocr_embeddings = torch.load(ocr_emb_path, weights_only=True)
            ocr_confidences = torch.load(ocr_conf_path, weights_only=True)
            logger.info("Feature store already complete: %d images, skipping", n)
            return {
                "image_embeddings": image_embeddings,
                "ocr_embeddings": ocr_embeddings,
                "ocr_confidences": ocr_confidences,
                "filenames": filenames,
            }

    embedder = CLIPEmbedder(cfg)

    # --- Image embeddings ---
    if img_emb_path.exists():
        logger.info("Loading cached image embeddings from %s", img_emb_path)
        image_embeddings = torch.load(img_emb_path, weights_only=True)
    else:
        logger.info("Computing image embeddings for %d images...", len(image_paths))
        image_embeddings = embedder.embed_images(image_paths)
        torch.save(image_embeddings, img_emb_path)

    # --- OCR text embeddings ---
    if ocr_emb_path.exists() and ocr_conf_path.exists():
        logger.info("Loading cached OCR embeddings")
        ocr_embeddings = torch.load(ocr_emb_path, weights_only=True)
        ocr_confidences = torch.load(ocr_conf_path, weights_only=True)
    else:
        logger.info("Computing OCR text embeddings...")
        ocr_texts = []
        confidences = []
        for fname in filenames:
            meta = ocr_metadata.get(fname, {"text": "", "confidence": 0.0})
            text = meta.get("text", "").strip()
            conf = meta.get("confidence", 0.0)

            if len(text) > 1 and conf > cfg.ocr_confidence_threshold:
                ocr_texts.append(text)
                confidences.append(conf)
            else:
                ocr_texts.append("")
                confidences.append(0.0)

        ocr_embeddings = torch.zeros(len(filenames), cfg.embedding_dim)
        ocr_confidences = torch.tensor(confidences, dtype=torch.float32)

        # Only embed non-empty texts
        non_empty = [(i, t) for i, t in enumerate(ocr_texts) if t]
        if non_empty:
            indices, texts = zip(*non_empty)
            text_embs = embedder.embed_texts(list(texts))
            for j, idx in enumerate(indices):
                ocr_embeddings[idx] = text_embs[j]

        torch.save(ocr_embeddings, ocr_emb_path)
        torch.save(ocr_confidences, ocr_conf_path)

    # Save filenames
    with open(filenames_path, "w") as f:
        json.dump(filenames, f)

    logger.info(
        "Feature store: %d images, %d OCR texts",
        image_embeddings.shape[0],
        ocr_embeddings.shape[0],
    )

    return {
        "image_embeddings": image_embeddings,
        "ocr_embeddings": ocr_embeddings,
        "ocr_confidences": ocr_confidences,
        "filenames": filenames,
    }
