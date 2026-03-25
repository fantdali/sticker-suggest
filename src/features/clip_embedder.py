"""CLIP embedding extraction using open_clip."""

import logging
from pathlib import Path

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from src.config import Config
from src.preprocessing.image_processor import load_image

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.clip_device)

        logger.info(
            "Loading CLIP %s (%s) on %s",
            cfg.clip_model,
            cfg.clip_pretrained,
            self.device,
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model,
            pretrained=cfg.clip_pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_images(self, image_paths: list[Path]) -> torch.Tensor:
        """
        Compute L2-normalized image embeddings. Shape: (N, 768).
        Skips unreadable images by substituting a blank white image.
        """
        all_embs = []

        for i in tqdm(
            range(0, len(image_paths), self.cfg.batch_size),
            desc="CLIP image embeddings",
        ):
            batch_paths = image_paths[i : i + self.cfg.batch_size]
            tensors = []

            for p in batch_paths:
                img = load_image(p)
                if img is None:
                    img = Image.new("RGB", (224, 224), (255, 255, 255))
                tensors.append(self.preprocess(img))

            batch = torch.stack(tensors).to(self.device)
            emb = self.model.encode_image(batch)
            emb /= emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Compute L2-normalized text embeddings. Shape: (N, 768)."""
        all_embs = []

        for i in range(0, len(texts), self.cfg.batch_size):
            batch_texts = texts[i : i + self.cfg.batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            emb = self.model.encode_text(tokens)
            emb /= emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single query. Returns (1, 768) tensor."""
        return self.embed_texts([query])
