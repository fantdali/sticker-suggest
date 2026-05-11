"""Text-to-sticker generation using SDXL Turbo and SAM post-processing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.config import Config
from src.generation.postprocess import (
    add_white_outline,
    alpha_mask_coverage,
    apply_alpha_mask,
    segment_main_object_with_sam,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    query: str
    prompt: str
    output_path: Path
    mask_coverage: float


def build_sticker_prompt(query: str) -> str:
    """Build the prompt that worked well in the generation notebook."""
    return (
        f"A single isolated sticker of {query}. "
        "Exactly one main subject, centered in the image, full body visible. "
        "No other characters, no extra faces, no icons, no repeated objects, "
        "no sticker sheet, no collage, no background elements. "
        "Plain white background, clean outline, simple composition."
    )


def build_negative_prompt() -> str:
    return (
        "multiple characters, many objects, sticker sheet, collage, grid, icons, "
        "repeated faces, extra heads, extra people, background scene, decorations, "
        "text, letters, watermark, logo, clutter, cropped body, cut off"
    )


def safe_filename(text: str, suffix: str = ".png") -> str:
    """Create a stable filesystem-safe filename from a query."""
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    stem = stem[:80] or "generated_sticker"
    return f"{stem}{suffix}"


def resolve_generation_device(device: str | None = None) -> str:
    """Resolve auto generation device independently from retrieval device."""
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StickerGenerator:
    """Lazy-loaded generator for manual generation and retrieval fallback."""

    def __init__(self, cfg: Config, device: str | None = None):
        self.cfg = cfg
        self.device = resolve_generation_device(device or cfg.generation_device)
        self._pipe = None
        self._sam_predictor = None

    def preload(self, postprocess: bool = True):
        """Load generation models before the first user-visible generation."""
        self._load_pipe()
        if postprocess:
            self._load_sam_predictor()
        logger.info("Generation models preloaded")

    def generate(
        self,
        query: str,
        output_path: Path | None = None,
        seed: int | None = None,
        postprocess: bool = True,
    ) -> GenerationResult:
        """Generate a transparent PNG sticker for a query."""
        self.cfg.ensure_dirs()
        if output_path is None:
            output_path = self.cfg.generated_dir / safe_filename(query)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = build_sticker_prompt(query)
        image = self._generate_base_image(prompt, seed=seed)

        if postprocess:
            image = self._postprocess(image)
        else:
            image = image.convert("RGBA")

        image.save(output_path)
        coverage = alpha_mask_coverage(image)
        logger.info("Generated sticker saved to %s", output_path)
        return GenerationResult(
            query=query,
            prompt=prompt,
            output_path=output_path,
            mask_coverage=coverage,
        )

    def _generate_base_image(self, prompt: str, seed: int | None = None) -> Image.Image:
        pipe = self._load_pipe()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        return pipe(
            prompt=prompt,
            negative_prompt=build_negative_prompt(),
            num_inference_steps=self.cfg.generation_steps,
            guidance_scale=self.cfg.generation_guidance_scale,
            width=self.cfg.generation_width,
            height=self.cfg.generation_height,
            generator=generator,
        ).images[0]

    def _postprocess(self, image: Image.Image) -> Image.Image:
        predictor = self._load_sam_predictor()
        mask = segment_main_object_with_sam(image, predictor)
        sticker_rgba = apply_alpha_mask(image, mask)
        return add_white_outline(sticker_rgba, outline_size=self.cfg.outline_size)

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        variant = self.cfg.generation_variant or None
        if self.device == "cpu":
            logger.warning(
                "Generation is running on CPU. This is very slow; use --generation-device mps/cuda if available."
            )
        logger.info("Loading generation model %s on %s", self.cfg.generation_model, self.device)
        try:
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                self.cfg.generation_model,
                torch_dtype=dtype,
                variant=variant,
                local_files_only=self.cfg.generation_local_files_only,
            )
        except OSError as e:
            if self.cfg.generation_local_files_only:
                raise RuntimeError(
                    "Generation model is not available in the local Hugging Face cache. "
                    "Run once with --allow-model-download, preferably with "
                    "--generation-device mps or cuda, to populate the fp16 cache."
                ) from e
            raise
        self._pipe = self._pipe.to(self.device)
        if hasattr(self._pipe, "set_progress_bar_config"):
            self._pipe.set_progress_bar_config(disable=False)
        return self._pipe

    def _load_sam_predictor(self):
        if self._sam_predictor is not None:
            return self._sam_predictor

        if not self.cfg.sam_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {self.cfg.sam_checkpoint}. "
                "Download it to models/ or pass a different checkpoint in Config."
            )

        from segment_anything import SamPredictor, sam_model_registry

        logger.info("Loading SAM %s from %s", self.cfg.sam_model_type, self.cfg.sam_checkpoint)
        sam = sam_model_registry[self.cfg.sam_model_type](
            checkpoint=str(self.cfg.sam_checkpoint)
        )
        sam.to(device=self.cfg.sam_device)
        self._sam_predictor = SamPredictor(sam)
        return self._sam_predictor
