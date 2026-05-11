"""Post-processing utilities for turning generated images into stickers."""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageFilter


def segment_main_object_with_sam(
    image_pil: Image.Image,
    predictor,
    box_scale: float = 0.9,
) -> np.ndarray:
    """Segment the centered main object with a SAM predictor."""
    image_rgb = image_pil.convert("RGB")
    image_np = np.array(image_rgb)
    predictor.set_image(image_np)

    h, w = image_np.shape[:2]
    box_w = int(w * box_scale)
    box_h = int(h * box_scale)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    input_box = np.array([x1, y1, x2, y2])

    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(np.uint8)


def apply_alpha_mask(image_pil: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply a binary mask as a softly blurred alpha channel."""
    image_rgba = image_pil.convert("RGBA")
    image_np = np.array(image_rgba)
    alpha = (mask * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    image_np[..., 3] = alpha
    return Image.fromarray(image_np)


def add_white_outline(
    rgba_image: Image.Image,
    outline_size: int = 3,
) -> Image.Image:
    """Add a white sticker-style outline around the non-transparent area."""
    rgba_image = rgba_image.convert("RGBA")
    alpha = rgba_image.getchannel("A")
    expanded_alpha = alpha.filter(ImageFilter.MaxFilter(outline_size * 2 + 1))
    outline = Image.new("RGBA", rgba_image.size, (255, 255, 255, 0))
    outline.putalpha(expanded_alpha)
    return Image.alpha_composite(outline, rgba_image)


def alpha_mask_coverage(rgba_image: Image.Image) -> float:
    """Return the share of pixels that are not fully transparent."""
    alpha = np.array(rgba_image.convert("RGBA").getchannel("A"))
    return float((alpha > 0).mean())
