"""Tests for lightweight generation helpers."""

import numpy as np
from PIL import Image

from src.generation import build_sticker_prompt, safe_filename
from src.generation.generator import resolve_generation_device
from src.generation.postprocess import (
    add_white_outline,
    alpha_mask_coverage,
    apply_alpha_mask,
)


def test_safe_filename_sanitizes_query():
    assert safe_filename("Cat waving!!!") == "cat_waving.png"
    assert safe_filename("   ") == "generated_sticker.png"


def test_resolve_generation_device_respects_explicit_value():
    assert resolve_generation_device("cpu") == "cpu"


def test_build_sticker_prompt_contains_query_and_constraints():
    prompt = build_sticker_prompt("cat waving")
    assert "cat waving" in prompt
    assert "single isolated sticker" in prompt
    assert "Plain white background" in prompt


def test_alpha_mask_and_outline_do_not_need_sam():
    image = Image.new("RGB", (16, 16), "red")
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:11, 5:11] = 1

    rgba = apply_alpha_mask(image, mask)
    outlined = add_white_outline(rgba, outline_size=2)

    assert rgba.mode == "RGBA"
    assert outlined.mode == "RGBA"
    assert 0.0 < alpha_mask_coverage(rgba) < 1.0
    assert alpha_mask_coverage(outlined) > alpha_mask_coverage(rgba)
