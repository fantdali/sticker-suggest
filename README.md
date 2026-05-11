# Sticker Suggest

CLI project for multimodal sticker retrieval and generation.

## Main Commands

Preprocess stickers and OCR text:

```bash
uv run python main.py preprocess --ocr-gpu
```

Build CLIP + OCR embeddings:

```bash
uv run python main.py index --device mps
```

Search:

```bash
uv run python main.py search --query "hello sticker" --top-k 5
```

Generate a new sticker:

```bash
uv run python main.py generate \
  --query "cat waving sticker" \
  --output outputs/generated/cat_waving.png \
  --show
```
