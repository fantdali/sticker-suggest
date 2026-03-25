"""CLI: Interactive sticker search."""

import argparse
import logging

from src.config import Config
from src.retrieval.searcher import StickerSearcher


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Search stickers by text query")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--query", type=str, default=None, help="Single query (or omit for interactive)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show results as images (matplotlib)"
    )
    args = parser.parse_args()

    cfg = Config(clip_device=args.device, top_k=args.top_k)
    searcher = StickerSearcher(cfg)

    if args.query:
        results = searcher.search(args.query)
        _print_results(args.query, results)
        if args.show:
            _show_results(args.query, results)
    else:
        print("Interactive sticker search. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            results = searcher.search(query)
            _print_results(query, results)
            if args.show:
                _show_results(query, results)


def _print_results(query: str, results):
    print(f"\n=== Results for: '{query}' ===\n")
    for i, r in enumerate(results, 1):
        ocr_info = f'  OCR: "{r.ocr_text}"' if r.ocr_text else ""
        print(
            f"  {i:2d}. {r.filename}  "
            f"score={r.score:.3f}  "
            f"(vis={r.visual_score:.3f}, ocr={r.ocr_score:.3f})"
            f"{ocr_info}"
        )
    print()


def _show_results(query: str, results):
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"'{query}'", fontsize=14)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, r in enumerate(results):
        row, col = divmod(i, cols)
        ax = axes[row][col]
        try:
            img = Image.open(r.image_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(
            f"{r.score:.2f} (v={r.visual_score:.2f} o={r.ocr_score:.2f})", fontsize=9
        )
        ax.axis("off")

    # Hide unused subplots
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
