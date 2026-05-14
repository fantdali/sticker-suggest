"""CLI: Interactive sticker search."""

import argparse
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import Config
from src.generation import StickerGenerator, resolve_generation_device
from src.retrieval.searcher import StickerSearcher


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Search stickers by text query")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--query", type=str, default=None, help="Single query (or omit for interactive)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show results as images (matplotlib)"
    )
    parser.add_argument(
        "--generate-on-low-score",
        action="store_true",
        help="Generate a new sticker when the best retrieval score is below --min-score",
    )
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--generation-output", type=str, default=None)
    parser.add_argument(
        "--generation-device",
        type=str,
        default="auto",
        help="Generation device: auto, mps, cuda, or cpu. Independent from search --device.",
    )
    parser.add_argument("--sam-device", type=str, default="cpu")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of using only the local cache",
    )
    parser.add_argument(
        "--preload-generator",
        choices=["background", "blocking", "off"],
        default="background",
        help=(
            "When fallback generation is enabled, load SDXL + SAM before they are needed. "
            "'background' is used for interactive mode; 'blocking' is best for one-shot demos."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run repeated searches and print latency percentiles",
    )
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument(
        "--ocr-alpha",
        type=float,
        default=0.01,
        help="Weight of OCR score in late fusion; use 0.0 for image-only baseline",
    )
    args = parser.parse_args()

    generation_device = resolve_generation_device(args.generation_device)
    cfg = Config(
        clip_device=args.device,
        top_k=args.top_k,
        generation_min_score=args.min_score,
        ocr_alpha=args.ocr_alpha,
        generation_device=generation_device,
        generation_local_files_only=not args.allow_model_download,
        sam_device=args.sam_device,
    )
    searcher = StickerSearcher(cfg)
    generator_cache = _setup_generator_cache(cfg, args)

    if args.query:
        if args.benchmark:
            _benchmark(searcher, args.query, args.top_k, args.benchmark_runs)
        results, timing = searcher.search_with_timing(args.query)
        _print_results(args.query, results)
        _print_timing(timing)
        generated = _maybe_generate(args.query, results, cfg, args, generator_cache)
        if args.show:
            _show_results(args.query, results, generated)
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
            results, timing = searcher.search_with_timing(query)
            _print_results(query, results)
            _print_timing(timing)
            generated = _maybe_generate(query, results, cfg, args, generator_cache)
            if args.show:
                _show_results(query, results, generated)


def _print_results(query: str, results):
    print(f"\n=== Results for: '{query}' ===\n")
    for i, r in enumerate(results, 1):
        ocr_info = f'  OCR: "{r.ocr_text}"' if r.ocr_text else ""
        print(
            f"  {r.rank:2d}. {r.filename}  "
            f"score={r.score:.3f}  "
            f"(vis={r.visual_score:.3f}, ocr={r.ocr_score:.3f})"
            f"{ocr_info}"
        )
    print()


def _print_timing(timing):
    print(
        "Timing: "
        f"query={timing.query_ms:.1f} ms, "
        f"scoring={timing.scoring_ms:.1f} ms, "
        f"total={timing.total_ms:.1f} ms\n"
    )


def _maybe_generate(query: str, results, cfg: Config, args, generator_cache):
    if not args.generate_on_low_score:
        return None
    lowest_score = results[-1].score if results else float("-inf")
    if lowest_score >= args.min_score:
        print(
            f"Lowest score {lowest_score:.3f} >= {args.min_score:.3f}; "
            "generation skipped.\n"
        )
        return None

    print(
        f"Lowest score {lowest_score:.3f} < {args.min_score:.3f}; "
        "generating fallback sticker..."
    )
    output_path = Path(args.generation_output) if args.generation_output else None
    generator = _get_generator(cfg, generator_cache)
    result = generator.generate(query, output_path=output_path)
    print(f"Generated fallback: {result.output_path}\n")
    return result


def _setup_generator_cache(cfg: Config, args):
    cache = {"generator": None, "future": None, "executor": None}
    if not args.generate_on_low_score or args.preload_generator == "off":
        return cache
    if args.query and args.preload_generator == "background":
        return cache

    generator = StickerGenerator(cfg, device=cfg.generation_device)
    cache["generator"] = generator

    if args.preload_generator == "blocking":
        print("Preloading generation models (SDXL + SAM)...")
        generator.preload(postprocess=True)
        print("Generation models ready.\n")
        return cache

    print("Preloading generation models in the background...")
    executor = ThreadPoolExecutor(max_workers=1)
    cache["executor"] = executor
    cache["future"] = executor.submit(generator.preload, True)
    return cache


def _get_generator(cfg: Config, generator_cache):
    if generator_cache["generator"] is None:
        generator_cache["generator"] = StickerGenerator(
            cfg, device=cfg.generation_device
        )

    future = generator_cache.get("future")
    if future is not None:
        print("Waiting for generation models to finish preloading...")
        future.result()
        generator_cache["future"] = None
        executor = generator_cache.get("executor")
        if executor is not None:
            executor.shutdown(wait=False)
            generator_cache["executor"] = None

    return generator_cache["generator"]


def _benchmark(searcher: StickerSearcher, query: str, top_k: int, runs: int):
    timings = []
    for _ in range(max(1, runs)):
        _, timing = searcher.search_with_timing(query, top_k=top_k)
        timings.append(timing)
    totals = sorted(t.total_ms for t in timings)
    scoring = sorted(t.scoring_ms for t in timings)
    print("\n=== Benchmark ===")
    print(f"runs={len(timings)} query='{query}' top_k={top_k}")
    print(
        f"total p50={statistics.median(totals):.1f} ms "
        f"p95={_percentile(totals, 0.95):.1f} ms"
    )
    print(
        f"scoring p50={statistics.median(scoring):.1f} ms "
        f"p95={_percentile(scoring, 0.95):.1f} ms\n"
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def _show_results(query: str, results, generated=None):
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(results) + (1 if generated else 0)
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
            img = Image.open(r.image_path).convert("RGBA")
            ax.imshow(_checkerboard_background(img.size))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(
            f"{r.score:.2f} (v={r.visual_score:.2f} o={r.ocr_score:.2f})", fontsize=9
        )
        ax.axis("off")

    if generated:
        row, col = divmod(len(results), cols)
        ax = axes[row][col]
        try:
            img = Image.open(generated.output_path).convert("RGBA")
            ax.imshow(_checkerboard_background(img.size))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(
            f"generated fallback\nalpha={generated.mask_coverage:.2f}", fontsize=9
        )
        ax.axis("off")

    # Hide unused subplots
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()


def _checkerboard_background(size: tuple[int, int], square: int = 24):
    import numpy as np

    w, h = size
    y, x = np.indices((h, w))
    pattern = ((x // square + y // square) % 2).astype(np.uint8)
    bg = np.where(pattern[..., None] == 0, 235, 255).astype(np.uint8)
    return np.repeat(bg, 3, axis=2)


if __name__ == "__main__":
    main()
