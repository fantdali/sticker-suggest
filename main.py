"""
Sticker Suggest — main entry point.

Usage:
    python main.py preprocess     # dedup + OCR
    python main.py index          # CLIP + OCR embeddings
    python main.py search         # interactive search
    python main.py generate       # create a new sticker
    python main.py evaluate       # retrieval quality metrics
    python main.py migrate        # convert existing .npy → .pt
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # shift args for sub-command parser

    if cmd == "preprocess":
        from cli.preprocess import main as run
    elif cmd == "index":
        from cli.index import main as run
    elif cmd == "search":
        from cli.search import main as run
    elif cmd == "generate":
        from cli.generate import main as run
    elif cmd == "evaluate":
        from cli.evaluate import main as run
    elif cmd == "migrate":
        from cli.migrate import main as run
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__.strip())
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()
