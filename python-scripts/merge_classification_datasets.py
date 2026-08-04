"""Backward-compatible CLI entry point for merging classification datasets."""

from classification_dataset import *  # noqa: F401,F403
from classification_dataset import __all__ as _BACKEND_ALL
from classification_dataset.cli import build_parser, main

__all__ = (*_BACKEND_ALL, "build_parser", "main")


if __name__ == "__main__":
    main()
