"""Direct-file compatibility entrypoint for the unified vision CLI."""

from vision_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
