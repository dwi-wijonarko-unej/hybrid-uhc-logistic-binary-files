#!/usr/bin/env python3
"""Reproduce every experiment output from a YAML configuration.

Runs the full pipeline for all files in the configured input directory and
writes the manifest, raw measurements, byte statistics, reconstruction
checks, key-material report, summary table, and figures under the configured
output directory. If the input directory is empty (and
``experiment.generate_synthetic_files_if_missing`` is true), the 12
deterministic synthetic placeholder files are generated first.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for entry in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from hybrid_crypto.config import load_config  # noqa: E402
from hybrid_crypto.experiment import run_experiment  # noqa: E402

logger = logging.getLogger("reproduce_experiments")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "experiment_config.yaml",
                        help="Path to the experiment YAML configuration (default: config/experiment_config.yaml)")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Force regeneration of the synthetic placeholder files before running")
    return parser.parse_args(argv)


def _input_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(path for path in input_dir.iterdir() if path.is_file() and not path.name.startswith("."))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    existing = _input_files(config.input_dir)

    if args.generate_synthetic or (not existing and config.generate_synthetic_files_if_missing):
        if existing and args.generate_synthetic:
            logger.info("Forcing regeneration of synthetic files (--generate-synthetic)")
        elif not existing:
            logger.info("Input directory %s is empty; generating synthetic placeholder files", config.input_dir)
        from generate_synthetic_test_files import generate_synthetic_files

        generate_synthetic_files(config)
    elif not existing:
        logging.error(
            "No input files in %s and synthetic generation is disabled "
            "(experiment.generate_synthetic_files_if_missing: false)", config.input_dir,
        )
        return 1

    try:
        output_dir = run_experiment(config)
    except FileNotFoundError as error:
        logging.error("%s", error)
        return 1
    logger.info("All experiment outputs are under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
