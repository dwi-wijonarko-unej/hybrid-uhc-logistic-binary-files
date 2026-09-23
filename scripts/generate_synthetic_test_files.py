#!/usr/bin/env python3
"""Generate the 12 deterministic synthetic placeholder test files.

The set consists of DOCX-like, PDF-like, and JPEG-like binary placeholders
(4 target sizes x 3 types by default, all defined in the YAML config). Each
file starts with the real format's magic-byte header, ends with the format's
trailer where applicable, and fills the remainder with bytes from a
fixed-seed SplitMix64 stream, so the generator is fully deterministic: the
same config always produces byte-identical files.

These placeholders are **not identical to the original article dataset**.
They reproduce the experiment *workflow* (tables, figures, checksums,
timings) but not the article's numeric values for byte statistics, because
the byte distributions differ from the original files.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import struct
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_crypto.config import ExperimentConfig, load_config  # noqa: E402
from hybrid_crypto.io_utils import ensure_directory, sha256_bytes, write_binary, write_csv  # noqa: E402

logger = logging.getLogger("generate_synthetic_test_files")

_MASK64 = (1 << 64) - 1

_HEADERS = {
    "docx": b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00" + b"synthetic/word/document.xml",
    "pdf": b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n",
    "jpeg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00",
}

_TRAILERS = {
    "docx": b"PK\x05\x06" + b"\x00" * 18,
    "pdf": b"\n%%EOF\n",
    "jpeg": b"\xff\xd9",
}

_MANIFEST_FIELDS = [
    "filename",
    "file_type",
    "nominal_size_label",
    "target_size_bytes",
    "exact_size_bytes",
    "sha256",
    "generator",
    "seed",
]


class SplitMix64:
    """Minimal deterministic SplitMix64 byte stream (no external dependencies)."""

    def __init__(self, seed: int) -> None:
        self._state = int(seed) & _MASK64

    def _next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def next_bytes(self, count: int) -> bytes:
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        chunks = bytearray()
        for _ in range((count + 7) // 8):
            chunks += struct.pack("<Q", self._next_u64())
        return bytes(chunks[:count])


def _file_seed(base_seed: int, file_type: str, size_bytes: int) -> int:
    """Derive a per-file seed deterministically from the base seed."""
    material = f"{base_seed}|{file_type}|{size_bytes}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def build_placeholder(file_type: str, size_bytes: int, rng: SplitMix64) -> bytes:
    """Build a placeholder of exactly ``size_bytes`` bytes with real magic bytes."""
    header = _HEADERS[file_type]
    trailer = _TRAILERS[file_type]
    filler_length = size_bytes - len(header) - len(trailer)
    if filler_length < 0:
        raise ValueError(
            f"target size {size_bytes} too small for '{file_type}' placeholder "
            f"(needs >= {len(header) + len(trailer)} bytes)"
        )
    return header + rng.next_bytes(filler_length) + trailer


def generate_synthetic_files(config: ExperimentConfig) -> list[Path]:
    """Create every synthetic file and its manifest; return the file paths."""
    synthetic = config.synthetic
    ensure_directory(synthetic.output_dir)
    ensure_directory(synthetic.manifest_dir)

    written: list[Path] = []
    rows: list[dict[str, object]] = []
    for file_type in synthetic.file_types:
        for size_bytes, label in zip(synthetic.sizes_bytes, synthetic.size_labels):
            seed = _file_seed(synthetic.seed, file_type, size_bytes)
            data = build_placeholder(file_type, size_bytes, SplitMix64(seed))
            path = synthetic.output_dir / f"synthetic_{file_type}_{label}.bin"
            write_binary(path, data)
            written.append(path)
            rows.append(
                {
                    "filename": path.name,
                    "file_type": file_type,
                    "nominal_size_label": label,
                    "target_size_bytes": size_bytes,
                    "exact_size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "generator": "splitmix64",
                    "seed": seed,
                }
            )
            logger.info("Synthetic file written: %s (%d bytes)", path.name, len(data))

    manifest_path = synthetic.manifest_dir / "synthetic_files_manifest.csv"
    write_csv(manifest_path, _MANIFEST_FIELDS, rows)
    logger.info("Synthetic manifest written: %s (%d files)", manifest_path, len(rows))
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "experiment_config.yaml",
                        help="Path to the experiment YAML configuration (default: config/experiment_config.yaml)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the synthetic_files.output_dir from the config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
        config = replace(config, synthetic=replace(config.synthetic, output_dir=output_dir))

    written = generate_synthetic_files(config)
    logger.info("Generated %d deterministic synthetic placeholder files", len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
