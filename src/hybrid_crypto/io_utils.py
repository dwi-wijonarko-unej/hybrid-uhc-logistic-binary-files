"""Filesystem, hashing, CSV/JSON, and logging helpers.

All path handling uses :class:`pathlib.Path`. Hashes are streamed in 1 MiB
chunks so large files never need to fit in memory for checksumming. SHA-256
is the primary checksum everywhere; MD5 is provided only as a secondary,
legacy-compatibility digest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

_HASH_CHUNK_BYTES = 1024 * 1024


def ensure_directory(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_binary(path: Path) -> bytes:
    """Read a file as raw bytes."""
    return Path(path).read_bytes()


def write_binary(path: Path, data: bytes) -> None:
    """Write raw bytes to a file, creating parent directories as needed."""
    path = Path(path)
    ensure_directory(path.parent)
    path.write_bytes(data)


def sha256_bytes(data: bytes) -> str:
    """SHA-256 digest of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 digest of a file."""
    return _digest_file(path, "sha256")


def md5_file(path: Path) -> str:
    """Streamed MD5 digest of a file (secondary checksum only)."""
    return _digest_file(path, "md5")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as a CSV file with a header line."""
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON document with stable key ordering."""
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger with a concise console (and optional file) handler.

    Replaces previously configured handlers so repeated calls are idempotent.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file is not None:
        ensure_directory(log_file.parent)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
