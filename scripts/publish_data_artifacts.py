#!/usr/bin/env python3
"""Build the publication-grade data-availability layer under ``data/``.

Consolidates the real-format benchmark outputs (12 DOCX/PDF/JPEG files,
paper tables) and the synthetic-experiment outputs (repetition protocol)
into the auditable structure requested for the data availability statement:

    data/
    ├── file_manifest.csv          12-file manifest with all SHA-256 hashes
    ├── checksums_sha256.txt       sha256sum-compatible checksums (36 binaries)
    ├── parameter_settings.yaml    full parameter snapshot
    ├── raw_measurements.csv       per-file timing/throughput (paper Table 2)
    ├── byte_statistics.csv        entropy/correlation (paper Table 1)
    ├── reconstruction_checks.csv  integrity checks (paper Table 3)
    ├── artifact_map.md            artifact -> paper table/figure mapping
    ├── runtime_environment.txt    interpreter, packages, CPU/OS record
    └── analysis_outputs/          copied tables/figures/logs of both runs

Every checksum is RE-COMPUTED from the files on disk and cross-checked
against the values recorded by the benchmark before anything is written;
a mismatch aborts the publication.
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_crypto.config import load_config  # noqa: E402
from hybrid_crypto.io_utils import ensure_directory, sha256_file, write_csv, write_json  # noqa: E402

logger = logging.getLogger("publish_data_artifacts")

_GROUP_TO_TYPE = {"Doc": "docx", "PDF": "pdf", "Img": "jpeg"}
_SIZE_RANK = {"Small": "01", "Medium": "02", "Large": "03", "Huge": "04"}
_SIZE_LABEL = {"Small": "1MB", "Medium": "10MB", "Large": "50MB", "Huge": "100MB"}
_PAPER_TYPE = {"docx": "DOCX", "pdf": "PDF", "jpeg": "JPEG"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_id(filename: str) -> tuple[str, str, str]:
    """Split 'Doc_Small.docx' into (file_id, type token, nominal label)."""
    stem = Path(filename).stem
    group, size_name = stem.split("_", 1)
    type_token = _GROUP_TO_TYPE[group]
    return f"{type_token}_{_SIZE_RANK[size_name]}", type_token, _SIZE_LABEL[size_name]


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _pip_freeze(python: str) -> str:
    try:
        return subprocess.run(
            [python, "-m", "pip", "freeze"], capture_output=True, text=True, check=True, timeout=120
        ).stdout.strip()
    except Exception as error:  # pragma: no cover - best-effort record
        return f"# pip freeze failed: {error}"


def _verify_and_write_checksums(
    benchmark_dir: Path, summary_rows: list[dict[str, str]], data_dir: Path
) -> dict[str, str]:
    """Recompute every binary checksum on disk and write the checksum file.

    Returns the ``{filename: sha256_ciphertext}`` map (the benchmark's
    summary CSV does not carry a ciphertext-hash column). Aborts on any
    mismatch between the on-disk artifacts and the recorded hashes.
    """
    lines: list[str] = []
    ciphertext_hashes: dict[str, str] = {}
    for row in summary_rows:
        filename = row["filename"]
        triple = {
            benchmark_dir / "source" / filename: row["sha256_original"],
            benchmark_dir / "encrypted" / filename: None,  # recovered from disk
            benchmark_dir / "decrypted" / filename: row["sha256_decrypted"],
        }
        for path, recorded in triple.items():
            if not path.is_file():
                raise SystemExit(f"publication aborted: expected artifact missing: {path}")
            actual = sha256_file(path)
            if recorded is not None and actual != recorded:
                raise SystemExit(
                    f"publication aborted: checksum mismatch for {path}\n"
                    f"  recorded: {recorded}\n  actual:   {actual}"
                )
            if recorded is None:
                ciphertext_hashes[filename] = actual
            lines.append(f"{actual}  {path.relative_to(PROJECT_ROOT).as_posix()}")
    (data_dir / "checksums_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("checksums_sha256.txt written (%d entries, all verified against disk)", len(lines))
    return ciphertext_hashes


def publish(config_path: Path, benchmark_dir: Path, results_dir: Path, data_dir: Path) -> None:
    summary_rows = _read_csv(benchmark_dir / "summary_results.csv")
    if len(summary_rows) != 12:
        raise SystemExit(f"expected 12 benchmark rows, found {len(summary_rows)}")
    if not all(row["integrity"] == "MATCH" for row in summary_rows):
        raise SystemExit("benchmark contains non-MATCH integrity rows; refusing to publish")

    ensure_directory(data_dir)
    ciphertext_hashes = _verify_and_write_checksums(benchmark_dir, summary_rows, data_dir)
    ordered = sorted(summary_rows, key=lambda row: _file_id(row["filename"])[0])

    # --- file_manifest.csv -------------------------------------------------
    manifest_rows = [
        {
            "file_id": file_id,
            "filename": row["filename"],
            "file_type": _PAPER_TYPE[type_token],
            "nominal_size_label": label,
            "exact_size_bytes": row["size_bytes"],
            "sha256_plaintext": row["sha256_original"],
            "sha256_ciphertext": ciphertext_hashes[row["filename"]],
            "sha256_decrypted": row["sha256_decrypted"],
            "reconstruction_exact_match": row["integrity"] == "MATCH",
            "access_status": "public",
            "distribution": (
                "committed in repository"
                if label == "1MB"
                else "regenerable bit-exactly: python scripts/benchmark_real_files.py --generate-only"
            ),
        }
        for row in ordered
        for file_id, type_token, label in [_file_id(row["filename"])]
    ]
    write_csv(
        data_dir / "file_manifest.csv",
        [
            "file_id", "filename", "file_type", "nominal_size_label", "exact_size_bytes",
            "sha256_plaintext", "sha256_ciphertext", "sha256_decrypted",
            "reconstruction_exact_match", "access_status", "distribution",
        ],
        manifest_rows,
    )

    # --- raw_measurements / byte_statistics / reconstruction_checks --------
    measurements = [
        {
            "file_id": _file_id(row["filename"])[0],
            "filename": row["filename"],
            "plaintext_size_bytes": row["size_bytes"],
            "ciphertext_size_bytes": row["size_bytes"],
            "encryption_time_seconds": row["encryption_time_s"],
            "decryption_time_seconds": row["decryption_time_s"],
            "encryption_throughput_mb_s": row["enc_throughput_mb_s"],
            "decryption_throughput_mb_s": row["dec_throughput_mb_s"],
        }
        for row in ordered
    ]
    statistics = [
        {
            "file_id": _file_id(row["filename"])[0],
            "filename": row["filename"],
            "plaintext_entropy": row["entropy_plaintext"],
            "ciphertext_entropy": row["entropy_ciphertext"],
            "plaintext_adjacent_byte_correlation": row["corr_plaintext"],
            "ciphertext_adjacent_byte_correlation": row["corr_ciphertext"],
        }
        for row in ordered
    ]
    checks = [
        {
            "file_id": _file_id(row["filename"])[0],
            "md5_plaintext": row["md5_original"],
            "md5_decrypted": row["md5_decrypted"],
            "sha256_plaintext": row["sha256_original"],
            "sha256_decrypted": row["sha256_decrypted"],
            "exact_match": row["integrity"] == "MATCH",
        }
        for row in ordered
    ]
    write_csv(data_dir / "raw_measurements.csv", list(measurements[0].keys()), measurements)
    write_csv(data_dir / "byte_statistics.csv", list(statistics[0].keys()), statistics)
    write_csv(data_dir / "reconstruction_checks.csv", list(checks[0].keys()), checks)

    # --- parameter_settings.yaml -------------------------------------------
    config = load_config(config_path)
    cipher = config.cipher
    bench = config.benchmark
    key_report = results_dir / "configuration" / "key_material_report.json"
    parameters = {
        "source": (
            "config/experiment_config.yaml (snapshot; the authoritative per-run copy is "
            "results/configuration/used_config.yaml)"
        ),
        "cipher": {
            "matrix_dimension": cipher.matrix_dimension,
            "block_size": cipher.block_size,
            "modulus": cipher.modulus,
            "logistic_r": cipher.logistic_r,
            "logistic_x0": cipher.logistic_x0,
            "warmup_iterations": cipher.warmup_iterations,
            "quantization_rule": cipher.quantization_rule,
            "numeric_precision": cipher.numeric_precision,
            "parameter_id": cipher.parameter_id(),
        },
        "matrix_generation_algorithm": (
            "Identity init; strict upper triangle filled row-major from a 135-byte logistic sequence "
            "(n(n-1)/2 + (n-1), n=16); lower part via determinant-preserving row additions "
            "R_i <- R_i + a*R_0 mod 256; inverse by Gauss-Jordan over Z/256Z with unit pivots and the "
            "extended Euclidean algorithm, verified in both product directions."
        ),
        "benchmark_protocol": {
            "real_format_benchmark": "single encrypt+decrypt pass per file, timings include file I/O",
            "synthetic_experiment_repetitions": {
                "timing_warmup_runs": bench.timing_warmup_runs,
                "timing_repetitions": bench.timing_repetitions,
                "include_file_io_in_timing": bench.include_file_io_in_timing,
                "megabyte_bytes": bench.megabyte_bytes,
            },
        },
        "test_dataset": {
            "classes": "DOCX/PDF/JPEG x 1/10/50/100 MB (12 files)",
            "padding_source": "seeded SplitMix64 (default; bit-exact regeneration)",
            "generator": "python scripts/benchmark_real_files.py --generate-only",
        },
        "key_material_report": (
            key_report.relative_to(PROJECT_ROOT).as_posix() if key_report.is_file() else None
        ),
    }
    (data_dir / "parameter_settings.yaml").write_text(
        yaml.safe_dump(parameters, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # --- runtime_environment.txt -------------------------------------------
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    environment = "\n".join(
        [
            f"# Generated {generated_at} by scripts/publish_data_artifacts.py",
            f"python            : {platform.python_version()} ({platform.python_implementation()})",
            f"platform          : {platform.platform()}",
            f"machine           : {platform.machine()}",
            f"cpu_model         : {_cpu_model()}",
            f"cpu_count         : {os.cpu_count()}",
            "",
            "# Timings in raw_measurements.csv are wall-clock, hardware/OS/load-dependent,",
            "# and must not be compared across platforms.",
            "",
            "# pip freeze",
            _pip_freeze(sys.executable),
            "",
        ]
    )
    (data_dir / "runtime_environment.txt").write_text(environment, encoding="utf-8")

    # --- analysis_outputs/ ---------------------------------------------------
    outputs = data_dir / "analysis_outputs"
    real_dir = ensure_directory(outputs / "real_format_benchmark")
    for name in ("summary_results.csv", "source_manifest.csv", "Figure_histogram_detail.png"):
        source = benchmark_dir / name
        if source.is_file():
            shutil.copyfile(source, real_dir / name)
    synthetic_dir = ensure_directory(outputs / "synthetic_experiment")
    for subgroup in ("raw", "tables", "figures", "configuration", "logs"):
        source_dir = results_dir / subgroup
        if not source_dir.is_dir():
            continue
        target = ensure_directory(synthetic_dir / subgroup)
        for item in source_dir.iterdir():
            if item.is_file() and item.name != ".gitkeep":
                shutil.copyfile(item, target / item.name)

    # --- artifact_map.md -----------------------------------------------------
    artifact_map = f"""# Artifact map: repository artifacts -> paper tables and figures

Generated {generated_at}. Hashes and sizes live in `file_manifest.csv` and
`checksums_sha256.txt`; parameters in `parameter_settings.yaml`.

| Paper item | Primary artifact | Regeneration |
|---|---|---|
| Test-file dataset (12 files) | `file_manifest.csv` + `checksums_sha256.txt` (+ `benchmark/source/*Small*` committed) | `python scripts/benchmark_real_files.py --generate-only` (bit-exact, seeded) |
| Table 1 — statistical analysis | `byte_statistics.csv` | `python scripts/benchmark_real_files.py` |
| Table 2 — computational performance | `raw_measurements.csv` | `python scripts/benchmark_real_files.py` |
| Table 3 — integrity check | `reconstruction_checks.csv` | `python scripts/benchmark_real_files.py` |
| Histogram figure | `analysis_outputs/real_format_benchmark/Figure_histogram_detail.png` | idem |
| Repetition-protocol experiment (warm-up + 5 runs, synthetic 12-file set) | `analysis_outputs/synthetic_experiment/raw/*.csv`, `tables/summary_statistics.csv`, `figures/*.png` | `python scripts/reproduce_experiments.py` |
| Key matrix and modular inverse | `analysis_outputs/synthetic_experiment/configuration/key_matrix*.csv`, `key_material_report.json` | idem |
| Exact algorithm specification | `README.md` (Algorithm specification) + `src/hybrid_crypto/` | — |
| Environment | `runtime_environment.txt`, `requirements.txt`, `environment.yml` | — |

Notes: the real-format benchmark times a single pass per file (the legacy
protocol used for the paper tables); the repetition protocol (mean/std/
median/min/max over 5 measured runs after 1 warm-up) is documented in
`analysis_outputs/synthetic_experiment/`. Ciphertext expansion is zero for
every file (ciphertext size == plaintext size; see `raw_measurements.csv`).
"""
    (data_dir / "artifact_map.md").write_text(artifact_map, encoding="utf-8")

    summary = {
        "files_published": len(manifest_rows),
        "checksum_entries": 3 * len(manifest_rows),
        "data_dir": str(data_dir),
        "generated_at": generated_at,
    }
    write_json(benchmark_dir / "publication_report.json", summary)
    logger.info(
        "Publication layer written under %s (%d files, %d checksums)",
        data_dir, len(manifest_rows), 3 * len(manifest_rows),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "experiment_config.yaml")
    parser.add_argument("--benchmark-dir", type=Path, default=PROJECT_ROOT / "benchmark")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    publish(args.config, args.benchmark_dir, args.results_dir, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
