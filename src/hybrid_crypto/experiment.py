"""End-to-end experiment orchestration.

Given a validated :class:`~hybrid_crypto.config.ExperimentConfig`, this
module:

1. writes the used configuration and key-material artifacts;
2. runs the timed encrypt/decrypt benchmark protocol for every input file
   (warm-up runs first, excluded from the recorded measurements);
3. computes byte statistics and exact-reconstruction checks;
4. writes every CSV/JSON table and PNG figure through :mod:`reporting`.

Timings are wall-clock and hardware-dependent; they must not be compared
across platforms.
"""

from __future__ import annotations

import logging
import shutil
import statistics
import time
from pathlib import Path

from . import metrics, reporting
from .cipher import HybridHillXORCipher
from .io_utils import (
    ensure_directory,
    md5_file,
    read_binary,
    setup_logging,
    sha256_bytes,
    write_binary,
    write_csv,
    write_json,
)
from .matrix_generator import UnimodularKeyMaterial

logger = logging.getLogger(__name__)

_RESULT_SUBDIRS = ("configuration", "ciphertexts", "decrypted", "raw", "tables", "figures", "logs")


def _detect_file_type(data: bytes) -> str:
    """Classify a file by its leading magic bytes (synthetic files included)."""
    if data.startswith(b"PK\x03\x04"):
        return "docx"
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return "binary"


def _nominal_size_label(size_bytes: int, size_map: dict[int, str]) -> str:
    return size_map.get(size_bytes, f"{size_bytes}_bytes")


def _timed_run(
    cipher: HybridHillXORCipher,
    plaintext_path: Path,
    ciphertext_path: Path,
    decrypted_path: Path,
    include_io: bool,
) -> tuple[float, float, bytes, bytes, bytes]:
    """One full encrypt+decrypt pass.

    With ``include_io`` the stopwatch covers read + transform + write for
    both directions; otherwise only the in-memory transform is timed and
    file writes happen outside the measured interval. Artifacts are always
    (re)written so the final run's files remain on disk.
    """
    if include_io:
        start = time.perf_counter()
        plaintext = read_binary(plaintext_path)
        ciphertext = cipher.encrypt(plaintext)
        write_binary(ciphertext_path, ciphertext)
        encryption_seconds = time.perf_counter() - start

        start = time.perf_counter()
        ciphertext_read = read_binary(ciphertext_path)
        decrypted = cipher.decrypt(ciphertext_read)
        write_binary(decrypted_path, decrypted)
        decryption_seconds = time.perf_counter() - start
    else:
        plaintext = read_binary(plaintext_path)
        start = time.perf_counter()
        ciphertext = cipher.encrypt(plaintext)
        encryption_seconds = time.perf_counter() - start
        write_binary(ciphertext_path, ciphertext)

        start = time.perf_counter()
        decrypted = cipher.decrypt(ciphertext)
        decryption_seconds = time.perf_counter() - start
        write_binary(decrypted_path, decrypted)

    return encryption_seconds, decryption_seconds, plaintext, ciphertext, decrypted


def _nan_safe(function, data: bytes, label: str, file_id: str) -> float:
    try:
        return float(function(data))
    except ValueError:
        logger.warning("%s: %s is undefined for empty data; recorded as NaN", file_id, label)
        return float("nan")


def _byte_statistics_row(file_id: str, plaintext: bytes, ciphertext: bytes) -> dict[str, object]:
    try:
        chi_stat, chi_p = metrics.chi_square_uniform(ciphertext)
    except ValueError:
        logger.warning("%s: chi-square undefined for empty ciphertext; recorded as NaN", file_id)
        chi_stat, chi_p = float("nan"), float("nan")
    return {
        "file_id": file_id,
        "plaintext_entropy": _nan_safe(metrics.shannon_entropy, plaintext, "plaintext entropy", file_id),
        "ciphertext_entropy": _nan_safe(metrics.shannon_entropy, ciphertext, "ciphertext entropy", file_id),
        "plaintext_adjacent_byte_correlation": _nan_safe(
            metrics.adjacent_byte_correlation, plaintext, "plaintext correlation", file_id
        ),
        "ciphertext_adjacent_byte_correlation": _nan_safe(
            metrics.adjacent_byte_correlation, ciphertext, "ciphertext correlation", file_id
        ),
        "chi_square_statistic": float(chi_stat),
        "chi_square_p_value": float(chi_p),
    }


def run_experiment(config: ExperimentConfig) -> Path:
    """Execute the full experiment; returns the output directory."""
    output_dir = config.output_dir
    for subdirectory in _RESULT_SUBDIRS:
        ensure_directory(output_dir / subdirectory)
    setup_logging(output_dir / "logs" / "experiment.log")
    logger.info("Experiment '%s' started (output: %s)", config.name, output_dir)

    if config.source_path is not None:
        shutil.copyfile(config.source_path, output_dir / "configuration" / "used_config.yaml")

    params = config.cipher
    params.validate()
    key = UnimodularKeyMaterial(params)
    matrix_path = reporting.save_matrix_csv(key.matrix, output_dir / "configuration" / "key_matrix.csv")
    inverse_path = reporting.save_matrix_csv(
        key.inverse, output_dir / "configuration" / "key_matrix_inverse_mod256.csv"
    )
    write_json(
        output_dir / "configuration" / "key_material_report.json",
        reporting.build_key_material_report(params, key, matrix_path, inverse_path),
    )
    logger.info(
        "Key material ready: dimension=%d modulus=%d inverse_verified=%s",
        params.matrix_dimension,
        params.modulus,
        key.verification["verified"],
    )

    cipher = HybridHillXORCipher(
        key_matrix=key.matrix,
        inverse_matrix=key.inverse,
        logistic_r=params.logistic_r,
        logistic_x0=params.logistic_x0,
        warmup_iterations=params.warmup_iterations,
        block_size=params.block_size,
        modulus=params.modulus,
    )

    if not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    input_files = sorted(
        path for path in config.input_dir.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {config.input_dir}. Place test files there or run "
            "scripts/generate_synthetic_test_files.py first."
        )

    size_map = dict(zip(config.synthetic.sizes_bytes, config.synthetic.size_labels))
    manifest_rows: list[dict[str, object]] = []
    sources: dict[str, Path] = {}
    for index, path in enumerate(input_files, start=1):
        file_id = f"F{index:02d}"
        data = read_binary(path)
        manifest_rows.append(
            {
                "file_id": file_id,
                "filename": path.name,
                "file_type": _detect_file_type(data),
                "exact_size_bytes": len(data),
                "nominal_size_label": _nominal_size_label(len(data), size_map),
                "sha256_plaintext": sha256_bytes(data),
            }
        )
        sources[file_id] = path
    write_csv(output_dir / "raw" / "file_manifest.csv", reporting.FILE_MANIFEST_FIELDS, manifest_rows)
    logger.info("Manifest written: %d input files", len(manifest_rows))

    bench = config.benchmark
    measurement_rows: list[dict[str, object]] = []
    statistics_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    expansion_by_file: dict[str, float] = {}
    example: tuple[Path, Path, str] | None = None

    for manifest in manifest_rows:
        file_id = str(manifest["file_id"])
        source = sources[file_id]
        ciphertext_path = output_dir / "ciphertexts" / f"{file_id}_{source.name}.enc"
        decrypted_path = output_dir / "decrypted" / f"{file_id}_{source.name}.dec"

        total_runs = bench.timing_warmup_runs + bench.timing_repetitions
        last: tuple[float, float, bytes, bytes, bytes] | None = None
        for run in range(1, total_runs + 1):
            result = _timed_run(
                cipher, source, ciphertext_path, decrypted_path, bench.include_file_io_in_timing
            )
            if run <= bench.timing_warmup_runs:
                logger.debug(
                    "%s: warm-up run %d/%d excluded from measurements",
                    file_id, run, bench.timing_warmup_runs,
                )
                continue
            last = result
            encryption_seconds, decryption_seconds, plaintext, ciphertext, decrypted = result
            expansion_bytes, expansion_percent = metrics.ciphertext_expansion(
                len(plaintext), len(ciphertext)
            )
            measurement_rows.append(
                {
                    "file_id": file_id,
                    "run_id": run - bench.timing_warmup_runs,
                    "encryption_time_seconds": encryption_seconds,
                    "decryption_time_seconds": decryption_seconds,
                    "encryption_throughput_mb_s": metrics.throughput_mb_s(
                        len(plaintext), encryption_seconds, bench.megabyte_bytes
                    ),
                    "decryption_throughput_mb_s": metrics.throughput_mb_s(
                        len(ciphertext), decryption_seconds, bench.megabyte_bytes
                    ),
                    "plaintext_size_bytes": len(plaintext),
                    "ciphertext_size_bytes": len(ciphertext),
                    "expansion_bytes": expansion_bytes,
                    "expansion_percent": expansion_percent,
                    "sha256_plaintext": manifest["sha256_plaintext"],
                    "sha256_decrypted": sha256_bytes(decrypted),
                    "reconstruction_match": decrypted == plaintext,
                }
            )
        assert last is not None  # timing_repetitions >= 1 guarantees one measured run
        _, _, plaintext, ciphertext, decrypted = last

        reconstruction_match = (
            decrypted == plaintext and sha256_bytes(decrypted) == manifest["sha256_plaintext"]
        )
        reconstruction_rows.append(
            {
                "file_id": file_id,
                "md5_plaintext": md5_file(source),
                "md5_decrypted": md5_file(decrypted_path),
                "sha256_plaintext": manifest["sha256_plaintext"],
                "sha256_decrypted": sha256_bytes(decrypted),
                "exact_match": reconstruction_match,
            }
        )
        statistics_rows.append(_byte_statistics_row(file_id, plaintext, ciphertext))
        expansion_by_file[file_id] = statistics.fmean(
            [float(row["expansion_bytes"]) for row in measurement_rows if row["file_id"] == file_id]
        )
        if source.name == config.histogram_example_filename:
            example = (source, ciphertext_path, source.name)
        logger.info(
            "%s %s: size=%d B runs_recorded=%d reconstruction_match=%s",
            file_id,
            source.name,
            len(plaintext),
            bench.timing_repetitions,
            reconstruction_match,
        )

    write_csv(output_dir / "raw" / "raw_measurements.csv", reporting.RAW_MEASUREMENT_FIELDS, measurement_rows)
    write_csv(output_dir / "raw" / "byte_statistics.csv", reporting.BYTE_STATISTICS_FIELDS, statistics_rows)
    write_csv(
        output_dir / "raw" / "reconstruction_checks.csv",
        reporting.RECONSTRUCTION_CHECK_FIELDS,
        reconstruction_rows,
    )

    summary_rows = reporting.summarize_measurements(manifest_rows, measurement_rows)
    write_csv(output_dir / "tables" / "summary_statistics.csv", reporting.SUMMARY_STATISTICS_FIELDS, summary_rows)

    reporting.generate_figures(
        output_dir / "figures",
        manifest_rows,
        statistics_rows,
        summary_rows,
        expansion_by_file,
        example,
        bench.megabyte_bytes,
    )
    logger.info("Experiment complete; all outputs under %s", output_dir)
    return output_dir
