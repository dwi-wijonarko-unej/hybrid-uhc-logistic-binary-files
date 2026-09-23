"""CSV tables, key-material artifacts, and PNG figures for the experiment.

All figures are rendered through the non-interactive ``Agg`` backend so the
pipeline runs on headless machines. Plots contain no randomness; for the same
input rows they render identically.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must import after backend selection)
import numpy as np  # noqa: E402

from .config import CipherParameters  # noqa: E402
from .io_utils import sha256_file  # noqa: E402
from .matrix_generator import UnimodularKeyMaterial  # noqa: E402

logger = logging.getLogger(__name__)

FILE_MANIFEST_FIELDS = [
    "file_id",
    "filename",
    "file_type",
    "exact_size_bytes",
    "nominal_size_label",
    "sha256_plaintext",
]

RAW_MEASUREMENT_FIELDS = [
    "file_id",
    "run_id",
    "encryption_time_seconds",
    "decryption_time_seconds",
    "encryption_throughput_mb_s",
    "decryption_throughput_mb_s",
    "plaintext_size_bytes",
    "ciphertext_size_bytes",
    "expansion_bytes",
    "expansion_percent",
    "sha256_plaintext",
    "sha256_decrypted",
    "reconstruction_match",
]

BYTE_STATISTICS_FIELDS = [
    "file_id",
    "plaintext_entropy",
    "ciphertext_entropy",
    "plaintext_adjacent_byte_correlation",
    "ciphertext_adjacent_byte_correlation",
    "chi_square_statistic",
    "chi_square_p_value",
]

RECONSTRUCTION_CHECK_FIELDS = [
    "file_id",
    "md5_plaintext",
    "md5_decrypted",
    "sha256_plaintext",
    "sha256_decrypted",
    "exact_match",
]

_TIMED_SERIES = [
    "encryption_time_seconds",
    "decryption_time_seconds",
    "encryption_throughput_mb_s",
    "decryption_throughput_mb_s",
]

_STAT_SUFFIXES = ["mean", "std", "median", "min", "max"]

SUMMARY_STATISTICS_FIELDS = (
    ["file_id", "filename", "nominal_size_label", "plaintext_size_bytes", "n_runs"]
    + [f"{series}_{suffix}" for series in _TIMED_SERIES for suffix in _STAT_SUFFIXES]
)


def save_matrix_csv(matrix: np.ndarray, path: Path) -> Path:
    """Save an integer matrix as comma-separated values (no header)."""
    path = Path(path)
    np.savetxt(path, np.asarray(matrix, dtype=np.int64), fmt="%d", delimiter=",")
    return path


def build_key_material_report(
    params: CipherParameters,
    key: UnimodularKeyMaterial,
    matrix_csv: Path,
    inverse_csv: Path,
) -> dict[str, Any]:
    """Assemble key_material_report.json exactly as specified."""
    return {
        "matrix_dimension": params.matrix_dimension,
        "modulus": params.modulus,
        "logistic_r": params.logistic_r,
        "logistic_x0": params.logistic_x0,
        "warmup_iterations": params.warmup_iterations,
        "quantization_rule": params.quantization_rule,
        "numeric_precision_description": params.numeric_precision,
        "sequence_length_for_matrix_generation": key.sequence_length,
        "matrix_sha256": sha256_file(matrix_csv),
        "inverse_matrix_sha256": sha256_file(inverse_csv),
        "matrix_inverse_verification_result": key.verification,
        "per_file_nonce_or_diversification": False,
        "keystream_reuse_policy": (
            "Baseline experimental implementation: a single keystream derived from the fixed "
            "configuration parameters (logistic_r, logistic_x0, warmup_iterations) is reused "
            "for every file and every run. Per-file nonce diversification is disabled; this is "
            "a documented security limitation, not a production design."
        ),
        "parameter_id": params.parameter_id(),
        "matrix_csv_artifact": Path(matrix_csv).name,
        "inverse_matrix_csv_artifact": Path(inverse_csv).name,
        "matrix_sha256_definition": "SHA-256 over the exact bytes of the referenced CSV artifact.",
    }


def summarize_measurements(
    manifest_rows: Sequence[Mapping[str, Any]], measurement_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate per-file raw measurements into mean/std/median/min/max rows.

    The standard deviation is the sample standard deviation (ddof = 1) and is
    reported as 0.0 when only one run was measured.
    """
    by_file: dict[str, list[Mapping[str, Any]]] = {}
    for row in measurement_rows:
        by_file.setdefault(row["file_id"], []).append(row)
    manifest_by_id = {row["file_id"]: row for row in manifest_rows}

    summary: list[dict[str, Any]] = []
    for file_id in sorted(by_file):
        runs = by_file[file_id]
        manifest = manifest_by_id[file_id]
        entry: dict[str, Any] = {
            "file_id": file_id,
            "filename": manifest["filename"],
            "nominal_size_label": manifest["nominal_size_label"],
            "plaintext_size_bytes": manifest["exact_size_bytes"],
            "n_runs": len(runs),
        }
        for series in _TIMED_SERIES:
            values = [float(run[series]) for run in runs]
            entry[f"{series}_mean"] = statistics.fmean(values)
            entry[f"{series}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry[f"{series}_median"] = statistics.median(values)
            entry[f"{series}_min"] = min(values)
            entry[f"{series}_max"] = max(values)
        summary.append(entry)
    return summary


def _labels(manifest_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"{row['file_id']}\n{row['nominal_size_label']}" for row in manifest_rows]


def _grouped_bar_figure(
    labels: Sequence[str],
    series_a: Sequence[float],
    series_b: Sequence[float],
    name_a: str,
    name_b: str,
    ylabel: str,
    title: str,
    path: Path,
    ylim: tuple[float, float] | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(max(6.0, 0.75 * len(labels)), 4.8))
    positions = np.arange(len(labels))
    width = 0.4
    axis.bar(positions - width / 2, series_a, width, label=name_a)
    axis.bar(positions + width / 2, series_b, width, label=name_b)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_entropy_comparison(
    manifest_rows: Sequence[Mapping[str, Any]], stats_rows: Sequence[Mapping[str, Any]], path: Path
) -> Path:
    stats_by_id = {row["file_id"]: row for row in stats_rows}
    labels = _labels(manifest_rows)
    plain = [stats_by_id[row["file_id"]]["plaintext_entropy"] for row in manifest_rows]
    cipher = [stats_by_id[row["file_id"]]["ciphertext_entropy"] for row in manifest_rows]
    return _grouped_bar_figure(
        labels,
        plain,
        cipher,
        "Plaintext",
        "Ciphertext",
        "Shannon entropy (bits/byte)",
        "Shannon entropy: plaintext vs ciphertext",
        path,
        ylim=(0.0, 8.2),
    )


def plot_correlation_comparison(
    manifest_rows: Sequence[Mapping[str, Any]], stats_rows: Sequence[Mapping[str, Any]], path: Path
) -> Path:
    stats_by_id = {row["file_id"]: row for row in stats_rows}
    labels = _labels(manifest_rows)
    plain = [stats_by_id[row["file_id"]]["plaintext_adjacent_byte_correlation"] for row in manifest_rows]
    cipher = [stats_by_id[row["file_id"]]["ciphertext_adjacent_byte_correlation"] for row in manifest_rows]
    figure, axis = plt.subplots(figsize=(max(6.0, 0.75 * len(labels)), 4.8))
    positions = np.arange(len(labels))
    width = 0.4
    axis.bar(positions - width / 2, plain, width, label="Plaintext")
    axis.bar(positions + width / 2, cipher, width, label="Ciphertext")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Pearson r (adjacent bytes)")
    axis.set_title("Adjacent-byte correlation: plaintext vs ciphertext")
    axis.set_ylim(-1.1, 1.1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_throughput_vs_size(
    summary_rows: Sequence[Mapping[str, Any]], megabyte_bytes: int, path: Path
) -> Path:
    ordered = sorted(summary_rows, key=lambda row: row["plaintext_size_bytes"])
    sizes = [row["plaintext_size_bytes"] for row in ordered]
    enc = [row["encryption_throughput_mb_s_mean"] for row in ordered]
    dec = [row["decryption_throughput_mb_s_mean"] for row in ordered]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(sizes, enc, "o-", label="Encryption")
    axis.plot(sizes, dec, "s-", label="Decryption")
    axis.set_xlabel("Plaintext size (bytes)")
    axis.set_ylabel(f"Throughput (MB/s, 1 MB = {megabyte_bytes} B)")
    axis.set_title("Mean throughput versus file size")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_ciphertext_expansion(
    manifest_rows: Sequence[Mapping[str, Any]],
    expansion_by_file: Mapping[str, float],
    path: Path,
) -> Path:
    labels = _labels(manifest_rows)
    expansion = [float(expansion_by_file.get(row["file_id"], 0.0)) for row in manifest_rows]
    figure, axis = plt.subplots(figsize=(max(6.0, 0.75 * len(labels)), 4.2))
    positions = np.arange(len(labels))
    axis.bar(positions, expansion)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Ciphertext expansion (bytes)")
    axis.set_title("Ciphertext expansion per file (0 = length preserving)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_byte_histogram(plaintext: bytes, ciphertext: bytes, example_name: str, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for axis, data, label in zip(axes, (plaintext, ciphertext), ("Plaintext", "Ciphertext")):
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        axis.bar(np.arange(256), counts, width=1.0)
        axis.set_title(f"{label} — {example_name}")
        axis.set_xlabel("Byte value")
        axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def generate_figures(
    figures_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    stats_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    expansion_by_file: Mapping[str, float],
    example: tuple[Path, Path, str] | None,
    megabyte_bytes: int,
) -> list[Path]:
    """Render every experiment figure; returns the written paths."""
    figures_dir = Path(figures_dir)
    written: list[Path] = [
        plot_entropy_comparison(
            manifest_rows, stats_rows, figures_dir / "fig_entropy_comparison.png"
        ),
        plot_correlation_comparison(
            manifest_rows, stats_rows, figures_dir / "fig_adjacent_byte_correlation.png"
        ),
        plot_throughput_vs_size(
            summary_rows, megabyte_bytes, figures_dir / "fig_throughput_vs_file_size.png"
        ),
        plot_ciphertext_expansion(
            manifest_rows, expansion_by_file, figures_dir / "fig_ciphertext_expansion.png"
        ),
    ]
    if example is None:
        logger.warning(
            "histogram_example_filename not found among inputs; byte-frequency histogram figure skipped"
        )
    else:
        plaintext_path, ciphertext_path, example_name = example
        written.append(
            plot_byte_histogram(
                plaintext_path.read_bytes(),
                ciphertext_path.read_bytes(),
                example_name,
                figures_dir / "fig_byte_frequency_histogram.png",
            )
        )
    for path in written:
        logger.info("Figure written: %s", path)
    return written
