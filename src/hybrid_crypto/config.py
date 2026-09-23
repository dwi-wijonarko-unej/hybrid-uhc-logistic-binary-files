"""Configuration loading and validation.

Every experimental parameter comes from the YAML configuration file. The
loader deliberately has **no hidden defaults**: a missing key raises
:class:`ConfigError` instead of silently substituting a value. Relative
paths inside the configuration resolve against the project root (the parent
directory of the folder that contains the YAML file, i.e. ``<root>/config``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the configuration file is missing required values."""


def _require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in config or not isinstance(config[section], dict):
        raise ConfigError(f"Missing required configuration section: '{section}'")
    return config[section]


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required configuration key: '{section}.{key}'")
    return mapping[key]


@dataclass(frozen=True)
class CipherParameters:
    """Parameters of the hybrid unimodular Hill + logistic XOR scheme."""

    matrix_dimension: int
    block_size: int
    modulus: int
    logistic_r: float
    logistic_x0: float
    warmup_iterations: int
    quantization_rule: str
    numeric_precision: str

    def validate(self) -> None:
        if self.matrix_dimension < 2:
            raise ConfigError(f"cipher.matrix_dimension must be >= 2, got {self.matrix_dimension}")
        if self.block_size != self.matrix_dimension**2:
            raise ConfigError(
                "cipher.block_size must equal cipher.matrix_dimension ** 2 "
                f"({self.matrix_dimension**2}), got {self.block_size}"
            )
        if self.modulus != 256:
            raise ConfigError(f"cipher.modulus must be 256 (byte algebra), got {self.modulus}")
        if not 0.0 < self.logistic_x0 < 1.0:
            raise ConfigError(f"cipher.logistic_x0 must satisfy 0 < x0 < 1, got {self.logistic_x0}")
        if not 0.0 < self.logistic_r <= 4.0:
            raise ConfigError(f"cipher.logistic_r must satisfy 0 < r <= 4, got {self.logistic_r}")
        if self.warmup_iterations < 0:
            raise ConfigError(f"cipher.warmup_iterations must be >= 0, got {self.warmup_iterations}")

    @property
    def matrix_sequence_length(self) -> int:
        """Length of the logistic byte sequence consumed by key generation."""
        n = self.matrix_dimension
        return n * (n - 1) // 2 + (n - 1)

    def parameter_id(self) -> str:
        """Short deterministic identifier of the numeric parameter set."""
        payload = {
            "matrix_dimension": self.matrix_dimension,
            "block_size": self.block_size,
            "modulus": self.modulus,
            "logistic_r": self.logistic_r,
            "logistic_x0": self.logistic_x0,
            "warmup_iterations": self.warmup_iterations,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class BenchmarkParameters:
    """Timing protocol parameters."""

    timing_warmup_runs: int
    timing_repetitions: int
    include_file_io_in_timing: bool
    megabyte_bytes: int

    def validate(self) -> None:
        if self.timing_warmup_runs < 0:
            raise ConfigError(f"benchmark.timing_warmup_runs must be >= 0, got {self.timing_warmup_runs}")
        if self.timing_repetitions < 1:
            raise ConfigError(f"benchmark.timing_repetitions must be >= 1, got {self.timing_repetitions}")
        if self.megabyte_bytes < 1:
            raise ConfigError(f"benchmark.megabyte_bytes must be >= 1, got {self.megabyte_bytes}")


@dataclass(frozen=True)
class SyntheticFilesConfig:
    """Parameters of the deterministic synthetic placeholder generator."""

    output_dir: Path
    manifest_dir: Path
    seed: int
    file_types: tuple[str, ...]
    sizes_bytes: tuple[int, ...]
    size_labels: tuple[str, ...]

    def validate(self) -> None:
        allowed = {"docx", "pdf", "jpeg"}
        unknown = set(self.file_types) - allowed
        if unknown:
            raise ConfigError(f"synthetic_files.file_types must be a subset of {sorted(allowed)}, got {sorted(unknown)}")
        if not self.file_types:
            raise ConfigError("synthetic_files.file_types must not be empty")
        if len(self.sizes_bytes) != len(self.size_labels):
            raise ConfigError(
                "synthetic_files.sizes_bytes and synthetic_files.size_labels must have the same length "
                f"({len(self.sizes_bytes)} != {len(self.size_labels)})"
            )
        for size in self.sizes_bytes:
            if size < 1024:
                raise ConfigError(f"synthetic_files.sizes_bytes entries must be >= 1024, got {size}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully validated experiment configuration."""

    name: str
    input_dir: Path
    output_dir: Path
    generate_synthetic_files_if_missing: bool
    histogram_example_filename: str
    cipher: CipherParameters
    benchmark: BenchmarkParameters
    synthetic: SyntheticFilesConfig
    source_path: Path | None = None

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("experiment.name must be a non-empty string")
        self.cipher.validate()
        self.benchmark.validate()
        self.synthetic.validate()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, parse, and validate the experiment configuration YAML."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")

    project_root = config_path.resolve().parent.parent

    def _resolve_path(value: Any) -> Path:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else project_root / candidate

    experiment = _require_section(raw, "experiment")
    cipher = _require_section(raw, "cipher")
    benchmark = _require_section(raw, "benchmark")
    synthetic = _require_section(raw, "synthetic_files")

    config = ExperimentConfig(
        name=str(_require(experiment, "name", "experiment")),
        input_dir=_resolve_path(_require(experiment, "input_dir", "experiment")),
        output_dir=_resolve_path(_require(experiment, "output_dir", "experiment")),
        generate_synthetic_files_if_missing=bool(_require(experiment, "generate_synthetic_files_if_missing", "experiment")),
        histogram_example_filename=str(_require(experiment, "histogram_example_filename", "experiment")),
        source_path=config_path.resolve(),
        cipher=CipherParameters(
            matrix_dimension=int(_require(cipher, "matrix_dimension", "cipher")),
            block_size=int(_require(cipher, "block_size", "cipher")),
            modulus=int(_require(cipher, "modulus", "cipher")),
            logistic_r=float(_require(cipher, "logistic_r", "cipher")),
            logistic_x0=float(_require(cipher, "logistic_x0", "cipher")),
            warmup_iterations=int(_require(cipher, "warmup_iterations", "cipher")),
            quantization_rule=str(_require(cipher, "quantization_rule", "cipher")),
            numeric_precision=str(_require(cipher, "numeric_precision", "cipher")),
        ),
        benchmark=BenchmarkParameters(
            timing_warmup_runs=int(_require(benchmark, "timing_warmup_runs", "benchmark")),
            timing_repetitions=int(_require(benchmark, "timing_repetitions", "benchmark")),
            include_file_io_in_timing=bool(_require(benchmark, "include_file_io_in_timing", "benchmark")),
            megabyte_bytes=int(_require(benchmark, "megabyte_bytes", "benchmark")),
        ),
        synthetic=SyntheticFilesConfig(
            output_dir=_resolve_path(_require(synthetic, "output_dir", "synthetic_files")),
            manifest_dir=_resolve_path(_require(synthetic, "manifest_dir", "synthetic_files")),
            seed=int(_require(synthetic, "seed", "synthetic_files")),
            file_types=tuple(str(item) for item in _require(synthetic, "file_types", "synthetic_files")),
            sizes_bytes=tuple(int(item) for item in _require(synthetic, "sizes_bytes", "synthetic_files")),
            size_labels=tuple(str(item) for item in _require(synthetic, "size_labels", "synthetic_files")),
        ),
    )
    config.validate()
    return config
