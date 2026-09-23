"""Statistical metrics for the experiment.

Definitions (fixed by the experiment specification):

* Shannon entropy of the byte distribution: ``H = -sum(p_i * log2(p_i))``.
* Pearson correlation between adjacent bytes: ``corr(data[:-1], data[1:])``.
* Chi-square statistic against the uniform byte distribution with expected
  count ``N / 256`` per byte value and 255 degrees of freedom.
* Throughput: ``file_size_megabytes / elapsed_seconds`` where one megabyte
  is pinned in the configuration (default 1 MB = 1,048,576 bytes).
* Ciphertext expansion: ``ciphertext_size - plaintext_size`` in bytes and
  as a percentage of the plaintext size.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

#: Default megabyte convention used when the caller does not override it.
#: The benchmark configuration pins the value used in reports.
MEGABYTE_BYTES = 1024 * 1024

_BYTE_VALUES = 256


def as_byte_array(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Coerce input to a flat uint8 NumPy array without copying when possible."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    array = np.asarray(data)
    if array.dtype != np.uint8:
        raise TypeError(f"expected uint8 data, got dtype {array.dtype}")
    return array.reshape(-1)


def byte_counts(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Return the 256-element histogram of byte values."""
    array = as_byte_array(data)
    return np.bincount(array, minlength=_BYTE_VALUES)


def shannon_entropy(data: bytes | bytearray | memoryview | np.ndarray) -> float:
    """Shannon entropy of the byte distribution in bits per byte (0..8)."""
    array = as_byte_array(data)
    if array.size == 0:
        raise ValueError("entropy is undefined for empty data")
    probabilities = byte_counts(array) / array.size
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def adjacent_byte_correlation(data: bytes | bytearray | memoryview | np.ndarray) -> float:
    """Pearson correlation coefficient between adjacent bytes.

    Returns ``nan`` when either side of the pair sequence has zero variance
    (constant data); requires at least two bytes.
    """
    array = as_byte_array(data)
    if array.size < 2:
        raise ValueError("adjacent-byte correlation requires at least two bytes")
    left = array[:-1].astype(np.float64)
    right = array[1:].astype(np.float64)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def chi_square_uniform(data: bytes | bytearray | memoryview | np.ndarray) -> tuple[float, float]:
    """Chi-square statistic and p-value against the uniform byte distribution.

    Expected count per byte value is ``N / 256`` with 255 degrees of freedom.
    """
    array = as_byte_array(data)
    if array.size == 0:
        raise ValueError("chi-square statistic is undefined for empty data")
    observed = byte_counts(array).astype(np.float64)
    expected = array.size / _BYTE_VALUES
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(chi2.sf(statistic, df=_BYTE_VALUES - 1))
    return statistic, p_value


def throughput_mb_s(
    size_bytes: int, elapsed_seconds: float, megabyte_bytes: int = MEGABYTE_BYTES
) -> float:
    """Throughput in megabytes per second for the chosen megabyte convention."""
    if elapsed_seconds <= 0.0:
        raise ValueError(f"elapsed_seconds must be > 0, got {elapsed_seconds}")
    return (size_bytes / megabyte_bytes) / elapsed_seconds


def ciphertext_expansion(plaintext_size: int, ciphertext_size: int) -> tuple[int, float]:
    """Return ``(expansion_bytes, expansion_percent)`` for a ciphertext.

    ``expansion_bytes = ciphertext_size - plaintext_size``. The percentage is
    relative to the plaintext size; for an empty plaintext that maps to the
    empty ciphertext it is defined as 0.0.
    """
    expansion = int(ciphertext_size) - int(plaintext_size)
    if plaintext_size > 0:
        return expansion, 100.0 * expansion / plaintext_size
    return expansion, 0.0 if expansion == 0 else float("inf")
