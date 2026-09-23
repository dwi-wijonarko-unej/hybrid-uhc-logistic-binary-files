"""Exact modular linear algebra over Z/256Z.

``numpy.linalg.inv`` is intentionally **not** used anywhere in this project:
it computes a floating-point inverse, which is meaningless for modular
arithmetic. Instead, matrices are inverted with a Gauss-Jordan elimination
that is valid for the composite modulus 256:

* a pivot may only be selected when ``gcd(pivot, 256) == 1`` (a unit);
* if the current column entry is not a unit, rows are swapped to find one;
* if no unit pivot exists in a column where one is required, a clear error
  is raised (the matrix is not invertible modulo 256);
* the reciprocal of the pivot is computed with the extended Euclidean
  algorithm.

All arithmetic runs on signed 64-bit integers (or Python ints) and the
modulo is applied explicitly after every update, so there is no implicit
uint8 wrap-around before reduction.
"""

from __future__ import annotations

from math import gcd

import numpy as np


class ModularInverseError(ValueError):
    """Raised when a value or matrix has no inverse modulo the modulus."""


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``g = gcd(a, b)`` and ``a*x + b*y == g``."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s, old_t


def modular_inverse(value: int, modulus: int) -> int:
    """Return the multiplicative inverse of ``value`` modulo ``modulus``."""
    if modulus < 2:
        raise ValueError(f"modulus must be >= 2, got {modulus}")
    g, x, _ = extended_gcd(value % modulus, modulus)
    if g != 1:
        raise ModularInverseError(f"{value} is not invertible modulo {modulus} (gcd = {g})")
    return x % modulus


def modular_matrix_inverse(matrix: np.ndarray, modulus: int) -> np.ndarray:
    """Invert a square integer matrix modulo ``modulus`` via Gauss-Jordan.

    Uses unit pivots and row swaps only, which is correct for composite
    moduli such as 256. Raises :class:`ModularInverseError` when the matrix
    is singular modulo ``modulus``.
    """
    a = np.asarray(matrix, dtype=np.int64) % modulus
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got shape {a.shape}")
    n = a.shape[0]
    augmented = np.concatenate([a, np.eye(n, dtype=np.int64)], axis=1)

    for column in range(n):
        pivot_row = None
        for row in range(column, n):
            if gcd(int(augmented[row, column]), modulus) == 1:
                pivot_row = row
                break
        if pivot_row is None:
            raise ModularInverseError(
                f"No unit pivot available in column {column}; matrix is not invertible modulo {modulus}"
            )
        if pivot_row != column:
            augmented[[column, pivot_row]] = augmented[[pivot_row, column]]
        pivot_reciprocal = modular_inverse(int(augmented[column, column]), modulus)
        augmented[column, :] = (augmented[column, :] * pivot_reciprocal) % modulus
        for row in range(n):
            if row != column and augmented[row, column] != 0:
                factor = int(augmented[row, column])
                augmented[row, :] = (augmented[row, :] - factor * augmented[column, :]) % modulus

    return augmented[:, n:] % modulus


def verify_modular_inverse(matrix: np.ndarray, inverse: np.ndarray, modulus: int) -> tuple[bool, bool]:
    """Check ``(M @ Minv) % m == I`` and ``(Minv @ M) % m == I`` exactly."""
    m = np.asarray(matrix, dtype=np.int64)
    inv = np.asarray(inverse, dtype=np.int64)
    identity = np.eye(m.shape[0], dtype=np.int64)
    forward = bool(np.array_equal((m @ inv) % modulus, identity))
    backward = bool(np.array_equal((inv @ m) % modulus, identity))
    return forward, backward
