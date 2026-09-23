"""Deterministic unimodular key-matrix generation from the logistic map.

Construction (fixed by the experiment specification):

1. Generate a logistic byte sequence of length ``n*(n-1)/2 + (n-1)``.
2. Start from the ``n x n`` integer identity matrix.
3. Fill the strict upper triangle from the sequence, in row-major order.
4. Fill the lower part with determinant-preserving elementary row additions:
   for ``row_i`` in ``1..n-1``:
   ``M[row_i, :] = (M[row_i, :] + multiplier * M[0, :]) mod 256``.

Starting from the identity and only ever applying ``R_i <- R_i + a * R_0``
keeps the integer determinant equal to 1, so ``det(M) = 1 = 1 mod 256`` and
M is always invertible modulo 256. The inverse is computed exactly with
Gauss-Jordan over Z/256Z (see :mod:`hybrid_crypto.modular_algebra`) and both
product directions are verified against the identity.
"""

from __future__ import annotations

import numpy as np

from .config import CipherParameters
from .logistic_map import LogisticMapKeystream
from .modular_algebra import ModularInverseError, modular_matrix_inverse, verify_modular_inverse


def required_sequence_length(matrix_dimension: int) -> int:
    """Number of logistic bytes consumed by key generation: n(n-1)/2 + (n-1)."""
    if matrix_dimension < 2:
        raise ValueError(f"matrix_dimension must be >= 2, got {matrix_dimension}")
    n = matrix_dimension
    return n * (n - 1) // 2 + (n - 1)


def generate_unimodular_matrix(
    sequence: np.ndarray, matrix_dimension: int, modulus: int = 256
) -> np.ndarray:
    """Build the unimodular key matrix from a logistic byte sequence.

    The result is an ``int64`` matrix with entries in ``[0, modulus)``.
    """
    expected = required_sequence_length(matrix_dimension)
    seq = np.asarray(sequence)
    if seq.shape != (expected,):
        raise ValueError(f"sequence must have shape ({expected},), got {seq.shape}")

    matrix = np.eye(matrix_dimension, dtype=np.int64)
    index = 0
    # Strict upper triangle, row-major order.
    for i in range(matrix_dimension):
        for j in range(i + 1, matrix_dimension):
            matrix[i, j] = int(seq[index])
            index += 1
    # Lower part via determinant-preserving elementary row additions.
    for row in range(1, matrix_dimension):
        multiplier = int(seq[index])
        matrix[row, :] = (matrix[row, :] + multiplier * matrix[0, :]) % modulus
        index += 1
    return matrix


class UnimodularKeyMaterial:
    """Key matrix, its modular inverse, and the verification record.

    The same :class:`CipherParameters` always yield byte-identical key
    material on every platform: the logistic map runs in IEEE-754 binary64
    and the linear algebra is exact integer arithmetic.
    """

    def __init__(self, params: CipherParameters) -> None:
        params.validate()
        self.params = params
        self.sequence_length = params.matrix_sequence_length
        self.sequence = LogisticMapKeystream(
            params.logistic_r, params.logistic_x0, params.warmup_iterations
        ).next_bytes(self.sequence_length)
        self.matrix = generate_unimodular_matrix(self.sequence, params.matrix_dimension, params.modulus)
        self.inverse = modular_matrix_inverse(self.matrix, params.modulus)
        forward, backward = verify_modular_inverse(self.matrix, self.inverse, params.modulus)
        self.verification = {
            "forward_product_is_identity": forward,
            "backward_product_is_identity": backward,
            "verified": forward and backward,
        }
        if not self.verification["verified"]:
            raise ModularInverseError(  # pragma: no cover - defensive guard
                "Generated key matrix failed modular inverse verification"
            )
