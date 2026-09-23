"""Unit tests for the deterministic unimodular key-matrix generation."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_crypto.config import CipherParameters
from hybrid_crypto.matrix_generator import (
    UnimodularKeyMaterial,
    generate_unimodular_matrix,
    required_sequence_length,
)
from hybrid_crypto.modular_algebra import modular_matrix_inverse, verify_modular_inverse

DEFAULT_PARAMS = CipherParameters(
    matrix_dimension=16,
    block_size=256,
    modulus=256,
    logistic_r=3.923,
    logistic_x0=0.73911,
    warmup_iterations=1000,
    quantization_rule="k_i = floor((x_i * 1000) mod 256), stored as uint8",
    numeric_precision="IEEE-754 binary64 (Python float)",
)


def test_required_sequence_length():
    assert required_sequence_length(16) == 16 * 15 // 2 + 15  # 120 + 15 = 135
    assert required_sequence_length(4) == 4 * 3 // 2 + 3  # 6 + 3 = 9
    assert required_sequence_length(2) == 2
    with pytest.raises(ValueError):
        required_sequence_length(1)


def test_manual_construction_n3():
    sequence = np.array([10, 20, 30, 5, 7], dtype=np.uint8)
    matrix = generate_unimodular_matrix(sequence, 3)
    expected = np.array(
        [
            [1, 10, 20],                 # identity row 0 with filled upper triangle
            [5, 51, 130],                # e_1 + 5 * row_0
            [7, 70, 141],                # e_2 + 7 * row_0
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(matrix, expected)


def test_manual_construction_n4_row_order():
    sequence = np.arange(9, dtype=np.uint8)
    matrix = generate_unimodular_matrix(sequence, 4)
    expected = np.array(
        [
            [1, 0, 1, 2],                # upper triangle consumed in row-major order
            [6, 1, 9, 16],               # e_1 + 6 * row_0
            [7, 0, 8, 19],               # e_2 + 7 * row_0
            [8, 0, 8, 17],               # e_3 + 8 * row_0
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(matrix, expected)


def test_wrong_sequence_length_raises():
    with pytest.raises(ValueError):
        generate_unimodular_matrix(np.zeros(10, dtype=np.uint8), 16)


def test_full_pipeline_inverse_verified():
    key = UnimodularKeyMaterial(DEFAULT_PARAMS)
    assert key.matrix.shape == (16, 16)
    assert key.inverse.shape == (16, 16)
    assert key.matrix.min() >= 0 and key.matrix.max() <= 255
    forward, backward = verify_modular_inverse(key.matrix, key.inverse, 256)
    assert forward and backward
    assert key.verification["verified"] is True


def test_deterministic_generation():
    first = UnimodularKeyMaterial(DEFAULT_PARAMS)
    second = UnimodularKeyMaterial(DEFAULT_PARAMS)
    np.testing.assert_array_equal(first.sequence, second.sequence)
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.inverse, second.inverse)


def test_lower_rows_follow_elementary_row_addition():
    key = UnimodularKeyMaterial(DEFAULT_PARAMS)
    n = DEFAULT_PARAMS.matrix_dimension
    # Step 3 fills the strict upper triangle first, so the row additions in
    # step 4 act on the upper-triangular intermediate, not on the identity.
    intermediate = np.eye(n, dtype=np.int64)
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            intermediate[i, j] = key.sequence[index]
            index += 1
    for row in range(1, n):
        multiplier = int(key.sequence[index])
        index += 1
        expected = (intermediate[row, :] + multiplier * intermediate[0, :]) % 256
        np.testing.assert_array_equal(key.matrix[row, :], expected)


def test_generated_matrix_is_invertible_for_every_row_seed():
    # The construction must stay invertible even when every consumed byte is
    # an awkward value (e.g. all 255): R_i <- R_i + a * R_0 preserves det = 1.
    n = 4
    sequence = np.full(required_sequence_length(n), 255, dtype=np.uint8)
    matrix = generate_unimodular_matrix(sequence, n)
    inverse = modular_matrix_inverse(matrix, 256)
    forward, backward = verify_modular_inverse(matrix, inverse, 256)
    assert forward and backward
