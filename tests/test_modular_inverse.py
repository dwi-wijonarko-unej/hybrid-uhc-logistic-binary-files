"""Unit tests for exact modular linear algebra over Z/256Z."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hybrid_crypto.modular_algebra import (
    ModularInverseError,
    extended_gcd,
    modular_inverse,
    modular_matrix_inverse,
    verify_modular_inverse,
)


@pytest.mark.parametrize("a,b", [(240, 46), (256, 171), (7, 0), (0, 7), (1, 1), (128, 256), (99, 78)])
def test_extended_gcd_bezout_identity(a, b):
    g, x, y = extended_gcd(a, b)
    assert g == math.gcd(a, b)
    assert a * x + b * y == g


def test_modular_inverse_known_values():
    assert modular_inverse(3, 256) == 171  # 3 * 171 = 513 = 2*256 + 1
    assert modular_inverse(171, 256) == 3
    assert modular_inverse(1, 256) == 1
    assert modular_inverse(255, 256) == 255  # -1 is its own inverse


def test_modular_inverse_agrees_with_bruteforce():
    for value in range(1, 256, 2):  # odd values are exactly the units mod 256
        inverse = modular_inverse(value, 256)
        assert (value * inverse) % 256 == 1


@pytest.mark.parametrize("value", [0, 2, 4, 8, 64, 128, 254])
def test_modular_inverse_non_unit_raises(value):
    with pytest.raises(ModularInverseError):
        modular_inverse(value, 256)


def test_identity_matrix_is_its_own_inverse():
    identity = np.eye(5, dtype=np.int64)
    np.testing.assert_array_equal(modular_matrix_inverse(identity, 256), identity)


def test_pivot_requires_row_swap():
    # Column 0 starts with 2 (non-unit); the algorithm must swap with row 1.
    matrix = np.array([[2, 1], [1, 1]], dtype=np.int64)
    inverse = modular_matrix_inverse(matrix, 256)
    expected = np.array([[1, 255], [255, 2]], dtype=np.int64)
    np.testing.assert_array_equal(inverse, expected)
    forward, backward = verify_modular_inverse(matrix, inverse, 256)
    assert forward and backward


def test_singular_matrix_raises():
    singular = np.ones((3, 3), dtype=np.int64)  # rank 1, even determinant
    with pytest.raises(ModularInverseError):
        modular_matrix_inverse(singular, 256)


def test_even_determinant_matrix_raises():
    matrix = np.diag([1, 1, 2, 1]).astype(np.int64)  # det = 2, not a unit mod 256
    with pytest.raises(ModularInverseError):
        modular_matrix_inverse(matrix, 256)


def test_diagonal_unit_matrix_inverse():
    matrix = np.diag([1, 3, 5, 7]).astype(np.int64)
    inverse = modular_matrix_inverse(matrix, 256)
    forward, backward = verify_modular_inverse(matrix, inverse, 256)
    assert forward and backward
    for value, reciprocal in zip([1, 3, 5, 7], np.diag(inverse)):
        assert (value * int(reciprocal)) % 256 == 1


def test_unimodular_generator_output_invertible():
    from hybrid_crypto.config import CipherParameters
    from hybrid_crypto.matrix_generator import UnimodularKeyMaterial

    params = CipherParameters(
        matrix_dimension=16, block_size=256, modulus=256,
        logistic_r=3.923, logistic_x0=0.73911, warmup_iterations=1000,
        quantization_rule="k_i = floor((x_i * 1000) mod 256), stored as uint8",
        numeric_precision="IEEE-754 binary64 (Python float)",
    )
    key = UnimodularKeyMaterial(params)
    forward, backward = verify_modular_inverse(key.matrix, key.inverse, 256)
    assert forward and backward


def test_non_square_matrix_raises():
    with pytest.raises(ValueError):
        modular_matrix_inverse(np.zeros((3, 4), dtype=np.int64), 256)
