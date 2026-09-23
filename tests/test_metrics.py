"""Unit tests for the statistical metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hybrid_crypto import metrics


def test_entropy_uniform_distribution_is_eight_bits():
    data = np.arange(256, dtype=np.uint8).tobytes() * 4
    assert metrics.shannon_entropy(data) == pytest.approx(8.0, abs=1e-12)


def test_entropy_constant_data_is_zero():
    assert metrics.shannon_entropy(b"\x00" * 1000) == 0.0


def test_entropy_two_equally_likely_symbols_is_one_bit():
    assert metrics.shannon_entropy(b"\x00\x01" * 500) == pytest.approx(1.0, abs=1e-12)


def test_entropy_empty_data_raises():
    with pytest.raises(ValueError):
        metrics.shannon_entropy(b"")


def test_adjacent_correlation_strictly_increasing_is_plus_one():
    data = np.arange(200, dtype=np.uint8).tobytes()
    assert metrics.adjacent_byte_correlation(data) == pytest.approx(1.0, abs=1e-12)


def test_adjacent_correlation_alternating_is_minus_one():
    data = np.tile(np.array([0, 255], dtype=np.uint8), 500).tobytes()
    assert metrics.adjacent_byte_correlation(data) == pytest.approx(-1.0, abs=1e-12)


def test_adjacent_correlation_constant_data_is_nan():
    assert math.isnan(metrics.adjacent_byte_correlation(b"\x07" * 100))


def test_adjacent_correlation_requires_two_bytes():
    with pytest.raises(ValueError):
        metrics.adjacent_byte_correlation(b"\x01")


def test_chi_square_perfectly_uniform_data():
    data = np.arange(256, dtype=np.uint8).tobytes() * 10
    statistic, p_value = metrics.chi_square_uniform(data)
    assert statistic == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0, abs=1e-12)


def test_chi_square_constant_data_is_rejected():
    statistic, p_value = metrics.chi_square_uniform(b"\x42" * 1000)
    assert statistic > 1e4
    assert p_value < 1e-12


def test_chi_square_empty_data_raises():
    with pytest.raises(ValueError):
        metrics.chi_square_uniform(b"")


def test_throughput_one_mebibyte_in_half_second():
    assert metrics.throughput_mb_s(1024 * 1024, 0.5) == pytest.approx(2.0)


def test_throughput_respects_custom_megabyte_convention():
    assert metrics.throughput_mb_s(2_000_000, 1.0, megabyte_bytes=1_000_000) == pytest.approx(2.0)


def test_throughput_rejects_non_positive_elapsed_time():
    with pytest.raises(ValueError):
        metrics.throughput_mb_s(1024, 0.0)


def test_expansion_zero_when_lengths_match():
    assert metrics.ciphertext_expansion(1000, 1000) == (0, 0.0)


def test_expansion_nonzero_growth():
    assert metrics.ciphertext_expansion(1000, 1010) == (10, pytest.approx(1.0))


def test_expansion_negative_for_shrinking_ciphertext():
    expansion, percent = metrics.ciphertext_expansion(500, 250)
    assert expansion == -250
    assert percent == pytest.approx(-50.0)


def test_expansion_empty_to_empty_is_zero():
    assert metrics.ciphertext_expansion(0, 0) == (0, 0.0)


def test_byte_counts_shape_and_total():
    data = b"\x00\xff\x00"
    counts = metrics.byte_counts(data)
    assert counts.shape == (256,)
    assert counts[0] == 2 and counts[255] == 1
    assert counts.sum() == 3
