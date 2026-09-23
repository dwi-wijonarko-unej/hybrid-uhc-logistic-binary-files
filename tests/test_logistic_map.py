"""Unit tests for the stateful logistic-map keystream."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hybrid_crypto.logistic_map import LogisticMapKeystream, generate_logistic_byte_sequence

R = 3.923
X0 = 0.73911
WARMUP = 1000


def _stream(warmup: int = WARMUP, x0: float = X0, r: float = R) -> LogisticMapKeystream:
    return LogisticMapKeystream(r=r, x0=x0, warmup_iterations=warmup)


def test_deterministic_for_identical_parameters():
    first = _stream().next_bytes(1024)
    second = _stream().next_bytes(1024)
    np.testing.assert_array_equal(first, second)


def test_one_shot_wrapper_matches_stateful_stream():
    one_shot = generate_logistic_byte_sequence(R, X0, WARMUP, 300)
    stateful = _stream().next_bytes(300)
    np.testing.assert_array_equal(one_shot, stateful)


def test_stream_is_stateful_and_continuous():
    stream = _stream()
    first = stream.next_bytes(64)
    second = stream.next_bytes(64)
    combined = _stream().next_bytes(128)
    np.testing.assert_array_equal(np.concatenate([first, second]), combined)
    assert stream.bytes_emitted == 128


def test_warmup_length_changes_the_stream():
    default_warmup = _stream(WARMUP).next_bytes(64)
    one_less = _stream(WARMUP - 1).next_bytes(64)
    assert not np.array_equal(default_warmup, one_less)


def test_quantized_values_are_valid_uint8():
    values = _stream().next_bytes(5000)
    assert values.dtype == np.uint8
    assert values.min() >= 0
    assert values.max() <= 255


def test_quantization_formula_with_zero_warmup():
    r, x0 = 3.9, 0.5
    stream = LogisticMapKeystream(r=r, x0=x0, warmup_iterations=0)
    produced = int(stream.next_bytes(1)[0])
    x = r * x0 * (1 - x0)
    expected = math.floor((x * 1000.0) % 256.0)
    assert produced == expected


@pytest.mark.parametrize("r,x0,warmup", [
    (3.923, 0.0, 1000),
    (3.923, 1.0, 1000),
    (3.923, -0.5, 1000),
    (0.0, 0.5, 1000),
    (4.5, 0.5, 1000),
    (3.923, 0.5, -1),
])
def test_invalid_parameters_raise(r, x0, warmup):
    with pytest.raises(ValueError):
        LogisticMapKeystream(r=r, x0=x0, warmup_iterations=warmup)


def test_negative_count_raises():
    with pytest.raises(ValueError):
        _stream().next_bytes(-1)
