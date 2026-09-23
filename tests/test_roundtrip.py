"""End-to-end roundtrip tests for the hybrid Hill + logistic XOR cipher."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_crypto.cipher import HybridHillXORCipher
from hybrid_crypto.config import CipherParameters
from hybrid_crypto.logistic_map import LogisticMapKeystream
from hybrid_crypto.matrix_generator import UnimodularKeyMaterial

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


def _build_cipher(x0: float = DEFAULT_PARAMS.logistic_x0) -> HybridHillXORCipher:
    params = CipherParameters(
        matrix_dimension=DEFAULT_PARAMS.matrix_dimension,
        block_size=DEFAULT_PARAMS.block_size,
        modulus=DEFAULT_PARAMS.modulus,
        logistic_r=DEFAULT_PARAMS.logistic_r,
        logistic_x0=x0,
        warmup_iterations=DEFAULT_PARAMS.warmup_iterations,
        quantization_rule=DEFAULT_PARAMS.quantization_rule,
        numeric_precision=DEFAULT_PARAMS.numeric_precision,
    )
    key = UnimodularKeyMaterial(params)
    return HybridHillXORCipher(
        key_matrix=key.matrix,
        inverse_matrix=key.inverse,
        logistic_r=params.logistic_r,
        logistic_x0=params.logistic_x0,
        warmup_iterations=params.warmup_iterations,
        block_size=params.block_size,
        modulus=params.modulus,
    )


def _plaintext(length: int) -> bytes:
    """Deterministic, high-entropy test payload from an independent orbit."""
    return LogisticMapKeystream(r=3.99, x0=0.12345, warmup_iterations=10).next_bytes(length).tobytes()


@pytest.fixture(scope="module")
def cipher() -> HybridHillXORCipher:
    return _build_cipher()


@pytest.mark.parametrize("size", [0, 1, 15, 255, 256, 257, 511, 512, 1000, 4096, 4097])
def test_roundtrip_is_exact_for_every_size_class(cipher, size):
    plaintext = _plaintext(size)
    decrypted = cipher.decrypt(cipher.encrypt(plaintext))
    assert decrypted == plaintext


def test_ciphertext_length_equals_plaintext_length(cipher):
    for size in (0, 1, 255, 256, 257, 4097):
        plaintext = _plaintext(size)
        assert len(cipher.encrypt(plaintext)) == len(plaintext)


def test_ciphertext_differs_from_plaintext(cipher):
    plaintext = _plaintext(4096)
    assert cipher.encrypt(plaintext) != plaintext


def test_empty_file_roundtrip(cipher):
    assert cipher.encrypt(b"") == b""
    assert cipher.decrypt(b"") == b""


def test_tail_only_path_is_pure_xor(cipher):
    # A 100-byte plaintext never touches the Hill matrix; encryption must be
    # the keystream XOR, which also proves the tail consumes the orbit start.
    plaintext = _plaintext(100)
    keystream = LogisticMapKeystream(
        DEFAULT_PARAMS.logistic_r, DEFAULT_PARAMS.logistic_x0, DEFAULT_PARAMS.warmup_iterations
    ).next_bytes(100)
    xor = bytes(p ^ int(k) for p, k in zip(plaintext, keystream))
    assert cipher.encrypt(plaintext) == xor


def test_keystream_prefix_reuses_matrix_sequence_bytes(cipher):
    # Model A baseline property (see README limitations): the key-matrix
    # generator and the file-keystream generator are initialized
    # independently with the same parameters, so the first 135 keystream
    # bytes equal the matrix derivation sequence. Pinned here deliberately:
    # this overlap is an intentional, documented security limitation.
    key = UnimodularKeyMaterial(DEFAULT_PARAMS)
    keystream = LogisticMapKeystream(
        DEFAULT_PARAMS.logistic_r, DEFAULT_PARAMS.logistic_x0, DEFAULT_PARAMS.warmup_iterations
    ).next_bytes(135)
    np.testing.assert_array_equal(keystream, key.sequence)
    # And the cipher really XORs with those same prefix bytes:
    tail_only = _plaintext(135)
    expected = bytes(p ^ int(k) for p, k in zip(tail_only, key.sequence))
    assert cipher.encrypt(tail_only) == expected


def test_wrong_keystream_fails_to_reconstruct():
    right = _build_cipher(DEFAULT_PARAMS.logistic_x0)
    wrong = _build_cipher(0.73912)
    plaintext = _plaintext(2048)
    ciphertext = right.encrypt(plaintext)
    assert wrong.decrypt(ciphertext) != plaintext


def test_encryption_is_deterministic(cipher):
    plaintext = _plaintext(1024)
    assert cipher.encrypt(plaintext) == cipher.encrypt(plaintext)
