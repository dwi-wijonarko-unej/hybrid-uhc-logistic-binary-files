"""Hybrid unimodular Hill + logistic-XOR cipher for raw binary files.

Encryption of a byte string ``P`` of length ``L`` (``block_size = n*n``):

1. Split ``P`` into full blocks of ``block_size`` bytes plus a residual tail.
2. For every full block: reshape row-major to an ``n x n`` matrix ``P_i``,
   compute ``H_i = (M @ P_i) mod 256``, serialize ``H_i`` back to
   ``block_size`` bytes row-major, and XOR with the next ``block_size``
   keystream bytes.
3. The residual tail (fewer than ``block_size`` bytes) is never passed
   through the Hill matrix because its dimensions are incomplete; it is
   only XORed with the continuing keystream: ``C_tail = P_tail XOR K_tail``.

The keystream is a single stateful logistic-map orbit per operation: it is
never reset per block and runs continuously across the whole file, so full
blocks consume keystream bytes ``[0, n_full*block_size)`` and the tail
consumes the remaining bytes.

Decryption is symmetric: XOR with the identical keystream first, then apply
the exact modular inverse ``M^-1`` (Gauss-Jordan over Z/256Z) to every full
block. Ciphertext length always equals plaintext length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import read_binary, sha256_bytes, write_binary
from .logistic_map import LogisticMapKeystream
from .metrics import MEGABYTE_BYTES, throughput_mb_s
from .modular_algebra import verify_modular_inverse

# Files are processed in block-aligned chunks of this size so that int64
# temporaries stay bounded for very large inputs (4 MiB is a multiple of the
# default 256-byte block). Chunking never changes the output bytes.
_CHUNK_BYTES = 4 * 1024 * 1024


class HybridHillXORCipher:
    """Unimodular Hill transform over GF(256)-style byte algebra followed by XOR.

    Not modern authenticated encryption; an experimental proof of concept.
    """

    def __init__(
        self,
        key_matrix: np.ndarray,
        inverse_matrix: np.ndarray,
        logistic_r: float,
        logistic_x0: float,
        warmup_iterations: int,
        block_size: int = 256,
        modulus: int = 256,
    ) -> None:
        self.key_matrix = np.asarray(key_matrix, dtype=np.int64) % modulus
        self.inverse_matrix = np.asarray(inverse_matrix, dtype=np.int64) % modulus
        if self.key_matrix.ndim != 2 or self.key_matrix.shape[0] != self.key_matrix.shape[1]:
            raise ValueError(f"key_matrix must be square, got shape {self.key_matrix.shape}")
        if self.inverse_matrix.shape != self.key_matrix.shape:
            raise ValueError(
                f"inverse_matrix shape {self.inverse_matrix.shape} does not match key_matrix shape {self.key_matrix.shape}"
            )
        self.matrix_dimension = self.key_matrix.shape[0]
        if block_size != self.matrix_dimension**2:
            raise ValueError(
                f"block_size must equal matrix_dimension ** 2 ({self.matrix_dimension**2}), got {block_size}"
            )
        forward, backward = verify_modular_inverse(self.key_matrix, self.inverse_matrix, modulus)
        if not (forward and backward):
            raise ValueError("key matrix failed modular inverse verification; cannot build cipher")
        self.block_size = int(block_size)
        self.modulus = int(modulus)
        self.logistic_r = float(logistic_r)
        self.logistic_x0 = float(logistic_x0)
        self.warmup_iterations = int(warmup_iterations)

    def _stream(self) -> LogisticMapKeystream:
        """Fresh stateful keystream for one encrypt/decrypt operation."""
        return LogisticMapKeystream(
            self.logistic_r, self.logistic_x0, self.warmup_iterations
        )

    def _hill_blocks(self, flat: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply ``matrix`` to each row-major n x n block of a flat int64 array."""
        blocks = flat.reshape(-1, self.matrix_dimension, self.matrix_dimension)
        transformed = np.matmul(matrix, blocks) % self.modulus
        return transformed.reshape(-1)

    def _chunk_end(self, position: int, full_length: int) -> int:
        """Next chunk boundary: block-aligned, capped at ``full_length``."""
        step = (_CHUNK_BYTES // self.block_size) * self.block_size
        return min(position + step, full_length)

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt raw bytes; output length always equals input length.

        Processed in fixed-size block-aligned chunks with one continuous
        stateful keystream, so memory stays bounded for very large files
        while the output stays byte-identical to single-pass processing.
        """
        data = np.frombuffer(plaintext, dtype=np.uint8)
        stream = self._stream()
        ciphertext = np.empty(data.size, dtype=np.uint8)
        full_length = (data.size // self.block_size) * self.block_size
        position = 0
        while position < full_length:
            end = self._chunk_end(position, full_length)
            chunk = data[position:end].astype(np.int64)
            keystream = stream.next_bytes(end - position).astype(np.int64)
            hill = self._hill_blocks(chunk, self.key_matrix)
            ciphertext[position:end] = (hill ^ keystream).astype(np.uint8)
            position = end
        if data.size > full_length:
            tail = data[full_length:].astype(np.int64)
            keystream = stream.next_bytes(data.size - full_length).astype(np.int64)
            ciphertext[full_length:] = (tail ^ keystream).astype(np.uint8)
        return ciphertext.tobytes()

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt raw bytes produced by :meth:`encrypt`; exact inverse."""
        data = np.frombuffer(ciphertext, dtype=np.uint8)
        stream = self._stream()
        plaintext = np.empty(data.size, dtype=np.uint8)
        full_length = (data.size // self.block_size) * self.block_size
        position = 0
        while position < full_length:
            end = self._chunk_end(position, full_length)
            chunk = data[position:end].astype(np.int64)
            keystream = stream.next_bytes(end - position).astype(np.int64)
            unxor = chunk ^ keystream
            plaintext[position:end] = self._hill_blocks(unxor, self.inverse_matrix).astype(np.uint8)
            position = end
        if data.size > full_length:
            tail = data[full_length:].astype(np.int64)
            keystream = stream.next_bytes(data.size - full_length).astype(np.int64)
            plaintext[full_length:] = (tail ^ keystream).astype(np.uint8)
        return plaintext.tobytes()

    def encrypt_file(
        self, input_path: Path, output_path: Path, megabyte_bytes: int = MEGABYTE_BYTES
    ) -> "FileOperationResult":
        """Encrypt one file end to end (read + transform + write are timed)."""
        start = time.perf_counter()
        plaintext = read_binary(Path(input_path))
        ciphertext = self.encrypt(plaintext)
        write_binary(Path(output_path), ciphertext)
        elapsed = time.perf_counter() - start
        return FileOperationResult(
            operation="encrypt",
            input_path=Path(input_path),
            output_path=Path(output_path),
            num_bytes=len(plaintext),
            elapsed_seconds=elapsed,
            throughput_mb_s=throughput_mb_s(len(plaintext), elapsed, megabyte_bytes),
            sha256_input=sha256_bytes(plaintext),
            sha256_output=sha256_bytes(ciphertext),
        )

    def decrypt_file(
        self, input_path: Path, output_path: Path, megabyte_bytes: int = MEGABYTE_BYTES
    ) -> "FileOperationResult":
        """Decrypt one file end to end (read + transform + write are timed)."""
        start = time.perf_counter()
        ciphertext = read_binary(Path(input_path))
        plaintext = self.decrypt(ciphertext)
        write_binary(Path(output_path), plaintext)
        elapsed = time.perf_counter() - start
        return FileOperationResult(
            operation="decrypt",
            input_path=Path(input_path),
            output_path=Path(output_path),
            num_bytes=len(ciphertext),
            elapsed_seconds=elapsed,
            throughput_mb_s=throughput_mb_s(len(ciphertext), elapsed, megabyte_bytes),
            sha256_input=sha256_bytes(ciphertext),
            sha256_output=sha256_bytes(plaintext),
        )


@dataclass(frozen=True)
class FileOperationResult:
    """Outcome of a single file-level encrypt or decrypt operation."""

    operation: str
    input_path: Path
    output_path: Path
    num_bytes: int
    elapsed_seconds: float
    throughput_mb_s: float
    sha256_input: str
    sha256_output: str

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "num_bytes": self.num_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_mb_s": self.throughput_mb_s,
            "sha256_input": self.sha256_input,
            "sha256_output": self.sha256_output,
        }
