"""Stateful logistic-map keystream generator.

The recurrence

    x_(t+1) = r * x_t * (1 - x_t)

is evaluated in IEEE-754 binary64 (Python ``float``). After discarding
``warmup_iterations`` iterations, every orbit value is quantized to one byte
with the rule

    k_i = floor((x_i * 1000) mod 256)

and stored as ``uint8``. The generator is stateful: successive calls to
:meth:`LogisticMapKeystream.next_bytes` continue the same orbit, so the
keystream is continuous across blocks and across an entire file. The orbit is
never reset per block. Two instances constructed with identical parameters
produce byte-identical sequences (covered by unit tests).
"""

from __future__ import annotations

import math

import numpy as np


class LogisticMapKeystream:
    """Stateful byte keystream driven by the logistic map."""

    def __init__(self, r: float, x0: float, warmup_iterations: int) -> None:
        if not 0.0 < float(x0) < 1.0:
            raise ValueError(f"logistic x0 must satisfy 0 < x0 < 1, got {x0}")
        if not 0.0 < float(r) <= 4.0:
            raise ValueError(f"logistic r must satisfy 0 < r <= 4, got {r}")
        if int(warmup_iterations) < 0:
            raise ValueError(f"warmup_iterations must be >= 0, got {warmup_iterations}")
        self.r = float(r)
        self.warmup_iterations = int(warmup_iterations)
        self.bytes_emitted = 0
        self._x = float(x0)
        for _ in range(self.warmup_iterations):
            self._advance()

    @property
    def current_state(self) -> float:
        """Current orbit value (diagnostics/tests only)."""
        return self._x

    def _advance(self) -> None:
        # Evaluation order matches (r * x) * (1 - x) exactly; plain Python
        # float arithmetic is correctly rounded IEEE-754 binary64, which is
        # what makes the keystream reproducible across platforms.
        self._x = self.r * self._x * (1.0 - self._x)

    def next_bytes(self, count: int) -> np.ndarray:
        """Return the next ``count`` keystream bytes as a uint8 array.

        The internal state advances by exactly ``count`` iterations; the
        stream continues where the previous call stopped.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        output = np.empty(count, dtype=np.uint8)
        for index in range(count):
            self._advance()
            output[index] = math.floor((self._x * 1000.0) % 256.0)
        self.bytes_emitted += count
        return output


def generate_logistic_byte_sequence(r: float, x0: float, warmup_iterations: int, length: int) -> np.ndarray:
    """One-shot convenience wrapper returning ``length`` uint8 keystream bytes."""
    return LogisticMapKeystream(r, x0, warmup_iterations).next_bytes(length)
