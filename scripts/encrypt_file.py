#!/usr/bin/env python3
"""Encrypt a single file with the hybrid unimodular Hill + logistic XOR scheme.

All cipher parameters (matrix dimension, modulus, logistic r/x0, warm-up)
come from the YAML configuration file; nothing is prompted interactively.
The operation metadata (filename, exact bytes, parameter ID, SHA-256 of
plaintext and ciphertext, elapsed time, throughput) is written to a JSON
file when ``--metadata`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_crypto.cipher import HybridHillXORCipher  # noqa: E402
from hybrid_crypto.config import load_config  # noqa: E402
from hybrid_crypto.io_utils import write_json  # noqa: E402
from hybrid_crypto.matrix_generator import UnimodularKeyMaterial  # noqa: E402

logger = logging.getLogger("encrypt_file")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "experiment_config.yaml",
                        help="Path to the experiment YAML configuration (default: config/experiment_config.yaml)")
    parser.add_argument("--input", type=Path, required=True, help="Plaintext file to encrypt")
    parser.add_argument("--output", type=Path, required=True, help="Ciphertext output path (raw binary)")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Optional path for the JSON operation metadata")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    params = config.cipher
    params.validate()

    key = UnimodularKeyMaterial(params)
    cipher = HybridHillXORCipher(
        key_matrix=key.matrix,
        inverse_matrix=key.inverse,
        logistic_r=params.logistic_r,
        logistic_x0=params.logistic_x0,
        warmup_iterations=params.warmup_iterations,
        block_size=params.block_size,
        modulus=params.modulus,
    )
    result = cipher.encrypt_file(args.input, args.output, config.benchmark.megabyte_bytes)

    metadata = {
        "operation": "encrypt",
        "filename": Path(args.input).name,
        "exact_bytes": result.num_bytes,
        "parameter_id": params.parameter_id(),
        "sha256_plaintext": result.sha256_input,
        "sha256_ciphertext": result.sha256_output,
        "encryption_time_seconds": result.elapsed_seconds,
        "encryption_throughput_mb_s": result.throughput_mb_s,
        "megabyte_bytes": config.benchmark.megabyte_bytes,
        "cipher_parameters": {
            "matrix_dimension": params.matrix_dimension,
            "block_size": params.block_size,
            "modulus": params.modulus,
            "logistic_r": params.logistic_r,
            "logistic_x0": params.logistic_x0,
            "warmup_iterations": params.warmup_iterations,
            "quantization_rule": params.quantization_rule,
            "numeric_precision": params.numeric_precision,
        },
    }
    if args.metadata is not None:
        write_json(args.metadata, metadata)

    logger.info(
        "Encrypted %s -> %s (%d bytes, %.6f s, %.3f MB/s, sha256_ciphertext=%s)",
        args.input, args.output, result.num_bytes, result.elapsed_seconds,
        result.throughput_mb_s, result.sha256_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
