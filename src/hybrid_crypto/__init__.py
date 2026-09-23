"""Hybrid unimodular Hill + logistic-map XOR cipher for binary files.

Experimental proof of concept for the article
"Experimental Evaluation of a Hybrid Unimodular Hill and Logistic Map XOR
Scheme for Binary Files".

This package is NOT modern authenticated encryption. It exists to evaluate
reversibility, byte statistics, and runtime of the studied scheme.
"""

from .config import BenchmarkParameters, CipherParameters, ConfigError, ExperimentConfig, SyntheticFilesConfig, load_config
from .logistic_map import LogisticMapKeystream, generate_logistic_byte_sequence
from .matrix_generator import UnimodularKeyMaterial, generate_unimodular_matrix, required_sequence_length
from .modular_algebra import ModularInverseError, extended_gcd, modular_inverse, modular_matrix_inverse, verify_modular_inverse
from .cipher import FileOperationResult, HybridHillXORCipher

__version__ = "0.1.0"

__all__ = [
    "BenchmarkParameters",
    "CipherParameters",
    "ConfigError",
    "ExperimentConfig",
    "SyntheticFilesConfig",
    "load_config",
    "LogisticMapKeystream",
    "generate_logistic_byte_sequence",
    "UnimodularKeyMaterial",
    "generate_unimodular_matrix",
    "required_sequence_length",
    "ModularInverseError",
    "extended_gcd",
    "modular_inverse",
    "modular_matrix_inverse",
    "verify_modular_inverse",
    "FileOperationResult",
    "HybridHillXORCipher",
    "__version__",
]
