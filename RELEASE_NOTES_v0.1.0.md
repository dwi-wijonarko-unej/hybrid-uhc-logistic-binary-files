# v0.1.0 — Hybrid Unimodular Hill + Logistic Map XOR (baseline)

Experimental proof of concept accompanying the article
*"Experimental Evaluation of a Hybrid Unimodular Hill and Logistic Map XOR Scheme for Binary Files"*.

## What this release contains

- Hybrid cipher: unimodular 16 × 16 Hill transform per 256-byte block (mod 256)
  followed by byte-wise XOR with a stateful logistic-map keystream
  (`r = 3.923`, `x0 = 0.73911`, warm-up 1000, quantization `floor((x·1000) mod 256)`).
- Exact Gauss–Jordan modular inverse over Z/256Z (unit pivots + extended
  Euclidean algorithm), verified in both product directions — including a
  property test over 100 (r, x0) configurations.
- Reproducible experiment pipeline (`reproduce_experiments.py`): manifest,
  per-run raw measurements (1 warm-up + 5 measured runs per file), byte
  statistics (entropy, adjacent-byte correlation, chi-square), reconstruction
  checks (SHA-256 primary, MD5 secondary), key-material report, summary
  table, and 5 figures.
- Revised legacy benchmark (`benchmark_real_files.py`) on real-format
  DOCX/PDF/JPEG files with deterministic seeded padding (default), a
  `source_manifest.csv` audit trail, and pinned optional extras
  (`requirements-benchmark.txt`).
- 82 pytest tests, including roundtrip exactness for 11 size classes and the
  documented Model A keystream/matrix 135-byte prefix overlap.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python scripts/reproduce_experiments.py --config config/experiment_config.yaml
```

Verified on CPython 3.14 (code is Python 3.10-compatible); baseline pins for
Python 3.10 are in `requirements.txt` / `environment.yml`.

## Security disclaimer

This is **not** modern authenticated encryption and must not be used to
protect real data. No resistance to known-plaintext, chosen-plaintext, or any
modern cryptanalytic attack is claimed. There is no per-file nonce: every file
operation starts from the same keystream prefix, and the key matrix reuses the
first 135 keystream bytes (Model A). The evaluation covers reversibility,
byte statistics, and runtime only.

## Determinism

For the same input files, configuration, implementation revision, and
IEEE-754 binary64 execution path, ciphertexts, hashes, and byte-level
statistics are deterministic. Runtime and throughput figures are
hardware-, OS-, and load-dependent.
