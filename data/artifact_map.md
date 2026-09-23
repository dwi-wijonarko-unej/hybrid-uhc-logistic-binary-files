# Artifact map: repository artifacts -> paper tables and figures

Generated 2026-09-23T06:42:42+00:00. Hashes and sizes live in `file_manifest.csv` and
`checksums_sha256.txt`; parameters in `parameter_settings.yaml`.

| Paper item | Primary artifact | Regeneration |
|---|---|---|
| Test-file dataset (12 files) | `file_manifest.csv` + `checksums_sha256.txt` (+ `benchmark/source/*Small*` committed) | `python scripts/benchmark_real_files.py --generate-only` (bit-exact, seeded) |
| Table 1 — statistical analysis | `byte_statistics.csv` | `python scripts/benchmark_real_files.py` |
| Table 2 — computational performance | `raw_measurements.csv` | `python scripts/benchmark_real_files.py` |
| Table 3 — integrity check | `reconstruction_checks.csv` | `python scripts/benchmark_real_files.py` |
| Histogram figure | `analysis_outputs/real_format_benchmark/Figure_histogram_detail.png` | idem |
| Repetition-protocol experiment (warm-up + 5 runs, synthetic 12-file set) | `analysis_outputs/synthetic_experiment/raw/*.csv`, `tables/summary_statistics.csv`, `figures/*.png` | `python scripts/reproduce_experiments.py` |
| Key matrix and modular inverse | `analysis_outputs/synthetic_experiment/configuration/key_matrix*.csv`, `key_material_report.json` | idem |
| Exact algorithm specification | `README.md` (Algorithm specification) + `src/hybrid_crypto/` | — |
| Environment | `runtime_environment.txt`, `requirements.txt`, `environment.yml` | — |

Notes: the real-format benchmark times a single pass per file (the legacy
protocol used for the paper tables); the repetition protocol (mean/std/
median/min/max over 5 measured runs after 1 warm-up) is documented in
`analysis_outputs/synthetic_experiment/`. Ciphertext expansion is zero for
every file (ciphertext size == plaintext size; see `raw_measurements.csv`).
