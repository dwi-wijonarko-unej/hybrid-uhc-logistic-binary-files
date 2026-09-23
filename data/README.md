# Data directory — publication-grade data availability artifacts

This directory holds the auditable experiment record for the data
availability statement. Everything here is generated deterministically and
can be re-derived with the commands in [`artifact_map.md`](artifact_map.md).

## Structure

| Path | Content |
|---|---|
| `file_manifest.csv` | The 12-file test dataset (DOCX/PDF/JPEG × 1/10/50/100 MB): file id, type, nominal label, exact bytes, SHA-256 of plaintext/ciphertext/decrypted, reconstruction result, access status, and how each file is distributed. |
| `checksums_sha256.txt` | `sha256sum`-compatible checksums for all 36 binaries (12 × source/encrypted/decrypted). Verify with `sha256sum -c data/checksums_sha256.txt` from the repository root (after regenerating any large files, see below). |
| `parameter_settings.yaml` | Complete parameter snapshot: cipher parameters (r, x0, warm-up, quantization, block/modulus), matrix generation algorithm, benchmark protocol, dataset generator settings, parameter id. |
| `raw_measurements.csv` | Per-file encryption/decryption time and throughput (paper Table 2; single pass per file, timings include file I/O; 1 MB = 1,048,576 B). |
| `byte_statistics.csv` | Plaintext/ciphertext entropy and adjacent-byte correlation (paper Table 1). |
| `reconstruction_checks.csv` | SHA-256 (primary) and MD5 (secondary) of original vs decrypted plus exact-match result (paper Table 3). |
| `artifact_map.md` | Maps every paper table/figure to its artifact in this repository. |
| `runtime_environment.txt` | Interpreter, package versions (`pip freeze`), OS and CPU used for the recorded runs. Timings are hardware-dependent. |
| `analysis_outputs/real_format_benchmark/` | Benchmark outputs behind the paper tables: `summary_results.csv`, `source_manifest.csv`, histogram figure. |
| `analysis_outputs/synthetic_experiment/` | Repetition-protocol experiment on the synthetic 12-file set (1 warm-up + 5 measured runs per file): raw per-run measurements, summary statistics, figures, key matrices, config snapshot, log. |
| `input/` | Local drop-in directory for replacing the dataset with private/original files (never committed). |

## Which binaries are committed — and why the rest is still fully available

Git and GitHub are a poor fit for ~100 MB binaries, so only the **1 MB size
class** (source + encrypted + decrypted, 9 files ≈ 9 MB) is committed under
`benchmark/source`, `benchmark/encrypted`, and `benchmark/decrypted`.

The remaining files are **bit-exactly regenerable** because the generator is
deterministic (seeded SplitMix64 padding, normalized ZIP/PDF timestamps):

```bash
pip install -r requirements.txt -r requirements-benchmark.txt
python scripts/benchmark_real_files.py --generate-only     # recreates all 12 sources
python scripts/benchmark_real_files.py --skip-generate     # recreates enc + dec + tables
sha256sum -c data/checksums_sha256.txt                     # must report OK for all 36
```

Any regenerated file that does not match `checksums_sha256.txt` indicates a
different generator revision or seed — do not mix such files with the
published record.

## Replacing the dataset with the original article files

If the original (non-synthetic) files used for the manuscript become
shareable, drop them into `data/input/` and rerun
`scripts/reproduce_experiments.py`; the pipeline re-emits every manifest,
measurement, and figure for that dataset. Do not overwrite the hashes in
this directory with values for files that are not actually distributed —
record the real access conditions in `file_manifest.csv` (`access_status`
column) instead.

## Data terms

Files under this directory and everything generated under `results/` or
`benchmark/` are **not** automatically covered by the repository's MIT code
license. The committed dataset and analysis outputs in this release are
distributed under **CC0 1.0** (no rights reserved) unless stated otherwise
in `file_manifest.csv`; third-party materials remain under their own terms.
If you redistribute a frozen copy, keep this notice and the checksum file
together.
