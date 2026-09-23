# Data directory

This directory holds the experiment inputs and their metadata. **No test files
are committed to the repository.**

## `data/input/` — plaintext test files

Place the binary files you want to evaluate here. The experiment pipeline
(`scripts/reproduce_experiments.py`) picks up every regular file in this
directory, in sorted filename order, and assigns identifiers `F01, F02, ...`.

The original 12 files used in the article are **not** distributed with this
repository. Two options:

1. **Original dataset available.** Copy the 12 original files into
   `data/input/`, optionally record their checksums in a manifest under
   `data/manifest/` (see below), and run the reproduction script. This
   reproduces the article's numerical results exactly, because the pipeline
   is fully deterministic for a fixed file set and configuration.

2. **Original dataset unavailable.** Generate the 12 deterministic synthetic
   placeholder files instead:

   ```bash
   python scripts/generate_synthetic_test_files.py --config config/experiment_config.yaml
   ```

   The synthetic set consists of DOCX-like, PDF-like, and JPEG-like binary
   placeholders (4 sizes x 3 types) built from a fixed-seed SplitMix64
   stream. **These placeholders are not identical to the original files.**
   They reproduce the *workflow* (tables, figures, checksums, timings) but
   not the article's numeric values for byte statistics, since the byte
   distributions differ. If `experiment.generate_synthetic_files_if_missing`
   is `true`, the reproduction script generates them automatically when
   `data/input/` is empty.

## `data/manifest/` — input metadata and checksums

Generated input manifests (e.g. `synthetic_files_manifest.csv` with filename,
target/exact size, and SHA-256 per synthetic file) are written here. When you
supply your own test files, it is good practice to record
`filename,exact_size_bytes,sha256` rows in a CSV here so the dataset can be
audited later. The per-run authoritative checksums of everything the pipeline
reads or writes are always re-computed into `results/raw/` regardless.

## Small dummy binary for tests

The unit test suite (`tests/`) constructs small in-memory dummy byte strings
at runtime (deterministic logistic-map output). No large or binary fixture
files are stored in the repository.

## Data terms

Files placed in this directory and everything generated under `results/` or
`benchmark/` are **not** automatically covered by the repository's MIT code
license. If you distribute a frozen dataset or archived experiment outputs,
assign them an explicit data license (e.g., CC BY 4.0 or CC0) and state any
access restrictions here. Third-party test materials remain under their own
terms.
