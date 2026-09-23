#!/usr/bin/env python3
"""Revised benchmark program: Hybrid Unimodular Hill and Logistic Map XOR
scheme for binary files.

This script is the direct revision of the legacy monolithic benchmark program
(Unimodular Hill Cipher + Logistic Map + Shift Cipher). The algorithm changes
are real and structural:

* the Shift Cipher is gone entirely;
* the legacy tail-only chaotic XOR (with its separate ``x0 + 0.12345`` seed)
  is replaced by ONE continuous logistic-map keystream that XORs the ENTIRE
  Hill output — full 256-byte blocks and the residual tail alike;
* the Hill transform now operates per 16 x 16 (256-byte) row-major block,
  not over the whole file as one n x columns matrix;
* the modular inverse is computed exactly with Gauss-Jordan over Z/256Z
  (unit pivots + extended Euclidean algorithm) and verified in both product
  directions;
* parameters come from a frozen dataclass (constructor-configurable), never
  from interactive passwords;
* SHA-256 is the primary integrity checksum (MD5 kept as a secondary digest
  for legacy compatibility).

The cryptographic core is imported from the project library ``hybrid_crypto``
(single tested implementation, bounded-memory chunked processing). Kept from
the legacy program: realistic test-file generation (real DOCX via python-docx,
hand-built valid PDF, real JPEG via Pillow — each padded to target size with
``os.urandom``), the benchmark loop, the statistical tables, and the
plaintext/ciphertext histogram figure.

Note: ``os.urandom`` padding makes the generated test files realistic but NOT
bit-reproducible across runs; the cryptography itself is fully deterministic
for a given input file and parameter set.

Not modern authenticated encryption; experimental proof of concept only.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_crypto.cipher import HybridHillXORCipher  # noqa: E402
from hybrid_crypto.config import CipherParameters  # noqa: E402
from hybrid_crypto.io_utils import md5_file, sha256_file, write_csv  # noqa: E402
from hybrid_crypto.matrix_generator import UnimodularKeyMaterial  # noqa: E402

logger = logging.getLogger("benchmark_real_files")

MEGABYTE = 1024 * 1024

# ==============================================================================
# 0. CONFIGURATION (dataclass, constructor-configurable; no passwords)
# ==============================================================================

DEFAULT_CIPHER_PARAMETERS = CipherParameters(
    matrix_dimension=16,
    block_size=256,
    modulus=256,
    logistic_r=3.923,
    logistic_x0=0.73911,
    warmup_iterations=1000,
    quantization_rule="k_i = floor((x_i * 1000) mod 256), stored as uint8",
    numeric_precision="IEEE-754 binary64 (Python float)",
)

SIZE_NAMES = ["Small", "Medium", "Large", "Huge"]
FILE_GROUPS = {"Doc": ".docx", "PDF": ".pdf", "Img": ".jpg"}
TYPE_MAP = {".docx": "Document", ".pdf": "PDF", ".jpg": "Image", ".jpeg": "Image"}


@dataclass(frozen=True)
class BenchmarkConfig:
    """All benchmark knobs (cipher parameters live in CipherParameters)."""

    target_sizes_mb: tuple[float, ...] = (1.0, 10.0, 50.0, 100.0)
    root_dir: Path = PROJECT_ROOT / "benchmark"
    csv_name: str = "summary_results.csv"
    figure_name: str = "Figure_histogram_detail.png"

    @property
    def source_dir(self) -> Path:
        return self.root_dir / "source"

    @property
    def encrypted_dir(self) -> Path:
        return self.root_dir / "encrypted"

    @property
    def decrypted_dir(self) -> Path:
        return self.root_dir / "decrypted"

    @property
    def csv_path(self) -> Path:
        return self.root_dir / self.csv_name

    @property
    def figure_path(self) -> Path:
        return self.root_dir / self.figure_name

    def planned_files(self) -> list[tuple[Path, float]]:
        return [
            (self.source_dir / f"{group}_{name}{extension}", size_mb)
            for group, extension in FILE_GROUPS.items()
            for name, size_mb in zip(SIZE_NAMES, self.target_sizes_mb)
        ]


# ==============================================================================
# 1. REAL DOCX FILE GENERATOR (python-docx + os.urandom padding)
# ==============================================================================

SAMPLE_PARAGRAPHS = [
    "This document was automatically generated to simulate real-world Microsoft Word files for performance and scalability testing purposes.",
    "Performance testing is crucial for ensuring that systems can handle large documents efficiently.",
    "The DOCX format uses the Open XML standard. A DOCX file is essentially a ZIP archive containing XML files.",
    "Binary padding files cannot validate parsing logic, content extraction, or format handling capabilities.",
    "Scalability testing helps identify bottlenecks in document processing pipelines.",
]


def _import_python_docx():
    try:
        from docx import Document
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        from docx.shared import Pt
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx", "-q"])
        from docx import Document
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        from docx.shared import Pt
    return Document, Pt, WD_PARAGRAPH_ALIGNMENT


def _create_docx_document(target_size_mb: float):
    Document, Pt, ALIGNMENT = _import_python_docx()
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    title = doc.add_heading("Performance & Scalability Test Document", level=0)
    title.alignment = ALIGNMENT.CENTER
    meta = doc.add_paragraph()
    meta.alignment = ALIGNMENT.CENTER
    run = meta.add_run(f"Target Size: {target_size_mb} MB  |  Auto-Generated for Testing")
    run.italic = True
    doc.add_paragraph("─" * 60)
    for index in range(min(200, max(20, int(target_size_mb * 2)))):
        doc.add_heading(f"Section {index + 1}", level=1)
        doc.add_paragraph(SAMPLE_PARAGRAPHS[index % len(SAMPLE_PARAGRAPHS)])
    return doc


def generate_real_docx(filename: Path, target_size_mb: float) -> None:
    target_size_bytes = int(target_size_mb * MEGABYTE)
    document = _create_docx_document(target_size_mb)
    base_buffer = io.BytesIO()
    document.save(base_buffer)
    base_buffer.seek(0)
    padding_needed = max(0, target_size_bytes - len(base_buffer.getvalue()) - 120)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(base_buffer, "r") as source:
            for item in source.infolist():
                archive.writestr(item, source.read(item.filename))
        if padding_needed > 0:
            chunk_size = 10 * MEGABYTE
            chunk_number = 0
            total_written = 0
            while total_written < padding_needed:
                size = min(padding_needed - total_written, chunk_size)
                info = zipfile.ZipInfo(f"_padding/pad_{chunk_number:04d}.bin")
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, os.urandom(size))
                total_written += size
                chunk_number += 1
    logger.info("[ok] %s -> %s bytes", filename.name, f"{os.path.getsize(filename):,}")


# ==============================================================================
# 2. REAL PDF FILE GENERATOR (hand-built valid PDF + os.urandom padding)
# ==============================================================================

def generate_real_pdf(filename: Path, target_size_mb: float) -> None:
    target_bytes = int(target_size_mb * MEGABYTE)
    num_pages = min(100, max(5, int(target_size_mb * 1.5)))
    offsets: dict[int, int] = {}
    with open(filename, "wb") as handle:
        handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        page_objs = list(range(3, 3 + num_pages))
        content_objs = list(range(3 + num_pages, 3 + 2 * num_pages))
        font_obj = 3 + 2 * num_pages
        offsets[1] = handle.tell()
        handle.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n")
        kids = " ".join(f"{n} 0 R" for n in page_objs)
        offsets[2] = handle.tell()
        handle.write(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj\n\n".encode())
        offsets[font_obj] = handle.tell()
        handle.write(
            f"{font_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\n".encode()
        )
        for index in range(num_pages):
            stream_data = f"BT\n/F1 10 Tf\n1 0 0 1 54 750 Tm\n(Page {index + 1}) Tj\nET\n".encode()
            content_obj = content_objs[index]
            offsets[content_obj] = handle.tell()
            handle.write(f"{content_obj} 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode())
            handle.write(stream_data)
            handle.write(b"\nendstream\nendobj\n\n")
            page_obj = page_objs[index]
            offsets[page_obj] = handle.tell()
            handle.write(
                f"{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_obj} 0 R /Resources << /Font << /F1 {font_obj} 0 R >> >> >>\nendobj\n\n".encode()
            )
        padding_needed = max(0, target_bytes - handle.tell() - 300)
        if padding_needed > 0:
            obj_num = font_obj + 1
            total_written = 0
            while total_written < padding_needed:
                size = min(padding_needed - total_written, 10 * MEGABYTE)
                offsets[obj_num] = handle.tell()
                handle.write(f"{obj_num} 0 obj\n<< /Length {size} >>\nstream\n".encode())
                handle.write(os.urandom(size))
                handle.write(b"\nendstream\nendobj\n\n")
                total_written += size
                obj_num += 1
            max_obj = obj_num
        else:
            max_obj = font_obj + 1
        xref_offset = handle.tell()
        handle.write(b"xref\n")
        handle.write(f"0 {max_obj} \n".encode())
        handle.write(b"0000000000 65535 f \n")
        for index in range(1, max_obj):
            if index in offsets:
                handle.write(f"{offsets[index]:010d} 00000 n \n".encode())
            else:
                handle.write(b"0000000000 00000 f \n")
        handle.write(b"trailer\n<< /Size ")
        handle.write(f"{max_obj} /Root 1 0 R >>\n".encode())
        handle.write(b"startxref\n")
        handle.write(f"{xref_offset}\n".encode())
        handle.write(b"%%EOF\n")
    logger.info("[ok] %s -> %s bytes", filename.name, f"{os.path.getsize(filename):,}")


# ==============================================================================
# 3. REAL JPEG IMAGE GENERATOR (Pillow + os.urandom COM padding)
# ==============================================================================

_MAX_COM_DATA = 65533


def _import_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image, ImageDraw
    return Image, ImageDraw


def _create_base_image():
    Image, ImageDraw = _import_pillow()
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (30, 40, 55))
    draw = ImageDraw.Draw(image)
    draw.text((width // 2 - 200, 60), "PERFORMANCE TEST IMAGE", fill=(255, 255, 255))
    return image


def generate_real_jpeg(filename: Path, target_size_mb: float, base_image) -> None:
    target_bytes = int(target_size_mb * MEGABYTE)
    jpeg_buffer = io.BytesIO()
    base_image.save(jpeg_buffer, format="JPEG", quality=85)
    base_bytes = jpeg_buffer.getvalue()
    padding_needed = max(0, target_bytes - len(base_bytes))
    with open(filename, "wb") as handle:
        handle.write(base_bytes[:2])
        remaining = padding_needed
        while remaining > 0:
            data_size = min(remaining, _MAX_COM_DATA)
            length = data_size + 2
            handle.write(b"\xff\xfe")
            handle.write(length.to_bytes(2, byteorder="big"))
            handle.write(os.urandom(data_size))
            remaining -= data_size
        handle.write(base_bytes[2:])
    logger.info("[ok] %s -> %s bytes", filename.name, f"{os.path.getsize(filename):,}")


def prepare_source_files(config: BenchmarkConfig, regenerate: bool, skip_generate: bool) -> list[Path]:
    """Generate every planned test file that is missing (unless skipped)."""
    config.source_dir.mkdir(parents=True, exist_ok=True)
    planned = config.planned_files()
    if skip_generate:
        missing = [path for path, _ in planned if not path.is_file()]
        if missing:
            logger.warning(
                "--skip-generate set but %d planned file(s) missing: %s",
                len(missing), ", ".join(p.name for p in missing),
            )
        return [path for path, _ in planned if path.is_file()]
    base_image = None
    for path, size_mb in planned:
        if path.is_file() and not regenerate:
            logger.info("exists, keeping: %s", path.name)
            continue
        suffix = path.suffix.lower()
        if suffix == ".docx":
            generate_real_docx(path, size_mb)
        elif suffix == ".pdf":
            generate_real_pdf(path, size_mb)
        elif suffix in (".jpg", ".jpeg"):
            if base_image is None:
                base_image = _create_base_image()
            generate_real_jpeg(path, size_mb, base_image)
        else:  # pragma: no cover - guarded by FILE_GROUPS
            raise ValueError(f"unsupported test file type: {path}")
    return [path for path, _ in planned]


# ==============================================================================
# 4. CRYPTOSYSTEM BENCHMARK & ANALYSIS (hybrid UHC + logistic XOR)
# ==============================================================================

def build_cipher(params: CipherParameters) -> HybridHillXORCipher:
    """Build the revised cipher from constructor parameters (no passwords)."""
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


def calc_entropy(data: np.ndarray) -> float:
    """Shannon entropy in bits/byte of a uint8 array."""
    from scipy.stats import entropy as scipy_entropy

    counts = np.bincount(data, minlength=256)
    probabilities = counts[counts > 0] / len(data)
    return float(scipy_entropy(probabilities, base=2))


def calc_correlation(data: np.ndarray, samples: int = 10000) -> float:
    """Pearson correlation of adjacent bytes on a fixed deterministic sample."""
    if len(data) <= samples + 1:
        samples = max(1, len(data) - 2)
    indices = np.random.default_rng(42).choice(len(data) - 1, size=samples, replace=False)
    x = data[indices].astype(np.float64)
    y = data[indices + 1].astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def run_benchmark(config: BenchmarkConfig, cipher_params: CipherParameters) -> list[dict[str, object]]:
    import filecmp

    cipher = build_cipher(cipher_params)
    config.encrypted_dir.mkdir(parents=True, exist_ok=True)
    config.decrypted_dir.mkdir(parents=True, exist_ok=True)

    files = prepare_source_files(config, regenerate=False, skip_generate=False)
    results: list[dict[str, object]] = []
    for index, source in enumerate(files, start=1):
        size_bytes = os.path.getsize(source)
        size_mb = size_bytes / MEGABYTE
        encrypted_path = config.encrypted_dir / source.name
        decrypted_path = config.decrypted_dir / source.name

        logger.info("[%d/%d] %s (%.2f MB): encrypting...", index, len(files), source.name, size_mb)
        enc = cipher.encrypt_file(source, encrypted_path)
        logger.info("[%d/%d] %s: decrypting...", index, len(files), source.name)
        dec = cipher.decrypt_file(encrypted_path, decrypted_path)

        plaintext = np.fromfile(source, dtype=np.uint8)
        ciphertext = np.fromfile(encrypted_path, dtype=np.uint8)
        sha256_original = sha256_file(source)
        sha256_decrypted = sha256_file(decrypted_path)
        integrity = "MATCH" if (
            sha256_original == sha256_decrypted and filecmp.cmp(source, decrypted_path, shallow=False)
        ) else "MISMATCH"

        results.append(
            {
                "name": source.name,
                "type": TYPE_MAP.get(Path(source).suffix.lower(), "Unknown"),
                "size": size_bytes,
                "size_mb": size_mb,
                "ent_p": calc_entropy(plaintext),
                "ent_c": calc_entropy(ciphertext),
                "cor_p": calc_correlation(plaintext),
                "cor_c": calc_correlation(ciphertext),
                "t_enc": enc.elapsed_seconds,
                "t_dec": dec.elapsed_seconds,
                "tp_enc": size_mb / enc.elapsed_seconds,
                "tp_dec": size_mb / dec.elapsed_seconds,
                "sha256_orig": sha256_original,
                "sha256_dec": sha256_decrypted,
                "md5_orig": md5_file(source),
                "md5_dec": md5_file(decrypted_path),
                "integrity": integrity,
                # Byte buffers kept only for the first (smallest) file: the
                # histogram figure needs them and holding all 12 would waste RAM.
                "p_data": plaintext if index == 1 else None,
                "c_data": ciphertext if index == 1 else None,
            }
        )
        logger.info(
            "[%d/%d] %s: enc %.4f s | dec %.4f s | integrity %s",
            index, len(files), source.name, enc.elapsed_seconds, dec.elapsed_seconds, integrity,
        )
    return results


# ==============================================================================
# 5. FIGURES
# ==============================================================================

def generate_histogram_figure(results: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = results[0]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(reference["p_data"], bins=256, range=(0, 256), color="#2196F3", alpha=0.8,
                 edgecolor="black", linewidth=0.2)
    axes[0].set_title(f'Plaintext Histogram\n{reference["name"]}', fontsize=11, fontweight="bold", color="#1565C0")
    axes[0].text(0.02, 0.95, f'Entropy: {reference["ent_p"]:.4f}\nCorrelation: {reference["cor_p"]:.4f}',
                 transform=axes[0].transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[1].hist(reference["c_data"], bins=256, range=(0, 256), color="#F44336", alpha=0.8,
                 edgecolor="black", linewidth=0.2)
    axes[1].set_title(f'Ciphertext Histogram\n{reference["name"]}', fontsize=11, fontweight="bold", color="#C62828")
    axes[1].text(0.02, 0.95, f'Entropy: {reference["ent_c"]:.4f}\nCorrelation: {reference["cor_c"]:.4f}',
                 transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    for axis in axes:
        axis.set_xlabel("Byte Value (0-255)")
        axis.set_ylabel("Frequency")
        axis.grid(linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    logger.info("Figure written: %s", path)


# ==============================================================================
# 6. RESULT TABLES AND CSV EXPORT
# ==============================================================================

_CSV_FIELDS = [
    "filename", "type", "size_bytes", "size_mb",
    "entropy_plaintext", "entropy_ciphertext",
    "corr_plaintext", "corr_ciphertext",
    "encryption_time_s", "decryption_time_s",
    "enc_throughput_mb_s", "dec_throughput_mb_s",
    "sha256_original", "sha256_decrypted", "md5_original", "md5_decrypted", "integrity",
]


def export_csv(results: list[dict[str, object]], path: Path) -> None:
    rows = [
        {
            "filename": r["name"], "type": r["type"], "size_bytes": r["size"], "size_mb": f'{r["size_mb"]:.4f}',
            "entropy_plaintext": f'{r["ent_p"]:.6f}', "entropy_ciphertext": f'{r["ent_c"]:.6f}',
            "corr_plaintext": f'{r["cor_p"]:+.6f}', "corr_ciphertext": f'{r["cor_c"]:+.6f}',
            "encryption_time_s": f'{r["t_enc"]:.6f}', "decryption_time_s": f'{r["t_dec"]:.6f}',
            "enc_throughput_mb_s": f'{r["tp_enc"]:.4f}', "dec_throughput_mb_s": f'{r["tp_dec"]:.4f}',
            "sha256_original": r["sha256_orig"], "sha256_decrypted": r["sha256_dec"],
            "md5_original": r["md5_orig"], "md5_decrypted": r["md5_dec"], "integrity": r["integrity"],
        }
        for r in results
    ]
    write_csv(path, _CSV_FIELDS, rows)
    logger.info("CSV written: %s", path)


def print_tables(results: list[dict[str, object]], cipher_params: CipherParameters) -> None:
    print("\n" + "=" * 96)
    print("  HYBRID UNIMODULAR HILL + LOGISTIC MAP XOR -- BENCHMARK RESULTS")
    print(f"  n={cipher_params.matrix_dimension}  block={cipher_params.block_size} B  "
          f"mod={cipher_params.modulus}  r={cipher_params.logistic_r}  "
          f"x0={cipher_params.logistic_x0}  warmup={cipher_params.warmup_iterations}")
    print("=" * 96)

    avg_ent_c = float(np.mean([r["ent_c"] for r in results]))
    avg_cor_c = float(np.mean([r["cor_c"] for r in results]))
    print("\n  TABLE 1. Statistical Security Analysis")
    print("=" * 96)
    print(f"{'File Name':<22} | {'PT Entr':<8} | {'CT Entr':<8} | {'PT Corr':<8} | {'CT Corr':<8}")
    print("-" * 96)
    for r in results:
        print(f'{r["name"]:<22} | {r["ent_p"]:<8.4f} | {r["ent_c"]:<8.4f} | {r["cor_p"]:<+8.4f} | {r["cor_c"]:<+8.4f}')
    print("-" * 96)
    print(f'{"Average":<22} | {"---":<8} | {avg_ent_c:<8.4f} | {"---":<8} | {avg_cor_c:<+8.4f}')

    tp_enc = [r["tp_enc"] for r in results]
    tp_dec = [r["tp_dec"] for r in results]
    print("\n  TABLE 2. Computational Performance (timings include file I/O; 1 MB = 1,048,576 B)")
    print("=" * 96)
    print(f"{'File Name':<22} | {'Size(MB)':<9} | {'Enc(s)':<8} | {'Dec(s)':<8} | {'Enc TP':<8} | {'Dec TP':<8}")
    print("-" * 96)
    for r in results:
        print(f'{r["name"]:<22} | {r["size_mb"]:<9.2f} | {r["t_enc"]:<8.4f} | {r["t_dec"]:<8.4f} | '
              f'{r["tp_enc"]:<8.2f} | {r["tp_dec"]:<8.2f}')
    print("-" * 96)
    print(f'{"Std Dev":<22} | {"---":<9} | {"---":<8} | {"---":<8} | {np.std(tp_enc):<8.2f} | {np.std(tp_dec):<8.2f}')

    print("\n  TABLE 3. Integrity Check (SHA-256 primary; MD5 secondary)")
    print("=" * 96)
    all_match = True
    md5_agree = True
    for r in results:
        md5_ok = r["md5_orig"] == r["md5_dec"]
        if not md5_ok:
            md5_agree = False
        if r["integrity"] != "MATCH":
            all_match = False
        print(f'{r["name"]:<22} | SHA-256 {r["integrity"]:<8} | MD5 {"agree" if md5_ok else "DIFFER"}')
    print("-" * 96)
    print(f"CONCLUSION: {'ALL MATCH (byte-identical reconstruction)' if all_match and md5_agree else 'FAILED'}")
    print("=" * 96)


# ==============================================================================
# 7. MAIN
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-sizes-mb", type=str, default="1,10,50,100",
                        help="Comma-separated target sizes in MB per file type (default: 1,10,50,100)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate source files even if they already exist")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Never generate files; benchmark only the planned files that exist")
    parser.add_argument("--cipher-logistic-x0", type=float, default=DEFAULT_CIPHER_PARAMETERS.logistic_x0,
                        help="Override the logistic map initial condition")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sizes = tuple(float(item) for item in args.target_sizes_mb.split(","))
    if not sizes or any(size <= 0 for size in sizes):
        raise SystemExit("--target-sizes-mb must contain positive sizes")
    if len(sizes) > len(SIZE_NAMES):
        raise SystemExit(f"at most {len(SIZE_NAMES)} sizes supported (names: {SIZE_NAMES})")

    config = BenchmarkConfig(target_sizes_mb=sizes)
    cipher_params = DEFAULT_CIPHER_PARAMETERS
    if args.cipher_logistic_x0 != cipher_params.logistic_x0:
        cipher_params = replace(cipher_params, logistic_x0=args.cipher_logistic_x0)
    cipher_params.validate()

    logger.info("Configuration: sizes=%s MB, output root=%s", list(sizes), config.root_dir)
    logger.info(
        "Cipher parameters: n=%d, block=%d, modulus=%d, r=%s, x0=%s, warmup=%d, parameter_id=%s",
        cipher_params.matrix_dimension, cipher_params.block_size, cipher_params.modulus,
        cipher_params.logistic_r, cipher_params.logistic_x0, cipher_params.warmup_iterations,
        cipher_params.parameter_id(),
    )

    if args.regenerate or args.skip_generate or not any(path.is_file() for path, _ in config.planned_files()):
        prepare_source_files(config, regenerate=args.regenerate, skip_generate=args.skip_generate)

    results = run_benchmark(config, cipher_params)
    if not results:
        logging.error("No test files available to benchmark")
        return 1

    export_csv(results, config.csv_path)
    generate_histogram_figure(results, config.figure_path)
    print_tables(results, cipher_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
