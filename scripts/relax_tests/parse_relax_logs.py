#!/usr/bin/env python3
"""Simple parser for driver.sh relax test logs using inferred rank and WG variants.

Parses baseline and relaxed logs for allreducepush and fusedgemmallreducepush,
using only metrics printed directly in test_output/*.log.

Edit the CONFIG section to choose which logical configurations to parse.
Each config is expanded into two concrete variants automatically:
- 8 ranks with the IPC backend
- 16 ranks with the GDA backend

For each concrete variant, the parser finds the most recently modified matching
log file and uses that file's num_wgs value for both the baseline and relaxed
results.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "test_output"
OUTPUT_CSV = LOG_DIR / "relax_log_summary_simple_latest.csv"

VARIANTS = [
    {"num_ranks": 8, "backend": "ipc"},
    {"num_ranks": 16, "backend": "gda"},
]

# allreducepush_n{ranks}_w{wgs}_z{threads}_{msg_size}B_{backend}[ _rlx ].log
ALLREDUCE_CONFIGS = [
    {"num_threads": 512, "msg_size_B": 128},
    {"num_threads": 512, "msg_size_B": 256},
    {"num_threads": 512, "msg_size_B": 512},
    {"num_threads": 512, "msg_size_B": 1024},
    {"num_threads": 512, "msg_size_B": 2048},
    {"num_threads": 512, "msg_size_B": 4096},
    {"num_threads": 512, "msg_size_B": 8192},
    {"num_threads": 512, "msg_size_B": 16384},
    {"num_threads": 512, "msg_size_B": 32768},
    {"num_threads": 512, "msg_size_B": 65536},
    {"num_threads": 512, "msg_size_B": 131072},
    {"num_threads": 512, "msg_size_B": 262144},
]

# fusedgemmallreducepush_n{ranks}_w{wgs}_z{threads}_m{gemm_m}_n{gemm_n}_{backend}[ _rlx ].log
FUSED_GEMM_ALLREDUCE_CONFIGS = [
    {"num_threads": 128, "gemm_m": 64, "gemm_n": 4096},
    {"num_threads": 128, "gemm_m": 64, "gemm_n": 8192},
    {"num_threads": 128, "gemm_m": 64, "gemm_n": 16384},
    {"num_threads": 128, "gemm_m": 128, "gemm_n": 4096},
    {"num_threads": 128, "gemm_m": 128, "gemm_n": 8192},
    {"num_threads": 128, "gemm_m": 128, "gemm_n": 16384},
    {"num_threads": 128, "gemm_m": 256, "gemm_n": 4096},
    {"num_threads": 128, "gemm_m": 256, "gemm_n": 8192},
    {"num_threads": 128, "gemm_m": 256, "gemm_n": 16384},
]

ALLREDUCE_RE = re.compile(
    r"^allreducepush_n(?P<num_ranks>\d+)_w(?P<num_wgs>\d+)_z(?P<num_threads>\d+)"
    r"_(?P<msg_size_B>\d+)B_(?P<backend>ipc|gda)(?P<relaxed>_rlx)?\.log$"
)

FUSED_RE = re.compile(
    r"^fusedgemmallreducepush_n(?P<num_ranks>\d+)_w(?P<num_wgs>\d+)_z(?P<num_threads>\d+)"
    r"_m(?P<gemm_m>\d+)_n(?P<gemm_n>\d+)_(?P<backend>ipc|gda)(?P<relaxed>_rlx)?\.log$"
)


def _to_float(value: str) -> Optional[float]:
    txt = value.strip().lower()
    if not txt:
        return None
    if txt in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if txt in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(txt)
    except ValueError:
        return None


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def _find_last_metric_row(lines: List[str]) -> Optional[Tuple[List[str], List[str]]]:
    header_idx = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("# Size (B),"):
            header_idx = index

    if header_idx is None:
        return None

    header_cols = [column.strip() for column in lines[header_idx].split(",")]
    while header_cols and not header_cols[-1]:
        header_cols.pop()

    row_re = re.compile(r"^\s*\d+\s*,")
    for index in range(len(lines) - 1, header_idx, -1):
        line = lines[index]
        if row_re.match(line):
            row_cols = [column.strip() for column in line.split(",")]
            while row_cols and not row_cols[-1]:
                row_cols.pop()
            return header_cols, row_cols

    return None


def _normalize_header(header: str) -> str:
    normalized = header.strip().lower()
    normalized = normalized.lstrip("#").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    metrics = {
        "latency_us": None,
        "nw_bandwidth_gbs": None,
        "hbm_bandwidth_gbs": None,
        "compute_throughput_gflops": None,
    }

    if not log_path.exists():
        return metrics

    parsed = _find_last_metric_row(log_path.read_text(errors="ignore").splitlines())
    if not parsed:
        return metrics

    header_cols, row_cols = parsed
    for header, value in zip(header_cols, row_cols):
        key = _normalize_header(header)
        parsed_value = _to_float(value)
        if key == "latency (us)":
            metrics["latency_us"] = parsed_value
        elif key in {"bandwidth (gb/s)", "nw bandwidth (gb/s)"}:
            metrics["nw_bandwidth_gbs"] = parsed_value
        elif key == "mem bw (gb/s)":
            metrics["hbm_bandwidth_gbs"] = parsed_value
        elif key == "comp bw (gflop/s)":
            metrics["compute_throughput_gflops"] = parsed_value

    return metrics


def speedup(base_latency: Optional[float], rlx_latency: Optional[float]) -> Optional[float]:
    if base_latency is None or rlx_latency is None:
        return None
    if rlx_latency == 0:
        return math.inf
    return base_latency / rlx_latency


def allreduce_base(cfg: Dict[str, int], num_ranks: int, backend: str, num_wgs: int) -> str:
    return (
        f"allreducepush_n{num_ranks}_w{num_wgs}_z{cfg['num_threads']}"
        f"_{cfg['msg_size_B']}B_{backend}"
    )


def fused_base(cfg: Dict[str, int], num_ranks: int, backend: str, num_wgs: int) -> str:
    return (
        f"fusedgemmallreducepush_n{num_ranks}_w{num_wgs}_z{cfg['num_threads']}"
        f"_m{cfg['gemm_m']}_n{cfg['gemm_n']}_{backend}"
    )


def parse_log_metadata(path: Path) -> Optional[Dict[str, object]]:
    for test_name, pattern in (("allreducepush", ALLREDUCE_RE), ("fusedgemmallreducepush", FUSED_RE)):
        match = pattern.match(path.name)
        if not match:
            continue

        groups = match.groupdict()
        metadata: Dict[str, object] = {
            "test_name": test_name,
            "path": path,
            "relaxed": bool(groups.get("relaxed")),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for key, value in groups.items():
            if key == "relaxed" or value is None:
                continue
            if key == "backend":
                metadata[key] = value
            else:
                metadata[key] = int(value)
        return metadata

    return None


def collect_log_metadata() -> List[Dict[str, object]]:
    metadata_entries: List[Dict[str, object]] = []
    for log_path in LOG_DIR.glob("*.log"):
        metadata = parse_log_metadata(log_path)
        if metadata is not None:
            metadata_entries.append(metadata)
    return metadata_entries


def _matches_config(metadata: Dict[str, object], cfg: Dict[str, int], rank_backend: Dict[str, object]) -> bool:
    if metadata["num_ranks"] != rank_backend["num_ranks"]:
        return False
    if metadata["backend"] != rank_backend["backend"]:
        return False

    for key, value in cfg.items():
        if metadata.get(key) != value:
            return False

    return True


def find_latest_num_wgs(
    metadata_entries: Iterable[Dict[str, object]],
    test_name: str,
    cfg: Dict[str, int],
    rank_backend: Dict[str, object],
) -> Optional[int]:
    candidates = [
        entry
        for entry in metadata_entries
        if entry["test_name"] == test_name and _matches_config(entry, cfg, rank_backend)
    ]
    if not candidates:
        return None

    latest = max(candidates, key=lambda entry: int(entry["mtime_ns"]))
    return int(latest["num_wgs"])


def variant_metrics(base_name: str) -> Dict[str, Optional[float]]:
    baseline = parse_metrics(LOG_DIR / f"{base_name}.log")
    relaxed = parse_metrics(LOG_DIR / f"{base_name}_rlx.log")
    return {
        "nw_bandwidth_unrelaxed_gbs": baseline["nw_bandwidth_gbs"],
        "nw_bandwidth_relaxed_gbs": relaxed["nw_bandwidth_gbs"],
        "hbm_bandwidth_unrelaxed_gbs": baseline["hbm_bandwidth_gbs"],
        "hbm_bandwidth_relaxed_gbs": relaxed["hbm_bandwidth_gbs"],
        "compute_unrelaxed_gflops": baseline["compute_throughput_gflops"],
        "compute_relaxed_gflops": relaxed["compute_throughput_gflops"],
        "relaxation_speedup": speedup(baseline["latency_us"], relaxed["latency_us"]),
    }


def make_row(test_name: str, cfg: Dict[str, int], metadata_entries: List[Dict[str, object]]) -> Dict[str, str]:
    row: Dict[str, str] = {
        "test_name": test_name,
        "num_threads": str(cfg["num_threads"]),
        "msg_size_B": str(cfg.get("msg_size_B", "")),
        "gemm_m": str(cfg.get("gemm_m", "")),
        "gemm_n": str(cfg.get("gemm_n", "")),
        "num_wgs_8": "",
        "num_wgs_16": "",
        "nw_bandwidth_8_unrelaxed_gbs": "",
        "nw_bandwidth_8_relaxed_gbs": "",
        "nw_bandwidth_16_unrelaxed_gbs": "",
        "nw_bandwidth_16_relaxed_gbs": "",
        "hbm_bandwidth_8_unrelaxed_gbs": "",
        "hbm_bandwidth_8_relaxed_gbs": "",
        "hbm_bandwidth_16_unrelaxed_gbs": "",
        "hbm_bandwidth_16_relaxed_gbs": "",
        "compute_throughput_8_unrelaxed_gflops": "",
        "compute_throughput_8_relaxed_gflops": "",
        "compute_throughput_16_unrelaxed_gflops": "",
        "compute_throughput_16_relaxed_gflops": "",
        "relaxation_speedup_8": "",
        "relaxation_speedup_16": "",
    }

    for variant in VARIANTS:
        num_ranks = int(variant["num_ranks"])
        num_wgs = find_latest_num_wgs(metadata_entries, test_name, cfg, variant)
        if num_wgs is None:
            continue

        suffix = str(num_ranks)
        row[f"num_wgs_{suffix}"] = str(num_wgs)
        if test_name == "allreducepush":
            base_name = allreduce_base(cfg, num_ranks, str(variant["backend"]), num_wgs)
        else:
            base_name = fused_base(cfg, num_ranks, str(variant["backend"]), num_wgs)

        metrics = variant_metrics(base_name)
        row[f"nw_bandwidth_{suffix}_unrelaxed_gbs"] = _fmt(metrics["nw_bandwidth_unrelaxed_gbs"])
        row[f"nw_bandwidth_{suffix}_relaxed_gbs"] = _fmt(metrics["nw_bandwidth_relaxed_gbs"])
        row[f"hbm_bandwidth_{suffix}_unrelaxed_gbs"] = _fmt(metrics["hbm_bandwidth_unrelaxed_gbs"])
        row[f"hbm_bandwidth_{suffix}_relaxed_gbs"] = _fmt(metrics["hbm_bandwidth_relaxed_gbs"])
        row[f"compute_throughput_{suffix}_unrelaxed_gflops"] = _fmt(metrics["compute_unrelaxed_gflops"])
        row[f"compute_throughput_{suffix}_relaxed_gflops"] = _fmt(metrics["compute_relaxed_gflops"])
        row[f"relaxation_speedup_{suffix}"] = _fmt(metrics["relaxation_speedup"])

    return row


def main() -> None:
    metadata_entries = collect_log_metadata()
    rows: List[Dict[str, str]] = []

    for cfg in ALLREDUCE_CONFIGS:
        rows.append(make_row("allreducepush", cfg, metadata_entries))

    for cfg in FUSED_GEMM_ALLREDUCE_CONFIGS:
        rows.append(make_row("fusedgemmallreducepush", cfg, metadata_entries))

    fieldnames = [
        "test_name",
        "num_threads",
        "msg_size_B",
        "gemm_m",
        "gemm_n",
        "num_wgs_8",
        "num_wgs_16",
        "nw_bandwidth_8_unrelaxed_gbs",
        "nw_bandwidth_8_relaxed_gbs",
        "nw_bandwidth_16_unrelaxed_gbs",
        "nw_bandwidth_16_relaxed_gbs",
        "hbm_bandwidth_8_unrelaxed_gbs",
        "hbm_bandwidth_8_relaxed_gbs",
        "hbm_bandwidth_16_unrelaxed_gbs",
        "hbm_bandwidth_16_relaxed_gbs",
        "compute_throughput_8_unrelaxed_gflops",
        "compute_throughput_8_relaxed_gflops",
        "compute_throughput_16_unrelaxed_gflops",
        "compute_throughput_16_relaxed_gflops",
        "relaxation_speedup_8",
        "relaxation_speedup_16",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    writer_stdout = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer_stdout.writeheader()
    writer_stdout.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
