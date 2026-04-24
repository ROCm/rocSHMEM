#!/usr/bin/env python3
"""Lightweight validation for the bench_all2all DeepEP container."""

import importlib.util
import os
import subprocess
import sys


def main() -> int:
    print("=" * 72)
    print("bench_all2all environment check")
    print("=" * 72)

    try:
        import torch
    except Exception as exc:
        print(f"PyTorch import failed: {exc}")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"  GPU {idx}: {torch.cuda.get_device_name(idx)}")

    deep_ep_spec = importlib.util.find_spec("deep_ep")
    if deep_ep_spec is None:
        print("DeepEP import: FAIL")
        return 1

    import deep_ep  # noqa: F401

    print("DeepEP import: OK")
    print(f"DeepEP module path: {deep_ep_spec.origin}")

    relax_bin = "/usr/local/bin/rocshmem_relax_tests"
    print(f"rocSHMEM relax_tests binary: {'found' if os.path.exists(relax_bin) else 'missing'}")

    if shutil_which("ibv_devices"):
        print("")
        print("InfiniBand devices:")
        subprocess.run(["ibv_devices"], check=False)

    print("")
    print("Environment looks ready for the official DeepEP low-latency benchmark.")
    return 0


def shutil_which(cmd: str) -> str | None:
    for path in os.getenv("PATH", "").split(":"):
        candidate = os.path.join(path, cmd)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


if __name__ == "__main__":
    sys.exit(main())
