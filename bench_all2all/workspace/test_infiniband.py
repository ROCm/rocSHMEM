#!/usr/bin/env python3
"""Test InfiniBand and GPU connectivity in ROCm container"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and display output"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("ROCm + InfiniBand Environment Test")
    print("=" * 60)

    tests = [
        # InfiniBand tests
        ("ibstat", "InfiniBand devices status"),
        ("ibv_devices", "IB verbs devices"),
        ("ibv_devinfo", "IB device information"),

        # ROCm tests
        ("rocm-smi --showproductname", "ROCm GPU detection"),

        # PyTorch tests
        ("python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")'",
         "PyTorch GPU detection"),

        # RCCL (ROCm Collective Communications Library) test
        ("python3 -c 'import torch; import torch.distributed as dist; print(\"RCCL/NCCL backend available:\", dist.is_nccl_available())'",
         "RCCL/NCCL availability"),
    ]

    results = {}
    for cmd, description in tests:
        results[description] = run_command(cmd, description)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! Environment is ready for distributed training.")
    else:
        print("\n✗ Some tests failed. Check configuration.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
