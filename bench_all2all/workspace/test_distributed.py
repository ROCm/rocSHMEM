#!/usr/bin/env python3
"""
Test ROCm + InfiniBand distributed training environment
This script tests whether multi-GPU communication works correctly
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from datetime import timedelta

def setup_distributed(rank, world_size):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')

    # Initialize process group - RCCL/NCCL will automatically use InfiniBand
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=60)
    )

def cleanup():
    """Clean up distributed process group"""
    dist.destroy_process_group()

def test_allreduce(rank, world_size):
    """Test AllReduce operation - this will use InfiniBand communication"""
    device = torch.device(f'cuda:{rank}')

    # Create a tensor
    tensor = torch.ones(1000, 1000, device=device) * (rank + 1)

    print(f"[Rank {rank}] Before AllReduce: tensor sum = {tensor.sum().item():.0f}")

    # AllReduce operation - sum tensors across all GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = sum(range(1, world_size + 1)) * 1000 * 1000
    print(f"[Rank {rank}] After AllReduce: tensor sum = {tensor.sum().item():.0f} (expected: {expected})")

    # Verify result
    assert abs(tensor.sum().item() - expected) < 1e-3, f"AllReduce failed on rank {rank}"

    return True

def test_bandwidth(rank, world_size):
    """Test bandwidth - use large tensor to test communication performance"""
    device = torch.device(f'cuda:{rank}')

    # Create a large tensor (1GB)
    size = 256 * 1024 * 1024  # 1GB / 4 bytes per float
    tensor = torch.randn(size, device=device)

    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Timing
    torch.cuda.synchronize()
    import time
    start = time.time()

    iterations = 10
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Calculate bandwidth
    # AllReduce data amount = 2 * (N-1) / N * data_size (ring algorithm)
    data_size_gb = (tensor.numel() * 4) / (1024**3)  # Convert to GB
    algorithm_factor = 2 * (world_size - 1) / world_size
    total_data_gb = data_size_gb * algorithm_factor * iterations
    bandwidth_gbps = total_data_gb / elapsed

    if rank == 0:
        print(f"\n=== Bandwidth Test ===")
        print(f"Tensor size: {data_size_gb:.2f} GB")
        print(f"Iterations: {iterations}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"Per-GPU bus bandwidth: {bandwidth_gbps / world_size:.2f} GB/s")

    return True

def run_worker(rank, world_size):
    """Worker process running on each GPU"""
    try:
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(rank)}")

        # Initialize distributed
        setup_distributed(rank, world_size)

        print(f"[Rank {rank}] Distributed initialized (backend: {dist.get_backend()})")

        # Test AllReduce
        print(f"\n[Rank {rank}] Testing AllReduce...")
        test_allreduce(rank, world_size)

        # Synchronize
        dist.barrier()

        # Test bandwidth only on rank 0
        if rank == 0:
            print(f"\n[Rank 0] Testing bandwidth...")
        test_bandwidth(rank, world_size)

        # Synchronize
        dist.barrier()

        if rank == 0:
            print(f"\n✓ All distributed tests passed!")

        cleanup()

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main function"""
    print("="*60)
    print("ROCm + InfiniBand Distributed Training Test")
    print("="*60)

    # Check GPU count
    if not torch.cuda.is_available():
        print("Error: No CUDA/ROCm devices available")
        return

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")

    if world_size < 2:
        print("Warning: Need at least 2 GPUs for meaningful distributed testing")
        print("Running single GPU test...")
        world_size = 1

    # Print environment variables
    print(f"\nEnvironment:")
    print(f"  NCCL_DEBUG: {os.getenv('NCCL_DEBUG', 'not set')}")
    print(f"  NCCL_IB_DISABLE: {os.getenv('NCCL_IB_DISABLE', 'not set')}")
    print(f"  NCCL_NET_GDR_LEVEL: {os.getenv('NCCL_NET_GDR_LEVEL', 'not set')}")
    print(f"  RCCL_NET_GDR_LEVEL: {os.getenv('RCCL_NET_GDR_LEVEL', 'not set')}")
    print()

    # Launch multiprocessing
    if world_size > 1:
        mp.spawn(
            run_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        run_worker(0, 1)

if __name__ == "__main__":
    main()
