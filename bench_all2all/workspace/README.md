# DeepEP Workspace

This directory is mounted to `/workspace` inside the bench_all2all container.
It contains only the scripts needed for the paper's DeepEP low-latency artifact.

## Main Workflow

Single-node, inside the container:

```bash
python3 /workspace/simple_test.py
export ROCSHMEM_HEAP_SIZE=8589934592
bash /workspace/run_official_ll_benchmark.sh --runs 7 --tokens "1 2 4 8 16 32 64 128 256"
python3 /workspace/parse_official_ll_results.py \
  /workspace/results/official_ll/strict_<timestamp> \
  /workspace/results/official_ll/relaxed_<timestamp>
```

Two-node, from the host:

```bash
bash bench_all2all/workspace/run_official_ll_multinode.sh \
  --worker <worker-hostname> \
  --image rocm-infiniband-private:latest \
  --runs 7 \
  --tokens "1 2 4 8 16 32 64 128 256"
```

## Files

- `build_all.sh`: optional rocSHMEM + DeepEP rebuild helper
- `run_official_ll_benchmark.sh`: single-node official DeepEP benchmark
- `run_official_ll_multinode.sh`: two-node official DeepEP benchmark
- `parse_official_ll_results.py`: parser for the benchmark logs
- `simple_test.py`: import and environment check
- `test_infiniband.py`: low-level InfiniBand and GPU visibility test
- `test_distributed.py`: RCCL/NCCL all-reduce validation
- `results/`: output directory for benchmark logs
