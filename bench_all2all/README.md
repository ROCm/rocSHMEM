# DeepEP Low-Latency Artifact

This directory contains the DeepEP application-level artifact referenced by the
paper. It is intentionally limited to the official DeepEP low-latency benchmark
workflow and builds DeepEP against the checked-out `targeted_ordering`
rocSHMEM tree in this repository.

## Scope

- Single-node official DeepEP low-latency benchmark
- Two-node official DeepEP low-latency benchmark launcher
- Log parser for strict-vs-relaxed comparisons
- Docker image that builds rocSHMEM and DeepEP together

The earlier comparison machinery for other communication libraries has been
removed because it is not part of the paper artifact.

## Prerequisites

- 8x AMD MI300X GPUs for the single-node run
- 2 nodes / 16 GPUs with InfiniBand for the multi-node run
- Docker with ROCm device pass-through
- Host InfiniBand drivers installed (`/dev/infiniband` present)

## Quick Start

```bash
# Build and start the single-node image
docker compose -f bench_all2all/docker-compose.yml build
docker compose -f bench_all2all/docker-compose.yml up -d
docker compose -f bench_all2all/docker-compose.yml exec rocm bash

# Validate the environment inside the container
python3 /workspace/simple_test.py

# Run the official single-node DeepEP low-latency sweep
export ROCSHMEM_HEAP_SIZE=8589934592
bash /workspace/run_official_ll_benchmark.sh --runs 7 --tokens "1 2 4 8 16 32 64 128 256"

# Parse the strict vs relaxed comparison
python3 /workspace/parse_official_ll_results.py \
  /workspace/results/official_ll/strict_<timestamp> \
  /workspace/results/official_ll/relaxed_<timestamp>
```

## Two-Node Run

Run the host-side helper from the repository root:

```bash
bash bench_all2all/workspace/run_official_ll_multinode.sh \
  --worker <worker-hostname> \
  --image rocm-infiniband-private:latest \
  --runs 7 \
  --tokens "1 2 4 8 16 32 64 128 256"
```

This helper assumes the corresponding `rocm-infiniband-private:latest` image is
already available on both nodes.

## Image Contents

The image is based on
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` and includes:

- the current checked-out rocSHMEM source tree copied to `/opt/src/rocSHMEM`
- DeepEP cloned at pinned commit `e84464e`
- the paper's `deepep_relaxed_ll.patch`

## Files

- `Dockerfile`: builds rocSHMEM and DeepEP together
- `docker-compose.yml`: single-node ROCm container launcher
- `workspace/build_all.sh`: optional rebuild/verification helper
- `workspace/run_official_ll_benchmark.sh`: single-node benchmark runner
- `workspace/run_official_ll_multinode.sh`: two-node benchmark runner
- `workspace/parse_official_ll_results.py`: strict-vs-relaxed parser
- `workspace/simple_test.py`: quick environment validation
