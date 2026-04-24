#!/bin/bash
#
# Run official DeepEP test_low_latency.py on 2 nodes (16 GPUs) for strict vs relaxed.
#
# Uses rocm-infiniband-private:latest image (explicit-ctx + relaxed rocSHMEM).
# Launches fresh containers on both nodes, runs the official test, collects results.
#
# Usage (from HOST):
#   bash bench_all2all/workspace/run_official_ll_multinode.sh [--runs N] [--tokens "1 2 ..."]

set -e

# =============================================================================
# Defaults
# =============================================================================
WORKER="banff20-29"
IMAGE="${IMAGE:-rocm-infiniband-private:latest}"
NUM_RUNS=${NUM_RUNS:-3}
TOKEN_COUNTS="${TOKEN_COUNTS:-1 2 4 8 16 32 64 128 256}"
GPUS_PER_NODE=8
MASTER_PORT="8361"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

HEAD_CONTAINER="bench-official-ll-head"
WORKER_CONTAINER="bench-official-ll-worker"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_HOST="${BENCH_DIR}/workspace/results/official_ll_multinode"
mkdir -p "$RESULTS_HOST"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs) NUM_RUNS="$2"; shift 2 ;;
        --tokens) TOKEN_COUNTS="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --worker) WORKER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Detect head IP
NET_IF=${NET_IF:-$(ip route show default | awk '/default/ {print $5}' | head -1)}
HEAD_IP=$(ip -4 addr show "$NET_IF" | grep inet | awk '{print $2}' | cut -d/ -f1)

echo "================================================================"
echo "Official DeepEP LL Benchmark: Multi-Node Strict vs Relaxed"
echo "================================================================"
echo "Head:        $(hostname -s) ($HEAD_IP)"
echo "Worker:      $WORKER"
echo "Image:       $IMAGE"
echo "Token counts: $TOKEN_COUNTS"
echo "Runs/config: $NUM_RUNS"
echo "Master port: $MASTER_PORT"
echo "Timestamp:   $TIMESTAMP"
echo "================================================================"

# =============================================================================
# Cleanup trap
# =============================================================================
cleanup() {
    echo ""
    echo "Cleaning up containers..."
    docker rm -f "$HEAD_CONTAINER" 2>/dev/null || true
    ssh "$WORKER" "docker rm -f $WORKER_CONTAINER 2>/dev/null" || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# =============================================================================
# Start containers on both nodes
# =============================================================================
start_containers() {
    local EXTRA_ENV="$1"

    # Stop any existing
    docker rm -f "$HEAD_CONTAINER" 2>/dev/null || true
    ssh "$WORKER" "docker rm -f $WORKER_CONTAINER 2>/dev/null" || true

    # Clean stale NCCL shared memory (prevents hipIpcGetMemHandle failures)
    docker run --rm --ipc host "$IMAGE" bash -c "rm -f /dev/shm/nccl-*" 2>/dev/null || true
    ssh "$WORKER" "docker run --rm --ipc host $IMAGE bash -c 'rm -f /dev/shm/nccl-*'" 2>/dev/null || true

    local DOCKER_RUN="docker run -d --rm \
        --network host \
        --device /dev/kfd \
        --device /dev/dri \
        --device /dev/infiniband \
        --group-add video \
        --group-add render \
        --ipc host \
        --shm-size 64gb \
        --cap-add SYS_PTRACE \
        --cap-add IPC_LOCK \
        --cap-add SYS_NICE \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864:67108864 \
        -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
        -v /sys/class/net:/sys/class/net:ro \
        -v /sys/bus/pci:/sys/bus/pci:ro \
        -e NCCL_DEBUG=WARN \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
        -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e ROCSHMEM_HEAP_SIZE=8589934592 \
        -e NCCL_SOCKET_IFNAME=${NET_IF} \
        -e GLOO_SOCKET_IFNAME=${NET_IF} \
        ${EXTRA_ENV} \
        ${IMAGE} sleep infinity"

    echo "  Starting head container..."
    docker run -d --rm --name "$HEAD_CONTAINER" \
        --network host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render \
        --ipc host --shm-size 64gb \
        --cap-add SYS_PTRACE --cap-add IPC_LOCK --cap-add SYS_NICE \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
        -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
        -v /sys/class/net:/sys/class/net:ro \
        -v /sys/bus/pci:/sys/bus/pci:ro \
        -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 \
        -e "NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e ROCSHMEM_HEAP_SIZE=8589934592 \
        -e "NCCL_SOCKET_IFNAME=${NET_IF}" -e "GLOO_SOCKET_IFNAME=${NET_IF}" \
        $EXTRA_ENV \
        "$IMAGE" sleep infinity

    echo "  Starting worker container on $WORKER..."
    ssh "$WORKER" "docker run -d --rm --name $WORKER_CONTAINER \
        --network host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render \
        --ipc host --shm-size 64gb \
        --cap-add SYS_PTRACE --cap-add IPC_LOCK --cap-add SYS_NICE \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
        -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
        -v /sys/class/net:/sys/class/net:ro \
        -v /sys/bus/pci:/sys/bus/pci:ro \
        -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 \
        -e 'NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7' \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e ROCSHMEM_HEAP_SIZE=8589934592 \
        -e 'NCCL_SOCKET_IFNAME=${NET_IF}' -e 'GLOO_SOCKET_IFNAME=${NET_IF}' \
        $EXTRA_ENV \
        $IMAGE sleep infinity"

    sleep 3
    echo "  Containers ready."
}

# =============================================================================
# Run official test for a single token count
# =============================================================================
run_test() {
    local NUM_TOKENS=$1
    local ORDERING=$2
    local RUN_NUM=$3

    local OUTFILE="${RESULTS_HOST}/${ORDERING}_${TIMESTAMP}/run${RUN_NUM}_tok${NUM_TOKENS}.log"
    mkdir -p "$(dirname "$OUTFILE")"

    local ENV_VARS="MASTER_ADDR=${HEAD_IP} MASTER_PORT=${MASTER_PORT} ROCSHMEM_HEAP_SIZE=8589934592"
    if [ "$ORDERING" = "relaxed" ]; then
        ENV_VARS="$ENV_VARS ROCSHMEM_RELAXED_ORDER=1"
    fi

    echo "--- ${ORDERING} | run ${RUN_NUM}/${NUM_RUNS} | ${NUM_TOKENS} tokens ---"

    # Launch worker (node rank=1) in background
    ssh "$WORKER" "docker exec $WORKER_CONTAINER bash -c '
        export ${ENV_VARS} RANK=1 WORLD_SIZE=2
        cd /opt/src/DeepEP/tests
        python3 test_low_latency.py \
            --num-tokens ${NUM_TOKENS} \
            --num-experts 288 \
            --hidden 7168 \
            --num-topk 8 \
            --num-processes 8 2>&1
    '" > /dev/null 2>&1 &
    WORKER_PID=$!

    # Launch head (node rank=0) in foreground
    docker exec "$HEAD_CONTAINER" bash -c "
        export ${ENV_VARS} RANK=0 WORLD_SIZE=2
        cd /opt/src/DeepEP/tests
        python3 test_low_latency.py \
            --num-tokens ${NUM_TOKENS} \
            --num-experts 288 \
            --hidden 7168 \
            --num-topk 8 \
            --num-processes 8 2>&1
    " | tee "$OUTFILE"

    # Wait for worker
    wait $WORKER_PID 2>/dev/null || true

    echo "  Saved: $OUTFILE"
}

# =============================================================================
# Main loop
# =============================================================================
for ORDERING in strict relaxed; do
    echo ""
    echo "================================================================"
    echo "  ORDERING: ${ORDERING}"
    echo "================================================================"

    EXTRA_ENV=""
    if [ "$ORDERING" = "relaxed" ]; then
        EXTRA_ENV="-e ROCSHMEM_RELAXED_ORDER=1"
    fi

    for RUN_NUM in $(seq 1 $NUM_RUNS); do
        # Start fresh containers for each run to avoid hipIpcGetMemHandle failures
        # (GPU IPC state accumulates across test invocations in the same container)
        start_containers "$EXTRA_ENV"

        for NUM_TOKENS in ${TOKEN_COUNTS}; do
            run_test "$NUM_TOKENS" "$ORDERING" "$RUN_NUM" || {
                echo "  FAILED: ${ORDERING} run${RUN_NUM} tok${NUM_TOKENS}"
                # Kill any hung processes and continue
                docker exec "$HEAD_CONTAINER" pkill -9 python 2>/dev/null || true
                ssh "$WORKER" "docker exec $WORKER_CONTAINER pkill -9 python 2>/dev/null" || true
                sleep 2
            }
        done
    done
done

echo ""
echo "================================================================"
echo "All multi-node runs complete."
echo "  Strict:  ${RESULTS_HOST}/strict_${TIMESTAMP}/"
echo "  Relaxed: ${RESULTS_HOST}/relaxed_${TIMESTAMP}/"
echo "================================================================"
