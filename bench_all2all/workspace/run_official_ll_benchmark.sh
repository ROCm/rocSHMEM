#!/usr/bin/env bash
# Run official DeepEP test_low_latency.py for strict vs relaxed ordering comparison.
#
# The official test runs:
#   1. Correctness checks (FP8, BF16, hook/no-hook, round_scale, etc.)
#   2. Performance benchmark via bench() from tests/utils.py
#      - 50 warmup + 50 timed iterations, L2 cache flush, CUDA events
#      - Reports D+C bandwidth/latency, dispatch/combine breakdown via Kineto
#
# Usage (inside container):
#   bash /workspace/run_official_ll_benchmark.sh [--runs N] [--tokens "1 2 4 ..."]
#
# Env vars:
#   ROCSHMEM_HEAP_SIZE  - rocSHMEM heap (default 8GB for large token counts)

set -uo pipefail

# Defaults
NUM_RUNS=${NUM_RUNS:-3}
TOKEN_COUNTS="${TOKEN_COUNTS:-1 2 4 8 16 32 64 128 256}"
RESULTS_BASE="/workspace/results/official_ll"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}  # 8GB

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs) NUM_RUNS="$2"; shift 2 ;;
        --tokens) TOKEN_COUNTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Port counter for unique ports
PORT_COUNTER=0

echo "================================================================"
echo "Official DeepEP LL Benchmark: Strict vs Relaxed"
echo "================================================================"
echo "Token counts: ${TOKEN_COUNTS}"
echo "Runs per config: ${NUM_RUNS}"
echo "Heap size: ${ROCSHMEM_HEAP_SIZE}"
echo "Timestamp: ${TIMESTAMP}"
echo "================================================================"

FAIL_COUNT=0

for ORDERING in strict relaxed; do
    echo ""
    echo "================================================================"
    echo "  ORDERING: ${ORDERING}"
    echo "================================================================"

    if [ "$ORDERING" = "relaxed" ]; then
        export ROCSHMEM_RELAXED_ORDER=1
        echo "  ROCSHMEM_RELAXED_ORDER=1"
    else
        unset ROCSHMEM_RELAXED_ORDER 2>/dev/null || true
        echo "  ROCSHMEM_RELAXED_ORDER unset (strict)"
    fi

    for RUN_NUM in $(seq 1 $NUM_RUNS); do
        for NUM_TOKENS in ${TOKEN_COUNTS}; do
            RESULT_DIR="${RESULTS_BASE}/${ORDERING}_${TIMESTAMP}"
            mkdir -p "${RESULT_DIR}"
            OUTFILE="${RESULT_DIR}/run${RUN_NUM}_tok${NUM_TOKENS}.log"

            echo ""
            echo "--- ${ORDERING} | run ${RUN_NUM}/${NUM_RUNS} | ${NUM_TOKENS} tokens ---"

            # Use a unique port per run to avoid EADDRINUSE from lingering sockets
            PORT_COUNTER=$((PORT_COUNTER + 1))
            export MASTER_PORT=$((8361 + PORT_COUNTER))

            cd /opt/src/DeepEP/tests

            # Run with retry: if first attempt fails (IPC/port issue), wait and retry once
            RUN_OK=0
            for ATTEMPT in 1 2; do
                if python3 test_low_latency.py \
                    --num-tokens "${NUM_TOKENS}" \
                    --num-experts 288 \
                    --hidden 7168 \
                    --num-topk 8 \
                    --num-processes 8 \
                    2>&1 | tee "${OUTFILE}"; then
                    RUN_OK=1
                    break
                else
                    echo "  WARN: attempt ${ATTEMPT} failed (exit code $?)"
                    if [ "$ATTEMPT" -eq 1 ]; then
                        echo "  Waiting 5s before retry..."
                        sleep 5
                        PORT_COUNTER=$((PORT_COUNTER + 1))
                        export MASTER_PORT=$((8361 + PORT_COUNTER))
                    fi
                fi
            done

            if [ "$RUN_OK" -eq 1 ]; then
                echo "  Saved: ${OUTFILE}"
            else
                echo "  FAILED: ${OUTFILE} (both attempts failed)"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi

            # Allow TCP sockets and GPU IPC handles to fully release
            sleep 3
        done
    done
done

echo ""
echo "================================================================"
echo "All runs complete. Failures: ${FAIL_COUNT}"
echo "  Strict results: ${RESULTS_BASE}/strict_${TIMESTAMP}/"
echo "  Relaxed results: ${RESULTS_BASE}/relaxed_${TIMESTAMP}/"
echo "================================================================"
