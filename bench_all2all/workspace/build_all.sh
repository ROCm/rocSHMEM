#!/bin/bash
#
# Verify or rebuild the rocSHMEM + DeepEP stack inside the bench_all2all image.
#
# The Dockerfile already builds both components during image creation. This
# script is mainly a convenience entry point for verification and optional
# rebuilds after editing sources inside the container.
#

set -euo pipefail

FORCE_REBUILD=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE_REBUILD=1
fi

if [[ ! -f /.dockerenv ]]; then
    echo "Error: this script must run inside the Docker container."
    echo "Run: docker compose -f bench_all2all/docker-compose.yml exec rocm bash"
    exit 1
fi

SRC_DIR="${SRC_DIR:-/opt/src}"
ROCSHMEM_SRC="${SRC_DIR}/rocSHMEM"
DEEPEP_SRC="${SRC_DIR}/DeepEP"

echo "=========================================="
echo "bench_all2all build helper"
echo "=========================================="
echo "Source directory: ${SRC_DIR}"
echo "Force rebuild:    ${FORCE_REBUILD}"
echo ""

echo "[1/2] rocSHMEM"
if [[ ${FORCE_REBUILD} -eq 1 || ! -d /root/rocshmem ]]; then
    ROCSHMEM_BUILD_DIR=/tmp/rocshmem_build
    rm -rf "${ROCSHMEM_BUILD_DIR}"
    mkdir -p "${ROCSHMEM_BUILD_DIR}"
    cd "${ROCSHMEM_BUILD_DIR}"

    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/root/rocshmem \
        -DCMAKE_VERBOSE_MAKEFILE=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_RELAX_TESTS=ON \
        -DBUILD_FUNCTIONAL_TESTS=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DDEBUG=OFF \
        -DPROFILE=OFF \
        -DUSE_GDA=ON \
        -DUSE_RO=ON \
        -DUSE_IPC=ON \
        -DGDA_MLX5=ON \
        -DUSE_THREADS=OFF \
        -DUSE_WF_COAL=OFF \
        -DUSE_HDP_FLUSH=OFF \
        -DUSE_HDP_FLUSH_HOST_SIDE=OFF \
        -DUSE_EXTERNAL_MPI=OFF \
        "${ROCSHMEM_SRC}"
    cmake --build . --parallel "$(nproc)"
    cmake --install .

    if [[ -f tests/relax_tests/rocshmem_relax_tests ]]; then
        cp tests/relax_tests/rocshmem_relax_tests /usr/local/bin/
    fi
    if [[ -f tests/relax_tests/rocshmem_relax_driver.sh ]]; then
        cp tests/relax_tests/rocshmem_relax_driver.sh /usr/local/bin/
        chmod +x /usr/local/bin/rocshmem_relax_driver.sh
    fi
else
    echo "rocSHMEM already installed at /root/rocshmem"
fi

echo ""
echo "[2/2] DeepEP"
DEEPEP_PRESENT="$(python3 -c 'import importlib.util; print(importlib.util.find_spec("deep_ep") is not None)' 2>/dev/null || echo False)"
if [[ ${FORCE_REBUILD} -eq 1 || "${DEEPEP_PRESENT}" != "True" ]]; then
    cd "${DEEPEP_SRC}"
    export PYTORCH_ROCM_ARCH=gfx942
    export ROCSHMEM_DIR=/root/rocshmem
    python3 setup.py --variant rocm --nic cx7 build install
else
    echo "DeepEP already installed"
fi

echo ""
echo "Verification"
python3 -c "import deep_ep; print('DeepEP import: OK')"
if [[ -x /usr/local/bin/rocshmem_relax_tests ]]; then
    echo "rocSHMEM relax_tests binary: /usr/local/bin/rocshmem_relax_tests"
fi
echo ""
echo "Build helper completed successfully."
