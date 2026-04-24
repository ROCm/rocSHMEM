FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Install InfiniBand tools, build dependencies, and utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # InfiniBand tools
    infiniband-diags \
    libibverbs-dev \
    ibverbs-utils \
    libibmad5 \
    libibnetdisc5t64 \
    libibumad3 \
    # Build dependencies (for DeepEP and rocSHMEM)
    cmake \
    libpci-dev \
    libopenmpi-dev \
    openmpi-bin \
    libgoogle-glog-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Remove pip-installed cmake to avoid conflicts with system cmake
RUN pip uninstall -y cmake 2>/dev/null || true

# ============================================================================
# Copy the checked-out rocSHMEM tree from this repository.
# The docker-compose file sets the build context to the repository root.
# ============================================================================
COPY CMakeLists.txt /opt/src/rocSHMEM/
COPY AUTHORS.md CHANGELOG.md CONTRIBUTING.md LICENSE.md README.md /opt/src/rocSHMEM/
COPY cmake /opt/src/rocSHMEM/cmake
COPY docs /opt/src/rocSHMEM/docs
COPY examples /opt/src/rocSHMEM/examples
COPY include /opt/src/rocSHMEM/include
COPY scripts /opt/src/rocSHMEM/scripts
COPY src /opt/src/rocSHMEM/src
COPY tests /opt/src/rocSHMEM/tests
COPY utils /opt/src/rocSHMEM/utils

# ============================================================================
# Build rocSHMEM with IPC, GDA, MLX5, and relax_tests support
# Installed to /root/rocshmem/ (matches ROCSHMEM_DIR convention)
# ============================================================================
RUN mkdir -p /tmp/rocshmem_build && cd /tmp/rocshmem_build && \
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
        /opt/src/rocSHMEM && \
    cmake --build . --parallel $(nproc) && \
    cmake --install . && \
    # Install relax_tests binary and driver script
    cp /tmp/rocshmem_build/tests/relax_tests/rocshmem_relax_tests /usr/local/bin/ && \
    cp /tmp/rocshmem_build/tests/relax_tests/rocshmem_relax_driver.sh /usr/local/bin/ && \
    chmod +x /usr/local/bin/rocshmem_relax_tests /usr/local/bin/rocshmem_relax_driver.sh && \
    # Clean up build artifacts
    rm -rf /tmp/rocshmem_build

# ============================================================================
# Copy DeepEP patch (fence + wait_until relaxed ordering support)
# ============================================================================
COPY bench_all2all/patches/deepep_relaxed_ll.patch /opt/patches/deepep_relaxed_ll.patch

# ============================================================================
# Clone DeepEP, build it against rocSHMEM, and clean up
# ============================================================================
WORKDIR /opt/src
RUN git clone --depth 1 https://github.com/ROCm/DeepEP.git && \
    cd DeepEP && \
    git fetch --depth 1 origin e84464ecae4805ff88088c8496480d51e4092bab && \
    git checkout e84464ecae4805ff88088c8496480d51e4092bab && \
    git apply /opt/patches/deepep_relaxed_ll.patch && \
    # Build DeepEP against modified rocSHMEM
    cd /opt/src/DeepEP && \
    PYTORCH_ROCM_ARCH=gfx942 ROCSHMEM_DIR=/root/rocshmem \
    python3 setup.py --variant rocm --nic cx7 build install && \
    # Clean up .git dirs and pip cache
    rm -rf /opt/src/DeepEP/.git && \
    find /opt/src -name ".git" -type d -exec rm -rf {} + 2>/dev/null; \
    pip cache purge; rm -rf /root/.cache/pip

WORKDIR /workspace
