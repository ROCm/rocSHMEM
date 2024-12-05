# ROCm OpenSHMEM (rocSHMEM)

The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD Research
initiative to provide a unified runtime for heterogeneous systems.
rocSHMEM supports both host-centric (a traditional host-driven
OpenSHMEM runtime) and GPU-centric networking (provided a GPU kernel
the ability to perform network operations) through an
OpenSHMEM-like interface. This intra-kernel networking simplifies application
code complexity and enables more fine-grained communication/computation
overlap than traditional host-driven networking.

rocSHMEM's primary target is heterogeneous computing; hence, for both
CPU-centric and GPU-centric communications, rocSHMEM uses a single
symmetric heap (SHEAP) that is allocated on GPU memories.

rocSHMEM's GPU-centric communication has two different backend designs.
The backends primarily differ in their implementations of
intra-kernel networking.

The first design will be referred to as the the GPU InfiniBand (GPU-IB)
backend.  This backend implements a lightweight InfiniBand verbs interface
on the GPU.  The GPU itself is responsible with building commands and ringing
the doorbell on the NIC to send network commands.  GPU-IB is the default and
preferred backend design that offers the best performance.

The second design will be referred to as the Reverse Offload (RO) backend. With
the RO backend, the GPU runtime forwards rocSHMEM networking operations to the
host-side runtime, which calls into a traditional MPI or OpenSHMEM
implementation.  This forwarding of requests is transparent to the
programmer, who only sees the GPU-side interface.

Both designs of the GPU-centric interface coexist seamlessly with the
CPU-centric interface of the unified runtime. rocSHMEM ensures that CPU-centric
updates to the SHEAP are consistent and visible to a GPU kernel that is executing
in parallel to host-initiated communication.

## Limitations

rocSHMEM is an experimental prototype from AMD Research and not an official
ROCm product.  The software is provided as-is with no guarantees of support
from AMD or AMD Research.

rocSHMEM base requirements:
* ROCm version 4.3.1 onwards
    *  May work with other versions, but not tested
* AMD GFX9 GPUs (e.g.: MI25, Vega 56, Vega 64, MI50, MI60, MI100, Radeon VII)
* AMD MI200 GPUs: To enable the support on MI200, please configure the library
 with USE_COHERENT_HEAP
* ROCm-aware MPI as described in
  [Building the Dependencies](#building-the-dependencies)
* InfiniBand adaptor compatable with ROCm RDMA technology
* UCX 1.6 or greater with ROCm support

rocSHMEM optional requirements
 * For Documentation:
     *  Doxygen

rocSHMEM only supports HIP applications. There are no plans to port to
OpenCL.

## Building and Installation

rocSHMEM uses the CMake build system. The CMakeLists file contains
additional details about library options.

To create an out-of-source build:

    mkdir build
    cd build

Next, choose one configuration from the build_configs subdirectory. These
scripts pass configuration options to CMake to setup canonical builds which
are regularly tested:

    ../scripts/build_configs/dc_single
    ../scripts/build_configs/dc_multi
    ../scripts/build_configs/rc_single
    ../scripts/build_configs/rc_multi
    ../scripts/build_configs/rc_multi_wf_coal
    ../scripts/build_configs/ro_net_basic

By default, the library is installed in `~/rocshmem`. You may provide a
custom install path by supplying it as an argument. For example:

    ../scripts/build_configs/rc_single /path/to/install

## Compiling/linking and Running with rocSHMEM

rocSHMEM is built as a host and device side library that can be statically
linked to your application during compilation using hipcc.

During the compilation of your application, include the rocSHMEM header files
and the rocSHMEM library when using hipcc:

    -I/path/to/rocshmem/install/include
    -L/path/to/rocshmem/install/lib -lrocshmem

NOTE: rocSHMEM depends on MPI for its host code. So, you will need to link
to an MPI library. Since you must use the hipcc compiler, the arguments for
MPI linkage must be added manually as opposed to using mpicc. Similary,
rocSHMEM depends on Verbs for its device code. So, you will need to link
to a Verbs library.

When using hipcc directly (as opposed to through a build system), we
recommend performing the compilation and linking steps separately.
Here are the steps to build a standalone program, say
rocshmem_hello.cpp.

```
# Compile
/opt/rocm/bin/hipcc ./rocshmem_hello.cpp -I/path/to/rocshmem/install/include -fgpu-rdc -o ./rocshmem_hello.o -c

# Link
/opt/rocm/bin/hipcc ./rocshmem_hello.o /path/to/rocshmem/install/lib/librocshmem.a -lmpi -lmlx5 -libverbs -lhsa-runtime64 -fgpu-rdc -o rocshmem_hello

```

If your project uses cmake, please refer to the CMakeLists.txt files
in the clients directory for examples. You may also find the
[Using CMake with AMD ROCm](https://rocmdocs.amd.com/en/latest/conceptual/cmake-packages.html)
page useful.

## Runtime Parameters

    ROCSHMEM_HEAP_SIZE (default : 1 GB)
                        Defines the size of the OpenSHMEM symmetric heap
                        Note the heap is on the GPU memory.

    ROCSHMEM_SQ_SIZE   (default 1024)
                        Defines the size of the SQ as number of network
                        packet (WQE). Each WQE is 64B. This only for
                        GPU-IB conduit

    ROCSHMEM_USE_CQ_GPU_MEM  (default : 1)
                        Set the placement of CQ on GPU memory (1)
                        or CPU memory (0)

    ROCSHMEM_USE_SQ_GPU_MEM  (default : 1)
                        Set the placement of SQ on GPU memory (1)
                        or CPU memory (0)

    RO_NET_CPU_QUEUE    (default: not set)
                        Force producer/consumer queues between CPU and GPU to
                        be in CPU memory. RO backend only.

rocSHMEM also requires the following environment variable be set for ROCm:

    export HSA_FORCE_FINE_GRAIN_PCIE=1

## Documentation

To generate doxygen documentation for rocSHMEM's API, run the following
from the library's build directory:

    make docs

The doxygen output will be in the `docs` folder of the build directory.

## Examples

rocSHMEM is similar to OpenSHMEM and should be familiar to programmers who
have experience with OpenSHMEM or other PGAS network programming APIs in the
context of CPUs. The best way to learn how to use rocSHMEM is to read the
autogenerated doxygen documentation for functions described in
`rocshmem/rocshmem.hpp`, or to look at the provided sample applications in the
`tests/` folder. rocSHMEM is shipped with a basic test suite for the
supported rocSHMEM API. The examples test Puts, Gets, nonblocking Puts,
nonblocking Gets, Quiets, Atomics, Tests, Wai-untils, Broadcasts, and
Reductions.

To run the examples, you may use the driver scripts provided in respective
folders of device- or host-initiated communication examples. Simply
executing `./driver.sh` will show the help message on how to use the script.
Here are some example uses of the driver script:

    ./scripts/functional_tests/driver.sh ./build/rocshmem_example_driver single_thread ./build   (for device-initiated communication)
    ./scripts/sos_tests/driver.sh ./build short                                           (for host-initiated communication)

## Building the Dependencies

rocSHMEM requires an MPI runtime on the host that supports ROCm-Aware MPI.
Currently all ROCm-Aware MPI runtimes require the usage of ROCm-Aware UCX.

To build and configure ROCm-Aware UCX, you need to:
 1. Download the latest UCX
 2. Configure and build UCX with ROCm support: --with-rocm=/opt/rocm

Then, you need to build your MPI (OpenMPI or MPICH CH4) with UCX support.

For more information on OpenMPI-UCX support, please visit:
https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX

For more information on MPICH-UCX support, please visit:
https://www.mpich.org/about/news/
