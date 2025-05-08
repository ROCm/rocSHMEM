.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _install-rocshmem:

---------------------------
Installing rocSHMEM
---------------------------

This topic describes how to install rocSHMEM.

The file `README.md <https://github.com/ROCm/rocSHMEM/blob/rocm-6.4.0/README.md>`_ in the rocSHMEM sources may contain additional information.

Requirements
---------------------------

1. ROCm stack installed on the system (HIP runtime)

  * ROCm v6.4.0 or later

2. AMD GPUs

  * MI250X

  * MI300X

3. ROCm-aware Open MPI and UCX as described in Building Dependencies

Installing from a Package Manager
---------------------------------

On Ubuntu, rocSHMEM can be installed with the following command:

.. code-block:: bash

   apt install rocshmem-dev

.. note::

  This installation method requires ROCm 6.4 or newer. Dependencies
  (open MPI and UCX) still need to be built following the instructions
  in the next paragraph, as the distribution packaged versions do not
  include full accelerator support.

.. _install-dependencies:

Building Dependencies
---------------------------

rocSHMEM requires a ROCm-Aware Open MPI and UCX.
Other MPI implementations, such as MPICH,
*should* be compatible, if rocSHMEM is built from source,
but it has not been thoroughly tested.

To build and configure ROCm-Aware UCX (1.17.0 or later), you need to:

.. code-block:: bash

  git clone https://github.com/ROCm/ucx.git -b v1.17.x
  cd ucx
  ./autogen.sh
  ./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --enable-mt
  make -j 8
  make -j 8 install

Then, you need to build Open MPI (5.0.7 or later) with UCX support.

.. code-block:: bash

  git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
  cd ompi
  ./autogen.pl
  ./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --with-ucx=<ucx_path>
  make -j 8
  make -j 8 install

Alternatively, we have script to install dependencies.
Configuration options are platform dependent, so please review the script to
check for fitness with your system.

.. code-block:: bash

  export BUILD_DIR=/path/to/not_rocshmem_src_or_build/dependencies
  /path/to/rocshmem_src/scripts/install_dependencies.sh

For more information on OpenMPI-UCX support, please visit:
`GPU-enabled Message Passing Interface <https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html>`_

Installing rocSHMEM from Source
--------------------------------

The following method can be used to build and install rocSHMEM with the IPC
on-node, GPU-to-GPU backend:

.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/ipc_single

The build script passes configuration options to CMake to setup a canonical
build.
There are other scripts for experimental configurations in the
`./scripts/build_configs` directory, but currently, only `ipc_single`
is supported.

By default, the library is installed in `~/rocshmem`. You may provide a
custom install path by supplying it as an argument. For example:

.. code-block:: bash

  ../scripts/build_configs/ipc_single /path/to/install

