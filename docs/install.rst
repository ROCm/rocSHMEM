.. meta::
  :description: Instruction on how to install rocSHMEM.
  :keywords: rocSHMEM, ROCm, install, build, dependencies, MPI, UCX, Open MPI

.. _install-rocshmem:

---------------------------
Installing rocSHMEM
---------------------------

This topic describes how to install rocSHMEM.

Requirements
---------------------------

* ROCm 6.4.0 or later, including the :doc:`HIP runtime <hip:index>`. For more information, see `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_.

* AMD GPUs

  * MI250X

  * MI300X

* ROCm-aware Open MPI and UCX. For more information, see :ref:`install-dependencies`.

* Inter-node with the Reverse Offload backend is supported only for CX7 Infiniband.

Available network backends
--------------------------

rocSHMEM currently supports two different network backends:

1. The **IPC** backend (Inter-Process Communication) supports communication between GPUs on the same host (using ROCm interprocess GPU communication mechanisms). This is the fastest backend for on-node inter-GPU communication, but it cannot communicate inter-node.
2. The **RO** backend (Reverse-Offload) supports communication between GPUs on separate nodes connected through a NIC, using a host-based proxy to forward the communication orders to/from the GPU. In this release RO is the only inter-node communication backend, and is implemented on top of an MPI-RMA compatibility layer.

The IPC and RO backends can be activated in the same build of rocSHMEM, in which case intra-node communication will use IPC, and inter-node communication will use RO. Note that when RO is active, all atomic operations will use RO (even for intra-node communication).

Installing from a package manager
---------------------------------

On Ubuntu, you can install rocSHMEM by running:

.. code-block:: bash

   apt install rocshmem-dev

.. note::

  This installation method requires ROCm 6.4 or later. You must manually build dependencies such as Open MPI and UCX, because the distribution packaged versions don't include full accelerator support. For more information, see :ref:`install-dependencies`.

.. _install-dependencies:

Building dependencies
---------------------------

rocSHMEM requires ROCm-Aware Open MPI and UCX. Other MPI implementations, such as MPICH, have not been fully tested.

To build and configure ROCm-Aware UCX 1.17.0 or later, run:

.. code-block:: bash

  git clone https://github.com/ROCm/ucx.git -b v1.17.x
  cd ucx
  ./autogen.sh
  ./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --enable-mt
  make -j 8
  make -j 8 install

To build Open MPI 5.0.7 or later with UCX support, run:

.. code-block:: bash

  git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
  cd ompi
  ./autogen.pl
  ./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --with-ucx=<ucx_path>
  make -j 8
  make -j 8 install

Alternatively, you can use a script to install dependencies:

.. code-block:: bash

  export BUILD_DIR=/path/to/not_rocshmem_src_or_build/dependencies
  /path/to/rocshmem_src/scripts/install_dependencies.sh

.. note::

  Configuration options vary by platform. Review the script to ensure it is compatible with your system.

For more information about OpenMPI-UCX support, see
`GPU-enabled Message Passing Interface <https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html>`_.

Installing from source
--------------------------------

rocSHMEM currently has two communication backends that can be selected at build time: RO and IPC.
The default configuration enables both backends, and will use, at runtime, IPC for intra-node communication,
and RO for inter-node communication (rocSHMEM atomic operations always use RO in this configuration).

The IPC only configuration is still possible, a benefit of this setup is that it will benefit from
performing rocSHMEM atomic operation using the IPC backend.

RO+IPC backend build
^^^^^^^^^^^^^^^^^^^^

To build and install rocSHMEM with the hybrid RO-IPC off-node,on-node backends, run:

.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/ro_ipc

The build script passes configuration options to CMake to setup a canonical build.

.. note::

  The only supported and tested configuration for the RO backend is when using Open MPI and UCX with a CX7 Infiniband adapter (see :ref:`install-dependencies`). Using other configurations may be possible (notably when the MPI implementation is thread-safe and supports GPU buffers) but is considered experimental.


IPC only backend build
^^^^^^^^^^^^^^^^^^^^^^

To build and install rocSHMEM with the IPC on-node, GPU-to-GPU backend, run:

.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/ipc_single

The build script passes configuration options to CMake to setup a single-node build.
This is similar to the default build in ROCm 6.4.

.. note::

  The default configuration changed from IPC only in ROCm 6.4 (as built by script ``ipc_single``) to RO+IPC in ROCm 7.0 (as built by script ``ro_ipc``).
  Other experimental configuration scripts are available in ``./scripts/build_configs``, but only ``ipc_single`` and ``ro_ipc``
  are currently supported.

Installation prefix
^^^^^^^^^^^^^^^^^^^

By default, the build scripts install the library in ``~/rocshmem``. You can customize the installation path by running:

.. code-block:: bash

  ../scripts/build_configs/ro_ipc /path/to/install

or alternatively for an IPC only build:

.. code-block:: bash

  ../scripts/build_configs/ipc_single /path/to/install

