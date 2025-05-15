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

* ROCm 6.4.0 or later, including the :doc:`HIP runtime <hip:index>`.

* AMD GPUs

  * MI250X

  * MI300X

* ROCm-aware Open MPI and UCX. For more information, see :ref:`install-dependencies`.

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

Installing rocSHMEM from source
--------------------------------

To build and install rocSHMEM with the IPC on-node, GPU-to-GPU backend, run:

.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/ipc_single

The build script passes configuration options to CMake to setup a canonical build. 

.. note::

  Other experimental configuration scripts are available in ``./scripts/build_configs``, but only ``ipc_single`` is currently supported.


By default, the library is installed in ``~/rocshmem``. You can customize the installation path by running:

.. code-block:: bash

  ../scripts/build_configs/ipc_single /path/to/install

