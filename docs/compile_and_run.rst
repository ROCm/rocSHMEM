-------------------------
Running rocSHMEM Programs
-------------------------

Compiling and Linking with rocSHMEM
-----------------------------------

RocSHMEM is built as a library that can be statically
linked to your application during compilation using ``hipcc``.

During the compilation of your application, include the rocSHMEM header files
and the rocSHMEM library when using ``hipcc``.
Since rocSHMEM depends on MPI (in version 6.4.0, this requirement may be dropped
in future versions) you will need to link with an MPI library.
The arguments for MPI linkage must be added manually as opposed to using ``mpicc``.

When using ``hipcc`` directly (as opposed to through a build system), we
recommend performing the compilation and linking steps separately.

For example, one can refer to how to compile the examples files (``./examples/*`` in
the source tarball) with the following compile and link commands:

.. code-block:: bash

  # Compile
  hipcc -c -fgpu-rdc -x hip rocshmem_allreduce_test.cc \
    -I/opt/rocm/include                                \
    -I$ROCSHMEM_INSTALL_DIR/include                    \
    -I$OPENMPI_UCX_INSTALL_DIR/include/

  # Link
  hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test \
    $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                       \
    $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                        \
    -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

If your project uses cmake, you may refer to 
`Using CMake with AMD ROCm <https://rocmdocs.amd.com/en/latest/conceptual/cmake-packages.html>`_.

Running a rocSHMEM program
--------------------------

Program that use rocSHMEM will typically deploy multiple processes (Typically, one per GPU). 
The MPI launcher (e.g., ``mpiexec`` when using Open MPI) is used to start the required number
of processes. As an example, one may launch 2 getmem example processes (available when compiled from source) using the following command line:

.. code-block:: bash

  mpiexec --map-by numa --mca pml ucx --mca osc ucx -np 2 ./build/examples/rocshmem_getmem_test

Please refer to the Open MPI documentation for more information about ``mpiexec`` command line parameters.

.. note::
  Some systems may have multiple installs of MPI, some of which would not
  have GPU support enabled. Make sure you use the ``mpiexec`` from the expected
  MPI library, notably when using the MPI you built yourself
  as part of :ref:`install-dependencies`.

Environment Variables
---------------------

The behavior of rocSHMEM can be controlled with the following environment variables:

.. list-table:: Environment Variables
    :widths: 30 10 20
    :header-rows: 1

    * - Name
      - Default Value
      - Description
    * - ROCSHMEM_HEAP_SIZE
      - 1 GB
      - Defines the size of the rocSHMEM symmetric heap.
        Note the heap is on the GPU memory.
    * - ROCSHMEM_MAX_NUM_CONTEXTS
      - 1024
      - Defines the number of contexts an application can use
    * - ROCSHMEM_MAX_NUM_TEAMS
      - 40
      - Defines the number of teams an application can use
