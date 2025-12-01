.. meta::
  :description: rocSHMEM environment variables reference
  :keywords: rocSHMEM, ROCm, API, environment variables, environment, reference

********************************************************************
rocSHMEM environment variables
********************************************************************

This section describes the important environment variables used to
control the behavior of rocSHMEM.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``ROCSHMEM_HEAP_SIZE``
        | Defines the size of the rocSHMEM symmetric heap in bytes.
      - ``1073741824`` (1 GB)
      - | Size in bytes.
        | Note the heap is on the GPU memory.

    * - | ``ROCSHMEM_MAX_NUM_CONTEXTS``
        | Defines the number of contexts an application can use.
      - ``32``
      - Maximum number of contexts.

    * - | ``ROCSHMEM_MAX_NUM_TEAMS``
        | Defines the number of teams an application can use.
      - ``40``
      - Maximum number of teams.

    * - | ``ROCSHMEM_UNIQUEID_WITH_MPI``
        | Defines whether rocSHMEM is expected to use MPI when using the uniqueId based initialization.
      - ``0``
      - | 0: Do not use MPI.
        | 1: Use MPI.

    * - | ``ROCSHMEM_DISABLE_MIXED_IPC``
        | Defines whether to force using the network conduit even when IPC is available.
      - ``0``
      - | 0: Use IPC when available.
        | 1: Force network conduit.

    * - | ``ROCSHMEM_GDA_ALTERNATE_QP_PORTS``
        | Enables/Disables having QPs alternate their mappings across rocSHMEM contexts.
      - ``1``
      - | 0: Disabled.
        | 1: Enabled (helps saturate bandwidth on multiport bonded interfaces).

    * - | ``ROCSHMEM_GDA_PCIE_RELAXED_ORDERING``
        | Enables PCIe Relaxed Ordering when registering the symmetric heap with the RDMA NICs.
      - ``0``
      - | 0: Disabled.
        | 1: Enabled.
