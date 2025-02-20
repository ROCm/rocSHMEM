
---------------------
Environment Variables
---------------------

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
