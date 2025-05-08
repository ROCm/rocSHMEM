.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-coll:

---------------------------
Collective Routines
---------------------------

ROCSHMEM_BARRIER_ALL
--------------------

.. cpp:function:: __device__ void rocshmem_ctx_wg_barrier_all(rocshmem_ctx_t ctx)
.. cpp:function:: __device__ void rocshmem_wg_barrier_all()

  :param ctx: Context with which to perform this operation
  :returns:   None

**Description:**
Perform a collective barrier between all PEs in the system.
The caller is blocked until the barrier is resolved.

ROCSHMEM_TEAM_SYNC
------------------

.. cpp:function:: __device__ void rocshmem_ctx_wg_team_sync(rocshmem_ctx_t ctx, rocshmem_team_t team)
.. cpp:function:: __device__ void rocshmem_wg_team_sync(rocshmem_team_t team)

  :param ctx:  Context with which to perform this operation
  :param team: Team with which to perform this operation
  :returns:    None

**Description:**
Registers the arrival of a PE at a barrier.
The caller is blocked until the synchronization is resolved.

In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
completion and visibility of previously issued memory stores and does not
ensure completion of remote memory updates issued via OpenSHMEM routines.

ROCSHMEM_SYNC_ALL
-----------------

.. cpp:function:: __device__ void rocshmem_ctx_wg_sync_all(rocshmem_ctx_t ctx)
.. cpp:function:: __device__ void rocshmem_wg_sync_all()

  :param ctx:  Context with which to perform this operation
  :returns:    None

**Description:**
This routine is the same as ``rocshmem_wg_team_sync`` if were to be called on the world team.


ROSHMEM_ALLTOALL
----------------

.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_wg_alltoall(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems)

  :param team:   The team participating in the collective
  :param dest:   Destination address; Must be an address on the
                 symmetric heap
  :param source: Source address; Must be an address on the symmetric
                 heap
  :param nelems: Number of data blocks transferred per pair of PEs
  :returns:      None

**Description:**
Exchanges a fixed amount of contiguous data blocks between all pairs
of PEs participating in the collective routine.
This function must be called as a work-group collective.

Valid ``TYPENAME`` and ``TYPE`` values can be seen at :ref:`RMA_TYPES`.

ROCSHMEM_BROADCAST
------------------

.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_wg_broadcast(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int pe_root)

  :param ctx:    Context with which to perform this collective
  :param team:   The team participating in the collective
  :param dest:   Destination address; Must be an address on the
                 symmetric heap
  :param source: Source address; Must be an address on the symmetric
                 heap
  :param nelems: Number of data blocks transferred per pair of PEs
  :returns:      None

**Description:**
Perform a broadcast between PEs in the team.
The caller is blocked until the broadcast completes.

Valid ``TYPENAME`` and ``TYPE`` values can be seen at :ref:`RMA_TYPES`.

ROCSHMEM_FCOLLECT
-----------------

.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_wg_fcollect(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems)

  :param ctx:    Context with which to perform this collective
  :param team:   The team participating in the collective
  :param dest:   Destination address; Must be an address on the
                 symmetric heap
  :param source: Source address; Must be an address on the symmetric
                 heap
  :param nelems: Number of data blocks transferred per pair of PEs.
  :returns:      None

**Description:**
Concatenates blocks of data from multiple PEs to an array in every
PE participating in the collective routine.

ROCSHMEM_REDUCTION
------------------
.. cpp:function:: __device__ int rocshmem_ctx_TYPENAME_OPNAME_wg_reduce(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nreduce)

  :param ctx:     Context with which to perform this collective
  :param team:    The team participating in the collective
  :param dest:    Destination address; Must be an address on the
                  symmetric heap
  :param source:  Source address; Must be an address on the symmetric
                  heap
  :param nreduce: Number of data blocks transferred per pair of PEs
  :returns:       Zero on successful local completion; Nonzero otherwise


**Description:**
Perform an allreduce between PEs in the team.

Valid ``TYPENAME``, ``TYPE``, and ``OPNAME`` values can be seen at :ref:`REDUCE_TYPES`.

SUPPORTED REDUCTION TYPES AND OPERATIONS
----------------------------------------

.. _REDUCE_TYPES:

.. list-table:: Reduction Types, Names and Operations
    :widths: 20 20 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
      - OPNAME
      - Supported
    * - char
      - char
      - max, min, sum, prod
      - No
    * - signed char
      - schar
      - max, min, sum, prod
      - No
    * - short
      - short
      - max, min, sum, prod
      - Yes
    * - int
      - int
      - max, min, sum, prod
      - Yes
    * - long
      - long
      - max, min, sum, prod
      - Yes
    * - long long
      - longlong
      - max, min, sum, prod
      - Yes
    * - ptrdiff_t
      - ptrdiff
      - max, min, sum, prod
      - No
    * - unsigned char
      - uchar
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned short
      - ushort
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned int
      - uint
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned long
      - ulong
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned long long
      - ulonglong
      - and, or, xor, max, min, sum, prod
      - No
    * - int8_t
      - int8
      - and, or, xor, max, min, sum, prod
      - No
    * - int16_t
      - int16
      - and, or, xor, max, min, sum, prod
      - No
    * - int32_t
      - int32
      - and, or, xor, max, min, sum, prod
      - No
    * - int64_t
      - int64
      - and, or, xor, max, min, sum, prod
      - No
    * - uint8_t
      - uint8
      - and, or, xor, max, min, sum, prod
      - No
    * - uint16_t
      - uint16
      - and, or, xor, max, min, sum, prod
      - No
    * - uint32_t
      - uint32
      - and, or, xor, max, min, sum, prod
      - No
    * - uint64_t
      - uint64
      - and, or, xor, max, min, sum, prod
      - No
    * - size_t
      - size
      - and, or, xor, max, min, sum, prod
      - No
    * - float
      - float
      - max, min, sum, prod
      - Yes
    * - double
      - double
      - max, min, sum, prod
      - Yes
    * - long double
      - longdouble
      - max, min, sum, prod
      - No
    * - double _Complex
      - complexd
      - sum, prod
      - No
    * - float _Complex
      - complexf
      - sum, prod
      - No
