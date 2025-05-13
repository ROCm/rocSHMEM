.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-init:

---------------------------------------
Library setup, exit, and query routines
---------------------------------------

ROCSHMEM_INIT
-------------

.. cpp:function:: __host__ void rocshmem_init(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine initializes the rocSHMEM runtime and underlying transport layer.
Before ``rocshmem_init`` is called,
you must select the device that this PE is associated to by calling
`hipSetDevice
<https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___device.html#ga43c1e7f15925eeb762195ccb5e063eae>`_.

.. cpp:function:: __device__ void rocshmem_wg_init(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine initializes device-side rocSHMEM resources.
It must be called before any threads in this work-group invoke other rocSHMEM functions.
It must be called collectively by all threads in the work-group.

ROCSHMEM_FINALIZE
-----------------
.. cpp:function:: __host__ void rocshmem_finalize(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine finalizes the rocSHMEM runtime.

.. cpp:function:: __device__ void rocshmem_wg_finalize(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine finalizes device-side rocSHMEM resources.
It must be called before work-group completion if the work-group also called ``rocshmem_wg_init``.
It must be called collectively by all threads in the work-group.

ROCSHMEM_N_PES
--------------

.. cpp:function:: __host__ int rocshmem_n_pes(void)

  :Parameters: None.
  :returns: Total number of PEs.

**Description:**
This routine queries the total number of PEs.
It can be called before ``rocshmem_init``.

.. cpp:function:: __device__ int rocshmem_n_pes(void)
.. cpp:function:: __device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx)

  :param ctx: GPU side context handle.
  :returns: Total number of PEs.

**Description:**
This routine queries the total number of PEs for a given context.
It can be called per thread with no performance penalty.

ROCSHMEM_MY_PE
--------------

.. cpp:function:: __host__ int rocshmem_my_pe(void)

  :Parameters: None.
  :returns: PE ID of the caller.

**Description:**
This routine queries the PE ID of the caller.
It can be called before ``rocshmem_init``.

.. cpp:function:: __device__ int rocshmem_my_pe(void)
.. cpp:function:: __device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx)

  :param ctx: GPU side context handle.
  :returns: PE ID of the caller.

**Description:**
This routine queries the PE ID of the caller.
It can be called per thread with no performance penalty.
