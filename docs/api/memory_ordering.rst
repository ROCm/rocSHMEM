.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-memory-ordering:

---------------------------
Memory Ordering Routines
---------------------------

ROCSHMEM_FENCE
--------------

.. cpp:function:: __device__ void rocshmem_fence()
.. cpp:function:: __device__ void rocshmem_fence(int pe)
.. cpp:function:: __device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx)
.. cpp:function:: __device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx, int pe)

 :param ctx: Context with which to perform this operation
 :param pe:  Destination pe
 :returns:   None

**Description:**
Guarantees order between messages in this context in accordance with OpenSHMEM semantics.

ROCSHMEM_QUIET
--------------

.. cpp:function:: __device__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx)
.. cpp:function:: __device__ void rocshmem_quiet()

 :param ctx: Context with which to perform this operation
 :returns:   None

**Description:**
Completes all previous operations posted to this context.
