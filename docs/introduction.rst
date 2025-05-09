.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-introduction:

---------------------------
What is rocSHMEM?
---------------------------

The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD and AMD Research initiative
to provide GPU-centric networking through an OpenSHMEM-like interface.
This intra-kernel networking library simplifies application code complexity and
enables more fine-grained communication/computation overlap
than traditional host-driven networking.
rocSHMEM uses a single symmetric heap (SHEAP) that is allocated on GPU memories.

The code is open and hosted at `<https://github.com/ROCm/rocSHMEM>`_.

The rocSHMEM Programming Model
-------------------------------

Defining how OpenSHMEM applications interact with GPUs remains an
ongoing active discussion within the OpenSHMEM community, and the OpenSHMEM
specification has yet to coalesce on this topic.
rocSHMEM extends beyond the OpenSHMEM specification to add semantic that
support GPU kernel communication, while maintaining close resemblance to
the original OpenSHMEM specification semantics. 

Applications that use HIP can be easily interface with rocSHMEM.
As per the HIP programming model,
rocSHMEM has `__host__` APIs which are to be called from host code,
and `__device__` APIs which can be called within GPU Kernels.
Any device APIs which do not have any special suffixes/infixes (e.g. `_wg` or `_wave`)
must be called by a single thread.
GPU specific `_wg` and `_wave` APIs are expected to be called from multiple GPU threads
and block until the calling scope completes.
These APIs can be called in divergent code paths but this is not recommended.

Wavefront APIs
==============
The wavefront APIs are any API calls that have the suffix `_wave`.
The parameters in which these routines are called must be
the same for every thread in the wavefront.
If any thread calls these routines with differing parameters, the behavior is undefined.
These APIs will block until the calling wavefront completes.

Workgroup APIs
==============
The workgroup APIs are any API calls that have the suffix `_wg` or infix `_wg_`.
The parameters in which these routines are called must be
the same for every thread in the workgroup.
If any thread calls these routines with differing parameters, the behavior is undefined.
These APIs will block until the calling workgroup completes.
