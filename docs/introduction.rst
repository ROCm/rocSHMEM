.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-introduction:

---------------------------
Introduction
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

How OpenSHMEM applications should intereact with GPUs is currently undefined and
is an active discussion topic within the OpenSHMEM community.
rocSHMEM is based upon the OpenSHMEM specification
and it tries to adheres to the specifiction the best it can with regards to GPU semantics.

Applications that use HIP can be easily interface with rocSHMEM.
As per the HIP programing model,
rocSHMEM has `__host__` APIs which are to be called from host code,
and `__device__` APIs which can be called within GPU Kernels.
Any device APIs which do not have any special suffixes/infixes (e.g. `_wg` or `_wave`)
can be called by single GPU thread.
Each thread can call into these APIs with a different parameters and
will block until the calling wavefront completes.
These APIs can be called in divergent code paths but it is not recommened.

Wavefront APIs
==============
The wavefront APIs are any API calls that have the suffix `_wave`.
The parameters in which these routines are called must be
the same for every thread in the wavefront.
If each thread calls these routines with different parameters, the behaviour will be undefined.
These APIs will block until the calling wavefront completes.

Workgroup APIs
==============
The workgroup APIs are any API calls that have the suffix `_wg` or infix `_wg_`.
The parameters in which these routines are called must be
the same for every thread in the workgroup.
If each thread calls these routines with different parameters, the behaviour will be undefined.
These APIs will block until the calling workgroup completes.
