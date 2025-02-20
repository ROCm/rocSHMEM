.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

****************************
rocSHMEM Documentation
****************************

The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD and AMD Research initiative
to provide GPU-centric networking through an OpenSHMEM-like interface.
This intra-kernel networking library simplifies application code complexity and
enables more fine-grained communication/computation overlap
than traditional host-driven networking.
rocSHMEM uses a single symmetric heap (SHEAP) that is allocated on GPU memories.

The code is open and hosted at `<https://github.com/ROCm/rocSHMEM>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`install-rocshmem`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
