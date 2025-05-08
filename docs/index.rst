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
rocSHMEM uses a single symmetric heap (SHEAP) that is allocated on GPU memories. To learn more, see :doc:`introduction`

The code is open and hosted at `<https://github.com/ROCm/rocSHMEM>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

      * :doc:`Install rocSHMEM <./install>`

  .. grid-item-card:: How to

      * :doc:`Compile and Run rocSHMEM Programs <./compile_and_run>`

  .. grid-item-card:: API Reference

      * :doc:`Library Setup, Exit, and Query Routines <./api/init>`
      * :doc:`Memory Management Routines <./api/memory_management>`
      * :doc:`Team Management Routines <./api/teams>`
      * :doc:`Context Management Routines <./api/ctx>`
      * :doc:`Remote Memory Access Routines <./api/rma>`
      * :doc:`Atomic Memory Operations <./api/amo>`
      * :doc:`Signaling Operations <./api/sigops>`
      * :doc:`Collective Routines <./api/coll>`
      * :doc:`Point-to-Point Synchronization Routines <./api/pt2pt_sync>`
      * :doc:`Memory Ordering Routines <./api/memory_ordering>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
