.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-amo:

---------------------------
Atomic Memory Operations
---------------------------

- These functions can be called from divergent control paths at per-thread
  granularity.

ROSHMEM_ATOMIC_FETCH
--------------------
.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_fetch(TYPE *source, int pe)
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_fetch(rocshmem_ctx_t ctx, TYPE *source, int pe)

 :param ctx:  Context with which to perform this operation
 :param dest: Destination address; Must be an address on the symmetric heap
 :param pe:   PE of the remote process

 :returns:    The value of dest

**Description:**
Atomically return the value of dest to the calling PE.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in EXTENDED_AMO_TYPES_.


SHMEM_ATOMIC_SET
----------------
.. cpp:function:: __device__ void rocshmem_TYPENAME_atomic_set(TYPE *dest, TYPE value, int pe);
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_atomic_set(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);

 :param ctx:  Context with which to perform this operation
 :param dest: Destination address; Must be an address on the symmetric heap
 :param val:  The value to be atomically set
 :param pe:   PE of the remote process

 :returns:    None

**Description:**
Atomically set the value val to dest on pe.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in EXTENDED_AMO_TYPES_.

SHMEM_ATOMIC_COMPARE_SWAP
-------------------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_compare_swap(rocshmem_ctx_t ctx, TYPE *dest, TYPE cond, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param cond:    The value to be compare with
  :param val:     The value to be atomically swapped
  :param pe:      PE of the remote process

  :return:        The old value of dest

**Description:**
Atomically compares if the value in dest with cond is equal then put val in dest.
The operation returns the older value of dest to the calling PE.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in STANDARD_AMO_TYPES_.

SHMEM_ATOMIC_SWAP
-----------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_swap(TYPE *dest, TYPE value, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_swap(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param val:     The value to be atomically swapped
  :param pe:      PE of the remote process

  :return:        The old value of dest

**Description:**
Atomically swaps the value val to dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in EXTENDED_AMO_TYPES_.

SHMEM_ATOMIC_FETCH_INC
----------------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_fetch_inc(TYPE *dest, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_fetch_inc(rocshmem_ctx_t ctx, TYPE *dest, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param pe:      PE of the remote process

  :return:        The old value of dest

**Description:**
Atomically adds 1 to dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in STANDARD_AMO_TYPES_.

SHMEM_ATOMIC_INC
----------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_atomic_inc(TYPE *dest, TYPE pe);
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_atomic_inc(rocshmem_ctx_t ctx, TYPE *dest, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param pe:      PE of the remote process

  :return:        None

**Description:**
Atomically adds 1 to dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in STANDARD_AMO_TYPES_.

SHMEM_ATOMIC_FETCH_ADD
----------------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_fetch_add(TYPE *dest, TYPE value, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_fetch_add(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically added
  :param pe:      PE of the remote process

  :return:        The old value of dest

**Description:**
Atomically adds value to dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in STANDARD_AMO_TYPES_.

SHMEM_ATOMIC_ADD
----------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_atomic_add(TYPE *dest, TYPE value, TYPE pe);
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_atomic_add(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically added
  :param pe:      PE of the remote process

  :return:        None

**Description:**
Atomically adds value to dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in STANDARD_AMO_TYPES_.

SHMEM_ATOMIC_FETCH_AND
----------------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_fetch_and(TYPE *dest, TYPE value, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_fetch_and(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically AND
  :param pe:      PE of the remote process

  :return:        The old value of dest.

**Description:**
Atomically bitwise-and value to the value at dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in BITWISE_AMO_TYPES_.

SHMEM_ATOMIC_AND
----------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_and(TYPE *dest, TYPE value, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_and(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically AND
  :param pe:      PE of the remote process

  :return:        None

**Description:**
Atomically bitwise-and value to the value at dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in BITWISE_AMO_TYPES_.

SHMEM_ATOMIC_FETCH_OR
----------------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_fetch_or(TYPE *dest, TYPE value, TYPE pe)
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_fetch_or(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe)


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically OR
  :param pe:      PE of the remote process

  :return:        The old value of dest

**Description:**
Atomically bitwise-or value to the value at dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in BITWISE_AMO_TYPES_.

SHMEM_ATOMIC_OR
---------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_or(TYPE *dest, TYPE value, TYPE pe)
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_or(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe)


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically OR
  :param pe:      PE of the remote process

  :return:        None. 

**Description:**
Atomically bitwise-or value to the value at dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in BITWISE_AMO_TYPES_.

SHMEM_ATOMIC_FETCH_XOR
----------------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_fetch_xor(TYPE *dest, TYPE value, TYPE pe);
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_fetch_xor(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe);


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically XOR
  :param pe:      PE of the remote process

  :return:        The old value of dest

**Description:**
Atomically bitwise-xor value to the value at dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in BITWISE_AMO_TYPES_.

SHMEM_ATOMIC_XOR
----------------

.. cpp:function:: __device__ TYPE rocshmem_TYPENAME_atomic_xor(TYPE *dest, TYPE value, TYPE pe)
.. cpp:function:: __device__ TYPE rocshmem_ctx_TYPENAME_atomic_xor(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, TYPE pe)


  :param ctx:     Context with which to perform this operation
  :param dest:    Destination address; Must be an address on the symmetric heap
  :param value:   The value to be atomically XOR
  :param pe:      PE of the remote process

  :return:        None

**Description:**
Atomically bitwise-xor value to the value at dest on pe.
The operation is blocking.

Valid ``TYPENAME`` and ``TYPE`` values can be seen in BITWISE_AMO_TYPES_.

SUPPORTED AMO DATA TYPES
------------------------

.. _STANDARD_AMO_TYPES:

.. list-table:: Standard AMO Datatypes
    :widths: 10 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
      - Supported
    * - int
      - int
      - Yes
    * - long
      - long
      - Yes
    * - long long
      - longlong
      - Yes
    * - unsigned int
      - uint
      - Yes
    * - unsigned long
      - ulong
      - Yes
    * - unsigned long long
      - ulonglong
      - Yes
    * - int32_t
      - int32
      - Yes
    * - int64_t
      - int64
      - Yes
    * - uint32_t
      - uint32
      - Yes
    * - uint64_t
      - uint64
      - Yes
    * - size_t
      - size
      - Yes
    * - ptrdiff_t
      - ptrdiff
      - Yes

.. _EXTENDED_AMO_TYPES:

.. list-table:: Extended AMO Datatypes
    :widths: 10 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
      - Supported
    * - float
      - float
      - Yes
    * - double
      - double
      - Yes
    * - int
      - int
      - Yes
    * - long
      - long
      - Yes
    * - long long
      - longlong
      - Yes
    * - unsigned int
      - uint
      - Yes
    * - unsigned long
      - ulong
      - Yes
    * - unsigned long long
      - ulonglong
      - Yes
    * - int32_t
      - int32
      - Yes
    * - int64_t
      - int64
      - Yes
    * - uint32_t
      - uint32
      - Yes
    * - uint64_t
      - uint64
      - Yes
    * - size_t
      - size
      - Yes
    * - ptrdiff_t
      - ptrdiff
      - Yes

.. _BITWISE_AMO_TYPES:

.. list-table:: Bitwise AMO Datatypes
    :widths: 10 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
      - Supported
    * - unsigned int
      - uint
      - Yes
    * - unsigned long
      - ulong
      - Yes
    * - unsigned long long
      - ulonglong
      - Yes
    * - int32_t
      - int32
      - Yes
    * - int64_t
      - int64
      - Yes
    * - uint32_t
      - uint32
      - Yes
    * - uint64_t
      - uint64
      - Yes

