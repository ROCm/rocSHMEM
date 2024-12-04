/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "context_ib_device.hpp"

#include <hip/hip_runtime.h>

#include "config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "../backend_type.hpp"
#include "../context_incl.hpp"
#include "backend_ib.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

__host__ GPUIBContext::GPUIBContext(Backend *backend, bool option, int idx)
    : Context(backend, option) {
  GPUIBBackend *b{static_cast<GPUIBBackend *>(backend)};
  ctx_idx = idx;
  networkImpl = b->networkImpl;
  base_heap = b->heap.get_heap_bases().data();
  networkImpl.networkHostInit(this, idx);

  barrier_sync = b->barrier_sync;
  ipcImpl_.ipc_bases = b->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = b->ipcImpl.shm_size;
}

__device__ void GPUIBContext::ctx_create() {
  /* Nothing to do in the GPU_IB backend */
  return;
}

/*
 * TODO(bpotter): these will go in a policy class based on DC/RC.
 * I am not completely sure at this point what else is needed in said class,
 * so just leave them up here for now.
 */
__device__ __host__ QueuePair *GPUIBContext::getQueuePair(int pe) {
  return networkImpl.getQueuePair(device_qp_proxy, pe);
}

__device__ __host__ int GPUIBContext::getNumQueuePairs() {
  return networkImpl.getNumQueuePairs();
}

__device__ __host__ int GPUIBContext::getNumDest() {
  return networkImpl.getNumDest();
}

__device__ void GPUIBContext::fence() {
#ifdef USE_SINGLE_NODE
  threadfence_system();
#else

  for (int k = 0; k < getNumDest(); k++) {
    getQueuePair(k)->fence(k);
  }

  fence_.flush();
#endif
}

__device__ void GPUIBContext::fence(int pe) {
#ifdef USE_SINGLE_NODE
  threadfence_system();
#else
  getQueuePair(pe)->fence(pe);
  fence_.flush();
#endif
}

__device__ void GPUIBContext::putmem_nbi(void *dest, const void *source,
                                         size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - base_heap[my_pe];

  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset,
                     const_cast<void *>(source), nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }

    auto *qp = getQueuePair(pe);
    qp->put_nbi<THREAD>(base_heap[pe] + L_offset, source, nelems, pe, true);
  }
}

__device__ void GPUIBContext::getmem_nbi(void *dest, const void *source,
                                         size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }

    auto *qp = getQueuePair(pe);
    qp->get_nbi<THREAD>(base_heap[pe] + L_offset, dest, nelems, pe, true);
  }
}

__device__ void GPUIBContext::quiet() {
#ifdef USE_SINGLE_NODE
  threadfence_system();
  for (int pe = 0; pe < ipcImpl_.shm_size; pe++) {
    if (pe != my_pe) {
      ipcImpl_.zero_byte_read(pe);
    }
  }
#else

  for (int k = 0; k < getNumDest(); k++) {
    getQueuePair(k)->quiet_single_heavy<THREAD>(k);
  }
  fence_.flush();

#endif
}

__device__ void *GPUIBContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
    int local_pe = pe % ipcImpl_.shm_size;
    ret = ipcImpl_.ipc_bases[local_pe] + L_offset;
  }
  return ret;
}

__device__ void GPUIBContext::threadfence_system() {
  int thread_id = get_flat_block_id();

  if (thread_id % WF_SIZE == lowerID()) {
#ifdef USE_SINGLE_NODE
    // Flush current PE HDP
    HdpPolicy::hdp_flush(
        reinterpret_cast<unsigned int *>(networkImpl.hdp_address));

    // Flush the rest of the HDPs
    for (int pe = 0; pe < ipcImpl_.shm_size; pe++) {
      auto target_address = networkImpl.hdp_address;
      const int value = HdpPolicy::HDP_FLUSH_VAL;
      if (pe != my_pe) {
        const int value = HdpPolicy::HDP_FLUSH_VAL;
        auto mapped_address =
            shmem_ptr(reinterpret_cast<void *>(target_address), pe);
        __hip_atomic_store(static_cast<int *>(mapped_address), value,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
#else
    getQueuePair(my_pe)->hdp_policy->flushCoherency();
#endif
  }

  __threadfence_system();
}

__device__ void GPUIBContext::getmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    auto *qp = getQueuePair(pe);
    qp->get_nbi_cqe<THREAD>(base_heap[pe] + L_offset, dest, nelems, pe, true);
    qp->quiet_single<THREAD>();
  }
  fence_.flush();
}

__device__ void GPUIBContext::putmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset,
                     const_cast<void *>(source), nelems);

    threadfence_system();
    ipcImpl_.zero_byte_read(pe);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    auto *qp = getQueuePair(pe);
    qp->put_nbi_cqe<THREAD>(base_heap[pe] + L_offset, source, nelems, pe, true);
    qp->quiet_single<THREAD>();
  }
  fence_.flush();
}

/******************************************************************************
 ************************ WORKGROUP/WAVE-LEVEL RMA API ************************
 *****************************************************************************/
__device__ void GPUIBContext::putmem_nbi_wg(void *dest, const void *source,
                                            size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset,
                        const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_block()) {
      auto *qp = getQueuePair(pe);
      qp->put_nbi<WG>(base_heap[pe] + L_offset, source, nelems, pe, true);
    }
  }
  __syncthreads();
}

__device__ void GPUIBContext::putmem_nbi_wave(void *dest, const void *source,
                                              size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset,
                          const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      auto *qp = getQueuePair(pe);
      qp->put_nbi<WAVE>(base_heap[pe] + L_offset, source, nelems, pe, true);
    }
  }
}

__device__ void GPUIBContext::putmem_wg(void *dest, const void *source,
                                        size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset,
                        const_cast<void *>(source), nelems);
    __syncthreads();
    threadfence_system();
    ipcImpl_.zero_byte_read(pe);
  } else {
    auto *qp = getQueuePair(pe);
    if (is_thread_zero_in_block()) {
      qp->put_nbi_cqe<WG>(base_heap[pe] + L_offset, source, nelems, pe, true);
    }
    qp->quiet_single<WG>();
  }
  __syncthreads();
  fence_.flush();
}

__device__ void GPUIBContext::putmem_wave(void *dest, const void *source,
                                          size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset,
                          const_cast<void *>(source), nelems);
    threadfence_system();
    ipcImpl_.zero_byte_read(pe);
  } else {
    if (is_thread_zero_in_wave()) {
      qp->put_nbi_cqe<WAVE>(base_heap[pe] + L_offset, source, nelems, pe, true);
    }
    qp->quiet_single<WAVE>();
  }
  fence_.flush();
}

__device__ void GPUIBContext::getmem_wg(void *dest, const void *source,
                                        size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    if (is_thread_zero_in_block()) {
      qp->get_nbi_cqe<WG>(base_heap[pe] + L_offset, dest, nelems, pe, true);
    }
    qp->quiet_single<WG>();
  }
  __syncthreads();
  fence_.flush();
}

__device__ void GPUIBContext::getmem_wave(void *dest, const void *source,
                                          size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset,
                          nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      qp->get_nbi_cqe<WAVE>(base_heap[pe] + L_offset, dest, nelems, pe, true);
    }
    qp->quiet_single<WAVE>();
  }
  fence_.flush();
}

__device__ void GPUIBContext::getmem_nbi_wg(void *dest, const void *source,
                                            size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    if (is_thread_zero_in_block()) {
      auto *qp = getQueuePair(pe);
      qp->get_nbi<WG>(base_heap[pe] + L_offset, dest, nelems, pe, true);
    }
  }
  __syncthreads();
}

__device__ void GPUIBContext::getmem_nbi_wave(void *dest, const void *source,
                                              size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_typed) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset,
                          nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      auto *qp = getQueuePair(pe);
      qp->get_nbi<WAVE>(base_heap[pe] + L_offset, dest, nelems, pe, true);
    }
  }
}

}  // namespace rocshmem
