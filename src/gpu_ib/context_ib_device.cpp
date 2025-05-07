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
#include <rocshmem/rocshmem.hpp>

#include "context_incl.hpp"
#include "backend_ib.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

GPUIBContext::GPUIBContext(GPUIBBackend *backend, int idx)
    : Context(backend) {
  networkImpl = backend->networkImpl;
  base_heap = backend->heap.get_heap_bases().data();
  networkImpl.networkHostInit(this, idx);
  barrier_sync = backend->barrier_sync;
}

__device__ __host__ QueuePair *GPUIBContext::getQueuePair(int pe) {
  return networkImpl.getQueuePair(device_qp_proxy, pe);
}

__device__ void GPUIBContext::quiet() {
  for (int k = 0; k < networkImpl.num_pes; k++) {
    getQueuePair(k)->quiet();
  }
}

__device__ void *GPUIBContext::shmem_ptr(const void *dest, int pe) {
  return nullptr;
}

__device__ void GPUIBContext::putmem(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qp->put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
      qp->quiet();
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

__device__ void GPUIBContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qp->put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

__device__ void GPUIBContext::putmem_wave(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  qp->put_nbi_wave(base_heap[pe] + L_offset, source, nelems, pe);
  qp->quiet();
}

__device__ void GPUIBContext::putmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  if (is_thread_zero_in_wave()) {
    auto *qp = getQueuePair(pe);
    qp->put_nbi_wave(base_heap[pe] + L_offset, source, nelems, pe);
  }
}

}  // namespace rocshmem
