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

#ifndef LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_DEVICE_HPP_

#include <rocshmem/rocshmem.hpp>

#include "context_ib_device.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

template <typename T>
__device__ void GPUIBContext::p(T *dest, T value, int pe) {
  putmem_nbi(dest, &value, sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::put(T *dest, const T *source, size_t nelems, int pe) {
  putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_add(void *dst, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  return qp->atomic_fetch(base_heap[pe] + L_offset, value, 0, pe, MLX5_OPCODE_ATOMIC_FA);
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  return qp->atomic_fetch(base_heap[pe] + L_offset, value, cond, pe, MLX5_OPCODE_ATOMIC_CS);
}

template <typename T>
__device__ void GPUIBContext::amo_add(void *dst, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  qp->atomic_nofetch(base_heap[pe] + L_offset, value, 0, pe, MLX5_OPCODE_ATOMIC_FA);
}

template <typename T>
__device__ void GPUIBContext::amo_set(void *dst, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];

  auto *qp = getQueuePair(pe);

  T ret_val;
  T cond = 0;
  while ((ret_val = qp->atomic_fetch(base_heap[pe] + L_offset, value, cond, pe, MLX5_OPCODE_ATOMIC_CS))) {
    if (ret_val == cond) {
      break;
    }
    cond = ret_val;
  }
}

template <typename T>
__device__ T GPUIBContext::amo_swap(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void GPUIBContext::amo_cas(void *dst, T value, T cond, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  auto *qp = getQueuePair(pe);
  qp->atomic_nofetch(base_heap[pe] + L_offset, value, cond, pe, MLX5_OPCODE_ATOMIC_CS);
}

template <typename T>
__device__ void GPUIBContext::put_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_DEVICE_HPP_
