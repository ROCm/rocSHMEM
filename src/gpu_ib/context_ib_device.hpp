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

#ifndef LIBRARY_SRC_GPU_IB_CONTEXT_IB_DEVICE_HPP_
#define LIBRARY_SRC_GPU_IB_CONTEXT_IB_DEVICE_HPP_

#include "context.hpp"
#include "network_policy.hpp"

namespace rocshmem {

class QueuePair;

class GPUIBContext : public Context {
 public:
  __host__ GPUIBContext(GPUIBBackend *b, int idx);

  __device__ __host__ QueuePair *getQueuePair(int pe);

  __device__ __attribute__((noinline)) void threadfence_system();

  __device__ void ctx_destroy();

  __device__ void putmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void putmem_nbi(void *dest, const void *source, size_t nelems, int pe);

  __device__ void fence();

  __device__ void fence(int pe);

  __device__ void quiet();

  __device__ void *shmem_ptr(const void *dest, int pe);

  __device__ void barrier_all();

  __device__ void sync_all();

  __device__ void sync(rocshmem_team_t team);

  template <typename T>
  __device__ void amo_add(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_set(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_swap(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ T amo_fetch_add(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ void p(T *dest, T value, int pe);

  template <typename T>
  __device__ void put(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi(T *dest, const T *source, size_t nelems, int pe);

  __device__ void putmem_wave(void *dest, const void *source, size_t nelems, int pe);

  __device__ void putmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

 private:
  __device__ void internal_direct_barrier(int pe, int PE_start, int stride, int n_pes, int64_t *pSync);

  __device__ void internal_atomic_barrier(int pe, int PE_start, int stride, int n_pes, int64_t *pSync);

  __device__ void internal_sync(int pe, int PE_start, int stride, int PE_size, int64_t *pSync);

  __device__ void quiet_single(int cq_num);

 public:
  /*
   * Collection of queue pairs that are currently checked out by this
   * context from GPUIBBackend.
   */
  QueuePair *device_qp_proxy{nullptr};

  /*
   * Array of char * pointers corresponding to the heap base pointers VA for
   * each PE that we can communicate with.
   */
  char *const *base_heap{nullptr};

  /*
   * Buffer used to store the results of a *_g operation. These ops do not
   * provide a destination buffer, so the runtime must manage one.
   */
  char *g_ret{nullptr};

  NetworkImpl networkImpl{};

  /*
   * Temporary scratchpad memory used by internal barrier algorithms.
   */
  int64_t *barrier_sync{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_CONTEXT_IB_DEVICE_HPP_
