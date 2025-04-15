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

#ifndef LIBRARY_SRC_CONTEXT_HPP_
#define LIBRARY_SRC_CONTEXT_HPP_

#include <hip/hip_runtime.h>

#include "host/host.hpp"
#include "wf_coal_policy.hpp"

namespace rocshmem {

class GPUIBBackend;

/**
 * @file context.hpp
 * @brief Context class corresponds directly to an OpenSHMEM context.
 *
 * GPUs perform networking operations on a context that is created by the
 * application programmer or a "default context" managed by the runtime.
 */
class Context {
 public:
  __host__ Context(GPUIBBackend* handle);

  __device__ Context(GPUIBBackend* handle);

  /**************************************************************************
   ***************************** DEVICE METHODS *****************************
   *************************************************************************/
  template <typename T>
  __device__
  void wait_until(T *ivars, int cmp, T val);

  template <typename T>
  __device__
  void wait_until_all(T *ivars, size_t nelems, const int *status, int cmp, T val);

  template <typename T>
  __device__
  size_t wait_until_any(T *ivars, size_t nelems, const int *status, int cmp, T val);

  template <typename T>
  __device__
  size_t wait_until_some(T *ivars, size_t nelems, size_t* indices, const int *status, int cmp, T val);

  template <typename T>
  __device__
  int test(T *ivars, int cmp, T val);

  __device__
  void ctx_create();

  __device__
  void ctx_destroy();

  __device__
  void putmem(void* dest, const void* source, size_t nelems, int pe);

  __device__
  void putmem_nbi(void* dest, const void* source, size_t nelems, int pe);

  __device__
  void quiet();

  __device__
  void* shmem_ptr(const void* dest, int pe);

  __device__
  void barrier_all();

  __device__
  void barrier(rocshmem_team_t team);

  __device__
  void sync_all();

  __device__
  void sync(rocshmem_team_t team);

  template <typename T>
  __device__
  void amo_add(void* dst, T value, int pe);

  template <typename T>
  __device__
  void amo_set(void* dst, T value, int pe);

  template <typename T>
  __device__
  T amo_swap(void* dst, T value, int pe);

  template <typename T>
  __device__
  void amo_cas(void* dst, T value, T cond, int pe);

  template <typename T>
  __device__
  T amo_fetch_add(void* dst, T value, int pe);

  template <typename T>
  __device__
  T amo_fetch_cas(void* dst, T value, T cond, int pe);

  template <typename T>
  __device__
  void p(T* dest, T value, int pe);

  template <typename T>
  __device__
  void put(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__
  void put_nbi(T* dest, const T* source, size_t nelems, int pe);

  __device__
  void putmem_wave(void* dest, const void* source, size_t nelems, int pe);

  __device__
  void putmem_nbi_wave(void* dest, const void* source, size_t nelems, int pe);

  template <typename T>
  __device__
  void put_wave(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__
  void put_nbi_wave(T* dest, const T* source, size_t nelems, int pe);

  /**************************************************************************
   ****************************** HOST METHODS ******************************
   *************************************************************************/
  template <typename T>
  __host__
  void p(T* dest, T value, int pe);

  template <typename T>
  __host__
  void put(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __host__
  void put_nbi(T* dest, const T* source, size_t nelems, int pe);

  __host__
  void putmem(void* dest, const void* source, size_t nelems, int pe);

  __host__
  void putmem_nbi(void* dest, const void* source, size_t nelems, int pe);

  template <typename T>
  __host__
  void amo_add(void* dst, T value, int pe);

  template <typename T>
  __host__
  void amo_set(void* dst, T value, int pe);

  template <typename T>
  __host__
  T amo_swap(void* dst, T value, int pe);

  template <typename T>
  __host__
  void amo_cas(void* dst, T value, T cond, int pe);

  template <typename T>
  __host__
  T amo_fetch_add(void* dst, T value, int pe);

  template <typename T>
  __host__
  T amo_fetch_cas(void* dst, T value, T cond, int pe);

  __host__
  void quiet();

  __host__
  void barrier_all();

  __host__
  void sync_all();

  template <typename T>
  __host__
  void wait_until(T *ivars, int cmp, T val);

  template <typename T>
  __host__
  void wait_until_all(T *ivars, size_t nelems, const int *status, int cmp, T val);

  template <typename T>
  __host__
  size_t wait_until_any(T *ivars, size_t nelems, const int *status, int cmp, T val);

  template <typename T>
  __host__
  size_t wait_until_some(T *ivars, size_t nelems, size_t* indices, const int *status, int cmp, T val);

  template <typename T>
  __host__
  int test(T *ivars, int cmp, T val);

 public:
  /**************************************************************************
   ***************************** PUBLIC MEMBERS *****************************
   *************************************************************************/
  /**
   * @brief Duplicated local copy of backend's num_pes
   */
  int num_pes{0};

  /**
   * @brief Duplicated local copy of backend's my_pe
   */
  int my_pe{-1};

 protected:
  /**************************************************************************
   ***************************** POLICY MEMBERS *****************************
   *************************************************************************/

  /**
   * @brief Coalesce policy for 'multi' configuration builds
   */
  WavefrontCoalescer wf_coal_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTEXT_HPP_
