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

#ifndef LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP_
#define LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP_

#include "rocshmem_config.h"
#include "gpu_ib/context_ib_host.hpp"

namespace rocshmem {

template <typename T>
__host__
void Context::p(T *dest, T value, int pe) {
  static_cast<GPUIBHostContext*>(this)->p(dest, value, pe);
}

template <typename T>
__host__
T Context::g(const T *source, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->g(source, pe);
  return ret_val;
}

template <typename T>
__host__
void Context::put(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->put(dest, source, nelems, pe);
}

template <typename T>
__host__
void Context::get(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->get(dest, source, nelems, pe);
}

template <typename T>
__host__
void Context::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->put_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ 
void Context::get_nbi(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->get_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ 
T Context::amo_fetch_add(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->amo_fetch_add(dst, value, pe);
  return ret_val;
}

template <typename T>
__host__ 
void Context::amo_add(void *dst, T value, int pe) {
  static_cast<GPUIBHostContext*>(this)->amo_add(dst, value, pe);
}

template <typename T>
__host__ 
void Context::amo_set(void *dst, T value, int pe) {
  static_cast<GPUIBHostContext*>(this)->amo_set(dst, value, pe);
}

template <typename T>
__host__ 
T Context::amo_swap(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->amo_swap(dst, value, pe);
  return ret_val;
}

template <typename T>
__host__ 
T Context::amo_fetch_and(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->amo_fetch_and(dst, value, pe);
  return ret_val;
}

template <typename T>
__host__ 
void Context::amo_and(void *dst, T value, int pe) {
  static_cast<GPUIBHostContext*>(this)->amo_and(dst, value, pe);
}

template <typename T>
__host__ 
T Context::amo_fetch_or(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->amo_fetch_or(dst, value, pe);
  return ret_val;
}

template <typename T>
__host__ 
void Context::amo_or(void *dst, T value, int pe) {
  static_cast<GPUIBHostContext*>(this)->amo_or(dst, value, pe);
}

template <typename T>
__host__ 
T Context::amo_fetch_xor(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->amo_fetch_xor(dst, value, pe);
  return ret_val;
}

template <typename T>
__host__ 
void Context::amo_xor(void *dst, T value, int pe) {
  static_cast<GPUIBHostContext*>(this)->amo_xor(dst, value, pe);
}

template <typename T>
__host__ 
T Context::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->amo_fetch_cas(dst, value, cond, pe);
  return ret_val;
}

template <typename T>
__host__ 
void Context::amo_cas(void *dst, T value, T cond, int pe) {
  static_cast<GPUIBHostContext*>(this)->amo_cas(dst, value, cond, pe);
}

template <typename T>
__host__ 
void Context::broadcast(T *dest, const T *source, int nelems, int pe_root, int pe_start, int log_pe_stride, int pe_size, long *p_sync) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->broadcast<T>(dest, source, nelems, pe_root, pe_start, log_pe_stride, pe_size, p_sync);
}

template <typename T>
__host__ 
void Context::broadcast(rocshmem_team_t team, T *dest, const T *source, int nelems, int pe_root) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->broadcast<T>(team, dest, source, nelems, pe_root);
}

template <typename T, ROCSHMEM_OP Op>
__host__ 
void Context::to_all(T *dest, const T *source, int nreduce, int PE_start, int logPE_stride, int PE_size, T *pWrk, long *pSync) {
  if (nreduce == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->to_all<T, Op>(dest, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
}

template <typename T, ROCSHMEM_OP Op>
__host__ 
int Context::reduce(rocshmem_team_t team, T *dest, const T *source, int nreduce) {
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }
  auto ret_val = static_cast<GPUIBHostContext*>(this)->reduce<T, Op>(team, dest, source, nreduce);
  return ret_val;
}

template <typename T>
__host__ 
void Context::wait_until(T *ivars, int cmp, T val) {
  static_cast<GPUIBHostContext*>(this)->wait_until<T>(ivars, cmp, val);
}

template <typename T>
__host__ 
size_t Context::wait_until_any(T *ivars, size_t nelems, const int* status, int cmp, T val) {
  return static_cast<GPUIBHostContext*>(this)->wait_until_any<T>(ivars, nelems, status, cmp, val);
}

template <typename T>
__host__ 
void Context::wait_until_all(T *ivars, size_t nelems, const int* status, int cmp, T val) {
  static_cast<GPUIBHostContext*>(this)->wait_until_all<T>(ivars, nelems, status, cmp, val);
}

template <typename T>
__host__ 
size_t Context::wait_until_some(T *ivars, size_t nelems, size_t* indices, const int* status, int cmp, T val) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->wait_until_some<T>(ivars, nelems, indices, status, cmp, val);
  return ret_val;
}

template <typename T>
__host__ 
void Context::wait_until_all_vector(T *ivars, size_t nelems, const int *status, int cmp, T* vals) {
  static_cast<GPUIBHostContext*>(this)->wait_until_all_vector<T>(ivars, nelems, status, cmp, vals);
}

template <typename T>
__host__ 
size_t Context::wait_until_any_vector(T *ivars, size_t nelems, const int *status, int cmp, T* vals) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->wait_until_any_vector<T>(ivars, nelems, status, cmp, vals);
  return ret_val;
}

template <typename T>
__host__
size_t Context::wait_until_some_vector(T *ivars, size_t nelems, size_t* indices, const int *status, int cmp, T* vals) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->wait_until_some_vector<T>(ivars, nelems, indices, status, cmp, vals);
  return ret_val;
}

template <typename T>
__host__
int Context::test(T *ivars, int cmp, T val) {
  auto ret_val = static_cast<GPUIBHostContext*>(this)->test<T>(ivars, cmp, val);
  return ret_val;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP_
