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

#ifndef LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_HOST_HPP_
#define LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_HOST_HPP_

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "host/host_templates.hpp"

namespace rocshmem {

template <typename T>
__host__ void GPUIBHostContext::p(T *dest, T value, int pe) {
  host_interface->p<T>(dest, value, pe, context_window_info);
}

template <typename T>
__host__ void GPUIBHostContext::put(T *dest, const T *source, size_t nelems,
                                    int pe) {
  host_interface->put<T>(dest, source, nelems, pe, context_window_info);
}

template <typename T>
__host__ void GPUIBHostContext::put_nbi(T *dest, const T *source, size_t nelems,
                                        int pe) {
  host_interface->put_nbi<T>(dest, source, nelems, pe, context_window_info);
}

template <typename T>
__host__ void GPUIBHostContext::amo_add(void *dst, T value, int pe) {
  host_interface->amo_add(dst, value, pe, context_window_info);
}

template <typename T>
__host__ void GPUIBHostContext::amo_cas(void *dst, T value, T cond, int pe) {
  host_interface->amo_cas(dst, value, cond, pe, context_window_info);
}

template <typename T>
__host__ T GPUIBHostContext::amo_fetch_add(void *dst, T value, int pe) {
  return host_interface->amo_fetch_add(dst, value, pe, context_window_info);
}

template <typename T>
__host__ T GPUIBHostContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  return host_interface->amo_fetch_cas(dst, value, cond, pe,
                                       context_window_info);
}

template <typename T>
__host__ void GPUIBHostContext::wait_until(T *ivars, int cmp, T val) {
  host_interface->wait_until<T>(ivars, cmp, val, context_window_info);
}

template <typename T>
__host__ void GPUIBHostContext::wait_until_all(T *ivars, size_t nelems,
                                               const int* status,
                                               int cmp, T val) {
  host_interface->wait_until_all<T>(ivars, nelems, status, cmp, val, context_window_info);
}

template <typename T>
__host__ size_t GPUIBHostContext::wait_until_any(T *ivars, size_t nelems,
                                               const int* status,
                                               int cmp, T val) {
  return host_interface->wait_until_any<T>(ivars, nelems, status, cmp, val, context_window_info);
}

template <typename T>
__host__ size_t GPUIBHostContext::wait_until_some(T *ivars, size_t nelems,
                                                 size_t* indices,
                                                 const int* status,
                                                 int cmp, T val) {
  return host_interface->wait_until_some<T>(ivars, nelems, indices, status, cmp, val, context_window_info);
}

template <typename T>
__host__ int GPUIBHostContext::test(T *ivars, int cmp, T val) {
  return host_interface->test<T>(ivars, cmp, val, context_window_info);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_HOST_HPP_
