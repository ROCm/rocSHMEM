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

#ifndef LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP_

#include "rocshmem_config.h"
#include "gpu_ib/context_ib_device.hpp"

namespace rocshmem {

template <typename T>
__device__
void Context::p(T *dest, T value, int pe) {
  static_cast<GPUIBContext*>(this)->p(dest, value, pe);
}

template <typename T>
__device__
void Context::put(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->put(dest, source, nelems, pe);
}

template <typename T>
__device__
void Context::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->put_nbi(dest, source, nelems, pe);
}

template <typename T>
__device__ __forceinline__
void Context::wait_until(T *ivars, int cmp, T val) {
  while (!test(ivars, cmp, val)) {
  }
}

__device__ __forceinline__
size_t status_entry(size_t nelems, const int *status) {
  size_t i{0};
  while (i < nelems) {
    if (status[i] == 0) {
      return i;
    }
    i++;
  }
  return i;
}

template <typename T>
__device__ __forceinline__
size_t Context::wait_until_any(T *ivars, size_t nelems, const int *status, int cmp, T val) {
  // zero nelems error condition
  if (!nelems) {
    return SIZE_MAX;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return SIZE_MAX;
  }

  while (true) {
    for (size_t i{pos}; i < nelems; i++) {
      // skip entries marked with non-zero status
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val)) {
        return i;
      }
    }
  }
}

template <typename T>
__device__ __forceinline__
void Context::wait_until_all(T *ivars, size_t nelems, const int *status, int cmp, T val) {
  // zero nelems error condition
  if (!nelems) {
    return;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return;
  }

  for (size_t i{pos}; i < nelems; i++) {
    if (status[i]) {
      continue;
    }
    while (!test(ivars + i, cmp, val)) {
    }
  }
}

template <typename T>
__device__ __forceinline__
size_t Context::wait_until_some(T *ivars, size_t nelems, size_t* indices, const int *status, int cmp, T val) {
  // zero nelems error condition
  if (!nelems) {
    return 0;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return 0;
  }

  bool done {false};
  size_t ncompleted {0};
  while (!done) {
    for (size_t i{pos}; i < nelems; i++) {
      // skip entries marked with non-zero status
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val)) {
        done = true;
        indices[ncompleted] = i;
        ncompleted++;
      }
    }
  }

  return ncompleted;
}

template <typename T>
__device__ __forceinline__
int Context::test(T *ivars, int cmp, T val) {
  int ret = 0;
  volatile T *vol_ivars = reinterpret_cast<T *>(ivars);
  switch (cmp) {
    case ROCSHMEM_CMP_EQ:
      if (uncached_load(vol_ivars) == val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_NE:
      if (uncached_load(vol_ivars) != val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_GT:
      if (uncached_load(vol_ivars) > val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_GE:
      if (uncached_load(vol_ivars) >= val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_LT:
      if (uncached_load(vol_ivars) < val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_LE:
      if (uncached_load(vol_ivars) <= val) {
        ret = 1;
      }
      break;
    default:
      break;
  }
  return ret;
}

template <typename T>
__device__
void Context::put_wave(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->put_wave(dest, source, nelems, pe);
}

template <typename T>
__device__
void Context::put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->put_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__
T Context::amo_fetch_add(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBContext*>(this)->amo_fetch_add(dst, value, pe);
  return ret_val;
}

template <typename T>
__device__
void Context::amo_add(void *dst, T value, int pe) {
  static_cast<GPUIBContext*>(this)->amo_add(dst, value, pe);
}

template <typename T>
__device__
void Context::amo_set(void *dst, T value, int pe) {
  static_cast<GPUIBContext*>(this)->amo_set(dst, value, pe);
}

template <typename T>
__device__
T Context::amo_swap(void *dst, T value, int pe) {
  auto ret_val = static_cast<GPUIBContext*>(this)->amo_swap(dst, value, pe);
  return ret_val;
}

template <typename T>
__device__
T Context::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  auto ret_val = static_cast<GPUIBContext*>(this)->amo_fetch_cas(dst, value, cond, pe);
  return ret_val;
}

template <typename T>
__device__
void Context::amo_cas(void *dst, T value, T cond, int pe) {
  static_cast<GPUIBContext*>(this)->amo_cas(dst, value, cond, pe);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP_
