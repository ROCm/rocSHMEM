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

#ifndef LIBRARY_SRC_GPU_IB_THREAD_POLICY_HPP_
#define LIBRARY_SRC_GPU_IB_THREAD_POLICY_HPP_

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "util.hpp"

namespace rocshmem {

class QueuePair;

class SingleThreadImpl {
 public:
  uint32_t cq_lock = 0;
  uint32_t sq_lock = 0;

  __device__ void quiet(QueuePair *handle);

  __device__ void quiet_heavy(QueuePair *handle, int pe);

  __device__ void decQuietCounter(uint32_t *quiet_counter, int num);

  template <bool cqe>
  __device__ void finishPost(QueuePair *handle, bool ring_db, int num_wqes,
                             int pe, uint16_t le_sq_counter, uint8_t opcode);

  __device__ void postLock(QueuePair *handle, int pe);

  template <typename T>
  __device__ T threadAtomicAdd(T *val, T value = 1);
};

typedef SingleThreadImpl ThreadImpl;

class THREAD {
 public:
  ThreadImpl threadImpl;

  __device__ void quiet(QueuePair *handle) { threadImpl.quiet(handle); }

  __device__ void quiet_heavy(QueuePair *handle, int pe) {
    threadImpl.quiet_heavy(handle, pe);
  }

  __device__ void decQuietCounter(uint32_t *quiet_counter, int num) {
    threadImpl.decQuietCounter(quiet_counter, num);
  }

  template <bool cqe>
  __device__ void finishPost(QueuePair *handle, bool ring_db, int num_wqes,
                             int pe, uint16_t le_sq_counter, uint8_t opcode) {
    threadImpl.finishPost<cqe>(handle, ring_db, num_wqes, pe, le_sq_counter,
                               opcode);
  }

  __device__ void postLock(QueuePair *handle, int pe) {
    threadImpl.postLock(handle, pe);
  }

  template <typename T>
  __device__ T threadAtomicAdd(T *val, T value = 1) {
    T tmp = threadImpl.threadAtomicAdd(val, value);
    return tmp;
  }
};

class WAVE {
 public:
  __device__ void quiet(QueuePair *handle);

  __device__ void quiet_heavy(QueuePair *handle, int pe);

  __device__ void decQuietCounter(uint32_t *quiet_counter, int num);

  template <bool cqe>
  __device__ void finishPost(QueuePair *handle, bool ring_db, int num_wqes,
                             int pe, uint16_t le_sq_counter, uint8_t opcode);

  __device__ void postLock(QueuePair *handle, int pe);

  template <typename T>
  __device__ T threadAtomicAdd(T *val, T value = 1);
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_THREAD_POLICY_HPP_
