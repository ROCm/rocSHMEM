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

#include "thread_policy.hpp"

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "queue_pair.hpp"

namespace rocshmem {

__device__ void SingleThreadImpl::quiet(QueuePair *handle) {
  handle->quiet_internal<THREAD>();
}

__device__ void SingleThreadImpl::quiet_heavy(QueuePair *handle, int pe) {
  handle->zero_b_rd<THREAD>(pe);
  handle->quiet_internal<THREAD>();
}

__device__ void WAVE::quiet(QueuePair *handle) {
  int thread_id = get_flat_block_id();
  if (thread_id % WF_SIZE == 0) {
    while (atomicCAS(&(handle->threadImpl.cq_lock), 0, 1) == 1) {
    }
    handle->quiet_internal<WAVE>();
    __threadfence();
    handle->threadImpl.cq_lock = 0;
  }
}

__device__ void WAVE::quiet_heavy(QueuePair *handle, int pe) {
  int thread_id = get_flat_block_id();
  if (thread_id % WF_SIZE == 0) {
    handle->zero_b_rd<THREAD>(pe);
    while (atomicCAS(&(handle->threadImpl.cq_lock), 0, 1) == 1) {
    }
    handle->quiet_internal<WAVE>();
    __threadfence();
    handle->threadImpl.cq_lock = 0;
  }
}

__device__ void SingleThreadImpl::decQuietCounter(uint32_t *quiet_counter, int num) {
  *quiet_counter -= num;
}

__device__ void WAVE::decQuietCounter(uint32_t *quiet_counter, int num) {
  *quiet_counter -= num;
}

template <bool cqe>
__device__ void SingleThreadImpl::finishPost(QueuePair *handle, bool ring_db,
                                             int num_wqes, int pe,
                                             uint16_t le_sq_counter,
                                             uint8_t opcode) {
  if (ring_db) {
    uint64_t db_val = handle->db_val;
    handle->compute_db_val_opcode(&db_val, le_sq_counter, opcode);
    handle->update_wqe_ce_single<cqe>(num_wqes);
    handle->ring_doorbell(db_val);
  }
}

template <bool cqe>
__device__ void WAVE::finishPost(QueuePair *handle, bool ring_db, int num_wqes,
                                 int pe, uint16_t le_sq_counter,
                                 uint8_t opcode) {
  if (ring_db) {
    uint64_t db_val = handle->sq_buf[8 * ((handle->sq_counter - num_wqes) % handle->sq_wqe_cnt)];
    handle->update_wqe_ce_thread<cqe>(num_wqes);
    handle->ring_doorbell(db_val);
  }
  handle->threadImpl.sq_lock = 0;
}

__device__ void SingleThreadImpl::postLock(QueuePair *handle, int pe) {
  handle->waitSQSpace(1);
}

__device__ void WAVE::postLock(QueuePair *handle, int pe) {
  while (atomicCAS(&(handle->threadImpl.sq_lock), 0, 1) == 1) {
  }
  handle->waitSQSpace(1);
}

template <typename T>
__device__ T SingleThreadImpl::threadAtomicAdd(T *val, T value) {
  T old_val = *val;
  *val += value;
  return old_val;
}

template <typename T>
__device__ T WAVE::threadAtomicAdd(T *val, T value) {
  return atomicAdd(val, value);
}

#define TYPE_GEN(T)                                                            \
  template __device__ T SingleThreadImpl::threadAtomicAdd<T>(T * val,          \
                                                             T value);         \
  template __device__ T WAVE::threadAtomicAdd<T>(T * val, T value);

TYPE_GEN(float)
TYPE_GEN(double)
TYPE_GEN(int)
TYPE_GEN(unsigned int)
TYPE_GEN(unsigned long long)  // NOLINT(runtime/int)

#define TYPE_BOOL(T)                                                \
  template __device__ void SingleThreadImpl::finishPost<T>(         \
      QueuePair * handle, bool ring_db, int num_wqes, int pe,       \
      uint16_t le_sq_counter, uint8_t opcode);                      \
  template __device__ void WAVE::finishPost<T>(                     \
      QueuePair * handle, bool ring_db, int num_wqes, int pe,       \
      uint16_t le_sq_counter, uint8_t opcode);

TYPE_BOOL(true)
TYPE_BOOL(false)

}  // namespace rocshmem
