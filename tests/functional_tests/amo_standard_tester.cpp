/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "amo_standard_tester.hpp"
#include "tester.hpp"

#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* Declare the global kernel template with a generic implementation */
template <typename T>
__global__ void AMOStandardTest(int loop, int skip, long long int *start_time,
                                long long int *end_time, char *r_buf,
                                T *s_buf, T *ret_val, TestType type,
                                ShmemContextType ctx_type) {
  return;
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
AMOStandardTester<T>::AMOStandardTester(TesterArguments args) : Tester(args) {
  CHECK_HIP(hipMalloc((void **)&_ret_val, args.max_msg_size * args.num_wgs));
  _r_buf = (char *)rocshmem_malloc(args.max_msg_size);
  _s_buf = (T *)rocshmem_malloc(args.max_msg_size * args.num_wgs);
}

template <typename T>
AMOStandardTester<T>::~AMOStandardTester() {
  rocshmem_free(_r_buf);
  CHECK_HIP(hipFree(_ret_val));
}

template <typename T>
void AMOStandardTester<T>::resetBuffers(size_t size) {
  memset(_r_buf, 0, args.max_msg_size);
  memset(_ret_val, 0, args.max_msg_size * args.num_wgs);
  memset(_s_buf, 0, args.max_msg_size * args.num_wgs);
}

template <typename T>
void AMOStandardTester<T>::launchKernel(dim3 gridsize, dim3 blocksize, int loop,
                                        size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(AMOStandardTest, gridsize, blocksize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, _r_buf, _s_buf,
                     _ret_val, _type, _shmem_context);

  _gridSize = gridsize;
  num_msgs = (loop + args.skip) * gridsize.x * blocksize.x;
  num_timed_msgs = loop * gridsize.x * blocksize.x;
}

template <typename T>
void AMOStandardTester<T>::verifyResults(size_t size) {
  T ret;
  if (args.myid == 0) {
    T expected_val = 0;
    switch (_type) {
      case AMO_FAddTestType:
        expected_val = 2 * (num_msgs - 1);
        break;
      case AMO_FIncTestType:
        expected_val = num_msgs - 1;
        break;
      case AMO_AddTestType:
        expected_val = 2 * num_msgs;
        break;
      case AMO_IncTestType:
        expected_val = num_msgs;
        break;
      case AMO_FCswapTestType:
        expected_val = (num_msgs - 2) / _gridSize.x;
        break;
      default:
        break;
    }

    int fetch_op = (_type == AMO_FAddTestType || _type == AMO_FIncTestType || _type == AMO_FCswapTestType) ? 1 : 0;

    if (fetch_op == 1) {
      ret = *std::max_element(_ret_val, _ret_val + args.num_wgs);
    } else {
      ret = *std::max_element(_s_buf, _s_buf + args.num_wgs);
    }
    if (ret != expected_val) {
      std::cerr << "data validation error\n";
      std::cerr << "got " << ret << ", expected " << expected_val << std::endl;
      exit(-1);
    }
  }
}

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                                                                    \
  template <>                                                                                                             \
  __global__ void AMOStandardTest<T>(int loop, int skip, long long int *start_time, long long int *end_time, char *r_buf, \
                                     T *s_buf, T *ret_val, TestType type, ShmemContextType ctx_type) {                    \
    __shared__ rocshmem_ctx_t ctx;                                                                                        \
    rocshmem_wg_init();                                                                                                   \
    rocshmem_wg_ctx_create(ctx_type, &ctx);                                                                               \
    __shared__ long long int wf_start_time[16];                                                                           \
    __shared__ long long int wf_ret_val[16];                                                                              \
    int wg_id = get_flat_grid_id();                                                                                       \
    int t_id  = get_flat_block_id();                                                                                      \
    int wf_size = 64;                                                                                                     \
    int wf_id = t_id / wf_size;                                                                                           \
    wf_ret_val[wf_id] = 0;                                                                                                \
    T ret = 0;                                                                                                            \
    T cond = 0;                                                                                                           \
    for (int i = 0; i < loop + skip; i++) {                                                                               \
      if (i == skip) {                                                                                                    \
        wf_start_time[wf_id] = wall_clock64();                                                                            \
      }                                                                                                                   \
      switch (type) {                                                                                                     \
        case AMO_FAddTestType:                                                                                            \
          ret = rocshmem_ctx_##TNAME##_atomic_fetch_add(ctx, (T *)r_buf, 2, 1);                                           \
          break;                                                                                                          \
        case AMO_FCswapTestType:                                                                                          \
          ret = rocshmem_ctx_##TNAME##_atomic_compare_swap(ctx, (T *)r_buf, cond, (T)i, 1);                               \
          cond = i;                                                                                                       \
          break;                                                                                                          \
        case AMO_FIncTestType:                                                                                            \
          ret = rocshmem_ctx_##TNAME##_atomic_fetch_inc(ctx, (T *)r_buf, 1);                                              \
          break;                                                                                                          \
        case AMO_AddTestType:                                                                                             \
          rocshmem_ctx_##TNAME##_atomic_add(ctx, (T *)r_buf, 2, 1);                                                       \
          break;                                                                                                          \
        case AMO_IncTestType:                                                                                             \
          rocshmem_ctx_##TNAME##_atomic_inc(ctx, (T *)r_buf, 1);                                                          \
          break;                                                                                                          \
        default:                                                                                                          \
          break;                                                                                                          \
      }                                                                                                                   \
    }                                                                                                                     \
    rocshmem_ctx_quiet(ctx);                                                                                              \
    end_time[wg_id] = wall_clock64();                                                                                     \
    rocshmem_ctx_getmem(ctx, &s_buf[wg_id], r_buf, sizeof(T), 1);                                                         \
    __hip_atomic_fetch_max(&wf_ret_val[wf_id], ret, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);                      \
    __syncthreads();                                                                                                      \
    int num_wfs = (get_flat_block_size() - 1 ) / wf_size + 1;                                                             \
    for (int i = num_wfs / 2; i > 0; i >>= 1 ) {                                                                          \
      if (t_id < i) {                                                                                                     \
        wf_ret_val[t_id] = max(wf_ret_val[t_id], wf_ret_val[t_id + i]);                                                   \
      }                                                                                                                   \
    }                                                                                                                     \
    ret_val[wg_id] = wf_ret_val[0];                                                                                       \
    for (int i = num_wfs / 2; i > 0; i >>= 1 ) {                                                                          \
      if (t_id < i) {                                                                                                     \
        wf_start_time[t_id] = min(wf_start_time[t_id], wf_start_time[t_id + i]);                                          \
      }                                                                                                                   \
    }                                                                                                                     \
    __syncthreads();                                                                                                      \
    if (t_id == 0) {                                                                                                      \
      start_time[wg_id] = wf_start_time[0];                                                                               \
    }                                                                                                                     \
    rocshmem_wg_ctx_destroy(&ctx);                                                                                        \
    rocshmem_wg_finalize();                                                                                               \
  }                                                                                                                       \
  template class AMOStandardTester<T>;

AMO_STANDARD_DEF_GEN(int, int)
AMO_STANDARD_DEF_GEN(long, long)
AMO_STANDARD_DEF_GEN(long long, longlong)
AMO_STANDARD_DEF_GEN(unsigned int, uint)
AMO_STANDARD_DEF_GEN(unsigned long, ulong)
AMO_STANDARD_DEF_GEN(unsigned long long, ulonglong)
