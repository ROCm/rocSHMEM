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

#include "amo_self_tester.hpp"

#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* Declare the global kernel template with a generic implementation */
template <typename T>
__global__ void AMOSelfTest(int loop, int skip, long long int *start_time,
                            long long int *end_time, char *r_buf,
                            T *s_buf, T *ret_val, TestType type,
                            ShmemContextType ctx_type) {
  return;
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
AMOSelfTester<T>::AMOSelfTester(TesterArguments args) : Tester(args) {
  CHECK_HIP(hipMalloc((void **)&_ret_val, args.max_msg_size * args.num_wgs));
  _r_buf = (char *)rocshmem_malloc(args.max_msg_size);
}

template <typename T>
AMOSelfTester<T>::~AMOSelfTester() {
  rocshmem_free(_r_buf);
  CHECK_HIP(hipFree(_ret_val));
}

template <typename T>
void AMOSelfTester<T>::resetBuffers(uint64_t size) {
  memset(_r_buf, 0, args.max_msg_size);
  memset(_ret_val, 0, args.max_msg_size * args.num_wgs);
}

template <typename T>
void AMOSelfTester<T>::launchKernel(dim3 gridsize, dim3 blocksize, int loop,
                                    uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(AMOSelfTest, gridsize, blocksize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, _r_buf, (T*)nullptr,
                     _ret_val, _type, _shmem_context);

  _gridSize = gridsize;
  num_msgs = (loop + args.skip) * gridsize.x * blocksize.x;
  num_timed_msgs = loop * gridsize.x * blocksize.x;
}

template <typename T>
void AMOSelfTester<T>::verifyResults(uint64_t size) {
  T ret;
  T expected_val = 0;
  switch (_type) {
    case AMO_FAddSelfTestType:
      expected_val = num_msgs - 1;
      break;
    case AMO_AddSelfTestType:
      expected_val = num_msgs;
      break;
    default:
      break;
  }

  int fetch_op = (_type == AMO_FAddSelfTestType) ? 1 : 0;

  if (fetch_op == 1) {
    ret = *std::max_element(_ret_val, _ret_val + args.num_wgs);
  } else {
    ret = *(T*)_r_buf;
  }

  if (ret != expected_val) {
    std::cerr << "data validation error\n";
    std::cerr << "got " << ret << ", expected " << expected_val << std::endl;
    exit(-1);
  }
}

#define AMO_SELF_DEF_GEN(T, TNAME)                                                                                        \
  template <>                                                                                                             \
  __global__ void AMOSelfTest<T>(int loop, int skip, long long int *start_time, long long int *end_time, char *r_buf,     \
                                 T *s_buf, T *ret_val, TestType type, ShmemContextType ctx_type) {                        \
    __shared__ long long int wf_start_time[16];                                                                           \
    __shared__ long long int wf_ret_val[16];                                                                              \
    int wg_id = get_flat_grid_id();                                                                                       \
    int t_id  = get_flat_block_id();                                                                                      \
    int wf_size = 64;                                                                                                     \
    int wf_id = t_id / wf_size;                                                                                           \
    wf_ret_val[wf_id] = 0;                                                                                                \
    T ret = 0;                                                                                                            \
    T cond = 0;                                                                                                           \
    int my_pe = rocshmem_my_pe();                                                                                         \
    for (int i = 0; i < loop + skip; i++) {                                                                               \
      if (i == skip) {                                                                                                    \
        wf_start_time[wf_id] = wall_clock64();                                                                            \
      }                                                                                                                   \
      switch (type) {                                                                                                     \
        case AMO_FAddSelfTestType:                                                                                        \
          ret = rocshmem_##TNAME##_atomic_fetch_add((T *)r_buf, 1, my_pe);                                                \
          break;                                                                                                          \
        case AMO_AddSelfTestType:                                                                                         \
          rocshmem_##TNAME##_atomic_add((T *)r_buf, 1, my_pe);                                                            \
          break;                                                                                                          \
        default:                                                                                                          \
          break;                                                                                                          \
      }                                                                                                                   \
      end_time[wg_id] = wall_clock64();                                                                                   \
      __hip_atomic_fetch_max(&wf_ret_val[wf_id], ret, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);                    \
    }                                                                                                                     \
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
  }                                                                                                                       \
  template class AMOSelfTester<T>;

AMO_SELF_DEF_GEN(int, int)
AMO_SELF_DEF_GEN(long, long)
AMO_SELF_DEF_GEN(long long, longlong)
AMO_SELF_DEF_GEN(unsigned int, uint)
AMO_SELF_DEF_GEN(unsigned long, ulong)
AMO_SELF_DEF_GEN(unsigned long long, ulonglong)
