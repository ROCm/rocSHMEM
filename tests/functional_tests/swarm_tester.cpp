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

#include "swarm_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void GetSwarmTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, char *s_buf,
                             char *r_buf, int size, ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  int provided;
  rocshmem_wg_init_thread(ROCSHMEM_THREAD_MULTIPLE, &provided);
  assert(provided == ROCSHMEM_THREAD_MULTIPLE);

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  __syncthreads();

  int index = hipThreadIdx_x * size;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      start_time[wg_id] = wall_clock64();
    }
    rocshmem_ctx_getmem(ctx, &r_buf[index], &s_buf[index], size, 1);

    __syncthreads();
  }

  // atomicAdd((unsigned long long *)&timer[hipBlockIdx_x],
  //           rocshmem_timer() - start);

  end_time[wg_id] = wall_clock64();

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
GetSwarmTester::GetSwarmTester(TesterArguments args) : PrimitiveTester(args) {}

GetSwarmTester::~GetSwarmTester() {}

void GetSwarmTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(GetSwarmTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, s_buf, r_buf, size,
                     _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void GetSwarmTester::verifyResults(uint64_t size) {
  if (args.myid == 0) {
    for (uint64_t i = 0; i < size * args.wg_size; i++) {
      if (r_buf[i] != '0') {
        fprintf(stderr, "Data validation error at idx %lu\n", i);
        fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
        exit(-1);
      }
    }
  }
}
