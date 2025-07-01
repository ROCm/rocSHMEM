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

#include "workgroup_primitives_a2a.hpp"

#include <rocshmem/rocshmem.hpp>

#include <numeric>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void WorkGroupPrimitiveA2ATest(int loop, int skip,
                                          long long int *start_time,
                                          long long int *end_time, char *source,
                                          char *dest, int size, TestType type,
                                          ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  
  auto npes = rocshmem_n_pes();

  // gridDim.x > npes && gridDim.x % npes != 0 // case 1
  // gridDim.x > npes && gridDim.x % npes == 0 // case 2
  // gridDim.x = npes // case 3
  // gridDim.x < npes // case 4
  //
  // only implement case 2, 3, and 4 - don't care about fixing this for case 1 right now
  auto num_tb_per_target_pe = (gridDim.x / npes == 0) ? gridDim.x / npes : 1;

  int wg_id = get_flat_grid_id();
  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int target_pe = wg_id / num_tb_per_target_pe;

  // Calculate start index for each work group
  uint64_t offset = size * (wg_id % num_tb_per_target_pe);
  source += offset;
  dest += offset;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      // Ensures all RMA calls from the skip loops are completed
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case WGGetTestType:
        rocshmem_ctx_getmem_wg(ctx, dest, source, size, target_pe);
        break;
      case WGGetNBITestType:
        rocshmem_ctx_getmem_nbi_wg(ctx, dest, source, size, target_pe);
        break;
      case WGPutTestType:
        rocshmem_ctx_putmem_wg(ctx, dest, source, size, target_pe);
        break;
      case WGPutNBITestType:
        rocshmem_ctx_putmem_nbi_wg(ctx, dest, source, size, target_pe);
        break;
      default:
        break;
    }
  }

  if (is_thread_zero_in_block()) {
    rocshmem_ctx_quiet(ctx);
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
WorkGroupPrimitiveA2ATester::WorkGroupPrimitiveA2ATester(TesterArguments args)
    : Tester(args) {
  size_t buff_size = args.max_msg_size * args.num_wgs;
  source = (char *)rocshmem_malloc(buff_size);
  dest = (char *)rocshmem_malloc(buff_size);

  if (source == nullptr || dest == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source << ", dest: " << dest << std::endl;
    if (source) {
      rocshmem_free(source);
    }
    if (dest) {
      rocshmem_free(dest);
    }
    rocshmem_global_exit(1);
  }

  for(size_t i = 0; i < buff_size; i++) {
    source[i] = static_cast<char>('a' + i % 26);
  }
}

WorkGroupPrimitiveA2ATester::~WorkGroupPrimitiveA2ATester() {
  rocshmem_free(source);
  rocshmem_free(dest);
}

void WorkGroupPrimitiveA2ATester::resetBuffers(uint64_t size) {
  size_t buff_size = size * args.num_wgs;
  memset(dest, '1', buff_size);
}

void WorkGroupPrimitiveA2ATester::launchKernel(dim3 gridSize, dim3 blockSize,
                                               int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(WorkGroupPrimitiveA2ATest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     source, dest, size, _type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void WorkGroupPrimitiveA2ATester::verifyResults(uint64_t size) {
}
