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

#include "primitive_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PrimitiveTest(int loop, int skip, long long int *start_time,
                              long long int *end_time, char *source,
                              char *dest, int size, TestType type,
                              ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  /**
   * Calculate start index for each thread within the grid
   */
  uint64_t idx = size * get_flat_id();
  source += idx;
  dest += idx;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case GetTestType:
        rocshmem_ctx_getmem(ctx, dest, source, size, 1);
        break;
      case GetNBITestType:
        rocshmem_ctx_getmem_nbi(ctx, dest, source, size, 1);
        break;
      case PutTestType:
        rocshmem_ctx_putmem(ctx, dest, source, size, 1);
        break;
      case PutNBITestType:
        rocshmem_ctx_putmem_nbi(ctx, dest, source, size, 1);
        break;
      case PTestType:
        for (int s = 0; s < size; s++) {
          char val = source[s];
          rocshmem_ctx_char_p(ctx, &dest[s], val, 1);
        }
        break;
      case GTestType:
        for (int s = 0; s < size; s++) {
          char ret = rocshmem_ctx_char_g(ctx, &source[s], 1);
          dest[s] = ret;
        }
        break;
      default:
        break;
    }
  }

  rocshmem_ctx_quiet(ctx);

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PrimitiveTester::PrimitiveTester(TesterArguments args) : Tester(args) {
  size_t buff_size = args.max_msg_size * args.wg_size * args.num_wgs;
  source = (char *)rocshmem_malloc(buff_size);
  dest = (char *)rocshmem_malloc(buff_size);

  if (source == nullptr || dest == nullptr) {
    std::cout << "Error allocating memory from symmetric heap" << std::endl;
    std::cout << "source: " << source << ", dest: " << dest << std::endl;
    rocshmem_global_exit(1);
  }

  for(size_t i = 0; i < buff_size; i++) {
    source[i] = static_cast<char>('a' + i % 26);
  }
}

PrimitiveTester::~PrimitiveTester() {
  rocshmem_free(source);
  rocshmem_free(dest);
}

void PrimitiveTester::resetBuffers(uint64_t size) {
  size_t buff_size = size * args.wg_size * args.num_wgs;
  memset(dest, '1', buff_size);
}

void PrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                   uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PrimitiveTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, source, dest,
                     size, _type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void PrimitiveTester::verifyResults(uint64_t size) {
  int check_id =
      (_type == GetTestType || _type == GetNBITestType || _type == GTestType)
          ? 0
          : 1;

  if (args.myid == check_id) {
    size_t buff_size = size * args.wg_size * args.num_wgs;
    for (uint64_t i = 0; i < buff_size; i++) {
      if (dest[i] != source[i]) {
        std::cerr << "Data validation error at idx " << i << std::endl;
        std::cerr << " Got " << dest[i] << ", Expected "
                  << source[i] << std::endl;
        exit(-1);
      }
    }
  }
}
