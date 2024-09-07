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

#include "wave_level_primitives.hpp"

#include <roc_shmem/roc_shmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNELS
 *****************************************************************************/
__global__ void WaveLevelPrimitiveTest(int loop, int skip, uint64_t *timer,
                                      char *s_buf, char *r_buf, int size,
                                      TestType type, ShmemContextType ctx_type,
                                      int wf_size) {
  __shared__ roc_shmem_ctx_t ctx;
  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(ctx_type, &ctx);

  uint64_t start;
  int wf_id = get_flat_block_id() / wf_size;
  int offset = size * get_flat_grid_id() * (get_flat_block_size() / wf_size);
  int idx = wf_id * size + offset;
  s_buf += idx;
  r_buf += idx;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) start = roc_shmem_timer();

    switch (type) {
      case WAVEGetTestType:
        roc_shmemx_ctx_getmem_wave(ctx, r_buf, s_buf, size, 1);
        break;
      case WAVEGetNBITestType:
        roc_shmemx_ctx_getmem_nbi_wave(ctx, r_buf, s_buf, size, 1);
        break;
      case WAVEPutTestType:
        roc_shmemx_ctx_putmem_wave(ctx, r_buf, s_buf, size, 1);
        break;
      case WAVEPutNBITestType:
        roc_shmemx_ctx_putmem_nbi_wave(ctx, r_buf, s_buf, size, 1);
        break;
      default:
        break;
    }
  }

  roc_shmem_ctx_quiet(ctx);

  if (hipThreadIdx_x == 0) {
    timer[hipBlockIdx_x] = roc_shmem_timer() - start;
  }

  roc_shmem_wg_ctx_destroy(&ctx);
  roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
WaveLevelPrimitiveTester::WaveLevelPrimitiveTester(TesterArguments args)
    : Tester(args) {
  s_buf = (char *)roc_shmem_malloc(args.max_msg_size * args.num_wgs
           * num_warps);
  r_buf = (char *)roc_shmem_malloc(args.max_msg_size * args.num_wgs
           * num_warps);
}

WaveLevelPrimitiveTester::~WaveLevelPrimitiveTester() {
  roc_shmem_free(s_buf);
  roc_shmem_free(r_buf);
}

void WaveLevelPrimitiveTester::resetBuffers(uint64_t size) {
  memset(s_buf, '0', size * args.num_wgs * num_warps);
  memset(r_buf, '1', size * args.num_wgs * num_warps);
}

void WaveLevelPrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(WaveLevelPrimitiveTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, timer, s_buf, r_buf, size, _type,
                     _shmem_context, deviceProps.warpSize);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void WaveLevelPrimitiveTester::verifyResults(uint64_t size) {
  int check_id = (_type == WAVEGetTestType || _type == WAVEGetNBITestType)
                     ? 0
                     : 1;

  if (args.myid == check_id) {
    for (int i = 0; i < size * args.num_wgs * num_warps; i++) {
      if (r_buf[i] != '0') {
        fprintf(stderr, "Data validation error at idx %d\n", i);
        fprintf(stderr, "Got %c, Expected %c \n", r_buf[i], '0');
        exit(-1);
      }
    }
  }
}
