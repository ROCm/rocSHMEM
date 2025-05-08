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

#include "put_a2a_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PutA2aTest(int loop, int skip, long long int *start_time,
                            long long int *end_time, int *r_buf, int *s_buf,
                            ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;

  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int num_pe {rocshmem_ctx_n_pes(ctx)};
  int num_wg {get_flat_grid_size()};
  int num_wl {get_flat_block_size()};
  int my_pe {rocshmem_ctx_my_pe(ctx)};
  int wg_id {get_flat_grid_id()};
  int wl_id {get_flat_block_id()};

  auto wl_offset {wg_id * num_wl + wl_id};
  auto tgt_offset {my_pe * num_wg * num_wl + wl_offset};

//  printf("%02d:%02d:%02d: num_pe=%d, num_wg=%d, num_wl=%d, wl_offset=%d, tgt_offset=%d\n", my_pe, wg_id, wl_id, num_pe, num_wg, num_wl, wl_offset, tgt_offset);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      start_time[wg_id] = wall_clock64();
    }

    for (int j{0}; j < num_pe; j++) {
      // shuffle ordering so that threads in the wave put to a
      // different pe 'simultaneously'
      auto pe = (wl_id + j) % num_pe;
      rocshmem_ctx_putmem(ctx, &r_buf[tgt_offset], &s_buf[wl_offset], sizeof(int), pe);
    }
    __syncthreads();
    if (is_thread_zero_in_wave()) {
      rocshmem_ctx_quiet(ctx);
    }
  }
  __syncthreads();
  if (is_thread_zero_in_wave()) {
    end_time[wg_id] = wall_clock64();
  }
  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PutA2aTester::PutA2aTester(TesterArguments args) : Tester(args) {
  int num_pes {rocshmem_n_pes()};
  int my_pe {rocshmem_my_pe()};
  s_buf = (int *)rocshmem_malloc(sizeof(int) * args.num_wgs * args.wg_size);
  printf("%02d:xx:xx: num_wgs=%d, wg_size=%d\n", my_pe, args.num_wgs, args.wg_size);
  for(int wg = 0; wg < args.num_wgs; wg++) for(int lane = 0; lane < args.wg_size; lane++) {
    s_buf[wg * args.wg_size + lane] = (my_pe<<24) + (wg<<16) + lane; // set value for verification
  }
  r_buf = (int *)rocshmem_malloc(sizeof(int) * args.num_wgs * args.wg_size * num_pes);
}

PutA2aTester::~PutA2aTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
}

void PutA2aTester::resetBuffers(uint64_t size) {
  int num_pes {rocshmem_n_pes()};
  memset(r_buf, 0, sizeof(int) * args.num_wgs * args.wg_size * num_pes);
}

void PutA2aTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                uint64_t size) {
  size_t shared_bytes = 0;
  int num_pes {rocshmem_n_pes()};

  hipLaunchKernelGGL(PutA2aTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, r_buf, s_buf,
                     _shmem_context);


  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x * num_pes;
  num_timed_msgs = loop * gridSize.x * blockSize.x * num_pes;
}

void PutA2aTester::verifyResults(uint64_t size) {
  int num_pes {rocshmem_n_pes()};
  int my_pe {rocshmem_my_pe()};
  if (num_pes > 256 || args.num_wgs > 256 || args.wg_size > 64*1024) {
    // can't check
  }
#if 0
  // TODO: write is as a kernel
  for(int pe = 0; pe < num_pes; pe++)
    for(int wg = 0; wg < args.num_wgs; wg++)
      for(int lane = 0; lane < args.wg_size; lane++) {
    if(
  }
#endif
}
