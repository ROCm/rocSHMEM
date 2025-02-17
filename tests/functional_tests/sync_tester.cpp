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

#include "sync_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;
rocshmem_team_t team_sync_world_dup;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void SyncTest(int loop, int skip, long long int *start_time,
                         long long int *end_time, TestType type,
                         ShmemContextType ctx_type, rocshmem_team_t team) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  for (int i = 0; i < loop + skip; i++) {
    if (hipThreadIdx_x == 0 && i == skip) {
      start_time[wg_id] = wall_clock64();
    }

    __syncthreads();
    switch (type) {
      case SyncAllTestType:
        rocshmem_ctx_wg_sync_all(ctx);
        break;
      case SyncTestType:
        rocshmem_ctx_wg_team_sync(ctx, team);
        break;
      default:
        break;
    }
  }
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
SyncTester::SyncTester(TesterArguments args) : Tester(args) {}

SyncTester::~SyncTester() {}

void SyncTester::resetBuffers(uint64_t size) {}

void SyncTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                              uint64_t size) {
  size_t shared_bytes = 0;

  int n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  team_sync_world_dup = ROCSHMEM_TEAM_INVALID;
  rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &team_sync_world_dup);

  hipLaunchKernelGGL(SyncTest, gridSize, blockSize, shared_bytes, stream, loop,
                     args.skip, start_time, end_time, _type, _shmem_context,
                     team_sync_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void SyncTester::verifyResults(uint64_t size) {}
