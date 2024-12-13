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

#include "team_ctx_primitive_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

rocshmem_team_t team_primitive_world_dup;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void TeamCtxPrimitiveTest(int loop, int skip, uint64_t *timer,
                                     char *s_buf, char *r_buf, int size,
                                     TestType type, ShmemContextType ctx_type,
                                     rocshmem_team_t team) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_init();
  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  if (hipThreadIdx_x == 0) {
    uint64_t start;

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) start = rocshmem_timer();

      switch (type) {
        case TeamCtxGetTestType:
          rocshmem_ctx_getmem(ctx, r_buf, s_buf, size, 1);
          break;
        case TeamCtxGetNBITestType:
          rocshmem_ctx_getmem_nbi(ctx, r_buf, s_buf, size, 1);
          break;
        case TeamCtxPutTestType:
          rocshmem_ctx_putmem(ctx, r_buf, s_buf, size, 1);
          break;
        case TeamCtxPutNBITestType:
          rocshmem_ctx_putmem_nbi(ctx, r_buf, s_buf, size, 1);
          break;
        default:
          break;
      }
    }

    rocshmem_ctx_quiet(ctx);

    timer[hipBlockIdx_x] = rocshmem_timer() - start;
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamCtxPrimitiveTester::TeamCtxPrimitiveTester(TesterArguments args)
    : Tester(args) {
  s_buf = (char *)rocshmem_malloc(args.max_msg_size * args.wg_size);
  r_buf = (char *)rocshmem_malloc(args.max_msg_size * args.wg_size);
}

TeamCtxPrimitiveTester::~TeamCtxPrimitiveTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
}

void TeamCtxPrimitiveTester::resetBuffers(uint64_t size) {
  memset(s_buf, '0', args.max_msg_size * args.wg_size);
  memset(r_buf, '1', args.max_msg_size * args.wg_size);
}

void TeamCtxPrimitiveTester::preLaunchKernel() {
  int n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  team_primitive_world_dup = ROCSHMEM_TEAM_INVALID;
  rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &team_primitive_world_dup);
}

void TeamCtxPrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                          int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(TeamCtxPrimitiveTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, timer, s_buf, r_buf, size, _type,
                     _shmem_context, team_primitive_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void TeamCtxPrimitiveTester::postLaunchKernel() {
  rocshmem_team_destroy(team_primitive_world_dup);
}

void TeamCtxPrimitiveTester::verifyResults(uint64_t size) {
  int check_id =
      (_type == TeamCtxGetTestType || _type == TeamCtxGetNBITestType) ? 0 : 1;

  if (args.myid == check_id) {
    for (uint64_t i = 0; i < size; i++) {
      if (r_buf[i] != '0') {
        fprintf(stderr, "Data validation error at idx %lu\n", i);
        fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
        exit(-1);
      }
    }
  }
}
