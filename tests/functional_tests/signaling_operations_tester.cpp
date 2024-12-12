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

#include "signaling_operations_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void SignalingOperationsTest(int loop, int skip, uint64_t *timer, char *s_buf,
                              char *r_buf, int size, uint64_t *sig_addr,
                              uint64_t *fetched_value,
                              TestType type, ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  uint64_t start;
  uint64_t signal = 0;
  int sig_op = ROCSHMEM_SIGNAL_SET;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start = rocshmem_timer();
    }

    switch (type) {
      case PutSignalTestType:
        rocshmem_ctx_putmem_signal(ctx, r_buf, s_buf, size, sig_addr, signal, sig_op, 1);
        break;
      case WGPutSignalTestType:
        rocshmem_ctx_putmem_signal_wg(ctx, r_buf, s_buf, size, sig_addr, signal, sig_op, 1);
        break;
      case WAVEPutSignalTestType:
        rocshmem_ctx_putmem_signal_wave(ctx, r_buf, s_buf, size, sig_addr, signal, sig_op, 1);
        break;
      case PutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi(ctx, r_buf, s_buf, size, sig_addr, signal, sig_op, 1);
        break;
      case WGPutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi_wg(ctx, r_buf, s_buf, size, sig_addr, signal, sig_op, 1);
        break;
      case WAVEPutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi_wave(ctx, r_buf, s_buf, size, sig_addr, signal, sig_op, 1);
        break;
      case SignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch(sig_addr);
        break;
      case WGSignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch_wg(sig_addr);
        break;
      case WAVESignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch_wave(sig_addr);
        break;
      default:
        break;
    }
  }

  rocshmem_ctx_quiet(ctx);

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    timer[hipBlockIdx_x] = rocshmem_timer() - start;
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SignalingOperationsTester::SignalingOperationsTester(TesterArguments args) : Tester(args) {
  s_buf = (char *)rocshmem_malloc(args.max_msg_size * args.wg_size);
  r_buf = (char *)rocshmem_malloc(args.max_msg_size * args.wg_size);
  sig_addr = (uint64_t *)rocshmem_malloc(sizeof(uint64_t));
  CHECK_HIP(hipMallocManaged(&fetched_value, sizeof(uint64_t), hipMemAttachHost));
}

SignalingOperationsTester::~SignalingOperationsTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  rocshmem_free(sig_addr);
  CHECK_HIP(hipFree(fetched_value));
}

void SignalingOperationsTester::resetBuffers(uint64_t size) {
  memset(s_buf, '0', args.max_msg_size * args.wg_size);
  memset(r_buf, '1', args.max_msg_size * args.wg_size);
  *fetched_value = -1;
  *sig_addr = args.myid + 123;
}

void SignalingOperationsTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                             uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(SignalingOperationsTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, timer, s_buf, r_buf, size, sig_addr, fetched_value,
                     _type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void SignalingOperationsTester::verifyResults(uint64_t size) {
  int check_data_id = (_type == PutSignalTestType ||
                       _type == PutSignalNBITestType ||
                       _type == WAVEPutSignalTestType ||
                       _type == WAVEPutSignalNBITestType ||
                       _type == WGPutSignalTestType ||
                       _type == WGPutSignalNBITestType)
                    ? 1 : -1; // do not check if it doesn't match a test

  int check_fetched_value_id = (_type == SignalFetchTestType ||
                                _type == WAVESignalFetchTestType ||
                                _type == WGSignalFetchTestType)
                             ? 0 : -1; // do not check if it doesn't match a test

  if (args.myid == check_data_id) {
    for (uint64_t i = 0; i < size; i++) {
      if (r_buf[i] != '0') {
        fprintf(stderr, "Data validation error at idx %lu\n", i);
        fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
        exit(-1);
      }
    }
  }

  if (args.myid == check_fetched_value_id) {
    uint64_t value = *fetched_value;
    uint64_t expected_value = (args.myid + 123);
    if (value != expected_value) {
      fprintf(stderr, "Fetched Value %lu, Expected %lu\n", value, expected_value);
      exit(-1);
    }
  }
}
