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

#ifndef _TESTER_HPP_
#define _TESTER_HPP_

#include <rocshmem/rocshmem.hpp>
#include <vector>
#include <climits>

#include "tester_arguments.hpp"
#include "../src/util.hpp"
#include "verify_results_kernels.hpp"

/******************************************************************************
 * TESTER CLASS TYPES
 *****************************************************************************/
enum TestType {
  AllreducePushTestType=0,
  FusedGemmAllreducePushTestType,
};

enum OpType { PutType = 0, GetType = 1 };

typedef int ShmemContextType;

/******************************************************************************
 * TESTER INTERFACE
 *****************************************************************************/
struct BandwidthMetrics {
  double nw_gbs      = 0.0;  // network bandwidth (GB/s)
  double mem_gbs     = 0.0;  // memory bandwidth (GB/s)
  double comp_gflops = 0.0;  // compute throughput (GFLOP/s)
};

class Tester {
 public:
  explicit Tester(TesterArguments args);
  virtual ~Tester();

  void execute();

  static std::vector<Tester *> create(TesterArguments args);

 protected:
  virtual void resetBuffers(uint64_t size) = 0;

  virtual void preLaunchKernel() {}

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) = 0;

  virtual void postLaunchKernel() {}

  virtual void verifyResults(uint64_t size) = 0;

  // Returns bandwidth metrics for the most recent timed run.
  // Base implementation computes generic network BW from total transfer size.
  // Override to report test-specific memory and compute demands.
  virtual BandwidthMetrics computeBandwidth(uint64_t size, double time_s);

  int num_msgs = 0;
  int num_timed_msgs = 0;
  int num_warps = 0;
  int bw_factor = 1;
  int device_id = 0;
  int wall_clk_rate = 0; //in kilohertz
  int wf_size = 0;

  TesterArguments args;

  TestType _type;
  ShmemContextType _shmem_context = 8;  // SHMEM_CTX_WP_PRIVATE

  hipStream_t stream;
  hipDeviceProp_t deviceProps;

  long long int *timer = nullptr;
  long long int *start_time = nullptr;
  long long int *end_time = nullptr;
  long long int min_start_time = 0;
  long long int max_end_time = 0;
  uint32_t num_timers = 0;

  bool *verification_error;

 protected:
  bool _print_results = true;

 private:
  bool _print_header = true;
  void print(uint64_t size);

  void barrier();

  double gpuCyclesToMicroseconds(long long int cycles);

  double timerAvgInMicroseconds();

  bool peLaunchesKernel();

  hipEvent_t start_event;
  hipEvent_t stop_event;
};

//TODO remove altogether? THere is a small difference in print format
#undef CHECK_HIP
#define CHECK_HIP(instr) do {                                               \
  hipError_t error = (instr);                                               \
  if (error != hipSuccess) {                                                \
    fprintf(stderr, "error: " #instr ": %s (%d) at %s:%d\n",                \
      hipGetErrorString(error), error, __FILE__, __LINE__);                 \
    abort();                                                                \
  }                                                                         \
} while(0)

#endif /* _TESTER_HPP */
