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

#include "tester.hpp"

#include <hip/hip_runtime.h>

#include <functional>
#include <iostream>
#include <thread>
#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "allreduce_push_tester.hpp"
#include "fused_gemm_allreduce_push_tester.hpp"
#include "wavefront_primitives.hpp"
#include "workgroup_primitives.hpp"
#include "backend_bc.hpp"

using namespace rocshmem;

extern Backend* backend;

Tester::Tester(TesterArguments args) : args(args) {
  _type = (TestType)args.algorithm;
  _shmem_context = args.shmem_context;
  CHECK_HIP(hipGetDevice(&device_id));
  CHECK_HIP(hipGetDeviceProperties(&deviceProps, device_id));
  wf_size = deviceProps.warpSize;
  num_warps = (args.wg_size - 1) / wf_size + 1;
  CHECK_HIP(hipStreamCreate(&stream));
  CHECK_HIP(hipEventCreate(&start_event));
  CHECK_HIP(hipEventCreate(&stop_event));
  CHECK_HIP(hipDeviceGetAttribute(&wall_clk_rate,
    hipDeviceAttributeWallClockRate, device_id));
  num_timers = args.num_wgs;
  //switch (_type) {
  //  case WAVEGetTestType:
  //  case WAVEGetNBITestType:
  //  case WAVEPutTestType:
  //  case WAVEPutNBITestType:
  //    num_timers = args.num_wgs * num_warps;
  //    break;
  //  default:
  //    break;
  //}
  CHECK_HIP(hipMalloc((void**)&timer, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&start_time, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&end_time, sizeof(long long int) * num_timers));
  CHECK_HIP(hipHostMalloc((void**)&verification_error, sizeof(bool)));
  *verification_error = false;
}

Tester::~Tester() {
  CHECK_HIP(hipFree(end_time));
  CHECK_HIP(hipFree(start_time));
  CHECK_HIP(hipFree(timer));
  CHECK_HIP(hipEventDestroy(stop_event));
  CHECK_HIP(hipEventDestroy(start_event));
  CHECK_HIP(hipStreamDestroy(stream));
  CHECK_HIP(hipFree(verification_error));
}

std::vector<Tester*> Tester::create(TesterArguments args) {
  int rank = args.myid;
  std::vector<Tester*> testers;

  if (rank == 0) std::cout << "### Creating Test: ";

  BackendType backend_type = get_backend_type();
  TestType type = (TestType)args.algorithm;

  switch (type) {
    case AllreducePushTestType:
      if (rank == 0) std::cout << "AllReducePush ###" << std::endl;
      testers.push_back(new AllreducePushTester(args));
      return testers;
    case FusedGemmAllreducePushTestType:
      if (rank == 0) std::cout << "FusedGemmAllReducePush ###" << std::endl;
      testers.push_back(new FusedGemmAllreducePushTester(args));
      return testers;
    default:
      if (rank == 0) std::cout << "Empty Test ###" << std::endl;
      return testers;
  }
  return testers;
}

void Tester::execute() {
  //if (_type == InitTestType) return;

  int num_loops = args.loop;

  /**
   * Some tests loop through data sizes in powers of 2 and report the
   * results for those ranges.
   */
  for (size_t size = args.min_msg_size; size <= args.max_msg_size;
       size <<= 1) {
    resetBuffers(size);
    
    // sleep for a couple secs to avoid thermal/clock impact between runs
    std::this_thread::sleep_for(std::chrono::seconds(3));

    /**
     * Restricts the number of iterations of really large messages.
     */
    if (size > args.large_message_size) num_loops = args.loop_large;

    barrier();

    preLaunchKernel();

    /**
     * This conditional launches the HIP kernel.
     *
     * Some tests may only launch a single kernel. These kernels will
     * be kicked off by the initiator (denoted by the args.myid check).
     *
     * Other tests will initiate of both sides and launch from both
     * rocshmem pes.
     */
    if (peLaunchesKernel()) {
      memset(timer, 0, sizeof(uint64_t) * args.num_wgs);

      const dim3 blockSize(args.wg_size, 1, 1);
      const dim3 gridSize(args.num_wgs, 1, 1);

      CHECK_HIP(hipEventRecord(start_event, stream));

      launchKernel(gridSize, blockSize, num_loops, size);

      CHECK_HIP(hipEventRecord(stop_event, stream));

      hipError_t err = hipStreamSynchronize(stream);
      if (err != hipSuccess) {
        printf("error = %d \n", err);
      }
    }

    barrier();

    postLaunchKernel();

    // data validation
    verifyResults(size);

    barrier();

    //if (_type != TeamCtxInfraTestType       &&
    //    _type != TeamCtxInfraTestSingleType &&
    //    _type != TeamCtxInfraTestBlockType  &&
    //    _type != TeamCtxInfraTestOddEvenType ) {
      print(size);
    //}
  }
}

bool Tester::peLaunchesKernel() {
  /**
   * The PE assigned 0 is always active in these tests.
   */
  bool is_launcher = (args.myid == 0);

  /**
   * Some test types are active on both sides.
   */
  switch (_type) {
    case AllreducePushTestType:
    case FusedGemmAllreducePushTestType:
      is_launcher = true;
    default:
      break;
  }

  return is_launcher;
}

BandwidthMetrics Tester::computeBandwidth(uint64_t size, double time_s) {
  // each pe has num_wgs that send size bytes to all other pes in each iter
  uint64_t total_size = size * num_timed_msgs * args.num_wgs * args.numprocs;
  return {static_cast<double>(total_size * bw_factor) / time_s / pow(2, 30), 0.0, 0.0};
}

void Tester::print(uint64_t size) {
  if (args.myid != 0 || !_print_results) {
    return;
  }

  double timer_avg = timerAvgInMicroseconds();

  double time_us = gpuCyclesToMicroseconds(max_end_time - min_start_time);
  double time_s = time_us / 1e6;

  double latency_avg = time_us / num_timed_msgs;
  double avg_msg_rate = num_timed_msgs / time_s;

  BandwidthMetrics metrics = computeBandwidth(size, time_s);
  bool extended = (metrics.mem_gbs > 0.0 || metrics.comp_gflops > 0.0);

  float total_kern_time_ms;
  CHECK_HIP(hipEventElapsedTime(&total_kern_time_ms, start_event, stop_event));
  float total_kern_time_s = total_kern_time_ms / 1000;

  int field_width = 20;
  int float_precision = 2;

  if (_print_header) {
    if (extended) {
      printf("%-*s%-*s%*s%*s%*s%*s%*s",
             15, "# Size (B),",
             15, "# of timed Msgs,",
             field_width, "Latency (us),",
             field_width, "NW Bandwidth (GB/s),",
             field_width, "Msg Rate (Msg/s),",
             field_width, "Mem BW (GB/s),",
             field_width + 1, "Comp BW (GFLOP/s),\n");
    } else {
      printf("%-*s%-*s%*s%*s%*s",
             15, "# Size (B),",
             15, "# of timed Msgs,",
             field_width, "Latency (us),",
             field_width, "Bandwidth (GB/s),",
             field_width + 1, "Msg Rate (Msg/s),\n");
    }
    _print_header = 0;
  }

  if (extended) {
    printf("%-*lu,%-*d,%*.*f,%*.*f,%*.*f,%*.*f,%*.*f,\n",
           15, size,
           15, num_timed_msgs,
           field_width, float_precision, latency_avg,
           field_width, float_precision+2, metrics.nw_gbs,
           field_width, float_precision, avg_msg_rate,
           field_width, float_precision, metrics.mem_gbs,
           field_width, float_precision, metrics.comp_gflops);
  } else {
    printf("%-*lu,%-*d,%*.*f,%*.*f,%*.*f,\n",
           15, size,
           15, num_timed_msgs,
           field_width, float_precision, latency_avg,
           field_width, float_precision+2, metrics.nw_gbs,
           field_width, float_precision, avg_msg_rate);
  }

  fflush(stdout);
}

void flush_hdp() {
  int hip_dev_id{};
  unsigned int* hdp_flush_ptr_{nullptr};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  CHECK_HIP(hipDeviceGetAttribute(reinterpret_cast<int*>(&hdp_flush_ptr_),
                        hipDeviceAttributeHdpMemFlushCntl, hip_dev_id));
  __atomic_store_n(hdp_flush_ptr_, 0x1, __ATOMIC_SEQ_CST);
}

void Tester::barrier() {
  rocshmem_barrier_all();
  flush_hdp();
}

double Tester::gpuCyclesToMicroseconds(long long int cycles) {
  return static_cast<double>(cycles) /
         (static_cast<double>(wall_clk_rate) * 1e-3);
}

double Tester::timerAvgInMicroseconds() {
  double sum = 0;
  min_start_time = LLONG_MAX;
  max_end_time = 0;

  for (uint32_t i = 0; i < num_timers; i++) {
    timer[i] = end_time[i] - start_time[i];
    sum += gpuCyclesToMicroseconds(timer[i]);
    min_start_time = (start_time[i] < min_start_time)
                     ? start_time[i]
                     : min_start_time;
    max_end_time = (end_time[i] > max_end_time)
                     ? end_time[i]
                     : max_end_time;
  }

  return sum / num_timers;
}
