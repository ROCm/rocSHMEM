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

#include "tester_arguments.hpp"

#include <cstdlib>
#include <iostream>
#include <rocshmem/rocshmem.hpp>

#include "tester.hpp"

using namespace rocshmem;

TesterArguments::TesterArguments(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-t") {
      i++;
      num_threads = atoi(argv[i]);
    } else if (arg == "-w") {
      i++;
      num_wgs = atoi(argv[i]);
    } else if (arg == "-s") {
      i++;
      max_msg_size = atoll(argv[i]);
    } else if (arg == "-a") {
      i++;
      algorithm = atoi(argv[i]);
    } else if (arg == "-z") {
      i++;
      wg_size = atoi(argv[i]);
    } else if (arg == "-c") {
      i++;
      coal_coef = atoi(argv[i]);
    } else if (arg == "-o") {
      i++;
      op_type = atoi(argv[i]);
    } else if (arg == "-ta") {
      i++;
      thread_access = atoi(argv[i]);
    } else if (arg == "-x") {
      i++;
      shmem_context = atoi(argv[i]);
    } else if (arg == "-m") {
      int atomics_addr_mode = atoi(argv[i]);
      if(atomics_addr_mode >= static_cast<int>(AddrMode::PerGrid) &&
         atomics_addr_mode <= static_cast<int>(AddrMode::PerBlock)) {
         addr_mode = static_cast<AddrMode>(atomics_addr_mode);
      }
      i++;
    } else if (arg == "-n") {
      i++;
      loop = atoi(argv[i]);
      loop_large = loop;
    } else if (arg == "-nloop") {
      i++;
      loop = atoi(argv[i]);
    } else if (arg == "-nlarge") {
      i++;
      loop_large = atoi(argv[i]);
    } else if (arg == "-nskip") {
      i++;
      skip = atoi(argv[i]);
    } else if (arg == "-ro") {
      rlx_order = true;
    } else if (arg == "-to") {
      tgt_order = true;
    } else if (arg == "-dg") {
      disable_gemm = true;
    } else if (arg == "-drs") {
      disable_rs = true;
    } else if (arg == "-gm") {
      i++;
      gemm_m = atoi(argv[i]);
    } else if (arg == "-gn") {
      i++;
      gemm_max_n = atoi(argv[i]);
    } else {
      show_usage(argv[0]);
      exit(-1);
    }
  }

  TestType type = (TestType)algorithm;

}

void TesterArguments::show_usage(std::string executable_name) {
  std::cout << "Usage: " << executable_name << std::endl;
  std::cout << "\t-t <number of rocshmem service threads>\n";
  std::cout << "\t-w <number of workgroups>\n";
  std::cout << "\t-s <maximum message size (in bytes)>\n";
  std::cout << "\t-a <algorithm number to test>\n";
  std::cout << "\t-z <WorkGroup Size>\n";
  std::cout << "\t-c <Coalescing Coefficient>\n";
  std::cout << "\t-o <Operation type for the random_access test>\n";
  std::cout << "\t-ta <Number of Thread Accessing the communication>\n";
  std::cout << "\t-x <shmem context>\n";
  std::cout << "\t-m Atomics Address mode\n";
  std::cout << "\t-n Set both loop and loop_large count\n";
  std::cout << "\t-nloop Set loop count\n";
  std::cout << "\t-nlarge Set loop_large count\n";
  std::cout << "\t-nskip Set skip/warmup count\n";
  std::cout << "\t-ro Use relaxed ordering rocshmem\n";
  std::cout << "\t-to Use targeted ordering rocshmem\n";
  std::cout << "\t-dg Disable GEMM (reduce-scatter only)\n";
  std::cout << "\t-drs Disable reduce-scatter (GEMM only, kernel launched skip+loop times)\n";
}

void TesterArguments::get_arguments() {
  numprocs = rocshmem_n_pes();
  myid = rocshmem_my_pe();

  TestType type = (TestType)algorithm;
  // Check if test requires exactly 2 PEs
  // Tests that support arbitrary number of PEs are excluded
  bool requires_two_pes = true;
  switch (type) {
    // Collective/barrier tests - support any number of PEs
    case AllreducePushTestType:
    case FusedGemmAllreducePushTestType:
      requires_two_pes = false;
      break;
    default:
      break;
  }

  // min_msg_size determines the starting size for n, and we set it
  // so that it is a power of two, and so each thread has at
  // least one gemm output to store:
  // m * n >= wg_size*num_wgs*numprocs
  size_t min_n = static_cast<int>(std::ceil(static_cast<double>(wg_size*num_wgs*numprocs/gemm_m)));
  if (type==FusedGemmAllreducePushTestType) {
    if (min_n > max_msg_size) {
      printf("Error: minimum valid n=%d*%d*%d/%d=%d > maximum n=%d. Aborting...\n", wg_size, num_wgs, numprocs, gemm_m, min_n, gemm_max_n);
      abort();
    }
  }
  switch (type) {
    case AllreducePushTestType:
      min_msg_size = max_msg_size;
      break;
    case FusedGemmAllreducePushTestType:
      min_msg_size = gemm_m;
      while (min_msg_size < min_n) {
        min_msg_size = min_msg_size << 1;
      }
      min_msg_size = gemm_max_n;
      break;
    default:
      break;
  }

  if (requires_two_pes && numprocs != 2) {
    if (myid == 0) {
      std::cerr << "This test requires exactly two processes, we have "
                << numprocs << "\n";
    }
    exit(-1);
  }
}
