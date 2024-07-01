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

#include "tester.hpp"

#include <functional>
#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <roc_shmem/roc_shmem.hpp>

//#include "broadcast_tester.hpp"
#include "primitive_tester.hpp"

Tester::Tester(TesterArguments args)
    : args(args)
{
    _type = (TestType) args.algorithm;
    _shmem_context = args.shmem_context;
    hipStreamCreate(&stream);
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    hipMalloc((void**)&timer, sizeof(uint64_t) * args.num_wgs);
}

Tester::~Tester()
{
    hipFree(timer);
    hipEventDestroy(stop_event);
    hipEventDestroy(start_event);
    hipStreamDestroy(stream);
}

std::vector<Tester*>
Tester::create(TesterArguments args)
{
    int rank = args.myid;
    std::vector<Tester*> testers;

    if (rank == 0)
        std::cout << "*** Creating Test: ";

    TestType type = (TestType) args.algorithm;

    switch (type) {
        case AlltoAll_Put:
            if (rank == 0)
                std::cout << "AlltoAll Puts***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case AlltoAll_Get:
            if (rank == 0)
                std::cout << "AlltoAll Gets***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        default:
            if (rank == 0)
                std::cout << "Unknown***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
    }
    return testers;
}

void
Tester::execute()
{

    int num_loops = args.loop;

    /**
     * Some tests loop through data sizes in powers of 2 and report the
     * results for those ranges.
     */
    for (uint64_t size = args.min_msg_size;
         size <= args.max_msg_size;
         size <<= 1) {


        /**
         * Restricts the number of iterations of really large messages.
         */
        if (size > args.large_message_size)
            num_loops = args.loop_large;



            /**
             * TODO:
             * Verify that this timer type is actually uint64_t on the
             * device side.
             */
            memset(timer, 0, sizeof(uint64_t) * args.num_wgs);

            const dim3 blockSize(args.wg_size, 1, 1);
            const dim3 gridSize(args.num_wgs, 1, 1);

            hipEventRecord(start_event, stream);

            launchKernel(gridSize, blockSize, num_loops, size, args.numprocs, args.myid);

            hipEventRecord(stop_event, stream);
            hipError_t err = hipStreamSynchronize(stream);
            if (err != hipSuccess) {
                printf("error = %d \n", err);
            }

//            roc_shmem_dump_stats();
      //      roc_shmem_reset_stats();



        // data validation
        verifyResults(size);

        barrier();
        resetBuffers(size);

        print(size);
    }
}


void
Tester::print(uint64_t size)
{
    if (args.myid != 0) {
        return;
    }

 //   uint64_t timer_avg = timerAvgInMicroseconds();
 //   double latency_avg = static_cast<double>(timer_avg) / num_timed_msgs;
 //   double avg_msg_rate = num_timed_msgs / (timer_avg / 1e6);

    float total_kern_time_ms;
    hipEventElapsedTime(&total_kern_time_ms, start_event, stop_event);
    float total_kern_time_s = total_kern_time_ms / 1000;
    double bandwidth_avg_gbs = num_timed_msgs * size * bw_factor / total_kern_time_s / pow(2, 30);

    float latency_us = (total_kern_time_ms *1000) /num_timed_msgs;

    int field_width = 20;
    int float_precision = 2;

    printf("\n##### Message Size %lu #####\n", size);

    printf("%*s%*s\n",
           field_width + 1, "Latency AVG (us)",
           field_width + 1, "Bandwidth (GB/s)");

    printf("%*.*f %*.*f \n",
           field_width, float_precision, latency_us,
           field_width, float_precision, bandwidth_avg_gbs);

    fflush(stdout);
}

void
Tester::barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

uint64_t
Tester::gpuCyclesToMicroseconds(uint64_t cycles)
{
    /**
     * The dGPU asm core timer runs at 27MHz. This is different from the
     * core clock returned by HIP. For an APU, this is different and might
     * need adjusting.
     */
    uint64_t gpu_frequency_MHz = 27;

    /**
     * hipDeviceGetAttribute(&gpu_frequency_khz,
     *                       hipDeviceAttributeClockRate,
     *                       0);
     */

    return cycles / gpu_frequency_MHz;
}

uint64_t
Tester::timerAvgInMicroseconds()
{
    uint64_t sum = 0;

    for (int i = 0; i < args.num_wgs; i++) {
       sum += gpuCyclesToMicroseconds(timer[i]);
    }

    return sum / args.num_wgs;
}
