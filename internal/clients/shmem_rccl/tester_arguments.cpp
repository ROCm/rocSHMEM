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
#include "tester_arguments.hpp"

#include <cstdlib>
#include <iostream>

#include <roc_shmem/roc_shmem.hpp>

using namespace rocshmem;

TesterArguments::TesterArguments(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w") {
            i++;
            num_wgs = atoi(argv[i]);
        } else if (arg == "-S") {
            i++;
            max_msg_size = atoll(argv[i]);
        } else if (arg == "-s") {
            i++;
            min_msg_size = atoll(argv[i]);
        } else if (arg == "-a") {
            i++;
            algorithm = atoi(argv[i]);
        } else if (arg == "-z") {
            i++;
            wg_size = atoi(argv[i]);
        } else if (arg == "-x") {
            i++;
            shmem_context = atoi(argv[i]);
        } else {
            show_usage(argv[0]);
            exit(-1);
        }
    }

}

void
TesterArguments::show_usage(std::string executable_name)
{
    std::cout << "Usage: " << executable_name << std::endl;
    std::cout << "\t-t <number of roc_shmem service threads>\n";
    std::cout << "\t-w <number of workgroups>\n";
    std::cout << "\t-s <maximum message size (in bytes)>\n";
    std::cout << "\t-a <algorithm number to test>\n";
    std::cout << "\t-z <WorkGroup Size>\n";
    std::cout << "\t-c <Coalescing Coefficient>\n";
    std::cout << "\t-o <Operation type for the random_access test>\n";
    std::cout << "\t-ta <Number of Thread Accessing the communication>\n";
    std::cout << "\t-x <shmem context>\n";
}

void
TesterArguments::get_rocshmem_arguments()
{
    numprocs = roc_shmem_n_pes();
    myid = roc_shmem_my_pe();

}
