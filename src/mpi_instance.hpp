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

#ifndef LIBRARY_SRC_MPI_INSTANCE_HPP_
#define LIBRARY_SRC_MPI_INSTANCE_HPP_

#include <mpi.h>

#include <memory>

/**
 * @file mpi_instance.hpp
 *
 * @brief Contains MPI library initialization code
 */

namespace rocshmem {

class MPIInstance {
  public:
    /**
     * @brief Primary constructor
     */
    MPIInstance(MPI_Comm comm);

    /**
     * @brief Destructor
     */
    ~MPIInstance();

    /**
     * @brief Accessor for my COMM_WORLD rank identifier
     *
     * @return My COMM_WORLD rank identifier
     */
    int get_rank();

    /**
     * @brief Accessor for number or processes in COMM_WORLD
     *
     * @return Number of processes in COMM_WORLD
     */
    int get_nprocs();

  private:
    /**
     * @brief My MPI rank identifier
     */
    int my_rank_{-1};

    /**
     * @brief Number of MPI processes
     */
    int nprocs_{-1};

    /**
     * @brief Was MPI initialized in this class
     */
    int init_in_this_class{0};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MPI_INSTANCE_HPP_
