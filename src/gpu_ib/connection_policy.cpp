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

#include "connection_policy.hpp"

#include <infiniband/mlx5dv.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "queue_pair.hpp"

namespace rocshmem {

RCConnectionImpl::RCConnectionImpl([[maybe_unused]] Connection* conn,
                                   [[maybe_unused]] uint32_t* _vec_rkey) {}

__device__ uint32_t RCConnectionImpl::getNumWqesImpl([
    [maybe_unused]] uint8_t opcode) {
  return 1;
}

__device__ bool RCConnectionImpl::updateConnectionSegmentImpl(
    [[maybe_unused]] ib_mlx5_base_av_t* wqe, [[maybe_unused]] int pe) {
  return false;
}

__device__ void RCConnectionImpl::setRkeyImpl([[maybe_unused]] uint32_t* rkey,
                                              [[maybe_unused]] int pe) {}

}  // namespace rocshmem
