/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LIBRARY_SRC_GPU_IB_BNXT_GDA_PROVIDER_HPP_
#define LIBRARY_SRC_GPU_IB_BNXT_GDA_PROVIDER_HPP_

extern "C" {
#include <linux/types.h>
#include "bnxt_re-abi.h"

#define EXPERIMENTAL_APIS
#include "bnxt_re_dv.h"
}

#include "bnxt_util.hpp"

#define GPUIB_DEFAULT_GID    0
#define GPUIB_MAX_ATOMIC     1
#define GPUIB_OP_RDMA_WRITE  BNXT_RE_WR_OPCD_RDMA_WRITE
#define GPUIB_OP_ATOMIC_FA   BNXT_RE_WR_OPCD_ATOMIC_FA
#define GPUIB_OP_ATOMIC_CS   BNXT_RE_WR_OPCD_ATOMIC_CS

// Should this be in bnxt_re-abi or _dv.h?
#define BNXT_CQE_SIZE       32

#endif  //LIBRARY_SRC_GPU_IB_BNXT_GDA_PROVIDER_HPP_
