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

#include "queue_pair.hpp"

#include <hip/hip_runtime.h>

#include "backend_gda.hpp"
#include "constants.hpp"

namespace rocshmem {

QueuePair::QueuePair(struct ibv_pd* pd, int gda_vendor) {
  int access = IBV_ACCESS_LOCAL_WRITE
             | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ
             | IBV_ACCESS_REMOTE_ATOMIC;

  allocator.allocate((void**)&nonfetching_atomic, 8);
  allocator.allocate((void**)&fetching_atomic, 8 * FETCHING_ATOMIC_CNT);
  allocator.allocate((void**)&fetching_atomic_freelist, sizeof(FreeListT*));
  new (fetching_atomic_freelist) FreeListT();

  CHECK_HIP(hipMemset(nonfetching_atomic, 0, 8));
  CHECK_HIP(hipMemset(fetching_atomic, 0, 8 * FETCHING_ATOMIC_CNT));

  mr_nonfetching_atomic = ibv_reg_mr(pd, nonfetching_atomic, 8, access);
  CHECK_NNULL(mr_nonfetching_atomic, "ibv_reg_mr");

  mr_fetching_atomic = ibv_reg_mr(pd, fetching_atomic, 8 * FETCHING_ATOMIC_CNT, access);
  CHECK_NNULL(mr_fetching_atomic, "ibv_reg_mr");

  if (gda_vendor == GDAVendor::MLX5) {
    nonfetching_atomic_lkey = htobe32(mr_nonfetching_atomic->lkey);
    fetching_atomic_lkey = htobe32(mr_fetching_atomic->lkey);
  } else {
    nonfetching_atomic_lkey = mr_nonfetching_atomic->lkey;
    fetching_atomic_lkey = mr_fetching_atomic->lkey;
  }

  for(int i{0}; i < FETCHING_ATOMIC_CNT; i+=WF_SIZE) {
    fetching_atomic_freelist->push_back(fetching_atomic + i);
  }

  /* Set Correct opcodes for each NIC */
  switch (gda_vendor) {
#if defined(GDA_IONIC)
  case GDAVendor::IONIC:
    gda_op_rdma_write = IONIC_V2_OP_RDMA_WRITE;
    gda_op_rdma_read  = IONIC_V2_OP_RDMA_READ;
    gda_op_atomic_fa  = IONIC_V2_OP_ATOMIC_FA;
    gda_op_atomic_cs  = IONIC_V2_OP_ATOMIC_CS;
    break;
#endif //defined(GDA_IONIC)
#if defined(GDA_BNXT)
  case GDAVendor::BNXT:
    gda_op_rdma_write = BNXT_RE_WR_OPCD_RDMA_WRITE;
    gda_op_rdma_read  = BNXT_RE_WR_OPCD_RDMA_READ;
    gda_op_atomic_fa  = BNXT_RE_WR_OPCD_ATOMIC_FA;
    gda_op_atomic_cs  = BNXT_RE_WR_OPCD_ATOMIC_CS;
    break;
#endif //defined(GDA_BNXT)
#if defined(GDA_MLX5)
  case GDAVendor::MLX5:
    gda_op_rdma_write = MLX5_OPCODE_RDMA_WRITE;
    gda_op_rdma_read  = MLX5_OPCODE_RDMA_READ;
    gda_op_atomic_fa  = MLX5_OPCODE_ATOMIC_FA;
    gda_op_atomic_cs  = MLX5_OPCODE_ATOMIC_CS;
    break;
#endif //defined(GDA_MLX5)
  default:
    assert(false /* invalid nic provider */);
  }
  gda_vendor_ = gda_vendor;
}

QueuePair::~QueuePair() {
  int err;

  err = ibv_dereg_mr(mr_nonfetching_atomic);
  CHECK_ZERO(err, "ibv_dereg_mr (nonfetching_atomic)");

  err = ibv_dereg_mr(mr_fetching_atomic);
  CHECK_ZERO(err, "ibv_dereg_mr (fetching_atomic)");

  allocator.deallocate((void*)nonfetching_atomic);
  allocator.deallocate((void*)fetching_atomic);

  fetching_atomic_freelist->~FreeListT();
  allocator.deallocate((void*)fetching_atomic_freelist);
}

/******************************************************************************
 ************************ PROVIDER-SPECIFIC HELPERS ***************************
 *****************************************************************************/
__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  switch (gda_vendor_) {
#if defined(GDA_MLX5)
  case GDAVendor::MLX5:
    mlx5_post_wqe_rma(pe, size, laddr, raddr, opcode);
    return;
#endif
#if defined(GDA_BNXT)
  case GDAVendor::BNXT:
    bnxt_post_wqe_rma(pe, size, laddr, raddr, opcode);
    return;
#endif
#if defined(GDA_IONIC)
  case GDAVendor::IONIC:
    ionic_post_wqe_rma(pe, size, laddr, raddr, opcode);
    return;
#endif
  default:
    assert(false /* invalid nic provider */);
  }
}

__device__ uint64_t QueuePair::post_wqe_amo(int pe, int32_t size, uintptr_t *raddr, uint8_t opcode,
                                            int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  switch (gda_vendor_) {
#if defined(GDA_MLX5)
  case GDAVendor::MLX5:
    return mlx5_post_wqe_amo(pe, size, raddr, opcode, atomic_data, atomic_cmp, fetching);
#endif
#if defined(GDA_BNXT)
  case GDAVendor::BNXT:
    return bnxt_post_wqe_amo(pe, size, raddr, opcode, atomic_data, atomic_cmp, fetching);
#endif
#if defined(GDA_IONIC)
  case GDAVendor::IONIC:
    return ionic_post_wqe_amo(pe, size, raddr, opcode, atomic_data, atomic_cmp, fetching);
#endif
  default:
    assert(false /* invalid nic provider */);
    return 0;
  }
}

__device__ void QueuePair::quiet() {
  switch (gda_vendor_) {
#if defined(GDA_MLX5)
  case GDAVendor::MLX5:
    mlx5_quiet();
    return;
#endif
#if defined(GDA_BNXT)
  case GDAVendor::BNXT:
    bnxt_quiet();
    return;
#endif
#if defined(GDA_IONIC)
  case GDAVendor::IONIC:
    ionic_quiet();
    return;
#endif
  default:
    assert(false /* invalid nic provider */);
  }
}

/******************************************************************************
 ****************************** SHMEM INTERFACE *******************************
 *****************************************************************************/
__device__ void QueuePair::put_nbi(void *dest, const void *source, size_t nelems, int pe) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, src, dst, gda_op_rdma_write);
}

__device__ void QueuePair::get_nbi(void *dest, const void *source, size_t nelems, int pe) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, dst, src, gda_op_rdma_read);
}

__device__ int64_t QueuePair::atomic_cas(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe) {
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  return post_wqe_amo(pe, sizeof(int64_t), dst, gda_op_atomic_cs, atomic_data, atomic_cmp, true);
}

__device__ int64_t QueuePair::atomic_cas_nofetch(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe) {
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  return post_wqe_amo(pe, sizeof(int64_t), dst, gda_op_atomic_cs, atomic_data, atomic_cmp, false);
}

__device__ int64_t QueuePair::atomic_fetch(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe) {
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  return post_wqe_amo(pe, sizeof(int64_t), dst, gda_op_atomic_fa, atomic_data, atomic_cmp, true);
}

__device__ void QueuePair::atomic_nofetch(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe) {
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_amo(pe, sizeof(int64_t), dst, gda_op_atomic_fa, atomic_data, atomic_cmp, false);
}

}  // namespace rocshmem
