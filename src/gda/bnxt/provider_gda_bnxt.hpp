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

#ifndef LIBRARY_SRC_GDA_BNXT_GDA_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_BNXT_GDA_PROVIDER_HPP_

extern "C" {
#include <infiniband/bnxt_re_dv.h>
#include <infiniband/bnxt_re_hsi.h>
}

#define GDA_DEFAULT_GID    3
#define GDA_MAX_ATOMIC     1
#define GDA_OP_RDMA_WRITE  BNXT_RE_WR_OPCD_RDMA_WRITE
#define GDA_OP_RDMA_READ   BNXT_RE_WR_OPCD_RDMA_READ
#define GDA_OP_ATOMIC_FA   BNXT_RE_WR_OPCD_ATOMIC_FA
#define GDA_OP_ATOMIC_CS   BNXT_RE_WR_OPCD_ATOMIC_CS

#define bnxt_re_get_cqe_sz() (sizeof(struct bnxt_re_req_cqe) + \
                              sizeof(struct bnxt_re_bcqe))

#define bnxt_re_is_cqe_valid(valid, phase)              \
        (((valid) & BNXT_RE_BCQE_PH_MASK) == (phase))

struct bnxt_device_wq {
  void *buf;
  uint32_t depth;
  uint32_t head;
  uint32_t tail;
  uint32_t flags;
  uint32_t id;

  uint32_t lock;

  uint32_t db_cnt {0};
} __attribute__((packed));

struct bnxt_device_cq : public bnxt_device_wq {
  uint32_t phase;
} __attribute__((packed));

struct bnxt_device_sq : public bnxt_device_wq {
  uint32_t psn;
  volatile uint32_t posted;

  void *msntbl;
  uint32_t msn;
  uint32_t msn_tbl_sz;
  uint32_t psn_sz_log2;
  uint64_t mtu;
} __attribute__((packed));

struct bnxt_host_cq {
  void *buf;
  void *handle;
  void *umem_handle;
  uint64_t length;
  uint32_t depth;
} __attribute__((packed));

struct bnxt_host_qp {
  struct bnxt_re_dv_qp_mem_info mem_info;
  struct bnxt_re_dv_qp_init_attr attr;
  void *sq_buf;
  void *rq_buf;
  void *msntbl;
  uint32_t msn_tbl_sz;
} __attribute__((packed));

/*****************************************************************************/

#endif  //LIBRARY_SRC_GDA_BNXT_GDA_PROVIDER_HPP_
