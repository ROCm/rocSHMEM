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

#include "gda/backend_gda.hpp"
#include "util.hpp"
#include <unistd.h> // getpagesize()

namespace rocshmem {

int GDABackend::ibv_mtu_to_int(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:  return 256;
    case IBV_MTU_512:  return 512;
    case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048;
    case IBV_MTU_4096: return 4096;
    default: {
      fprintf(stderr, "[ERROR] Invalid ibv_mtu\n");
      return 0;
    }
  }
}

void GDABackend::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  int err;

  ib_state = new ib_state_t;
  CHECK_NNULL(ib_state, "ib_state object create");

  ib_state->context = ibv_open_device(ib_dev);
  CHECK_NNULL(ib_state->context, "ibv_open_device");

  ib_state->pd_orig = ibv_alloc_pd(ib_state->context);
  CHECK_NNULL(ib_state->pd_orig, "ibv_alloc_pd");

  err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  CHECK_ZERO(err, "ibv_query_port");

  init_gid_index(port);
}

void GDABackend::init_qp_status(uint8_t port) {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));

  attr.qp_state        = IBV_QPS_INIT;
  attr.pkey_index      = 0;
  attr.port_num        = port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_LOCAL_WRITE
                       | IBV_ACCESS_REMOTE_READ
                       | IBV_ACCESS_REMOTE_ATOMIC;

  attr_mask = IBV_QP_STATE
            | IBV_QP_PKEY_INDEX
            | IBV_QP_PORT
            | IBV_QP_ACCESS_FLAGS;

  for (int i =0; i < qps.size() ; i++) {
    err = bnxt_re_dv_modify_qp(qps[i], &attr, attr_mask, 0, 0);
    CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
  }
}

void GDABackend::change_status_rtr(ibv_qp *qp, dest_info_t *dest, uint8_t port) {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state               = IBV_QPS_RTR;
  attr.path_mtu               = ib_state->portinfo.active_mtu;
  attr.rq_psn                 = dest->psn;
  attr.dest_qp_num            = dest->qpn;

  memcpy(&attr.ah_attr.grh.dgid, &dest->gid, 16);
  attr.ah_attr.grh.sgid_index = gid_index;
  attr.ah_attr.grh.hop_limit  = 1;
  attr.ah_attr.sl             = 1;
  attr.ah_attr.is_global      = 1;
  attr.ah_attr.port_num       = port;

  attr.max_dest_rd_atomic     = GDA_MAX_ATOMIC;
  attr.min_rnr_timer          = 12;

  attr_mask = IBV_QP_STATE
            | IBV_QP_PATH_MTU
            | IBV_QP_RQ_PSN
            | IBV_QP_DEST_QPN
            | IBV_QP_AV
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;

  err = bnxt_re_dv_modify_qp(qp, &attr, attr_mask, 0, 0);
  CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
}

void GDABackend::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state      = IBV_QPS_RTS;
  attr.sq_psn        = dest->psn;
  attr.max_rd_atomic = GDA_MAX_ATOMIC;
  attr.timeout       = 14;
  attr.retry_cnt     = 7;
  attr.rnr_retry     = 7;

  attr_mask = IBV_QP_STATE
            | IBV_QP_SQ_PSN
            | IBV_QP_MAX_QP_RD_ATOMIC
            | IBV_QP_TIMEOUT
            | IBV_QP_RETRY_CNT
            | IBV_QP_RNR_RETRY;

  err = bnxt_re_dv_modify_qp(qp, &attr, attr_mask, 0, 0);
  CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
}

void GDABackend::create_qps(uint8_t port, ibv_port_attr* ib_port_att) {
  int resize_length = (maximum_num_contexts_ + 1) * num_pes;

  cqs.resize(resize_length);
  bnxt_cqs.resize(resize_length);

  bnxt_qps.resize(resize_length);
  qps.resize(resize_length);

  create_cqs(qps.size(), sq_size);
  create_qps_impl(qps.size());
  init_qp_status(port);

  for (int i{0}; i < qps.size(); i++) {
    dest_info[i].lid = ib_port_att->lid;
    dest_info[i].qpn = qps[i]->qp_num;
    dest_info[i].psn = 0;
    dest_info[i].gid = gid;
  }
}

void GDABackend::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  struct bnxt_re_dv_obj dv_obj;
  struct bnxt_re_dv_cq dv_cq;
  struct bnxt_re_dv_qp dv_qp;
  struct ibv_context *context;
  struct ibv_qp *ib_qp;
  int err;

  context = ib_state->context;
  ib_qp = qps[conn_num];

  /* Export CQ */
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.cq.in  = cqs[conn_num];
  dv_obj.cq.out = &dv_cq;

  err = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(CQ)");

  memset(&gpu_qp->cq, 0, sizeof(bnxt_device_cq));
  gpu_qp->cq.buf   = bnxt_cqs[conn_num].buf;
  gpu_qp->cq.depth = bnxt_cqs[conn_num].depth;
  gpu_qp->cq.id    = dv_cq.cqn;
  gpu_qp->cq.phase = BNXT_RE_QUEUE_START_PHASE;

  /* Export QP */
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.qp.in  = ib_qp;
  dv_obj.qp.out = &dv_qp;

  err = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_QP);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(QP)");

  memset(&gpu_qp->sq, 0, sizeof(bnxt_device_sq));
  gpu_qp->sq.buf        = bnxt_qps[conn_num].sq_buf;
  gpu_qp->sq.depth      = bnxt_qps[conn_num].mem_info.sq_slots;

  if ((gpu_qp->sq.depth % BNXT_RE_STATIC_WQE_BB) != 0) {
    fprintf(stderr,
            "[WARNING] SQ depth not divisible by BNXT_RE_STATIC_WQE_BB. "
            "There may be runtime errors.\n");
  }

  gpu_qp->sq.id          = ib_qp->qp_num;
  gpu_qp->sq.msntbl      = bnxt_qps[conn_num].msntbl;
  gpu_qp->sq.msn_tbl_sz  = bnxt_qps[conn_num].msn_tbl_sz;
  gpu_qp->sq.psn_sz_log2 = std::log2(bnxt_qps[conn_num].mem_info.sq_psn_sz);
  gpu_qp->sq.mtu         = ibv_mtu_to_int(ib_state->portinfo.active_mtu);

  /* Export DB */
  err = bnxt_re_dv_get_default_db_region(context, &db_region_attr);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(QP)");

  CHECK_HIP(hipHostRegister(db_region_attr.dbr, getpagesize(), hipHostRegisterDefault));
  CHECK_HIP(hipHostGetDevicePointer((void**) &gpu_qp->dbr, db_region_attr.dbr, 0));

  /* Export Memory Keys */
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey = heap_rkey[conn_num % num_pes];
}

void GDABackend::create_cqs(int ncqs, int cqe) {
  struct bnxt_re_dv_cq_attr cq_attr;
  struct bnxt_re_dv_cq_init_attr cq_init_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;
  struct ibv_context *context;

  context = ib_state->context;

  for (int i = 0; i < ncqs; i++) {
    /* Allocate CQ mem */
    memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_attr));
    bnxt_cqs[i].handle = bnxt_re_dv_cq_mem_alloc(context, cqe, &cq_attr);
    CHECK_NNULL(bnxt_cqs[i].handle, "bnxt_re_dv_cq_mem_alloc");

    /* Allocate CQ UMEM */
    bnxt_cqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
    bnxt_cqs[i].depth  = cq_attr.ncqe;
    CHECK_HIP(hipExtMallocWithFlags(&bnxt_cqs[i].buf, bnxt_cqs[i].length, hipDeviceMallocUncached));

    /* Register CQ UMEM */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = bnxt_cqs[i].buf;
    umem_attr.size         = bnxt_cqs[i].length;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    bnxt_cqs[i].umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
    CHECK_NNULL(bnxt_cqs[i].umem_handle, "bnxt_re_dv_umem_reg(cq_buf)");

    /* Create CQ */
    memset(&cq_init_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
    cq_init_attr.cq_handle   = (uint64_t) bnxt_cqs[i].handle;
    cq_init_attr.umem_handle = bnxt_cqs[i].umem_handle;
    cq_init_attr.ncqe        = cq_attr.ncqe;

    cqs[i] = bnxt_re_dv_create_cq(context, &cq_init_attr);
    CHECK_NNULL(cqs[i], "bnxt_re_dv_create_cq");
  }
}

void GDABackend::create_qps_impl(int nqps) {
  struct ibv_pd *pd;
  struct ibv_context *context;
  struct ibv_qp_init_attr ib_qp_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;
  void *sq_ptr;
  void *rq_ptr;
  void* sq_umem_handle;
  void* rq_umem_handle;
  uint64_t msntbl_len;
  uint64_t msntbl_offset;
  int err;

  pd = ib_state->pd_orig;
  context = ib_state->context;

  for (int i = 0; i < nqps; i++) {
    /* IB QP Init Attr */
    memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_init_attr));
    ib_qp_attr.send_cq             = cqs[i];
    ib_qp_attr.recv_cq             = cqs[i];
    ib_qp_attr.cap.max_send_wr     = sq_size;
    ib_qp_attr.cap.max_recv_wr     = 0;
    ib_qp_attr.cap.max_send_sge    = 1;
    ib_qp_attr.cap.max_recv_sge    = 0;
    ib_qp_attr.cap.max_inline_data = 0;
    ib_qp_attr.qp_type             = IBV_QPT_RC;
    ib_qp_attr.sq_sig_all          = 0;

    /* Alloc qp_mem_info */
    memset(&bnxt_qps[i].mem_info, 0, sizeof(struct bnxt_re_dv_qp_mem_info));
    err = bnxt_re_dv_qp_mem_alloc(pd, &ib_qp_attr, &bnxt_qps[i].mem_info);
    CHECK_ZERO(err, "bnxt_re_dv_qp_mem_alloc");

    /* Alloc SQ */
    CHECK_HIP(hipExtMallocWithFlags(&sq_ptr, bnxt_qps[i].mem_info.sq_len, hipDeviceMallocUncached));
    bnxt_qps[i].mem_info.sq_va = (uint64_t) sq_ptr;
    bnxt_qps[i].sq_buf = sq_ptr;

    /* Obtain MSN Table Pointer */
    msntbl_len             = (bnxt_qps[i].mem_info.sq_psn_sz * bnxt_qps[i].mem_info.sq_npsn);
    msntbl_offset          = bnxt_qps[i].mem_info.sq_len - msntbl_len;
    bnxt_qps[i].msntbl     = (void*) ((char*) bnxt_qps[i].sq_buf + msntbl_offset);
    bnxt_qps[i].msn_tbl_sz = bnxt_qps[i].mem_info.sq_npsn;

    /* Alloc RQ */
    CHECK_HIP(hipExtMallocWithFlags(&rq_ptr, bnxt_qps[i].mem_info.rq_len, hipDeviceMallocUncached));
    bnxt_qps[i].mem_info.rq_va = (uint64_t) rq_ptr;
    bnxt_qps[i].rq_buf = rq_ptr;

    /* Register UMEM */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = (void*) bnxt_qps[i].mem_info.sq_va;
    umem_attr.size         = bnxt_qps[i].mem_info.sq_len;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    sq_umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
    CHECK_NNULL(sq_umem_handle, "bnxt_re_dv_umem_reg(sq)");

    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = (void*) bnxt_qps[i].mem_info.rq_va;
    umem_attr.size         = bnxt_qps[i].mem_info.rq_len;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    rq_umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
    CHECK_NNULL(rq_umem_handle, "bnxt_re_dv_umem_reg(rq)");

    /* IB DV QP Init Attr */
    memset(&bnxt_qps[i].attr, 0, sizeof(struct bnxt_re_dv_qp_init_attr));
    bnxt_qps[i].attr.send_cq         = ib_qp_attr.send_cq;
    bnxt_qps[i].attr.recv_cq         = ib_qp_attr.recv_cq;
    bnxt_qps[i].attr.max_send_wr     = ib_qp_attr.cap.max_send_wr;
    bnxt_qps[i].attr.max_recv_wr     = ib_qp_attr.cap.max_recv_wr;
    bnxt_qps[i].attr.max_send_sge    = ib_qp_attr.cap.max_send_sge;
    bnxt_qps[i].attr.max_recv_sge    = ib_qp_attr.cap.max_recv_sge;
    bnxt_qps[i].attr.max_inline_data = ib_qp_attr.cap.max_inline_data;
    bnxt_qps[i].attr.qp_type         = ib_qp_attr.qp_type;

    bnxt_qps[i].attr.qp_handle = bnxt_qps[i].mem_info.qp_handle;
    bnxt_qps[i].attr.sq_umem_handle = sq_umem_handle;
    bnxt_qps[i].attr.sq_len    = bnxt_qps[i].mem_info.sq_len;
    bnxt_qps[i].attr.sq_slots  = bnxt_qps[i].mem_info.sq_slots;
    bnxt_qps[i].attr.sq_wqe_sz = bnxt_qps[i].mem_info.sq_wqe_sz;
    bnxt_qps[i].attr.sq_psn_sz = bnxt_qps[i].mem_info.sq_psn_sz;
    bnxt_qps[i].attr.sq_npsn   = bnxt_qps[i].mem_info.sq_npsn;

    bnxt_qps[i].attr.rq_umem_handle = rq_umem_handle;
    bnxt_qps[i].attr.rq_len    = bnxt_qps[i].mem_info.rq_len;
    bnxt_qps[i].attr.rq_slots  = bnxt_qps[i].mem_info.rq_slots;
    bnxt_qps[i].attr.rq_wqe_sz = bnxt_qps[i].mem_info.rq_wqe_sz;
    bnxt_qps[i].attr.comp_mask = bnxt_qps[i].mem_info.comp_mask;

    /* Alloc QP */
    qps[i] = bnxt_re_dv_create_qp(pd, &bnxt_qps[i].attr);
    CHECK_NNULL(qps[i], "bnxt_re_dv_create_qp");
  }
}

}  // namespace rocshmem

