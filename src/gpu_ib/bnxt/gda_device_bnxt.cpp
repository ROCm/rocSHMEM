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

#include "../gda_device.hpp"
#include "../gpuib_macros.inl"

namespace rocshmem {

void GDADevice::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  ib_state = new ib_state_t;
  GPUIB_CHECK_NNULL(ib_state, "ib_state object create");

  ib_state->context = ibv_open_device(ib_dev);
  GPUIB_CHECK_NNULL(ib_state->context, "ib open device");

  ib_state->pd_orig = ibv_alloc_pd(ib_state->context);
  GPUIB_CHECK_NNULL(ib_state->pd_orig, "ib allocate pd");

  int err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  GPUIB_CHECK_ZERO(err, "ibv_query_port");
}

void GDADevice::init_qp_status(ibv_qp *qp, uint8_t port) {
  int err;
  struct ib_uverbs_qp_attr attr;

  memset(&attr, 0, sizeof(struct ib_uverbs_qp_attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = port;

  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_LOCAL_WRITE
                       | IBV_ACCESS_REMOTE_READ
                       | IBV_ACCESS_REMOTE_ATOMIC;

  attr.qp_attr_mask = IBV_QP_STATE
                    | IBV_QP_PKEY_INDEX
                    | IBV_QP_PORT
                    | IBV_QP_ACCESS_FLAGS;


  err = bnxt_re_dv_modify_qp(qp, &attr, 0, 0);
  GPUIB_CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
}

void GDADevice::change_status_rtr(ibv_qp *qp, dest_info_t *dest, uint8_t port) {
  int err;
  struct ib_uverbs_qp_attr attr;

  memset(&attr, 0, sizeof(struct ib_uverbs_qp_attr));
  attr.qp_attr_mask           = IBV_QP_STATE
                              | IBV_QP_PATH_MTU
                              | IBV_QP_RQ_PSN
                              | IBV_QP_DEST_QPN
                              | IBV_QP_AV
                              | IBV_QP_MAX_DEST_RD_ATOMIC
                              | IBV_QP_MIN_RNR_TIMER;

  attr.qp_state               = IBV_QPS_RTR;
  attr.path_mtu               = IBV_MTU_4096;
  attr.rq_psn                 = dest->psn;
  attr.dest_qp_num            = dest->qpn;

  memcpy(&attr.ah_attr.grh.dgid, &dest->gid, 16);
  attr.ah_attr.grh.sgid_index = GPUIB_DEFAULT_GID;
  attr.ah_attr.grh.hop_limit  = 1;
  attr.ah_attr.sl             = 1;
  attr.ah_attr.is_global      = 1;
  attr.ah_attr.port_num       = port;

  attr.max_dest_rd_atomic     = GPUIB_MAX_ATOMIC;
  attr.min_rnr_timer          = 12;

  err = bnxt_re_dv_modify_qp(qp, &attr, 0, 0);
  GPUIB_CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
}

void GDADevice::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  int err;
  struct ib_uverbs_qp_attr attr;

  memset(&attr, 0, sizeof(struct ib_uverbs_qp_attr));
  attr.qp_attr_mask  = IBV_QP_STATE
                     | IBV_QP_SQ_PSN
                     | IBV_QP_MAX_QP_RD_ATOMIC
                     | IBV_QP_TIMEOUT
                     | IBV_QP_RETRY_CNT
                     | IBV_QP_RNR_RETRY;

  attr.qp_state      = IBV_QPS_RTS;
  attr.sq_psn        = dest->psn;
  attr.max_rd_atomic = GPUIB_MAX_ATOMIC;
  attr.timeout       = 14;
  attr.retry_cnt     = 7;
  attr.rnr_retry     = 7;

  err = bnxt_re_dv_modify_qp(qp, &attr, 0, 0);
  GPUIB_CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
}

void GDADevice::create_qps(uint8_t port, ibv_port_attr* ib_port_att) {
  ibv_qp_cap cap{};
  cap.max_send_wr = sq_size;
  cap.max_send_sge = 1;
  cap.max_inline_data = 0;
  cqs.resize((maximum_num_contexts_ + 1) * num_pes);
  qps.resize((maximum_num_contexts_ + 1) * num_pes);
  for (int i{0}; i < qps.size(); i++) {
    cqs[i] = bnxt_re_dv_create_cq(ib_state->context, sq_size);
    GPUIB_CHECK_NNULL(cqs[i], "bnxt_re_dv_create_cq");
    qps[i] = create_qp(ib_state->pd_orig, cap, cqs[i]);
    GPUIB_CHECK_NNULL(qps[i], "create_qp");
    init_qp_status(qps[i], port);
    dest_info[i].lid = ib_port_att->lid;
    dest_info[i].qpn = qps[i]->qp_num;
    dest_info[i].psn = 0;
    union ibv_gid gid;
    int err = ibv_query_gid(ib_state->context, port, GPUIB_DEFAULT_GID, &gid);
    GPUIB_CHECK_ZERO(err, "ibv_query_gid");
    dest_info[i].gid = gid;
  }
}

void GDADevice::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  fprintf(stderr, "%s is not implemented\n", __func__);
}

struct ibv_qp* GDADevice::create_qp(struct ibv_pd *pd,
                                    struct ibv_qp_cap qp_cap,
                                    struct ibv_cq *cq) {
  struct ibv_qp_init_attr attr;
  struct ibv_qp *qp = nullptr;

  memset(&attr, 0, sizeof(struct ibv_qp_init_attr));

  attr.send_cq    = cq;
  attr.recv_cq    = cq;
  attr.cap        = qp_cap;
  attr.sq_sig_all = 0;
  attr.qp_type    = IBV_QPT_RC;

  qp = bnxt_re_dv_create_qp(pd, &attr);
  GPUIB_CHECK_NNULL(qp, "bnxt_re_dv_create_qp");
  return qp;
}

}  // namespace rocshmem

