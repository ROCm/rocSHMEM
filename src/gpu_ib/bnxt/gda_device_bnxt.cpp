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

#include "gpu_ib/gda_device.hpp"
#include "gpu_ib/gpuib_macros.inl"
#include <unistd.h> // getpagesize()

namespace rocshmem {

void GDADevice::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  int err;

  ib_state = new ib_state_t;
  GPUIB_CHECK_NNULL(ib_state, "ib_state object create");

  ib_state->context = ibv_open_device(ib_dev);
  GPUIB_CHECK_NNULL(ib_state->context, "ibv_open_device");

  ib_state->pd_orig = ibv_alloc_pd(ib_state->context);
  GPUIB_CHECK_NNULL(ib_state->pd_orig, "ibv_alloc_pd");

  err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  GPUIB_CHECK_ZERO(err, "ibv_query_port");

  err = ibv_query_gid(ib_state->context, port, GPUIB_DEFAULT_GID, &gid);
  GPUIB_CHECK_ZERO(err, "ibv_query_gid");
}

void GDADevice::init_qp_status(uint8_t port) {
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


  for (int i =0; i < qps.size() ; i++) {
    err = bnxt_re_dv_modify_qp(qps[i], &attr, 0, 0);
    GPUIB_CHECK_ZERO(err, "bnxt_re_dv_modify_qp");
  }
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
  cqs.resize((maximum_num_contexts_ + 1) * num_pes);
  qps.resize((maximum_num_contexts_ + 1) * num_pes);

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

void GDADevice::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  fprintf(stderr, "\n\n%s is not fully implemented\n\n", __func__);

  int err;
  uint64_t db_addr = 0;

  gpu_qp->dpi = nullptr; /* TODO */
  gpu_qp->cq_buf = (void*) ((char*) cq_buf + (conn_num * cq_buf_offset));
  gpu_qp->sq_buf = (void*) ((char*) qp_buf + (conn_num * qp_buf_offset));
  gpu_qp->rq_buf = (void*) ((char*) qp_buf + (conn_num * qp_buf_offset) + sq_buf_offset);

  err = bnxt_re_dv_query_dpi(ib_state->context, &db_addr);
  GPUIB_CHECK_ZERO(err, "bnxt_re_dv_query_dpi");

  host_dpi_ptr = (uint64_t*) db_addr;
}

void GDADevice::create_cqs(int ncqs, int cqe) {
  struct bnxt_re_dv_cq_init_attr cq_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;
  struct ibv_context *context;

  int dmabuf_fd = 0;
  uint64_t offset = 0;

  context = ib_state->context;

  /* From Thor 2 docs:
   * nqce = (max_send_wr + max_recv_wr) * num_qps associated to this cq
   * cq_slots = align(ncqe + 1) * 2
   * total_bytes = cq_slots * 16
   *
   * TODO: Adjust to use the correct amount. We currently use `cqe`.
   */

  cq_buf_offset = next_pow((cqe + 1), 2) * BNXT_CQE_SIZE;

  cq_buf = calloc(ncqs, cq_buf_offset);
  GPUIB_CHECK_NNULL(cq_buf, "calloc(cq_buf)");

  CHECK_HIP(hipHostRegister(cq_buf, (ncqs * cq_buf_offset), hipHostRegisterMapped));
  CHECK_HIP(hipHostGetDevicePointer((void**) &gpu_cq_buf, cq_buf, 0));

  memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
  umem_attr.addr = cq_buf;
  umem_attr.size = ncqs * cq_buf_offset;
  umem_attr.dmabuf_fd = dmabuf_fd;

  cq_umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
  GPUIB_CHECK_NNULL(cq_umem_handle, "bnxt_re_dv_umem_reg(cq_buf)");

  memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
  cq_attr.umem_handle    = cq_umem_handle;
  cq_attr.ncqe           = cqe;

  for (int i = 0; i < ncqs; i++) {
    cq_attr.cq_umem_offset = i * cq_buf_offset;

    cqs[i] = bnxt_re_dv_create_cq(context, &cq_attr);
    GPUIB_CHECK_NNULL(cqs[i], "bnxt_re_dv_create_cq");
  }
}

void GDADevice::create_qps_impl(int nqps) {
  struct ibv_pd *pd;
  struct ibv_context *context;
  struct bnxt_re_dv_qp_init_attr attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;
  int q_slots;
  int msn_table_slots;

  pd = ib_state->pd_orig;
  context = ib_state->context;

  /* From Thor 2 docs:
   * SQ_slots = align((max_send_wr + 1) * (2 + max sge per wqe), num_slots_per_4K_page)
   * MSN_tbl_slots = pow-of-2((max_send_wr + 1) * 8)
   * total_bytes = (SQ_slots + MSN tbl slots) * 16 bytes/slot
   */
  q_slots = next_pow((sq_size + 1) * (2 + 13), 256);
  msn_table_slots = (int) pow(2, ((sq_size + 1) * 8));
  sq_buf_offset = (q_slots + msn_table_slots) * 16;

  q_slots = next_pow((0 + 1) * (2 + 13), 256);
  msn_table_slots = (int) pow(2, ((0 + 1) * 8));
  rq_buf_offset = (q_slots + msn_table_slots) * 16;

  qp_buf_offset = sq_buf_offset + rq_buf_offset;

  qp_buf = calloc(nqps, qp_buf_offset);
  GPUIB_CHECK_NNULL(qp_buf, "calloc(qp_buf)");

  CHECK_HIP(hipHostRegister(qp_buf, (nqps * qp_buf_offset), hipHostRegisterMapped));
  CHECK_HIP(hipHostGetDevicePointer((void**) &gpu_qp_buf, qp_buf, 0));

  memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
  umem_attr.addr = qp_buf;
  umem_attr.size = nqps * qp_buf_offset;

  qp_umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
  GPUIB_CHECK_NNULL(qp_umem_handle, "bnxt_re_dv_umem_reg(qp_umem_handle)");

  memset(&attr, 0, sizeof(struct bnxt_re_dv_qp_init_attr));
  attr.qp_type        = IBV_QPT_RC;
  attr.max_send_wr    = sq_size;
  attr.max_send_sge   = 1;
  attr.sq_umem_handle = qp_umem_handle;
  attr.rq_umem_handle = qp_umem_handle;

  for (int i = 0; i < nqps; i++) {
    int base_offset = i * qp_buf_offset;
    attr.sq_umem_offset = base_offset;
    attr.rq_umem_offset = base_offset + sq_buf_offset;
    attr.send_cq        = cqs[i];
    attr.recv_cq        = cqs[i];

    qps[i] = bnxt_re_dv_create_qp(pd, &attr);
    GPUIB_CHECK_NNULL(qps[i], "bnxt_re_dv_create_qp");
  }
}

}  // namespace rocshmem

