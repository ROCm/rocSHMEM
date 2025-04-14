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

#include "connection.hpp"
#include "gpuib_macros.inl"

#include "backend_ib.hpp"
#include "queue_pair.hpp"
#include "util.hpp"

namespace rocshmem {

Connection::Connection(GPUIBBackend* b) : backend(b) {
  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_USE_IB_HCA"))) {
    requested_dev = value;
  }
  if ((value = getenv("ROCSHMEM_SQ_SIZE"))) {
    sq_size = atoi(value);
  }
}

Connection::~Connection() {
  delete ib_state;
}

void Connection::reg_mr(void* ptr, size_t size, ibv_mr** mr) {
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  *mr = ibv_reg_mr(ib_state->pd, ptr, size, access);
  GPUIB_CHECK_NNULL(*mr, "ibv_reg_mr");
}

unsigned Connection::total_number_connections() {
  return backend->maximum_num_contexts_ * backend->num_pes;
}

void Connection::initialize(int num_contexts) {
  dest_info.resize(backend->num_pes * num_contexts);
  int ib_devices{0};
  dev_list = ibv_get_device_list(&ib_devices);
  GPUIB_CHECK_NNULL(dev_list, "ibv_get_device");
  struct ibv_device* ib_dev = dev_list[0];
  if (requested_dev) {
    for (int i{0}; i < ib_devices; i++) {
      const char* select_dev{ibv_get_device_name(dev_list[i])};
      GPUIB_CHECK_NNULL(select_dev, "ibv_get_device_name");
      if (strstr(select_dev, requested_dev)) {
        ib_dev = dev_list[i];
        break;
      }
    }
  }
  uint8_t port{1};
  ib_init(ib_dev, port);
  int ib_fork_err = ibv_fork_init();
  GPUIB_CHECK_ZERO(ib_fork_err, "ibv_fork_init");
  create_qps(port, &ib_state->portinfo);
  MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t) * num_contexts, MPI_CHAR, dest_info.data(), sizeof(dest_info_t) * num_contexts, MPI_CHAR, backend->thread_comm);
  for (int i{0}; i < qps.size(); i++) {
    change_status_rtr(qps[i], &dest_info[i], port);
  }
  MPI_Barrier(backend->thread_comm);
  for (int i{0}; i < qps.size(); i++) {
    change_status_rts(qps[i], &dest_info[i]);
  }
  MPI_Barrier(backend->thread_comm);
}

void Connection::finalize() {
  ibv_free_device_list(dev_list);
  int ret = ibv_dereg_mr(backend->networkImpl.heap_mr);
  GPUIB_CHECK_ZERO(ret, "ibv_dereg_mr");
  ret = ibv_dereg_mr(backend->networkImpl.mr);
  GPUIB_CHECK_ZERO(ret, "ibv_dereg_mr");
}

void Connection::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  ib_state = new ib_state_t;
  GPUIB_CHECK_NNULL(ib_state, "ib_state object create");
  ib_state->context = ibv_open_device(ib_dev);
  GPUIB_CHECK_NNULL(ib_state->context, "ib open device");
  ib_state->pd = ibv_alloc_pd(ib_state->context);
  GPUIB_CHECK_NNULL(ib_state->pd, "ib allocate pd");
  ibv_parent_domain_init_attr pattr;
  init_parent_domain_attr(&pattr);
  ib_state->pd = ibv_alloc_parent_domain(ib_state->context, &pattr);
  GPUIB_CHECK_NNULL(ib_state->pd, "ibv_alloc_parent_domain");
  int err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  GPUIB_CHECK_ZERO(err, "ibv_query_port");
}

template <typename StateType>
void Connection::try_to_modify_qp(ibv_qp* qp, StateType state) {
  int err = ibv_modify_qp(qp, &state.exp_qp_attr, state.exp_attr_mask);
  GPUIB_CHECK_ZERO(err, "ibv_modify_qp");
}

void Connection::init_qp_status(ibv_qp* qp, uint8_t port) {
  try_to_modify_qp<InitQPState>(qp, initqp(port));
}

void Connection::change_status_rtr(ibv_qp* qp, dest_info_t* dest, uint8_t port) {
  try_to_modify_qp<RtrState>(qp, rtr(dest, port));
}

void Connection::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  try_to_modify_qp<RtsState>(qp, rts(dest));
}

void Connection::create_qps(uint8_t port, ibv_port_attr* ib_port_att) {
  ibv_qp_cap cap{};
  cap.max_send_wr = sq_size;
  cap.max_send_sge = 1;
  cap.max_inline_data = 4;
  QPInitAttr qp_init_attr{qpattr(cap)};
  cqs.resize(total_number_connections());
  qps.resize(total_number_connections());
  int max_num_cqe = qp_init_attr.attr.cap.max_send_wr;
  for (auto& entry : cqs) {
    entry = create_cq(ib_state->context, ib_state->pd, max_num_cqe);
    GPUIB_CHECK_NNULL(entry, "create_cq");
  }
  for (int i{0}; i < qps.size(); i++) {
    qps[i] = create_qp(ib_state->pd, ib_state->context, &qp_init_attr.attr, cqs[i]);
    GPUIB_CHECK_NNULL(qps[i], "create_qp");
    init_qp_status(qps[i], port);
    dest_info[i].lid = ib_port_att->lid;
    dest_info[i].qpn = qps[i]->qp_num;
    dest_info[i].psn = 0;
    union ibv_gid gid;
    int err = ibv_query_gid(ib_state->context, port, 0, &gid);
    GPUIB_CHECK_ZERO(err, "ibv_query_gid");
    dest_info[i].gid = gid;
  }
}

void Connection::set_rdma_seg(mlx5_wqe_raddr_seg* rdma, uint64_t address, uint32_t rkey) {
  rdma->raddr = htobe64(address);
  rdma->rkey = htobe32(rkey);
}

uint64_t* Connection::get_address_sq(int i) {
  mlx5dv_obj mlx_obj;
  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[i];
  mlx_obj.qp.out = &qp_out;
  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);
  return reinterpret_cast<uint64_t*>(qp_out.sq.buf);
}

void* Connection::buf_alloc([[maybe_unused]] struct ibv_pd* pd,
                            [[maybe_unused]] void* pd_context, size_t size,
                            [[maybe_unused]] size_t alignment,
                            [[maybe_unused]] uint64_t resource_type) {
  void* dev_ptr{nullptr};
#ifdef USE_FINEGRAINED_HEAP
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocFinegrained));
#endif
#ifdef USE_UNCACHED_HEAP
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
#endif
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void Connection::buf_release([[maybe_unused]] struct ibv_pd* pd,
                             [[maybe_unused]] void* pd_context, void* ptr,
                             [[maybe_unused]] uint64_t resource_type) {
  CHECK_HIP(hipFree(ptr));
}

void Connection::init_parent_domain_attr(ibv_parent_domain_init_attr* attr1) {
  attr1->pd = ib_state->pd;
  attr1->td = nullptr;
  attr1->comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  attr1->alloc = Connection::buf_alloc;
  attr1->free = Connection::buf_release;
  attr1->pd_context = nullptr;
}

ibv_cq* Connection::create_cq(ibv_context* context, ibv_pd* pd, int cqe) {
  ibv_cq_init_attr_ex cq_attr;
  memset(&cq_attr, 0, sizeof(ibv_cq_init_attr_ex));
  cq_attr.cqe = cqe;
  cq_attr.cq_context = nullptr;
  cq_attr.channel = nullptr;
  cq_attr.comp_vector = 0;
  cq_attr.flags = 0;  // see ibv_exp_cq_create_flags
  cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;
  cq_attr.parent_domain = pd;
  ibv_cq_ex* cq_ex = ibv_create_cq_ex(context, &cq_attr);
  GPUIB_CHECK_NNULL(cq_ex, "ibv_create_cq_ex");
  ibv_cq *cq = ibv_cq_ex_to_cq(cq_ex);
  GPUIB_CHECK_NNULL(cq, "ibv_cq_ex_to_cq");
  return cq;
}

void Connection::init_gpu_qp_from_connection(QueuePair* gpu_qp, int conn_num) {
  mlx5dv_cq cq_out;
  mlx5dv_obj mlx_obj;
  mlx_obj.cq.in = cqs[conn_num];
  mlx_obj.cq.out = &cq_out;
  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_CQ);

  /*
   * struct mlx5dv_cq {
   *   void                    *buf;
   *   __be32                  *dbrec;
   *   uint32_t                cqe_cnt;
   *   uint32_t                cqe_size;
   *   void                    *cq_uar;
   *   uint32_t                cqn;
   *   uint64_t                comp_mask;
   * };
  */
  gpu_qp->cq_buf_head = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
  gpu_qp->cq_buf = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
  gpu_qp->cq_cnt = cq_out.cqe_cnt;
  gpu_qp->cq_log_cnt = log2(cq_out.cqe_cnt);
  gpu_qp->cq_dbrec = reinterpret_cast<volatile uint32_t*>(cq_out.dbrec);

  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[conn_num];
  mlx_obj.qp.out = &qp_out;
  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);

  /*
   * struct mlx5dv_qp {
   *   __be32 *dbrec;
   *   struct {
   *     void *buf;
   *     uint32_t wqe_cnt;
   *     uint32_t stride;
   *   } sq;
   *   struct {
   *     void *buf;
   *     uint32_t wqe_cnt;
   *     uint32_t stride;
   *   } rq;
   *   struct {
   *     void *reg;
   *     uint32_t size;
   *   } bf;
   *   uint64_t comp_mask;
   *   off_t uar_mmap_offset;
   *   uint32_t tirn;
   *   uint32_t tisn;
   *   uint32_t rqn;
   *   uint32_t sqn;
   *   uint64_t tir_icm_addr;
   * };
   */
  volatile uint32_t* sq_dbrec = qp_out.dbrec;
  gpu_qp->sq_dbrec = reinterpret_cast<volatile uint32_t*>(sq_dbrec);
  gpu_qp->sq_buf_head = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
  gpu_qp->sq_buf = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
  gpu_qp->sq_wqe_cnt = qp_out.sq.wqe_cnt;
  gpu_qp->setDBval(*(reinterpret_cast<uint64_t*>(qp_out.sq.buf)));

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  void* gpu_ptr{nullptr};
  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr, hip_dev_id);
  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);

  uint32_t* sq = reinterpret_cast<uint32_t*>(qp_out.sq.buf);
  uint32_t ctrl_qp_sq = (reinterpret_cast<uint32_t*>(sq))[1];
  gpu_qp->ctrl_qp_sq = ctrl_qp_sq & 0xFFFFFF;
  gpu_qp->ctrl_sig = (reinterpret_cast<uint64_t*>(sq))[1];
  gpu_qp->rkey = (reinterpret_cast<uint32_t*>(sq))[6];
  gpu_qp->lkey = (reinterpret_cast<uint32_t*>(sq))[9];
}

ibv_qp* Connection::create_qp(ibv_pd* pd, ibv_context* context, ibv_qp_init_attr_ex* qp_attr, ibv_cq* cq) {
  ibv_qp* qp{nullptr};
  assert(pd);
  assert(context);
  assert(qp_attr);
  qp_attr->send_cq = cq;
  qp_attr->recv_cq = cq;
  qp_attr->pd = pd;
  qp_attr->comp_mask = IBV_QP_INIT_ATTR_PD;
  qp = ibv_create_qp_ex(context, qp_attr);
  GPUIB_CHECK_NNULL(qp, "ibv_create_qp_ex");
  return qp;
}

Connection::InitQPState Connection::initqp(uint8_t port) {
  InitQPState init{};
  init.exp_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  init.exp_qp_attr.port_num = port;
  init.exp_attr_mask |= IBV_QP_ACCESS_FLAGS;
  return init;
}

Connection::RtrState Connection::rtr(dest_info_t* dest, uint8_t port) {
  RtrState rtr{};
  rtr.exp_qp_attr.dest_qp_num = dest->qpn;
  rtr.exp_qp_attr.rq_psn = dest->psn;
  rtr.exp_qp_attr.ah_attr.port_num = port;
  if (ib_state->portinfo.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    rtr.exp_qp_attr.ah_attr.dlid = dest->lid;
  } else {
    rtr.exp_qp_attr.ah_attr.is_global = 1;
    rtr.exp_qp_attr.ah_attr.grh.dgid = dest->gid;
    rtr.exp_qp_attr.ah_attr.grh.sgid_index = 0;
    rtr.exp_qp_attr.ah_attr.grh.hop_limit = 1;
  }
  rtr.exp_attr_mask |= IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  return rtr;
}

Connection::RtsState Connection::rts(dest_info_t* dest) {
  RtsState rts{};
  rts.exp_qp_attr.sq_psn = dest->psn;
  rts.exp_attr_mask |= IBV_QP_SQ_PSN;
  return rts;
}

void Connection::initialize_rkey_handle(uint32_t** heap_rkey_handle, ibv_mr* mr) {
  CHECK_HIP(hipHostMalloc(heap_rkey_handle, sizeof(uint32_t) * backend->num_pes));
  (*heap_rkey_handle)[backend->my_pe] = mr->rkey;
}

void Connection::free_rkey_handle(uint32_t* heap_rkey_handle) {
  CHECK_HIP(hipHostFree(heap_rkey_handle));
}

Connection::QPInitAttr Connection::qpattr(ibv_qp_cap cap) {
  QPInitAttr qpattr(cap);
  qpattr.attr.qp_type = IBV_QPT_RC;
  return qpattr;
}

void Connection::post_dv_rc_wqe() {
  mlx5_wqe_ctrl_seg* ctrl;
  mlx5_wqe_raddr_seg* rdma;
  mlx5_wqe_data_seg* data;

  for (int i{0}; i < backend->num_pes; i++) {
    int num_contexts = backend->maximum_num_contexts_;
    for (int j{0}; j < num_contexts; j++) {
      int qp_index = i * num_contexts + j;
      uint64_t* ptr = get_address_sq(qp_index);
      uint8_t op_code = 8; // rdma_write
      uint8_t op_mod = 0; // operation modifier
      uint32_t qp_num = qps[qp_index]->qp_num;
      uint8_t fm_ce_se = 0; // fence,completion,solicited_event
      uint8_t ds = 3; // number segments (16B each)
      ctrl = reinterpret_cast<mlx5_wqe_ctrl_seg*>(ptr);
      mlx5dv_set_ctrl_seg(ctrl, 0, op_code, op_mod, qp_num, fm_ce_se, ds, 0, 0);
      ptr = ptr + 2; // 16B

      rdma = reinterpret_cast<mlx5_wqe_raddr_seg*>(ptr);
      const auto& heap_bases = backend->heap.get_heap_bases();
      auto temp = heap_bases[(backend->my_pe + 1) % 2];
      uint64_t r_address = reinterpret_cast<uint64_t>(temp);
      uint32_t rkey = backend->networkImpl.heap_rkey[i];
      set_rdma_seg(rdma, r_address, rkey);
      ptr = ptr + 2; // 16B

      data = reinterpret_cast<mlx5_wqe_data_seg*>(ptr);
      uint32_t lkey = backend->networkImpl.heap_mr->lkey;
      temp = heap_bases[backend->my_pe];
      uint64_t address = reinterpret_cast<uint64_t>(temp);
      mlx5dv_set_data_seg(data, 1, lkey, address);
      ptr = ptr + 4; // 32B
    }
  }
}

}  // namespace rocshmem
