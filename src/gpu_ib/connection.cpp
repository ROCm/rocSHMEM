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

#include <mpi.h>

#include <mutex>  // NOLINT(build/c++11)
#include <vector>

#include "backend_ib.hpp"
#include "queue_pair.hpp"
#include "util.hpp"

namespace rocshmem {

int Connection::use_gpu_mem = 0;
int Connection::coherent_cq = 0;

Connection::Connection(GPUIBBackend* b, int k) : backend(b), key_offset(k) {
  char* value = nullptr;

  if ((value = getenv("ROCSHMEM_USE_IB_HCA"))) {
    requested_dev = value;
  }

  if ((value = getenv("ROCSHMEM_SQ_SIZE"))) {
    sq_size = atoi(value);
  }

  if ((value = getenv("ROCSHMEM_USE_CQ_GPU_MEM")) != nullptr) {
    cq_use_gpu_mem = atoi(value);
  }

  if ((value = getenv("ROCSHMEM_USE_SQ_GPU_MEM")) != nullptr) {
    sq_use_gpu_mem = atoi(value);
  }
}

Connection::~Connection() { delete ib_state; }

void Connection::reg_mr(void* ptr, size_t size, ibv_mr** mr, bool managed) {
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
               IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  if (managed) {
    access |= IBV_ACCESS_ON_DEMAND;
  }

  *mr = ibv_reg_mr(ib_state->pd, ptr, size, access);
  GPUIB_CHECK_NNULL(*mr, "ibv_reg_mr");
}

unsigned Connection::total_number_connections() {
  int connections;
  get_remote_conn(&connections);
  return backend->num_blocks_ * connections;
}

void Connection::initialize(int num_block) {
  allocate_dynamic_members(num_block);

  int ib_devices{0};
  dev_list = ibv_get_device_list(&ib_devices);
  GPUIB_CHECK_NNULL(dev_list, "ibv_get_device");

  struct ibv_device* ib_dev = dev_list[0];
  if (requested_dev != nullptr) {
    for (int i = 0; i < ib_devices; i++) {
      const char* select_dev = ibv_get_device_name(dev_list[i]);
      GPUIB_CHECK_NNULL(select_dev, "ibv_get_device_name");

      if (strstr(select_dev, requested_dev) != nullptr) {
        ib_dev = dev_list[i];
        break;
      }
    }
  }

  uint8_t port = 1;
  ib_init(ib_dev, port);

  int ib_fork_err = ibv_fork_init();
  GPUIB_CHECK_ZERO(ib_fork_err, "ibv_fork_init");

  sq_post_dv = static_cast<sq_post_dv_t*>(
      malloc(sizeof(sq_post_dv_t) * total_number_connections()));

  if (sq_post_dv == nullptr) {
    abort();
  }

  create_qps(port, backend->my_pe, &ib_state->portinfo);
  initialize_1(port, num_block);

  MPI_Barrier(backend->thread_comm);
}

void Connection::finalize() {
  ibv_free_device_list(dev_list);

  int ret = ibv_dereg_mr(backend->networkImpl.heap_mr);
  GPUIB_CHECK_ZERO(ret, "ibv_dereg_mr");

  // comment until rocm 4.5
  ret = ibv_dereg_mr(backend->networkImpl.mr);
  GPUIB_CHECK_ZERO(ret, "ibv_dereg_mr");
}

void Connection::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  ib_state = new ib_state_t;
  if (!ib_state) {
    abort();
  }

  ib_state->context = ibv_open_device(ib_dev);
  if (!ib_state->context) {
    delete ib_state;
    abort();
  }

  ib_state->pd = ibv_alloc_pd(ib_state->context);
  if (!ib_state->pd) {
    delete ib_state;
    abort();
  }

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

/**
 * rtr stands for 'ready to receive'
 */
void Connection::change_status_rtr(ibv_qp* qp, dest_info_t* dest,
                                     uint8_t port) {
  try_to_modify_qp<RtrState>(qp, rtr(dest, port));
}

/**
 * rts stands for 'ready to send'
 */
void Connection::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  try_to_modify_qp<RtsState>(qp, rts(dest));
}

void Connection::create_qps(uint8_t port, int my_rank,
                              ibv_port_attr* ib_port_att) {
  ibv_qp_cap cap{};
  cap.max_send_wr = sq_size;
  cap.max_send_sge = 1;
  cap.max_inline_data = 4;

  QPInitAttr qp_init_attr = qpattr(cap);

  size_t qp_size = total_number_connections();
  cqs.resize(qp_size);
  qps.resize(qp_size);

  int cqe = qp_init_attr.attr.cap.max_send_wr;
  for (auto& entry : cqs) {
    entry = create_cq(ib_state->context, ib_state->pd, cqe);
    if (!entry) {
      abort();
    }
  }

  for (int i = 0; i < qps.size(); i++) {
    qps[i] =
        create_qp(ib_state->pd, ib_state->context, &qp_init_attr.attr, cqs[i]);
    if (!qps[i]) {
      abort();
    }

    create_qps_3(port, qps[i], i, ib_port_att);
  }
}

/*
 * Create and write the rdma segment to the SQ
 */
void Connection::set_rdma_seg(mlx5_wqe_raddr_seg* rdma, uint64_t address,
                              uint32_t rkey) {
  rdma->raddr = htobe64(address);
  rdma->rkey = htobe32(rkey);
}

/*
 * Retrieve the address of a SQ.
 * We used this address to write the WQE directly to the SQ.
 */
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
  if (use_gpu_mem) {
    void* dev_ptr;
    if (coherent_cq == 1) {
#if defined USE_COHERENT_HEAP
      CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&dev_ptr), size));
#else
  #ifdef HIP_SUPPORTS_MALLOC_UNCACHED
        CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size,
                                        hipDeviceMallocUncached));
  #else
        CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size,
                                        hipDeviceMallocFinegrained));
  #endif
#endif
    } else {
#ifdef HIP_SUPPORTS_MALLOC_UNCACHED
      CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size,
                                      hipDeviceMallocUncached));
#else
      CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size,
                                      hipDeviceMallocFinegrained));
#endif

    }
    memset(dev_ptr, 0, size);
    return dev_ptr;
  }
  return IBV_ALLOCATOR_USE_DEFAULT;
}

void Connection::buf_release([[maybe_unused]] struct ibv_pd* pd,
                             [[maybe_unused]] void* pd_context, void* ptr,
                             [[maybe_unused]] uint64_t resource_type) {
  if (use_gpu_mem) {
    CHECK_HIP(hipFree(ptr));
  } else {
    free(ptr);
  }
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
  use_gpu_mem = cq_use_gpu_mem;

  ibv_cq_init_attr_ex cq_attr;
  memset(&cq_attr, 0, sizeof(ibv_cq_init_attr_ex));
  cq_attr.cqe = cqe;
  cq_attr.cq_context = nullptr;
  cq_attr.channel = nullptr;
  cq_attr.comp_vector = 0;
  cq_attr.flags = 0;  // see ibv_exp_cq_create_flags
  cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;
  cq_attr.parent_domain = pd;

  coherent_cq = 1;
  ibv_cq_ex* cq_ex = ibv_create_cq_ex(context, &cq_attr);
  coherent_cq = 0;

  GPUIB_CHECK_NNULL(cq_ex, "ibv_create_cq_ex");

  ibv_cq *cq = ibv_cq_ex_to_cq(cq_ex);
  GPUIB_CHECK_NNULL(cq, "ibv_cq_ex_to_cq");
  return cq;
}

void Connection::init_gpu_qp_from_connection(QueuePair* gpu_qp,
                                               int conn_num) {
  int hip_dev_id = 0;
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  use_gpu_mem = cq_use_gpu_mem;

  mlx5dv_cq cq_out;
  mlx5dv_obj mlx_obj;
  mlx_obj.cq.in = cqs[conn_num];
  mlx_obj.cq.out = &cq_out;

  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_CQ);
  gpu_qp->cq_log_size = log2(cq_out.cqe_cnt);
  gpu_qp->cq_size = cq_out.cqe_cnt;

  void* gpu_ptr = nullptr;
  if (use_gpu_mem) {
    gpu_qp->current_cq_q = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
    gpu_qp->dbrec_cq = reinterpret_cast<volatile uint32_t*>(cq_out.dbrec);
  } else {
    rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(cq_out.buf),
                                   cq_out.cqe_cnt * 64, &gpu_ptr, hip_dev_id);
    gpu_qp->current_cq_q = reinterpret_cast<mlx5_cqe64*>(gpu_ptr);

    rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(cq_out.dbrec), 64,
                                   &gpu_ptr, hip_dev_id);

    gpu_qp->dbrec_cq = reinterpret_cast<volatile uint32_t*>(gpu_ptr);
  }
  gpu_qp->current_cq_q_H = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);

  use_gpu_mem = sq_use_gpu_mem;

  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[conn_num];
  mlx_obj.qp.out = &qp_out;

  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);

  gpu_qp->max_nwqe = (qp_out.sq.wqe_cnt);

  volatile uint32_t* dbrec_send = qp_out.dbrec + 1;

  if (use_gpu_mem) {
    gpu_qp->current_sq = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
    gpu_qp->dbrec_send = reinterpret_cast<volatile uint32_t*>(dbrec_send);
  } else {
    gpu_ptr = nullptr;
    rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(qp_out.sq.buf),
                                   qp_out.sq.wqe_cnt * 64, &gpu_ptr,
                                   hip_dev_id);

    gpu_qp->current_sq = reinterpret_cast<uint64_t*>(gpu_ptr);

    rocm_memory_lock_to_fine_grain(
        reinterpret_cast<void*>(const_cast<uint32_t*>(dbrec_send)), 32,
        &gpu_ptr, hip_dev_id);

    gpu_qp->dbrec_send = reinterpret_cast<volatile uint32_t*>(gpu_ptr);
  }

  gpu_qp->current_sq_H = reinterpret_cast<uint64_t*>(qp_out.sq.buf);

  gpu_qp->setDBval(*(reinterpret_cast<uint64_t*>(qp_out.sq.buf)));

  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr,
                                 hip_dev_id);

  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);

  uint32_t* sq = reinterpret_cast<uint32_t*>(qp_out.sq.buf);
  uint32_t ctrl_qp_sq = (reinterpret_cast<uint32_t*>(sq))[1];
  gpu_qp->ctrl_qp_sq = ctrl_qp_sq & 0xFFFFFF;
  gpu_qp->ctrl_sig = (reinterpret_cast<uint64_t*>(sq))[1];
  gpu_qp->rkey = (reinterpret_cast<uint32_t*>(sq))[6 + key_offset];
  gpu_qp->lkey = (reinterpret_cast<uint32_t*>(sq))[9 + key_offset];
}

ibv_qp* Connection::create_qp(ibv_pd* pd, ibv_context* context,
                              ibv_qp_init_attr_ex* qp_attr, ibv_cq* cq) {
  use_gpu_mem = sq_use_gpu_mem;

  ibv_qp* qp = nullptr;

  assert(pd);
  assert(context);
  assert(qp_attr);

  qp_attr->send_cq = cq;
  qp_attr->recv_cq = cq;
  qp_attr->pd = pd;

  qp_attr->comp_mask = IBV_QP_INIT_ATTR_PD;

  qp = create_qp_0(context, qp_attr);

  if (!qp) {
    printf("***** error ibv_create_qp failed %d m %m \n", errno, errno);
    ibv_destroy_cq(cq);
  }

  return qp;
}

Connection::InitQPState Connection::initqp(uint8_t port) {
  InitQPState init{};

  init.exp_qp_attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
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

  rtr.exp_attr_mask |= IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  return rtr;
}

Connection::RtsState Connection::rts(dest_info_t* dest) {
  RtsState rts{};

  rts.exp_qp_attr.sq_psn = dest->psn;

  rts.exp_attr_mask |= IBV_QP_SQ_PSN;

  return rts;
}

ibv_qp* Connection::create_qp_0(ibv_context* context,
                                        ibv_qp_init_attr_ex* qp_attr) {
  ibv_qp *qp = ibv_create_qp_ex(context, qp_attr);
  GPUIB_CHECK_NNULL(qp, "ibv_create_qp_ex");
  return qp;
}


void Connection::get_remote_conn(int* remote_conn) {
  *remote_conn = backend->num_pes;
}

void Connection::initialize_rkey_handle(uint32_t** heap_rkey_handle,
                                                  ibv_mr* mr) {
  CHECK_HIP(
      hipHostMalloc(heap_rkey_handle, sizeof(uint32_t) * backend->num_pes));
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

void Connection::post_dv_rc_wqe(int remote_conn) {
  mlx5_wqe_ctrl_seg* ctrl;
  mlx5_wqe_raddr_seg* rdma;
  mlx5_wqe_data_seg* data;

  for (int i = 0; i < remote_conn; i++) {
    int num_blocks = backend->num_blocks_;
    for (int j = 0; j < num_blocks; j++) {
      int qp_index = i * num_blocks + j;
      uint64_t* ptr = get_address_sq(qp_index);

      const uint16_t nb_post = 1;  // 4 * sq_size;
      for (uint16_t index = 0; index < nb_post; index++) {
        uint8_t op_mod = 0;
        uint8_t op_code = 8;
        uint32_t qp_num = qps[qp_index]->qp_num;
        uint8_t fm_ce_se = 0;
        uint8_t ds = 3;
        ctrl = reinterpret_cast<mlx5_wqe_ctrl_seg*>(ptr);
        mlx5dv_set_ctrl_seg(ctrl, index, op_code, op_mod, qp_num, fm_ce_se, ds,
                            0, 0);
        ptr = ptr + 2;

        rdma = reinterpret_cast<mlx5_wqe_raddr_seg*>(ptr);
        const auto& heap_bases = backend->heap.get_heap_bases();
        auto temp = heap_bases[(backend->my_pe + 1) % 2];
        uint64_t r_address = reinterpret_cast<uint64_t>(temp);
        uint32_t rkey = backend->networkImpl.heap_rkey[i];
        set_rdma_seg(rdma, r_address, rkey);
        ptr = ptr + 2;

        data = reinterpret_cast<mlx5_wqe_data_seg*>(ptr);
        uint32_t lkey = backend->networkImpl.heap_mr->lkey;
        temp = heap_bases[backend->my_pe];
        uint64_t address = reinterpret_cast<uint64_t>(temp);
        mlx5dv_set_data_seg(data, 1, lkey, address);
        ptr = ptr + 4;
      }
    }
  }
}

void Connection::post_wqes() {
  int remote_conn;
  get_remote_conn(&remote_conn);
  post_dv_rc_wqe(remote_conn);
}

void Connection::create_qps_3(int port, ibv_qp* qp, int offset,
                                        ibv_port_attr* ib_port_att) {
  init_qp_status(qp, port);

  all_qp[offset].lid = ib_port_att->lid;
  all_qp[offset].qpn = qp->qp_num;
  all_qp[offset].psn = 0;
  union ibv_gid gid;
  int err = ibv_query_gid(ib_state->context, port, 0, &gid);
  GPUIB_CHECK_ZERO(err, "ibv_query_gid");
  all_qp[offset].gid = gid;
}

void Connection::allocate_dynamic_members(int num_blocks) {
  all_qp.resize(backend->num_pes * num_blocks);
}

void Connection::initialize_1(int port, int num_blocks) {
  MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t) * num_blocks, MPI_CHAR,
               all_qp.data(), sizeof(dest_info_t) * num_blocks, MPI_CHAR,
               backend->thread_comm);

  for (int i = 0; i < qps.size(); i++) {
    change_status_rtr(qps[i], &all_qp[i], port);
  }

  MPI_Barrier(backend->thread_comm);

  for (int i = 0; i < qps.size(); i++) {
    change_status_rts(qps[i], &all_qp[i]);
  }
}

}  // namespace rocshmem
