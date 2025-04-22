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
#include "rocshmem_config.h"
#include "util.hpp"

namespace rocshmem {

static void dump_ibv_context(struct ibv_context* x) {
  /* 
   * struct ibv_context {
   *   struct ibv_device      *device;
   *   struct ibv_context_ops  ops;
   *   int                     cmd_fd;
   *   int                     async_fd;
   *   int                     num_comp_vectors;
   *   pthread_mutex_t         mutex;
   *   void                   *abi_compat;
   * };
   */
  printf("\n"
         "===============================================\n"
         "                MLX IBV_CONTEXT\n"
         "===============================================\n"
         "  (ibv_device*)        device              = %p\n"
         "  (int)                cmd_fd              = %d\n"
         "  (int)                async_fd            = %d\n"
         "  (int)                num_comp_vectors    = %d\n"
         "  (void*)              abi_compat          = %p\n",
	 x->device, x->cmd_fd, x->async_fd, x->num_comp_vectors, x->abi_compat);
};

static void dump_ibv_device(struct ibv_device* x) {
  /*
   * struct ibv_device {
   *   struct _ibv_device_ops  _ops;
   *   enum ibv_node_type node_type;
   *   enum ibv_transport_type transport_type;
   *   char name[IBV_SYSFS_NAME_MAX];
   *   char dev_name[IBV_SYSFS_NAME_MAX];
   *   char dev_path[IBV_SYSFS_PATH_MAX];
   *   char ibdev_path[IBV_SYSFS_PATH_MAX];
   * };
   */
  printf("\n"
         "===============================================\n"
         "               MLX IBV_DEVICE\n"
         "===============================================\n"
         "  (enum ibv_node_type)      node_type      = %d\n"
         "  (enum ibv_transport_type) transport_type = %d\n"
         "  (char[])                  name           = %s\n"
         "  (char[])                  dev_name       = %s\n"
         "  (char[])                  dev_path       = %s\n"
         "  (char[])                  ibdev_path     = %s\n",
	 x->node_type, x->transport_type, x->name, x->dev_name, x->dev_path, x->ibdev_path);
}

static void dump_ibv_pd(struct ibv_pd* x) {
  /*
   * struct ibv_pd {
   *   struct ibv_context     *context;
   *   uint32_t                handle;
   * };
   */
  printf("\n"
         "===============================================\n"
         "               MLX IBV_PD\n"
         "===============================================\n"
         "  (ibv_context*) context = %p\n"
         "  (uint32_t)     handle  = 0x%x\n",
	 x->context, x->handle);
}

static void dump_ibv_port_attr(struct ibv_port_attr* x) {
  /*
   * struct ibv_port_attr { 
   *   enum ibv_port_state     state; 
   *   enum ibv_mtu            max_mtu; 
   *   enum ibv_mtu            active_mtu; 
   *   int                     gid_tbl_len; 
   *   uint32_t                port_cap_flags; 
   *   uint32_t                max_msg_sz; 
   *   uint32_t                bad_pkey_cntr; 
   *   uint32_t                qkey_viol_cntr; 
   *   uint16_t                pkey_tbl_len; 
   *   uint16_t                lid; 
   *   uint16_t                sm_lid; 
   *   uint8_t                 lmc; 
   *   uint8_t                 max_vl_num; 
   *   uint8_t                 sm_sl; 
   *   uint8_t                 subnet_timeout; 
   *   uint8_t                 init_type_reply; 
   *   uint8_t                 active_width; 
   *   uint8_t                 active_speed; 
   *   uint8_t                 phys_state; 
   *   uint8_t                 link_layer; 
   *   uint8_t                 flags; 
   *   uint16_t                port_cap_flags2; 
   * }; 
   */
  printf("\n"
         "===============================================\n"
         "               MLX IBV_PORT_ATTR\n"
         "===============================================\n"
         "  (enum ibv_port_state) state           = %u\n"
         "  (enum ibv_mtu)        max_mtu         = %u\n"
         "  (enum ibv_mtu)        active_mtu      = %u\n"
         "  (int)                 gid_tbl_len     = %u\n"
         "  (uint32_t)            port_cap_flags  = 0x%x\n"
         "  (uint32_t)            max_msg_sz      = %u\n"
         "  (uint32_t)            bad_pkey_cntr   = %u\n"
         "  (uint32_t)            qkey_viol_cntr  = %u\n"
         "  (uint16_t)            pkey_tbl_len    = %u\n"
         "  (uint16_t)            lid             = 0x%x\n"
         "  (uint16_t)            sm_lid          = 0x%x\n"
         "  (uint8_t)             lmc             = 0x%x\n"
         "  (uint8_t)             max_vl_num      = 0x%x\n"
         "  (uint8_t)             sm_sl           = 0x%x\n"
         "  (uint8_t)             subnet_timeout  = 0x%x\n"
         "  (uint8_t)             init_type_reply = 0x%x\n"
         "  (uint8_t)             active_width    = 0x%x\n"
         "  (uint8_t)             active_speed    = 0x%x\n"
         "  (uint8_t)             phys_state      = 0x%x\n"
         "  (uint8_t)             link_layer      = 0x%x\n"
         "  (uint8_t)             flags           = 0x%x\n"
         "  (uint16_t)            port_cap_flags2 = 0x%x\n",
	 x->state, x->max_mtu, x->active_mtu, x->gid_tbl_len, x->port_cap_flags, x->max_msg_sz,
	 x->bad_pkey_cntr, x->qkey_viol_cntr, x->pkey_tbl_len, x->lid, x->sm_lid, x->lmc, x->max_vl_num,
	 x->sm_sl, x->subnet_timeout, x->init_type_reply, x->active_width, x->active_speed, x->phys_state,
	 x->link_layer, x->flags, x->port_cap_flags2);
}

void dump_ibv_qp(struct ibv_qp *qp, int conn_num) {
  /*
   * struct ibv_qp {
   *   struct ibv_context     *context;
   *   void                   *qp_context;
   *   struct ibv_pd          *pd;
   *   struct ibv_cq          *send_cq;
   *   struct ibv_cq          *recv_cq;
   *   struct ibv_srq         *srq;
   *   uint32_t                handle;
   *   uint32_t                qp_num;
   *   enum ibv_qp_state       state;
   *   enum ibv_qp_type        qp_type;
   *   pthread_mutex_t         mutex;
   *   pthread_cond_t          cond;
   *   uint32_t                events_completed;
   * };
   */
  printf("\n");
  printf("============== QP_DUMP CONNECTION#%d ==========\n", conn_num);
  printf("  (ibv_context*)      context          = %p\n",   qp->context);
  printf("  (void*)             qp_context       = %p\n",   qp->qp_context);
  printf("  (ibv_pd*)           pd               = %p\n",   qp->pd);
  printf("  (ibv_cq*)           send_cq          = %p\n",   qp->send_cq);
  printf("  (ibv_cq*)           recv_cq          = %p\n",   qp->recv_cq);
  printf("  (ibv_srq*)          srq              = %p\n",   qp->srq);
  printf("  (uint32_t)          handle           = 0x%x\n", qp->handle);
  printf("  (uint32_t)          qp_num           = 0x%x\n", qp->qp_num);
  printf("  (enum ibv_qp_state) state            = %u\n",   qp->state);
  printf("  (enum_ibv_qp_type)  qp_type          = %u\n",   qp->qp_type);
  printf("  (uint32_t)          events_completed = %u\n",   qp->events_completed);
  printf("=========== QP_DUMP_END CONNECTION#%d  ========\n", conn_num);
}

void dump_mlx5dv_qp(struct mlx5dv_qp *qp_dv, int conn_num) {
  printf("\n");
  printf("===============================================\n");
  printf("     INITIALIZED MLXDV_QP FOR CONNECTION#%d\n", conn_num);
  printf("===============================================\n");
  printf("=================== QP_DUMP ===================\n");
  printf("  (__be32*)  dbrec           = %p\n",     qp_dv->dbrec);
  printf("  (void*)    sq.buf          = %p\n",     qp_dv->sq.buf);
  printf("  (uint32_t) sq.wqe_cnt      = %u\n",     qp_dv->sq.wqe_cnt);
  printf("  (uint32_t) sq.stride       = %u\n",     qp_dv->sq.stride);
  printf("  (void*)    rq.buf          = %p\n",     qp_dv->rq.buf);
  printf("  (uint32_t) rq.wqe_cnt      = %u\n",     qp_dv->rq.wqe_cnt);
  printf("  (uint32_t) rq.stride       = %u\n",     qp_dv->rq.stride);
  printf("  (void*)    bf.reg          = %p\n",     qp_dv->bf.reg);
  printf("  (uint32_t) bf.size         = 0x%x\n",   qp_dv->bf.size);
  printf("  (uint64_t) comp_mask       = 0x%lx\n",  qp_dv->comp_mask);
  printf("  (off_t)    uar_mmap_offset = 0x%lx\n",  qp_dv->uar_mmap_offset);
  printf("  (uint32_t) tirn            = 0x%x\n",   qp_dv->tirn);
  printf("  (uint32_t) tisn            = 0x%x\n",   qp_dv->tisn);
  printf("  (uint32_t) rqn             = 0x%x\n",   qp_dv->rqn);
  printf("  (uint32_t) sqn             = 0x%x\n",   qp_dv->sqn);
  printf("  (uint64_t) tir_icm_addr    = 0x%lx\n",  qp_dv->tir_icm_addr);
  printf("================== QP_DUMP_END ================\n");
}

void dump_mlx5dv_cq(struct mlx5dv_cq *cq_dv, int conn_num) {
  printf("\n");
  printf("===============================================\n");
  printf("     INITIALIZED MLX5DV_CQ FOR CONNECTION#%d\n", conn_num);
  printf("===============================================\n");
  printf("=================== CQ_DUMP ===================\n");
  printf("  (void*)    buf             = %p\n",     cq_dv->buf);
  printf("  (__be32*)  dbrec           = %p\n",     cq_dv->dbrec);
  printf("  (uint32_t) cqe_cnt         = %u\n",     cq_dv->cqe_cnt);
  printf("  (uint32_t) cqe_size        = %u\n",     cq_dv->cqe_size);
  printf("  (void*)    cq_uar          = %p\n",     cq_dv->cq_uar);
  printf("  (uint32_t) cqn             = 0x%x\n",   cq_dv->cqn);
  printf("  (uint64_t) comp_mask       = 0x%lx\n",  cq_dv->comp_mask);
  printf("================== CQ_DUMP_END ================\n");
}

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
  printf("CALLING IBV_REG_MR\n");
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
  create_qps(port, &ib_state->portinfo);
  MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t) * num_contexts, MPI_CHAR, dest_info.data(), sizeof(dest_info_t) * num_contexts, MPI_CHAR, backend->thread_comm);
  for (int i{0}; i < qps.size(); i++) {
    change_status_rtr(qps[i], &dest_info[i], port);
  }
  MPI_Barrier(backend->thread_comm);
  for (int i{0}; i < qps.size(); i++) {
    change_status_rts(qps[i], &dest_info[i]);
    dump_ibv_qp(qps[i], i);
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
  dump_ibv_context(ib_state->context);
  dump_ibv_device(ib_state->context->device);

  ib_state->pd = ibv_alloc_pd(ib_state->context);
  GPUIB_CHECK_NNULL(ib_state->pd, "ib allocate pd");
  dump_ibv_pd(ib_state->pd);

  ibv_parent_domain_init_attr pattr;
  init_parent_domain_attr(&pattr);
  ib_state->pd = ibv_alloc_parent_domain(ib_state->context, &pattr);
  GPUIB_CHECK_NNULL(ib_state->pd, "ibv_alloc_parent_domain");
  dump_ibv_pd(ib_state->pd);

  int err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  GPUIB_CHECK_ZERO(err, "ibv_query_port");
  dump_ibv_port_attr(&ib_state->portinfo);
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
  cap.max_inline_data = 0;
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

void* Connection::buf_alloc([[maybe_unused]] struct ibv_pd* pd,
                            [[maybe_unused]] void* pd_context, size_t size,
                            [[maybe_unused]] size_t alignment,
                            [[maybe_unused]] uint64_t resource_type) {
  void* dev_ptr{nullptr};
  CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&dev_ptr), size, hipHostMallocDefault));
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
  dump_mlx5dv_cq(&cq_out, conn_num);

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

//  int hip_dev_id{-1};
//  CHECK_HIP(hipGetDevice(&hip_dev_id));
//  void* gpu_ptr{nullptr};
//  rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(cq_out.buf), cq_out.cqe_cnt * cq_out.cqe_size, &gpu_ptr, hip_dev_id);
//  assert(gpu_ptr);
//  gpu_qp->cq_buf_head = reinterpret_cast<mlx5_cqe64*>(gpu_ptr);
//  gpu_qp->cq_buf = reinterpret_cast<mlx5_cqe64*>(gpu_ptr);
  gpu_qp->cq_buf_head = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
  gpu_qp->cq_buf = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
//  gpu_ptr = nullptr;
  gpu_qp->cq_cnt = cq_out.cqe_cnt;
  gpu_qp->cq_log_cnt = log2(cq_out.cqe_cnt);
//  rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(cq_out.dbrec), sizeof(cq_out.dbrec), &gpu_ptr, hip_dev_id);
//  assert(gpu_ptr);
//  gpu_qp->cq_dbrec = reinterpret_cast<volatile uint32_t*>(gpu_ptr);
  gpu_qp->cq_dbrec = cq_out.dbrec;
//  gpu_ptr = nullptr;

  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[conn_num];
  mlx_obj.qp.out = &qp_out;
  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);
  dump_mlx5dv_qp(&qp_out, conn_num);
  monitor.register_queue(&qp_out, &cq_out, "connection_" + std::to_string(conn_num) + ".txt");

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

//  rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(const_cast<uint32_t*>(qp_out.dbrec)), sizeof(qp_out.dbrec), &gpu_ptr, hip_dev_id);
//  assert(gpu_ptr);
//  gpu_qp->dbrec = reinterpret_cast<volatile uint32_t*>(gpu_ptr);
  gpu_qp->dbrec = qp_out.dbrec;
//  gpu_ptr = nullptr;
//  rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(qp_out.sq.buf), qp_out.sq.wqe_cnt * qp_out.sq.stride, &gpu_ptr, hip_dev_id);
//  assert(gpu_ptr);
//  gpu_qp->sq_buf_head = reinterpret_cast<uint64_t*>(gpu_ptr);
//  gpu_qp->sq_buf = reinterpret_cast<uint64_t*>(gpu_ptr);
  gpu_qp->sq_buf_head = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
  gpu_qp->sq_buf = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
//  gpu_ptr = nullptr;
  gpu_qp->sq_wqe_cnt = qp_out.sq.wqe_cnt;
  gpu_qp->rkey = htobe32(backend->networkImpl.heap_rkey[conn_num % backend->num_pes]);
  gpu_qp->lkey = htobe32(backend->networkImpl.heap_mr->lkey);
  printf("\nASSIGNING RKEY:\n");
  for (int i {0}; i < backend->num_pes; i++) {
    printf("\t backend->networkImpl.heap_rkey index %d - %x\n", i, htobe32(backend->networkImpl.heap_rkey[i]));
  }
  printf("\tconnection# %d ASSIGNED RKEY %x\n", conn_num, htobe32(backend->networkImpl.heap_rkey[conn_num % backend->num_pes]));
  printf("\tconnection# %d ASSIGNED LKEY %x\n", conn_num, htobe32(backend->networkImpl.heap_mr->lkey));
  gpu_qp->qp_num = qps[conn_num]->qp_num;
  // The 2 in qp_out.bf.size * 2 below facilitates the switching between blue flame registers
  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  void* gpu_ptr{nullptr};
  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr, hip_dev_id);
  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);
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

}  // namespace rocshmem
