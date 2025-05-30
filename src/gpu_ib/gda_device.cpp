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

#include "gda_device.hpp"

#include <cstdio>
#include <cstdlib>
#include <endian.h>
#include <mpi.h>
#include <vector>
#include <mutex>
#include <rocshmem/rocshmem.hpp>
#include <unistd.h>

#include "context_incl.hpp"
#include "gpuib_macros.inl"
#include "gpu_ib_team.hpp"
#include "host/host.hpp"
#include "queue_pair.hpp"
#include "topology.hpp"

namespace rocshmem {

#define NET_CHECK(cmd) {                                     \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort();                                               \
    }                                                        \
  }

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

rocshmem_team_t get_external_team(GPUIBTeam *team) {
  return reinterpret_cast<rocshmem_team_t>(team);
}

int get_ls_non_zero_bit(char *bitmask, int mask_length) {
  int position{-1};
  for (int bit_i{0}; bit_i < mask_length; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    if (bitmask[byte_i] & (1 << (bit_i % CHAR_BIT))) {
      position = bit_i;
      break;
    }
  }

  return position;
}

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
  DPRINTF("\n"
         "===============================================\n"
         "                IBV_CONTEXT\n"
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
  DPRINTF("\n"
         "===============================================\n"
         "               IBV_DEVICE\n"
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
  DPRINTF("\n"
         "===============================================\n"
         "               IBV_PD\n"
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
  DPRINTF("\n"
         "===============================================\n"
         "               IBV_PORT_ATTR\n"
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
  DPRINTF("\n");
  DPRINTF("============== QP_DUMP CONNECTION#%d ==========\n", conn_num);
  DPRINTF("  (ibv_context*)      context          = %p\n",   qp->context);
  DPRINTF("  (void*)             qp_context       = %p\n",   qp->qp_context);
  DPRINTF("  (ibv_pd*)           pd               = %p\n",   qp->pd);
  DPRINTF("  (ibv_cq*)           send_cq          = %p\n",   qp->send_cq);
  DPRINTF("  (ibv_cq*)           recv_cq          = %p\n",   qp->recv_cq);
  DPRINTF("  (ibv_srq*)          srq              = %p\n",   qp->srq);
  DPRINTF("  (uint32_t)          handle           = 0x%x\n", qp->handle);
  DPRINTF("  (uint32_t)          qp_num           = 0x%x\n", qp->qp_num);
  DPRINTF("  (enum ibv_qp_state) state            = %u\n",   qp->state);
  DPRINTF("  (enum_ibv_qp_type)  qp_type          = %u\n",   qp->qp_type);
  DPRINTF("  (uint32_t)          events_completed = %u\n",   qp->events_completed);
  DPRINTF("=========== QP_DUMP_END CONNECTION#%d  ========\n", conn_num);
}

#if !defined(GPUIB_IONIC) && !defined(GPUIB_BNXT)
void dump_mlx5dv_qp(struct mlx5dv_qp *qp_dv, int conn_num) {
  DPRINTF("\n");
  DPRINTF("===============================================\n");
  DPRINTF("     INITIALIZED MLXDV_QP FOR CONNECTION#%d\n", conn_num);
  DPRINTF("===============================================\n");
  DPRINTF("=================== QP_DUMP ===================\n");
  DPRINTF("  (__be32*)  dbrec           = %p\n",     qp_dv->dbrec);
  DPRINTF("  (void*)    sq.buf          = %p\n",     qp_dv->sq.buf);
  DPRINTF("  (uint32_t) sq.wqe_cnt      = %u\n",     qp_dv->sq.wqe_cnt);
  DPRINTF("  (uint32_t) sq.stride       = %u\n",     qp_dv->sq.stride);
  DPRINTF("  (void*)    rq.buf          = %p\n",     qp_dv->rq.buf);
  DPRINTF("  (uint32_t) rq.wqe_cnt      = %u\n",     qp_dv->rq.wqe_cnt);
  DPRINTF("  (uint32_t) rq.stride       = %u\n",     qp_dv->rq.stride);
  DPRINTF("  (void*)    bf.reg          = %p\n",     qp_dv->bf.reg);
  DPRINTF("  (uint32_t) bf.size         = 0x%x\n",   qp_dv->bf.size);
  DPRINTF("  (uint64_t) comp_mask       = 0x%lx\n",  qp_dv->comp_mask);
  DPRINTF("  (off_t)    uar_mmap_offset = 0x%lx\n",  qp_dv->uar_mmap_offset);
  DPRINTF("  (uint32_t) tirn            = 0x%x\n",   qp_dv->tirn);
  DPRINTF("  (uint32_t) tisn            = 0x%x\n",   qp_dv->tisn);
  DPRINTF("  (uint32_t) rqn             = 0x%x\n",   qp_dv->rqn);
  DPRINTF("  (uint32_t) sqn             = 0x%x\n",   qp_dv->sqn);
  DPRINTF("  (uint64_t) tir_icm_addr    = 0x%lx\n",  qp_dv->tir_icm_addr);
  DPRINTF("================== QP_DUMP_END ================\n");
}

void dump_mlx5dv_cq(struct mlx5dv_cq *cq_dv, int conn_num) {
  DPRINTF("\n");
  DPRINTF("===============================================\n");
  DPRINTF("     INITIALIZED MLX5DV_CQ FOR CONNECTION#%d\n", conn_num);
  DPRINTF("===============================================\n");
  DPRINTF("=================== CQ_DUMP ===================\n");
  DPRINTF("  (void*)    buf             = %p\n",     cq_dv->buf);
  DPRINTF("  (__be32*)  dbrec           = %p\n",     cq_dv->dbrec);
  DPRINTF("  (uint32_t) cqe_cnt         = %u\n",     cq_dv->cqe_cnt);
  DPRINTF("  (uint32_t) cqe_size        = %u\n",     cq_dv->cqe_size);
  DPRINTF("  (void*)    cq_uar          = %p\n",     cq_dv->cq_uar);
  DPRINTF("  (uint32_t) cqn             = 0x%x\n",   cq_dv->cqn);
  DPRINTF("  (uint64_t) comp_mask       = 0x%lx\n",  cq_dv->comp_mask);
  DPRINTF("================== CQ_DUMP_END ================\n");
}
#endif // !GPUIB_IONIC

GDADevice::GDADevice(MPI_Comm _comm) : comm(_comm), heap(_comm) {
  CHECK_HIP(hipMalloc(&print_lock, sizeof(*print_lock)));
  *print_lock = 0;
  int* print_lock_addr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&print_lock_addr), HIP_SYMBOL(print_lock)));
  CHECK_HIP(hipMemcpy(print_lock_addr, &print_lock, sizeof(print_lock), hipMemcpyDefault));

  int* device_proxy_symbol_addr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&device_proxy_symbol_addr), HIP_SYMBOL(device_proxy)));
  GDADevice* this_temp_addr{this};
  CHECK_HIP(hipMemcpy(device_proxy_symbol_addr, &this_temp_addr, sizeof(this), hipMemcpyDefault));

  if (auto maximum_num_contexts_str = getenv("ROCSHMEM_MAX_NUM_CONTEXTS")) {
    std::stringstream sstream(maximum_num_contexts_str);
    sstream >> maximum_num_contexts_;
  }
  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_USE_IB_HCA"))) {
    requested_dev = strdup(value);
  } else {
    int gpu_dev = 0;
    CHECK_HIP(hipGetDevice(&gpu_dev));
    int nic_dev = rocshmem::GetClosestNicToGpu(gpu_dev, &requested_dev);
    assert (nic_dev != -1);
  }
  if ((value = getenv("ROCSHMEM_SQ_SIZE"))) {
    sq_size = atoi(value);
  }

  init_mpi_once(_comm);
  comm = _comm;
  NET_CHECK(MPI_Comm_size(comm, &num_pes));
  NET_CHECK(MPI_Comm_rank(comm, &my_pe));

  host_interface = new HostInterface(comm, &heap);
  setup_default_host_ctx();
  setup_team_world();

  init_teams();
  init_collective();

  NET_CHECK(MPI_Barrier(comm));
  dest_info.resize(num_pes * (maximum_num_contexts_ + 1));
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

  auto npes = num_pes;
  auto dinfo = dest_info.data();
  for (int i{0}; i < maximum_num_contexts_ + 1; i++) {
    MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t), MPI_CHAR, dinfo + i * npes, sizeof(dest_info_t), MPI_CHAR, comm);
  }

  for (int i{0}; i < qps.size(); i++) {
    change_status_rtr(qps[i], &dest_info[i], port);
  }
  MPI_Barrier(comm);
  for (int i{0}; i < qps.size(); i++) {
    change_status_rts(qps[i], &dest_info[i]);
    dump_ibv_qp(qps[i], i);
  }
  MPI_Barrier(comm);

  heap_memory_rkey();
  setup_gpu_qps();
  setup_ctxs();
  setup_default_ctx();
  NET_CHECK(MPI_Barrier(comm));
}

GDADevice::~GDADevice() {
  CHECK_HIP(hipFree(print_lock));

  destroy_teams();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));

  delete default_host_ctx_;
  CHECK_HIP(hipFree(default_ctx_->qps));
  CHECK_HIP(hipFree(default_ctx_));
  default_ctx_ = nullptr;

  delete host_interface;
  host_interface = nullptr;

  CHECK_HIP(hipFree(gpu_qps));
  gpu_qps = nullptr;

  CHECK_HIP(hipHostFree(heap_rkey));

  ibv_free_device_list(dev_list);

  int ret = ibv_dereg_mr(heap_mr);
  GPUIB_CHECK_ZERO(ret, "ibv_dereg_mr");

  CHECK_HIP(hipFree(ctx_array));

  heap.free(reinterpret_cast<void**>(&barrier_sync));

  delete ib_state;
  if (requested_dev != nullptr)
    free (requested_dev);
}

__device__ bool GDADevice::create_ctx(rocshmem_ctx_t *ctx) {
  GPUIBContext *ctx_;
  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;
  return true;
}

void GDADevice::create_ctx(void **ctx) {
  GPUIBHostContext *new_ctx = nullptr;
  new_ctx = new GPUIBHostContext(this);
  *ctx = new_ctx;
}

void GDADevice::destroy_ctx(Context *ctx) {
  GPUIBHostContext *gpu_ib_host_ctx = reinterpret_cast<GPUIBHostContext*>(ctx);
  delete gpu_ib_host_ctx;
}

__device__ void GDADevice::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<GPUIBContext*>(ctx->ctx_opaque));
}

__host__ void GDADevice::global_exit(int status) {
  MPI_Abort(comm, status);
}

void GDADevice::create_team(Team *parent_team, TeamInfo *team_info_wrt_parent, TeamInfo *team_info_wrt_world, int num_pes, int my_pe_in_new_team, MPI_Comm team_comm, rocshmem_team_t *new_team) {
  NET_CHECK(MPI_Allreduce(team_pool_bitmask_, team_reduced_bitmask_, team_bitmask_size_, MPI_CHAR, MPI_BAND, team_comm));

  auto max_num_teams{team_tracker.get_max_num_teams()};

  int common_index = get_ls_non_zero_bit(team_reduced_bitmask_, max_num_teams);
  if (common_index < 0) { abort(); }
  int byte = common_index / CHAR_BIT;
  team_pool_bitmask_[byte] &= ~(1 << (common_index % CHAR_BIT));

  GPUIBTeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(GPUIBTeam)));
  new (new_team_obj) GPUIBTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes, my_pe_in_new_team, team_comm, common_index);

  *new_team = get_external_team(new_team_obj);
}

void GDADevice::destroy_team(rocshmem_team_t team) {
  GPUIBTeam *team_obj = get_internal_gpu_ib_team(team);

  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  team_pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);

  team_obj->~GPUIBTeam();

  CHECK_HIP(hipFree(team_obj));
}


void GDADevice::init_collective() {
  size_t one_sync_size_bytes {sizeof(*barrier_sync)};
  size_t total_sync_elems {ROCSHMEM_BARRIER_SYNC_SIZE * (maximum_num_contexts_ + 1)};
  size_t sync_size_bytes {one_sync_size_bytes * total_sync_elems};

  heap.malloc(reinterpret_cast<void**>(&barrier_sync), sync_size_bytes);
  for (int i{0}; i < total_sync_elems; i++) {
    barrier_sync[i] = ROCSHMEM_SYNC_VALUE;
  }

  NET_CHECK(MPI_Barrier(comm));
}

void GDADevice::setup_default_host_ctx() {
  default_host_ctx_ = new GPUIBHostContext(this);
  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx_;
}

void GDADevice::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(GPUIBContext) * (maximum_num_contexts_ + 1)));
  for (int i = 0; i < maximum_num_contexts_; i++) {
    new (&ctx_array[i]) GPUIBContext(this, i);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

void GDADevice::setup_default_ctx() {
  CHECK_HIP(hipMalloc(&default_ctx_, sizeof(GPUIBContext)));
  new (default_ctx_) GPUIBContext(this, maximum_num_contexts_);

  int *symbol_address;
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_address), HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT)));

  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;
  rocshmem_ctx_t ctx_default_host{default_ctx_, tinfo};

  CHECK_HIP(hipMemcpy(symbol_address, &ctx_default_host, sizeof(rocshmem_ctx_t), hipMemcpyDefault));
}

void GDADevice::setup_team_world() {
  TeamInfo *team_info_wrt_parent;
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);

  TeamInfo *team_info_wrt_world;
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

  GPUIBTeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(GPUIBTeam)));
  new (team_world) GPUIBTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes, my_pe, comm, 0);

  team_tracker.set_team_world(team_world);
  ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
}

void GDADevice::init_mpi_once(MPI_Comm comm) {
  static std::mutex init_mutex;
  const std::lock_guard<std::mutex> lock(init_mutex);

  int init_done{0};
  NET_CHECK(MPI_Initialized(&init_done));
  if (init_done == 0) {
    int provided;
    NET_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
  }
}

void GDADevice::init_teams() {
  auto max_num_teams{team_tracker.get_max_num_teams()};
  size_t total_sync_elems = sizeof(long) * ROCSHMEM_BARRIER_SYNC_SIZE * max_num_teams;
  heap.malloc(reinterpret_cast<void**>(&barrier_pSync_pool), total_sync_elems);
  long *barrier_pSync;
  for (int team_i{0}; team_i < max_num_teams; team_i++) {
    barrier_pSync = reinterpret_cast<long*>(&barrier_pSync_pool[team_i * ROCSHMEM_BARRIER_SYNC_SIZE]);
    for (int i{0}; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
      barrier_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
  }

  team_bitmask_size_ = (max_num_teams % CHAR_BIT) ? (max_num_teams / CHAR_BIT + 1) : (max_num_teams / CHAR_BIT);
  team_pool_bitmask_ = reinterpret_cast<char*>(malloc(team_bitmask_size_));
  team_reduced_bitmask_ = reinterpret_cast<char*>(malloc(team_bitmask_size_));

  memset(team_pool_bitmask_, 0, team_bitmask_size_);
  memset(team_reduced_bitmask_, 0, team_bitmask_size_);
  for (int bit_i{1}; bit_i < max_num_teams; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    team_pool_bitmask_[byte_i] |= 1 << (bit_i % CHAR_BIT);
  }

  NET_CHECK(MPI_Barrier(comm));
}

void GDADevice::destroy_teams() {
  heap.free(barrier_pSync_pool);
  free(team_pool_bitmask_);
  free(team_reduced_bitmask_);
}

void GDADevice::heap_memory_rkey() {
  auto *base_heap = heap.get_local_heap_base();
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

#ifdef GPUIB_BNXT
  heap_mr = bnxt_re_dv_reg_mr(ib_state->pd_orig, base_heap, heap.get_size(), access);
  GPUIB_CHECK_NNULL(heap_mr, "bnxt_re_dv_reg_mr");
#else
  heap_mr = ibv_reg_mr(ib_state->pd_orig, base_heap, heap.get_size(), access);
  GPUIB_CHECK_NNULL(heap_mr, "ibv_reg_mr");
#endif

  const size_t rkeys_size = sizeof(uint32_t) * num_pes;
  uint32_t *host_rkey_cpy = reinterpret_cast<uint32_t*>(malloc(rkeys_size));
  if (!host_rkey_cpy) { abort(); }

  CHECK_HIP(hipHostMalloc(&heap_rkey, sizeof(uint32_t) * num_pes));
  heap_rkey[my_pe] = heap_mr->rkey;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(host_rkey_cpy, heap_rkey, rkeys_size, hipMemcpyDeviceToHost, stream));
  CHECK_HIP(hipStreamSynchronize(stream));

  MPI_Allgather(MPI_IN_PLACE, sizeof(uint32_t), MPI_CHAR, host_rkey_cpy, sizeof(uint32_t), MPI_CHAR, comm);

  CHECK_HIP(hipMemcpyAsync(heap_rkey, host_rkey_cpy, rkeys_size, hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));

  free(host_rkey_cpy);
}

void GDADevice::setup_gpu_qps() {
  CHECK_HIP(hipMalloc(&gpu_qps, sizeof(QueuePair) * (maximum_num_contexts_ + 1) * num_pes));
  for (int i{0}; i < (maximum_num_contexts_ + 1) * num_pes; i++) {
    QueuePair qp(ib_state->pd_orig);
    CHECK_HIP(hipMemcpy(&gpu_qps[i], &qp, sizeof(QueuePair), hipMemcpyDefault));
    initialize_gpu_qp(&gpu_qps[i], i);
  }
}

void GDADevice::initialize_context(GPUIBContext *ctx, int context_id) {
  CHECK_HIP(hipMalloc(&ctx->qps, sizeof(QueuePair) * num_pes));
  CHECK_HIP(hipMemset(ctx->qps, 0, sizeof(QueuePair) * num_pes));
  for (int i{0}; i < num_pes; i++) {
    int offset = num_pes * context_id + i;
    CHECK_HIP(hipMemcpy(&ctx->qps[i], &gpu_qps[offset], sizeof(QueuePair), hipMemcpyDefault));
    ctx->qps[i].base_heap = ctx->base_heap;
  }
}

#ifndef GPUIB_BNXT
void GDADevice::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  ib_state = new ib_state_t;
  GPUIB_CHECK_NNULL(ib_state, "ib_state object create");

  ib_state->context = ibv_open_device(ib_dev);
  GPUIB_CHECK_NNULL(ib_state->context, "ib open device");
  dump_ibv_context(ib_state->context);
  dump_ibv_device(ib_state->context->device);

  ib_state->pd_orig = ibv_alloc_pd(ib_state->context);
  GPUIB_CHECK_NNULL(ib_state->pd_orig, "ib allocate pd");
  dump_ibv_pd(ib_state->pd_orig);

  ibv_parent_domain_init_attr pattr{};
  init_parent_domain_attr(&pattr);
  ib_state->pd_parent = ibv_alloc_parent_domain(ib_state->context, &pattr);
  GPUIB_CHECK_NNULL(ib_state->pd_parent, "ibv_alloc_parent_domain");
  dump_ibv_pd(ib_state->pd_parent);

#ifdef GPUIB_IONIC
  ionic_dv_pd_set_sqcmb(ib_state->pd_parent, false, false, false);
  ionic_dv_pd_set_rqcmb(ib_state->pd_parent, false, false, false);

  for (int uxdma_i = 0; uxdma_i < 2; ++uxdma_i) {
    ib_state->pd_uxdma[uxdma_i] = ibv_alloc_parent_domain(ib_state->context, &pattr);
    GPUIB_CHECK_NNULL(ib_state->pd_uxdma[uxdma_i], "ibv_alloc_parent_domain (uxdma)");

    ionic_dv_pd_set_sqcmb(ib_state->pd_uxdma[uxdma_i], false, false, false);
    ionic_dv_pd_set_rqcmb(ib_state->pd_uxdma[uxdma_i], false, false, false);
    ionic_dv_pd_set_udma_mask(ib_state->pd_uxdma[uxdma_i], 1u << uxdma_i);
  }
#endif

  int err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  GPUIB_CHECK_ZERO(err, "ibv_query_port");
  dump_ibv_port_attr(&ib_state->portinfo);

#ifdef GPUIB_IONIC
  ionic_dv_ctx dvctx;
  ionic_dv_get_ctx(&dvctx, ib_state->context);

  int hip_dev_id = 0;
  CHECK_HIP(hipGetDevice(&hip_dev_id));

  void* gpu_db_page = nullptr;
  rocm_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id);

  uint64_t *db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
  uint64_t *gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);

  uint64_t *gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

  ib_state->gpu_db_page = gpu_db_page;
  ib_state->gpu_db_cq = &gpu_db_ptr[dvctx.cq_qtype];
  ib_state->gpu_db_sq = &gpu_db_ptr[dvctx.sq_qtype];
#endif
}

template <typename StateType>
void GDADevice::try_to_modify_qp(ibv_qp* qp, StateType state) {
  int err = ibv_modify_qp(qp, &state.exp_qp_attr, state.exp_attr_mask);
  GPUIB_CHECK_ZERO(err, "ibv_modify_qp");
}

void GDADevice::init_qp_status(ibv_qp* qp, uint8_t port) {
  try_to_modify_qp<InitQPState>(qp, initqp(port));
}

void GDADevice::change_status_rtr(ibv_qp* qp, dest_info_t* dest, uint8_t port) {
  try_to_modify_qp<RtrState>(qp, rtr(dest, port));
}

void GDADevice::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  try_to_modify_qp<RtsState>(qp, rts(dest));
}

void GDADevice::create_qps(uint8_t port, ibv_port_attr* ib_port_att) {
  ibv_qp_cap cap{};
  cap.max_send_wr = sq_size;
  cap.max_send_sge = 1;
  cap.max_inline_data = 0;
#ifdef GPUIB_IONIC
  // TODO allow zero sges in the driver
  cap.max_recv_sge = 1;
#endif
  QPInitAttr qp_init_attr{qpattr(cap)};
  cqs.resize((maximum_num_contexts_ + 1) * num_pes);
  qps.resize((maximum_num_contexts_ + 1) * num_pes);
  int max_num_cqe = qp_init_attr.attr.cap.max_send_wr;
  for (int i{0}; i < qps.size(); i++) {
#ifdef GPUIB_IONIC
    int uxdma_i = ((i + 1) / 2) & 1;
    cqs[i] = create_cq(ib_state->context, ib_state->pd_uxdma[uxdma_i], max_num_cqe << 1);
    GPUIB_CHECK_NNULL(cqs[i], "create_cq");
    qps[i] = create_qp(ib_state->pd_uxdma[uxdma_i], ib_state->context, &qp_init_attr.attr, cqs[i]);
#else
    cqs[i] = create_cq(ib_state->context, ib_state->pd_parent, max_num_cqe);
    GPUIB_CHECK_NNULL(cqs[i], "create_cq");
    qps[i] = create_qp(ib_state->pd_parent, ib_state->context, &qp_init_attr.attr, cqs[i]);
#endif
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
#endif

void* GDADevice::buf_alloc(struct ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type) {
  void* dev_ptr{nullptr};
#ifdef GPUIB_IONIC
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
#else
  CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&dev_ptr), size, hipHostMallocDefault));
#endif
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void GDADevice::buf_release(struct ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type) {
  CHECK_HIP(hipFree(ptr));
}

#ifndef GPUIB_BNXT
void GDADevice::init_parent_domain_attr(ibv_parent_domain_init_attr* attr1) {
  attr1->pd = ib_state->pd_orig;
  attr1->td = nullptr;
  attr1->comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  attr1->alloc = GDADevice::buf_alloc;
  attr1->free = GDADevice::buf_release;
  attr1->pd_context = nullptr;
}

ibv_cq* GDADevice::create_cq(ibv_context* context, ibv_pd* pd, int cqe) {
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

void GDADevice::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));

#ifdef GPUIB_IONIC
  uint8_t udma_idx = ionic_dv_qp_get_udma_idx(qps[conn_num]);

  ionic_dv_cq dvcq;
  ionic_dv_get_cq(&dvcq, cqs[conn_num], udma_idx);

  gpu_qp->cq_dbreg = ib_state->gpu_db_cq;
  gpu_qp->cq_dbval = dvcq.q.db_val;
  gpu_qp->cq_mask = dvcq.q.mask;

  gpu_qp->cq_buf = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);

  ionic_dv_qp dvqp;
  ionic_dv_get_qp(&dvqp, qps[conn_num]);

  gpu_qp->sq_dbreg = ib_state->gpu_db_sq;
  gpu_qp->sq_dbval = dvqp.sq.db_val;
  gpu_qp->sq_mask = dvqp.sq.mask;
  gpu_qp->sq_buf = reinterpret_cast<ionic_v1_wqe *>(dvqp.sq.ptr);

  gpu_qp->qp_num = qps[conn_num]->qp_num;
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey = heap_rkey[conn_num % num_pes];
  gpu_qp->inline_threshold = 32;
#else // !GPUIB_IONIC
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

  gpu_qp->cq_buf = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
  gpu_qp->cq_cnt = cq_out.cqe_cnt;
  gpu_qp->cq_log_cnt = log2(cq_out.cqe_cnt);
  gpu_qp->cq_dbrec = cq_out.dbrec;

  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[conn_num];
  mlx_obj.qp.out = &qp_out;
  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);
  dump_mlx5dv_qp(&qp_out, conn_num);

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

  gpu_qp->dbrec = &qp_out.dbrec[1]; // points to two pointers: 0 -> MLX5_REC_DBR, 1 -> MLX5_SND_DBR
  gpu_qp->sq_buf = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
  gpu_qp->sq_wqe_cnt = qp_out.sq.wqe_cnt;
  gpu_qp->rkey = htobe32(heap_rkey[conn_num % num_pes]);
  gpu_qp->lkey = htobe32(heap_mr->lkey);
  gpu_qp->qp_num = qps[conn_num]->qp_num;
  // The 2 in qp_out.bf.size * 2 below facilitates the switching between blue flame registers
  void* gpu_ptr{nullptr};
  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr, hip_dev_id);
  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);
#endif // !GPUIB_IONIC
}

ibv_qp* GDADevice::create_qp(ibv_pd* pd, ibv_context* context, ibv_qp_init_attr_ex* qp_attr, ibv_cq* cq) {
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

GDADevice::InitQPState GDADevice::initqp(uint8_t port) {
  InitQPState init{};
  init.exp_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  init.exp_qp_attr.port_num = port;
  init.exp_attr_mask |= IBV_QP_ACCESS_FLAGS;
  return init;
}

GDADevice::RtrState GDADevice::rtr(dest_info_t* dest, uint8_t port) {
  RtrState rtr{};
  rtr.exp_qp_attr.dest_qp_num = dest->qpn;
  rtr.exp_qp_attr.rq_psn = dest->psn;
  rtr.exp_qp_attr.ah_attr.port_num = port;
  if (ib_state->portinfo.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    rtr.exp_qp_attr.ah_attr.dlid = dest->lid;
  } else {
    rtr.exp_qp_attr.ah_attr.is_global = 1;
    rtr.exp_qp_attr.ah_attr.grh.dgid = dest->gid;
    rtr.exp_qp_attr.ah_attr.grh.sgid_index = GPUIB_DEFAULT_GID;
    rtr.exp_qp_attr.ah_attr.grh.hop_limit = 1;
  }
  rtr.exp_attr_mask |= IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  return rtr;
}

GDADevice::RtsState GDADevice::rts(dest_info_t* dest) {
  RtsState rts{};
  rts.exp_qp_attr.sq_psn = dest->psn;
  rts.exp_attr_mask |= IBV_QP_SQ_PSN;
  return rts;
}

GDADevice::QPInitAttr GDADevice::qpattr(ibv_qp_cap cap) {
  QPInitAttr qpattr(cap);
  qpattr.attr.qp_type = IBV_QPT_RC;
  return qpattr;
}
#endif

}  // namespace rocshmem

