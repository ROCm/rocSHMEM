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

#ifndef LIBRARY_SRC_GPU_IB_GDA_DEVICE_HPP_
#define LIBRARY_SRC_GPU_IB_GDA_DEVICE_HPP_

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "endian.hpp"
#include <infiniband/verbs.h>
extern "C" {
#ifdef GPUIB_IONIC
#include <infiniband/ionic_dv.h>
#include <infiniband/ionic_fw.h>
#else
#include <infiniband/mlx5dv.h>
#endif
}

#include "context_incl.hpp"
#include "containers/free_list_impl.hpp"
#include "memory/hip_allocator.hpp"
#include "memory/symmetric_heap.hpp"
#include "queue_pair.hpp"
#include "team_tracker.hpp"

namespace rocshmem {

class HostInterface;
class Team;
class TeamInfo;

class GPUIBContext;
class GPUIBHostContext;
class QueuePair;

class GDADevice {
 private:
  typedef struct ib_state {
    struct ibv_context* context;
    struct ibv_pd* pd_orig;
#ifndef GPUIB_BNXT
    struct ibv_pd* pd_parent;
#endif
#ifdef GPUIB_IONIC
    struct ibv_pd* pd_uxdma[2];
#endif
    struct ibv_mr* mr;
    struct ibv_port_attr portinfo;

#ifdef GPUIB_IONIC
    void *gpu_db_page;
    uint64_t *gpu_db_cq;
    uint64_t *gpu_db_sq;
#endif
  } ib_state_t;

  typedef struct dest_info {
    int lid;
    int qpn;
    int psn;
    union ibv_gid gid;
  } dest_info_t;

#ifndef GPUIB_BNXT
  class State {
   public:
    ibv_qp_attr exp_qp_attr{};
    uint64_t exp_attr_mask{};
  };

  class InitQPState : public State {
   public:
    InitQPState() {
      exp_qp_attr.qp_state = IBV_QPS_INIT;
      exp_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      exp_attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;
    }
  };

  class RtrState : public State {
   public:
    RtrState() {
      exp_qp_attr.qp_state = IBV_QPS_RTR;
      exp_qp_attr.path_mtu = IBV_MTU_4096;
      exp_qp_attr.ah_attr.sl = 1;
      exp_qp_attr.max_dest_rd_atomic = GPUIB_MAX_ATOMIC;
      exp_qp_attr.min_rnr_timer = 12;
      exp_attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU;
    }
  };

  class RtsState : public State {
   public:
    RtsState() {
      exp_qp_attr.qp_state = IBV_QPS_RTS;
      exp_qp_attr.timeout = 14;
      exp_qp_attr.retry_cnt = 7;
      exp_qp_attr.rnr_retry = 7;
      exp_qp_attr.max_rd_atomic = GPUIB_MAX_ATOMIC;
      exp_attr_mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    }
  };

  class QPInitAttr {
   public:
    explicit QPInitAttr(ibv_qp_cap cap) {
      attr.cap = cap;
      attr.sq_sig_all = 0;
    }
    ibv_qp_init_attr_ex attr{};
  };
#endif

 public:
  explicit GDADevice(MPI_Comm comm_in);

  ~GDADevice();

  void global_exit(int status);

  void create_team(Team* parent_team, TeamInfo* team_info_wrt_parent, TeamInfo* team_info_wrt_world, int num_pes, int my_pe_in_new_team, MPI_Comm team_comm, rocshmem_team_t* new_team);

  void destroy_team(rocshmem_team_t team);

  void create_ctx(void** ctx);

  __device__ bool create_ctx(rocshmem_ctx_t *ctx);

  void destroy_ctx(Context* ctx);

  __device__ void destroy_ctx(rocshmem_ctx_t* ctx);

  void init_mpi_once(MPI_Comm comm);

  void setup_default_ctx();

  void setup_ctxs();

  void setup_default_host_ctx();

  void setup_team_world();

  void init_collective();

  void init_teams();

  void destroy_teams();

  void heap_memory_rkey();

  void initialize_gpu_qp(QueuePair* qp, int conn_num);

#ifndef GPUIB_BNXT
  InitQPState initqp(uint8_t port);

  RtrState rtr(dest_info_t* dest, uint8_t port);

  RtsState rts(dest_info_t* dest);

  QPInitAttr qpattr(ibv_qp_cap cap);

  void init_qp_status(ibv_qp* qp, uint8_t port);
#endif

  void change_status_rtr(ibv_qp* qp, dest_info_t* dest, uint8_t port);

  void change_status_rts(ibv_qp* qp, dest_info_t* dest);

  void create_qps(uint8_t port, ibv_port_attr* ib_port_att);

#ifdef GPUIB_BNXT
  void init_qp_status(uint8_t port);

  void create_cqs(int ncqs, int cqe);

  void create_qps_impl(int nqps);
#else
  template <typename T>
  void try_to_modify_qp(ibv_qp* qp, T state);

  static void* buf_alloc(ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type);

  static void buf_release(ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type);

  void init_parent_domain_attr(ibv_parent_domain_init_attr* attr);

  ibv_cq* create_cq(ibv_context* context, ibv_pd* pd, int cqe);

  ibv_qp* create_qp(ibv_pd* pd, ibv_context* context, ibv_qp_init_attr_ex* qp_attr, ibv_cq* rcq);
#endif

  void ib_init(ibv_device* ib_dev, uint8_t port);

  void setup_gpu_qps();

  void initialize_context(GPUIBContext *ctx, int context_id);

  HostInterface *host_interface{nullptr};

  char* requested_dev{nullptr};

  ibv_device** dev_list{nullptr};

  ib_state_t* ib_state{nullptr};

  std::vector<dest_info_t> dest_info;

  char *team_pool_bitmask_{nullptr};

  char *team_reduced_bitmask_{nullptr};

  int team_bitmask_size_{-1};

  TeamTracker team_tracker{};

  long *barrier_pSync_pool{nullptr};

  int64_t *barrier_sync{nullptr};

  GPUIBContext *ctx_array{nullptr};

  size_t maximum_num_contexts_{32};

  GPUIBContext *default_ctx_{nullptr};

  GPUIBHostContext *default_host_ctx_{nullptr};

  FreeListProxy<HIPAllocator, GPUIBContext*> ctx_free_list{};

  QueuePair *gpu_qps{nullptr};

  std::vector<ibv_qp*> qps;

  std::vector<ibv_cq*> cqs;

  uint32_t sq_size{1024};

  uint32_t *heap_rkey{nullptr};

  ibv_mr *heap_mr{nullptr};

  int num_pes{0};

  int my_pe{-1};

  MPI_Comm comm{};

  SymmetricHeap heap;

#ifdef GPUIB_BNXT
  union ibv_gid gid;

  uint64_t *host_dpi_ptr;
  uint64_t *gpu_dpi_ptr;

  int cq_buf_offset;    /* Length of a single queue */
  void *cq_buf;         /* Host ptr */
  void *gpu_cq_buf;     /* Device ptr */
  void *cq_umem_handle;

  int sq_buf_offset;    /* Length of QP sq */
  int rq_buf_offset;    /* Length of QP rq */
  int qp_buf_offset;    /* Length of QP buf (sq + rq) */
  void *qp_buf;         /* Host ptr */
  void *gpu_qp_buf;     /* Device ptr */
  void *qp_umem_handle;
#endif
};

/**
 * @brief Global handle used by the device to access the proxy.
 */
extern __constant__ GDADevice* device_proxy;

}  // namespace rocshmem

#endif
