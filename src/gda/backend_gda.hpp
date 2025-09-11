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

#ifndef LIBRARY_SRC_GDA_BACKEND_HPP_
#define LIBRARY_SRC_GDA_BACKEND_HPP_

#include "backend_bc.hpp"
#include "containers/free_list_impl.hpp"
#include "hdp_proxy.hpp" //TODO useless?
#include "memory/hip_allocator.hpp"
#include "context_incl.hpp"
#include "gda_context_proxy.hpp"
#include "queue_pair.hpp"
#include "bootstrap/bootstrap.hpp"
#include "debug_gda.hpp"

namespace rocshmem {

class GDAContext;
class GDAHostContext;
class QueuePair;
class HostInterface;

class GDABackend : public Backend {
 private:
  typedef struct dest_info {
    int lid;
    int qpn;
    int psn;
    union ibv_gid gid;
  } dest_info_t;

  char *requested_dev = nullptr;
  struct ibv_context *context = nullptr;;
  struct ibv_pd *pd_orig = nullptr;

  struct ibv_port_attr portinfo;
  union ibv_gid gid;
  int port = 1;
  int gid_index;

  uint32_t *heap_rkey = nullptr;
  struct ibv_mr *heap_mr = nullptr;

  uint32_t sq_size = 1024;
  uint32_t inline_threshold = 8;
  QueuePair *gpu_qps = nullptr;
  std::vector<ibv_qp*> qps;
  std::vector<ibv_cq*> cqs;
  std::vector<dest_info_t> dest_info;

#ifdef GDA_BNXT
  std::vector<struct bnxt_host_qp> bnxt_qps;
  std::vector<struct bnxt_host_cq> bnxt_cqs;

  struct bnxt_re_dv_db_region_attr db_region_attr;
#else
  struct ibv_pd *pd_parent = nullptr;
#endif

#ifdef GDA_IONIC
  struct ibv_pd *pd_uxdma[2];
  void *gpu_db_page = nullptr;
  uint64_t *gpu_db_cq = nullptr;
  uint64_t *gpu_db_sq = nullptr;
#endif

 /**
   * @brief Common code invoked from the different constructors
   */
  void read_env();

 public:
  friend GDAContext;

  /**
   * @copydoc Backend::Backend(unsigned)
   */
  explicit GDABackend(MPI_Comm comm);
  explicit GDABackend(TcpBootstrap *bootstr);

  /**
   * @copydoc Backend::~Backend()
   */
  virtual ~GDABackend();

  __device__ bool create_ctx(int64_t options, rocshmem_ctx_t *ctx);

  /**
   * @brief Destroy a `rocshmem_ctx_t` context and returns it back to the
   * context free list.
   */
  __device__ void destroy_ctx(rocshmem_ctx_t *ctx);

  /**
   * @copydoc Backend::ctx_create
   */
  void ctx_create(int64_t options, void **ctx) override;

  /**
   * @copydoc Backend::ctx_destroy
   */
  void ctx_destroy(Context *ctx) override;

  /**
   * @brief Abort the application.
   *
   * @param[in] status Exit code.
   *
   * @return void.
   *
   * @note This routine terminates the entire application.
   */
  void global_exit(int status) override;

  /**
   * @copydoc Backend::create_new_team
   */
  void create_new_team(Team *parent_team, TeamInfo *team_info_wrt_parent,
                       TeamInfo *team_info_wrt_world, int num_pes,
                       int my_pe_in_new_team, MPI_Comm team_comm,
                       rocshmem_team_t *new_team) override;

  /**
   * @copydoc Backend::team_destroy(rocshmem_team_t)
   */
  void team_destroy(rocshmem_team_t team) override;

  /**
   * @brief Accessor for work/sync bases
   *
   * @return Vector containing the addresses of the work/sync bases
   */
  char** get_wrk_sync_bases() { return wrk_sync_pool_bases_; } //TODO UNUSED

  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the GDABackend
   */
  std::shared_ptr<HostInterface> host_interface{nullptr};

  /**
   * @brief Scratchpad for the internal barrier algorithms.
   */
  int64_t *barrier_sync{nullptr};

  /**
   * @brief Handle for raw memory for barrier sync
   */
  long *barrier_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for reduce sync
   */
  long *reduce_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for broadcast sync
   */
  long *bcast_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for alltoall sync
   */
  long *alltoall_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for work
   */
  void *pWrk_pool{nullptr};

  /**
   * @brief Handle for raw memory for alltoall
   */
  void *pAta_pool{nullptr};

  /**
   * @brief Handle for raw memory for fence/quiet
  */
  int *fence_pool{nullptr};

 protected:
   /**
   * @copydoc Backend::dump_backend_stats()
   */
  void dump_backend_stats() override;

  /**
   * @copydoc Backend::reset_backend_stats()
   */
  void reset_backend_stats() override;

  /**
   * @brief Allocates uncacheable host memory for the hdp policy.
   *
   * @note Internal data ownership is managed by the proxy
   */
  HdpProxy<HIPHostAllocator> hdp_proxy_{};

  /**
   * @brief Holds a copy of the default context for host functions
   */
  std::unique_ptr<GDAHostContext> default_host_ctx{nullptr};

  /**
   * @brief Allocate and initialize team world.
   */
  void setup_team_world();

  /**
   * @brief Initialize the resources required to support teams
   */
  void setup_teams();

  /**
   * @brief Destruct the resources required to support teams
   */
  void cleanup_teams();

  /**
   * @brief Allocation and initialization of backend contexts.
   */
  void setup_ctxs();
  void cleanup_ctxs();
  void setup_host_ctx();
  void setup_default_ctx();

  /**
   * @brief Allocate and initialize barrier operation addresses on
   * symmetric heap.
   *
   * When this method completes, the barrier_sync member will be available
   * for use.
   */
  void setup_collectives();

  /**
   * @brief Allocate buffer for fence/quiet operation
   */
  void setup_fence_buffer();

  void setup_heap_memory_rkey();
  void cleanup_heap_memory_rkey();

  void initialize_gpu_qp(QueuePair* qp, int conn_num);

  /**
   * @brief Setup InfiniBand Resources
   */
  void setup_ibv();

  /**
   * @brief Cleanup InfiniBand Resources
   */
  void cleanup_ibv();

  /**
   * @brief Open InfiniBand Device and create common structures
   */
  void open_ib_device();

  /**
   * @brief Selects the best GID index
   */
  void select_gid_index();

  /**
   * @brief Create all CQs and QPs
   */
  void create_queues();

  /**
   * @brief Create all CQs with a of length ncqes
   */
  void create_cqs(int ncqes);

  /**
   * @brief Create all QPs with a SQ of length sq_length
   */
  void create_qps(int sq_length);

  /**
   * @brief Exchange QP information for connection
   */
  void exchange_qp_dest_info();

  /**
   * @brief Modify all QPs from RESET to INIT state
   */
  void modify_qps_reset_to_init();

  /**
   * @brief Modify all QPs from INIT to RTR state
   */
  void modify_qps_init_to_rtr();

  /**
   * @brief Modify all QPs from RTR to RTs state
   */
  void modify_qps_rtr_to_rts();

  /**
   * @brief Converts an ibv_mtu to an integer
   */
  int ibv_mtu_to_int(enum ibv_mtu mtu);

#ifndef GDA_BNXT
  static void* pd_alloc(ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type);

  static void pd_release(ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type);

  void create_parent_domain();
#endif

  void setup_gpu_qps();
  void cleanup_gpu_qps();

 private:
  /**
   * @brief Common code invoked from the different constructors
   */
  void init();

  /**
   * @brief Proxy for the default context
   *
   * @note Internal data ownership is managed by the proxy
   */
  GDADefaultContextProxyT default_context_proxy_;  // init handled in constructor

  /**
   * @brief An array of @ref ROContexts that backs the context FreeList.
   */
  GDAContext *ctx_array{nullptr};

  /**
   * @brief A free-list containing contexts.
   */
  FreeListProxy<HIPAllocator, GDAContext *> ctx_free_list{};

  /**
   * @brief Holds maximum number of contexts used in library
   */
  size_t maximum_num_contexts_{32};

  /**
   * @brief The bitmask representing the availability of teams in the pool
   */
  char *team_pool_bitmask_{nullptr};

  /**
   * @brief Bitmask to store the reduced result of bitmasks on pariticipating
   * PEs
   *
   * With no thread-safety for this bitmask, multithreaded creation of teams is
   * not supported.
   */
  char *team_reduced_bitmask_{nullptr};

  /**
   * @brief Size of the bitmask
   */
  int team_bitmask_size_{-1};

  /**
   * Fine grained memory allocator for buffers used in collectives Routines
   */
  HIPDefaultFinegrainedAllocator fine_grained_allocator_ {};

  /**
   * @brief Collective routines work/sync buffer size
   */
  size_t wrk_sync_pool_size_{};

  /**
   * @brief Collective routines work/sync buffer base ptr
   */
  char* const wrk_sync_pool_{nullptr};

  /**
   * @brief Temporary buffer pointer pointing to the same address as
   * wrk_sync_pool_, used to calculate the starting addresses of
   * different work and sync buffers.
  */
  char *wrk_sync_pool_top_{nullptr};

  /**
   * @brief Array containing the addresses of the work/sync buffer bases
   * of other PEs
  */
  char** wrk_sync_pool_bases_{nullptr};//TODO UNUSED, maybe used again later when we decouple the sync from the main heap

  /**
   * @brief Initialize memory required for work/sync buffers and open GDA
   * handle on PE's wrk_sync_pool.
   */
  void setup_wrk_sync_buffer();

  /**
   * @brief Close GDA memory handles for work/sync buffers and deallocate
   * work/sync buffer.
  */
  void cleanup_wrk_sync_buffer();

  /**
   * @brief rte all-to-all
   */
  void Alltoall_char_inplace (char *inoutbuf, size_t num_bytes, rocshmem_team_t team);

  /**
   * @brief rte allreduce for teams
   */
  void Allreduce_char_BAND (char* inbuf, char *outbuf, size_t num_bytes, Team *team);

  /**
   * @brief rte barrier for initialization
   */
  void rte_barrier();
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_BACKEND_HPP_
