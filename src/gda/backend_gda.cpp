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

#include <cstring>

#include "backend_gda.hpp"
#include "gda_team.hpp"
#include "util.hpp"
#include "topology.hpp"

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cassert>

namespace rocshmem {

#define NET_CHECK(cmd) {                                     \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort();                                               \
    }                                                        \
  }

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

rocshmem_team_t get_external_team(GDATeam *team) {
  return reinterpret_cast<rocshmem_team_t>(team);
}

int get_ls_non_zero_bit(char *bitmask, int mask_length) {
  int position{-1};
  for (int bit_i = 0; bit_i < mask_length; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    if (bitmask[byte_i] & (1 << (bit_i % CHAR_BIT))) {
      position = bit_i;
      break;
    }
  }

  return position;
}

GDABackend::GDABackend(MPI_Comm comm):  Backend(comm) {
  init();
}

GDABackend::GDABackend(TcpBootstrap *bootstrap):  Backend(bootstrap) {
  init();
}

void GDABackend::init() {
  type = BackendType::GDA_BACKEND;
  int ret;

#if defined(GDA_BNXT)
  ret = bnxt_dv_dl_init();
  if (ret != ROCSHMEM_SUCCESS) {
    // Disable BNXT GDA support.
    DPRINTF("Initializing rocSHMEM BNXT GDA support failed\n");
    // We abort for now, but might remove that once we support
    // multiple NIC types in the same build
    abort();
  }
#endif
#if defined(GDA_MLX5)
  ret = mlx5_dv_dl_init();
  if (ret != ROCSHMEM_SUCCESS) {
    // Disable MLX5 GDA support.
    DPRINTF("Initializing rocSHMEM MLX5 GDA support failed\n");
    // We abort for now, but might remove that once we support
    // multiple NIC types in the same build
    abort();
  }
#endif

  read_env();

  //TODO setup_host_interface();
  /* Initialize the host interface */
  if (MPI_COMM_NULL != backend_comm)
    host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(), //TODO: need an hdp proxy?
                                                     backend_comm,
                                                     &heap);
  else
    host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(), //TODO: need an hdp proxy?
                                                     backend_bootstr,
                                                     &heap);

  setup_wrk_sync_buffer();
  setup_fence_buffer();
  setup_collectives();

  setup_teams();
  setup_team_world();
  rte_barrier();

  setup_ibv();
  setup_heap_memory_rkey();
  setup_gpu_qps();

  setup_ctxs();
  rte_barrier();
}

GDABackend::~GDABackend() {
  cleanup_ctxs();

  cleanup_teams();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));

  cleanup_wrk_sync_buffer();

  cleanup_gpu_qps();
  cleanup_heap_memory_rkey();
  cleanup_ibv();

#if defined(GDA_BNXT)
  if (bnxtdv_handle_ != nullptr)
    dlclose(bnxtdv_handle_);
#endif
#if defined(GDA_MLX5)
  if (mlx5dv_handle_ != nullptr)
    dlclose(mlx5dv_handle_);
#endif
}

void GDABackend::read_env() {
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
}


void GDABackend::setup_host_ctx() {
  default_host_ctx = std::make_unique<GDAHostContext>(this, 0);
  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();
}

void GDABackend::setup_default_ctx() {
  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;
  default_context_proxy_ = GDADefaultContextProxyT(this, tinfo);
}

void GDABackend::setup_ctxs() {
  setup_host_ctx();
  setup_default_ctx();

  CHECK_HIP(hipMalloc(&ctx_array, sizeof(GDAContext) * maximum_num_contexts_));
  // 0th context is default context
  for (size_t i = 0; i < maximum_num_contexts_; i++) {
    new (&ctx_array[i]) GDAContext(this, i + 1);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

void GDABackend::cleanup_ctxs() {
  ctx_free_list.~FreeListProxy();
  for (size_t i = 0; i < maximum_num_contexts_; i++) {
    ctx_array[i].~GDAContext();
  }

  CHECK_HIP(hipFree(ctx_array));
}

__device__ bool GDABackend::create_ctx(int64_t options, rocshmem_ctx_t *ctx) {
  GDAContext *ctx_{nullptr};

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;

  ctx_->tinfo = reinterpret_cast<TeamInfo *>(ctx->team_opaque);
  return true;
}

__device__ void GDABackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<GDAContext *>(ctx->ctx_opaque));
}

void GDABackend::setup_team_world() {
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  /**
   * Allocate device-side memory for team_world and construct a
   * GDA team in it.
   */
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));

  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

  GDATeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(GDATeam)));
  new (team_world) GDATeam(this, team_info_wrt_parent, team_info_wrt_world,
                           num_pes, my_pe, backend_comm, 0);
  team_tracker.set_team_world(team_world);

  /**
   * Copy the address to ROCSHMEM_TEAM_WORLD.
   */
  ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
}

void GDABackend::team_destroy(rocshmem_team_t team) {
  GDATeam *team_obj = get_internal_gda_team(team);

  /* Mark the pool as available */
  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  team_pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);

  team_obj->~GDATeam();
  CHECK_HIP(hipFree(team_obj));
}

//TODO: factorize somewhere else maybe backend_bc
void GDABackend::Alltoall_char_inplace (char *inoutbuf, size_t num_bytes, rocshmem_team_t team) {
  // Implement an Alltoall outside of MPI assuming in_place communication
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int num_pes = team_obj->num_pes;
  int my_pe = team_obj->my_pe;
  int *pes_in_world = new int[num_pes];

  int my_pe_in_world = team_obj->my_pe_in_world;
  for (int i = 0; i < num_pes; i++) {
      pes_in_world[i] = team_obj->get_pe_in_world(i);
  }

  // Since this is an in-place algorithm, allocate the temporary receive buffer first
  char *recv_buf = new char[num_bytes * num_pes];
  std::memset(recv_buf, 0, num_pes * num_bytes);

  // Perform pairwise exchange - local copy is ommitted
  for (int step = 1; step < num_pes; step++) {
    int sendto_team  = (my_pe + step) % num_pes;
    int recvfrom_team = (my_pe + num_pes - step) % num_pes;

    char *tmpsend = (char*)inoutbuf + (ptrdiff_t)sendto_team * num_bytes;
    char *tmprecv = (char*)recv_buf + (ptrdiff_t)recvfrom_team * num_bytes;

    // similarly to the allGather in the bootstrap code, we do send first
    // followed by the receive.
    // There is a chance for deadlock in my opinion for large messages.
    backend_bootstr->send(tmpsend, num_bytes, pes_in_world[sendto_team], step /* used as tag */);
    backend_bootstr->recv(tmprecv, num_bytes, pes_in_world[recvfrom_team], step);
  }
  //Since this is an in_place all-to-all, copy data back into the user buffer
  for (int step = 0; step < num_pes; step++) {
    if (step == my_pe) continue;
    std::memcpy(&inoutbuf[step*num_bytes], &recv_buf[step*num_bytes], num_bytes);
  }

  delete[] recv_buf;
  delete[] pes_in_world;
}

//TODO: factorize somewhere else, maybe backend_bc?
void GDABackend::Allreduce_char_BAND (char* inbuf, char *outbuf, size_t num_bytes,
                                      Team *team) {

  // Implement an Allreduce outside of MPI. This is specialized for the scenario
  // required for the team creation, i.e. assuming bytes and using BAND operation.
  // Implementation uses an Allgather operation followed a local reduction.

  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int num_pes = team_obj->num_pes;
  int my_pe = team_obj->my_pe;

  char *tmp_buffer = new char[num_pes * num_bytes];
  std::memset(tmp_buffer, 0, num_pes * num_bytes);
  std::memcpy (&tmp_buffer[my_pe * num_bytes], inbuf, num_bytes);

  if (num_pes == backend_bootstr->getNranks() ) {
    backend_bootstr->allGather(tmp_buffer, num_bytes);
  } else {
    printf("GDABackend::create_new_team: non-mpi version only supports parent_teams that contain all processes. Aborting.\n");
    abort();
  }

  for (int i = 0; i < num_bytes; i++) {
    outbuf[i] = tmp_buffer[i];
    for (int j = 1; j < num_pes; j++) {
      outbuf[i] &= tmp_buffer[j * num_bytes + i];
    }
  }

  delete[] tmp_buffer;
}

void GDABackend::create_new_team([[maybe_unused]] Team *parent_team,
                                TeamInfo *team_info_wrt_parent,
                                TeamInfo *team_info_wrt_world, int num_pes,
                                int my_pe_in_new_team, MPI_Comm team_comm,
                                rocshmem_team_t *new_team) {
  /**
   * Read the bit mask and find out a common index into
   * the pool of available work arrays.
   */
  if (team_comm != MPI_COMM_NULL) {
    NET_CHECK(MPI_Allreduce(team_pool_bitmask_, team_reduced_bitmask_, team_bitmask_size_,
                            MPI_CHAR, MPI_BAND, team_comm));
  } else {
    Allreduce_char_BAND (team_pool_bitmask_, team_reduced_bitmask_, team_bitmask_size_, parent_team);
  }

  /* Pick the least significant non-zero bit (logical layout) in the reduced
   * bitmask */
  auto max_num_teams{team_tracker.get_max_num_teams()};
  int common_index = get_ls_non_zero_bit(team_reduced_bitmask_, max_num_teams);
  if (common_index < 0) {
    /* No team available */
    printf("Could not create team, all bits in use. Aborting.\n");
    abort();
  }

  /* Mark the team as taken (by unsetting the bit in the pool bitmask) */
  int byte = common_index / CHAR_BIT;
  team_pool_bitmask_[byte] &= ~(1 << (common_index % CHAR_BIT));

  /**
   * Allocate device-side memory for team_world and
   * construct a GDA team in it
   */
  GDATeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(GDATeam)));
  new (new_team_obj)
      GDATeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes,
                my_pe_in_new_team, team_comm, common_index);

  *new_team = get_external_team(new_team_obj);
}

void GDABackend::ctx_create(int64_t options, void **ctx) {
  GDAHostContext *new_ctx{nullptr};
  new_ctx = new GDAHostContext(this, options);
  *ctx = new_ctx;
}

GDAHostContext *get_internal_gda_net_ctx(Context *ctx) {
  return reinterpret_cast<GDAHostContext *>(ctx);
}

void GDABackend::ctx_destroy(Context *ctx) {
  GDAHostContext *gda_host_ctx{get_internal_gda_net_ctx(ctx)};
  delete gda_host_ctx;
}

void GDABackend::reset_backend_stats() {
  assert(false);
}

void GDABackend::dump_backend_stats() {
  assert(false);
}

__host__ void GDABackend::global_exit(int status) {
  if (backend_comm != MPI_COMM_NULL)
    MPI_Abort(backend_comm, status);
  else
    abort();
}

void GDABackend::cleanup_teams() {
  free(team_pool_bitmask_);
  free(team_reduced_bitmask_);
}

void GDABackend::setup_wrk_sync_buffer() {
  /**
   * compute work/sync buffer size
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};

  /**
   * size of barrier sync
   */
  wrk_sync_pool_size_ += sizeof(*barrier_sync) * ROCSHMEM_BARRIER_SYNC_SIZE;

  /**
   * Size of sync arrays for the teams
  */
  wrk_sync_pool_size_ += sizeof(long) * max_num_teams *
                           (ROCSHMEM_BARRIER_SYNC_SIZE +
                            ROCSHMEM_REDUCE_SYNC_SIZE +
                            ROCSHMEM_BCAST_SYNC_SIZE +
                            ROCSHMEM_ALLTOALL_SYNC_SIZE);

  /**
   * Size of work arrays for the teams
   * Accommodate largest possible data type for pWrk
  */
  wrk_sync_pool_size_ += sizeof(double) * max_num_teams *
                           ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;

  /**
   * Size of fence array
   */
  wrk_sync_pool_size_ += sizeof(int) * num_pes; //TODO: do we need a fence array?

  /**
   * Allocate a buffer of size wrk_sync_pool_size_, using heap memory
   * (should be uncached fine-grained ideally)
  */
  heap.malloc((void**)&wrk_sync_pool_, wrk_sync_pool_size_);
  assert(wrk_sync_pool_);
  wrk_sync_pool_top_ = wrk_sync_pool_;
}

void GDABackend::cleanup_wrk_sync_buffer() {
  heap.free(wrk_sync_pool_);
}

void GDABackend::setup_fence_buffer() { //TODO is this used?
  /*
   * Reserve memory for fence
   */
  fence_pool = reinterpret_cast<int *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(int) * num_pes;
}

void GDABackend::setup_collectives() {
  /*
   * Allocate heap space for barrier_sync
   */
  size_t one_sync_size_bytes {sizeof(*barrier_sync)};
  size_t sync_size_bytes {one_sync_size_bytes * ROCSHMEM_BARRIER_SYNC_SIZE};

  barrier_sync = reinterpret_cast<int64_t*>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sync_size_bytes;

  /*
   * Initialize the barrier synchronization array with default values.
   */
  for (int i = 0; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
    barrier_sync[i] = ROCSHMEM_SYNC_VALUE;
  }

  /*
   * Make sure that all processing elements have done this before
   * continuing.
   */
  rte_barrier();
}

void GDABackend::setup_teams() {
  /**
   * Allocate pools for the teams sync and work arrary from the SHEAP.
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};

  barrier_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_BARRIER_SYNC_SIZE
                            * max_num_teams;

  reduce_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_REDUCE_SYNC_SIZE
                            * max_num_teams;

  bcast_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_BCAST_SYNC_SIZE
                            * max_num_teams;

  alltoall_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_BCAST_SYNC_SIZE
                            * max_num_teams;

  /* Accommodating for largest possible data type for pWrk */
  pWrk_pool = reinterpret_cast<void *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(double) * ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE
                            * max_num_teams;

  /**
   * Initialize the sync arrays in the pool with default values.
   */
  long *barrier_pSync, *reduce_pSync, *bcast_pSync, *alltoall_pSync;
  for (int team_i = 0; team_i < max_num_teams; team_i++) {
    barrier_pSync = reinterpret_cast<long *>(
        &barrier_pSync_pool[team_i * ROCSHMEM_BARRIER_SYNC_SIZE]);
    reduce_pSync = reinterpret_cast<long *>(
        &reduce_pSync_pool[team_i * ROCSHMEM_REDUCE_SYNC_SIZE]);
    bcast_pSync = reinterpret_cast<long *>(
        &bcast_pSync_pool[team_i * ROCSHMEM_BCAST_SYNC_SIZE]);
    alltoall_pSync = reinterpret_cast<long *>(
        &alltoall_pSync_pool[team_i * ROCSHMEM_ALLTOALL_SYNC_SIZE]);

    for (size_t i = 0; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
      barrier_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i++) {
      reduce_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_BCAST_SYNC_SIZE; i++) {
      bcast_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_ALLTOALL_SYNC_SIZE; i++) {
      alltoall_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
  }

  /**
   * Initialize bit mask
   *
   * Logical:
   * MSB..........................................................................LSB
   * Physical: MSB...1st least significant 8 bits...LSB  MSB...2nd least
   * signifant 8 bits...LSB
   *
   * Description shows only a 2-byte long mask but idea extends to any
   * arbitrary size.
   */
  team_bitmask_size_ = (max_num_teams % CHAR_BIT) ? (max_num_teams / CHAR_BIT + 1)
                                             : (max_num_teams / CHAR_BIT);
  team_pool_bitmask_ = reinterpret_cast<char *>(malloc(team_bitmask_size_));
  team_reduced_bitmask_ = reinterpret_cast<char *>(malloc(team_bitmask_size_));

  memset(team_pool_bitmask_, 0, team_bitmask_size_);
  memset(team_reduced_bitmask_, 0, team_bitmask_size_);
  /* Set all to available except the 0th one (reserved for TEAM_WORLD) */
  for (int bit_i = 1; bit_i < max_num_teams; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    team_pool_bitmask_[byte_i] |= 1 << (bit_i % CHAR_BIT);
  }

  /**
   * Make sure that all processing elements have done this before
   * continuing.
   */
  rte_barrier();
}

void GDABackend::rte_barrier() {
  if (backend_comm != MPI_COMM_NULL) {
    NET_CHECK(MPI_Barrier(backend_comm));
  } else {
    backend_bootstr->barrier();
  }
}

#if defined(GDA_MLX5)
int GDABackend::mlx5_dv_dl_init () {
  mlx5dv_handle_ = dlopen("libmlx5.so", RTLD_NOW);
  if (!mlx5dv_handle_) {
    printf("Could not open libmlx5.so. Returning\n");
    return ROCSHMEM_ERROR;
  }

  DLSYM_HELPER(mlx5dv_ftable_, mlx5dv_, mlx5dv_handle_, init_obj);
  return ROCSHMEM_SUCCESS;
}
#endif

void GDABackend::setup_ibv() {
  open_ib_device();

  create_queues();

  exchange_qp_dest_info();

  modify_qps_reset_to_init();
  modify_qps_init_to_rtr();
  modify_qps_rtr_to_rts();

  rte_barrier();
}

void GDABackend::cleanup_ibv() {
  int err;

#ifdef GDA_BNXT
  CHECK_HIP(hipHostUnregister(db_region_attr.dbr));

  for (int i = 0; i < qps.size(); i++) {
    err = bnxtdv_ftable_.destroy_qp(qps[i]);
    CHECK_ZERO(err, "bnxt_re_dv_destroy_qp");

    err = bnxtdv_ftable_.umem_dereg(bnxt_qps[i].attr.rq_umem_handle);
    CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (RQ)");

    err = bnxtdv_ftable_.umem_dereg(bnxt_qps[i].attr.sq_umem_handle);
    CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (SQ)");

    CHECK_HIP(hipFree(bnxt_qps[i].sq_buf));
    CHECK_HIP(hipFree(bnxt_qps[i].rq_buf));

    err = bnxtdv_ftable_.destroy_cq(cqs[i]);
    CHECK_ZERO(err, "bnxt_re_dv_destroy_cq");

    err = bnxtdv_ftable_.umem_dereg(bnxt_cqs[i].umem_handle);
    CHECK_ZERO(err, "bnxt_re_dv_umem_dereg");

    CHECK_HIP(hipFree(bnxt_cqs[i].buf));
  }
#else
  for (int i = 0; i < qps.size(); i++) {
    err = ibv_destroy_qp(qps[i]);
    CHECK_ZERO(err, "ibv_destroy_qp");

    err = ibv_destroy_cq(cqs[i]);
    CHECK_ZERO(err, "ibv_destroy_cqs");
  }

#ifdef GDA_IONIC
  err = ibv_dealloc_pd(pd_uxdma[0]);
  CHECK_ZERO(err, "ibv_dealloc_pd (uxdma[0])");

  err = ibv_dealloc_pd(pd_uxdma[1]);
  CHECK_ZERO(err, "ibv_dealloc_pd (uxdma[1])");
#endif

  err = ibv_dealloc_pd(pd_parent);
  CHECK_ZERO(err, "ibv_dealloc_pd (pd_parent)");
#endif

  err = ibv_dealloc_pd(pd_orig);
  CHECK_ZERO(err, "ibv_dealloc_pd (pd_orig)");

  err = ibv_close_device(context);
  CHECK_ZERO(err, "ibv_close_device");
}

void GDABackend::exchange_qp_dest_info() {
  for (int i = 0; i < qps.size(); i++) {
    dest_info[i].lid = portinfo.lid;
    dest_info[i].qpn = qps[i]->qp_num;
    dest_info[i].psn = 0;
    dest_info[i].gid = gid;
  }

  for (int i = 0; i < maximum_num_contexts_ + 1; i++) {
    if (backend_comm != MPI_COMM_NULL) {
      MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t), MPI_CHAR, dest_info.data() + i * num_pes, sizeof(dest_info_t), MPI_CHAR, backend_comm);
    } else {
      Alltoall_char_inplace(reinterpret_cast<char*>(dest_info.data() + i * num_pes), sizeof(dest_info_t), ROCSHMEM_TEAM_WORLD);
    }
  }
}

void GDABackend::setup_heap_memory_rkey() {
  auto *base_heap = heap.get_local_heap_base();
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  heap_mr = ibv_reg_mr(pd_orig, base_heap, heap.get_size(), access);
  CHECK_NNULL(heap_mr, "ibv_reg_mr");

  const size_t rkeys_size = sizeof(uint32_t) * num_pes;
  uint32_t *host_rkey_cpy = reinterpret_cast<uint32_t*>(malloc(rkeys_size));
  if (!host_rkey_cpy) { abort(); }

  CHECK_HIP(hipHostMalloc(&heap_rkey, sizeof(uint32_t) * num_pes));
  heap_rkey[my_pe] = heap_mr->rkey;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(host_rkey_cpy, heap_rkey, rkeys_size, hipMemcpyDeviceToHost, stream));
  CHECK_HIP(hipStreamSynchronize(stream));

  if (backend_comm != MPI_COMM_NULL)
    MPI_Allgather(MPI_IN_PLACE, sizeof(uint32_t), MPI_CHAR, host_rkey_cpy, sizeof(uint32_t), MPI_CHAR, backend_comm);
  else
    backend_bootstr->allGather(host_rkey_cpy, sizeof(uint32_t));

  CHECK_HIP(hipMemcpyAsync(heap_rkey, host_rkey_cpy, rkeys_size, hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));

  free(host_rkey_cpy);
}

void GDABackend::cleanup_heap_memory_rkey() {
  int ret = ibv_dereg_mr(heap_mr);
  CHECK_ZERO(ret, "ibv_dereg_mr");

  CHECK_HIP(hipHostFree(heap_rkey));
}

void GDABackend::setup_gpu_qps() {
  CHECK_HIP(hipMalloc(&gpu_qps, sizeof(QueuePair) * (maximum_num_contexts_ + 1) * num_pes));
  for (int i = 0; i < (maximum_num_contexts_ + 1) * num_pes; i++) {
    QueuePair qp(pd_orig);
    CHECK_HIP(hipMemcpy(&gpu_qps[i], &qp, sizeof(QueuePair), hipMemcpyDefault));
    initialize_gpu_qp(&gpu_qps[i], i);
  }
}

void GDABackend::cleanup_gpu_qps() {
  //TODO need to destruct qp[i]?
  CHECK_HIP(hipFree(gpu_qps));
  gpu_qps = nullptr;
}

//TODO this ifdef sequence should go in a nic-specific file, like it is for bnxt, maybe whats above too?
void GDABackend::open_ib_device() {
  struct ibv_device **device_list = nullptr;
  struct ibv_device *device = nullptr;
  int num_devices = 0;
  int err;

  device_list = ibv_get_device_list(&num_devices);
  CHECK_NNULL(device_list, "ibv_get_device_list");

  device = device_list[0]; //TODO default to HIP selected device?

  if (requested_dev) {
    for (int i = 0; i < num_devices; i++) {
      const char *select_device = ibv_get_device_name(device_list[i]);
      CHECK_NNULL(select_device, "ibv_get_device_name");

      if (strstr(select_device, requested_dev)) {
        device = device_list[i];
        break;
      }
    }
  }

  context = ibv_open_device(device);
  CHECK_NNULL(context, "ib open device");
  dump_ibv_context(context);
  dump_ibv_device(context->device);

  pd_orig = ibv_alloc_pd(context);
  CHECK_NNULL(pd_orig, "ib allocate pd");
  dump_ibv_pd(pd_orig);

#ifndef GDA_BNXT
  create_parent_domain();
#endif

  err = ibv_query_port(context, port, &portinfo);
  CHECK_ZERO(err, "ibv_query_port");
  dump_ibv_port_attr(&portinfo);

  /* Must init after querying port */
  select_gid_index();

  ibv_free_device_list(device_list);
}

void GDABackend::modify_qps_reset_to_init() {
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
#ifdef GDA_BNXT
    err = bnxtdv_ftable_.modify_qp(qps[i], &attr, attr_mask, 0, 0);
#else
    err = ibv_modify_qp(qps[i], &attr, attr_mask);
#endif
    CHECK_ZERO(err, "modify_qp (INIT)");
  }
}

void GDABackend::modify_qps_init_to_rtr() {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state               = IBV_QPS_RTR;
  attr.path_mtu               = portinfo.active_mtu;
  attr.max_dest_rd_atomic     = GDA_MAX_ATOMIC;
  attr.min_rnr_timer          = 12;
  attr.ah_attr.port_num       = port;

  if (portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.is_global      = 1;
    attr.ah_attr.grh.hop_limit  = 1;
    attr.ah_attr.sl             = 1;
  }

  attr_mask = IBV_QP_STATE
            | IBV_QP_PATH_MTU
            | IBV_QP_RQ_PSN
            | IBV_QP_DEST_QPN
            | IBV_QP_AV
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;

  for (int i = 0; i < qps.size(); i++) {
    attr.rq_psn      = dest_info[i].psn;
    attr.dest_qp_num = dest_info[i].qpn;

    if (portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      memcpy(&attr.ah_attr.grh.dgid, &dest_info[i].gid, 16);
    } else {
      attr.ah_attr.dlid = dest_info[i].lid;
    }

#ifdef GDA_BNXT
    err = bnxtdv_ftable_.modify_qp(qps[i], &attr, attr_mask, 0, 0);
#else
    err = ibv_modify_qp(qps[i], &attr, attr_mask);
#endif
    CHECK_ZERO(err, "modify_qp (RTR)");
  }
}

void GDABackend::modify_qps_rtr_to_rts() {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state      = IBV_QPS_RTS;
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

  for (int i = 0; i < qps.size(); i++) {
    attr.sq_psn = dest_info[i].psn;

#ifdef GDA_BNXT
    err = bnxtdv_ftable_.modify_qp(qps[i], &attr, attr_mask, 0, 0);
#else
    err = ibv_modify_qp(qps[i], &attr, attr_mask);
#endif
    CHECK_ZERO(err, "modify_qp (RTS)");
  }
}

void GDABackend::create_queues() {
  int ncqes;
  int resize_length;

#ifdef GDA_IONIC
  ncqes = sq_size << 1;
#else
  ncqes = sq_size;
#endif

  resize_length = (maximum_num_contexts_ + 1) * num_pes;

  dest_info.resize(resize_length);
  cqs.resize(resize_length);
  qps.resize(resize_length);

#ifdef GDA_BNXT
  bnxt_cqs.resize(resize_length);
  bnxt_qps.resize(resize_length);
#endif

  create_cqs(ncqes);
  create_qps(sq_size);
}

#ifndef GDA_BNXT
void* GDABackend::pd_alloc(struct ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type) {
  void* dev_ptr{nullptr};
  //TODO make this configurable, presumably we want it on device for all types?
#ifdef GDA_IONIC
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
#else
  CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&dev_ptr), size, hipHostMallocDefault));
#endif
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void GDABackend::pd_release(struct ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type) {
  CHECK_HIP(hipFree(ptr));
}

void GDABackend::create_parent_domain() {
  struct ibv_parent_domain_init_attr pattr;

  memset(&pattr, 0, sizeof(struct ibv_parent_domain_init_attr));
  pattr.pd         = pd_orig,
  pattr.td         = nullptr,
  pattr.comp_mask  = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS,
  pattr.alloc      = GDABackend::pd_alloc,
  pattr.free       = GDABackend::pd_release,
  pattr.pd_context = nullptr,

  pd_parent = ibv_alloc_parent_domain(context, &pattr);
  CHECK_NNULL(pd_parent, "ibv_alloc_parent_domain");
  dump_ibv_pd(pd_parent);

#ifdef GDA_IONIC
  ionic_dv_pd_set_sqcmb(pd_parent, false, false, false);
  ionic_dv_pd_set_rqcmb(pd_parent, false, false, false);

  for (int uxdma_i = 0; uxdma_i < 2; ++uxdma_i) {
    pd_uxdma[uxdma_i] = ibv_alloc_parent_domain(context, &pattr);
    CHECK_NNULL(pd_uxdma[uxdma_i], "ibv_alloc_parent_domain (uxdma)");

    ionic_dv_pd_set_sqcmb(pd_uxdma[uxdma_i], false, false, false);
    ionic_dv_pd_set_rqcmb(pd_uxdma[uxdma_i], false, false, false);
    ionic_dv_pd_set_udma_mask(pd_uxdma[uxdma_i], 1u << uxdma_i);
  }
#endif
}

void GDABackend::create_cqs(int cqe) {
  struct ibv_cq_init_attr_ex cq_attr;
  struct ibv_cq_ex *cq_ex;

  memset(&cq_attr, 0, sizeof(struct ibv_cq_init_attr_ex));
  cq_attr.cqe           = cqe;
  cq_attr.cq_context    = nullptr;
  cq_attr.channel       = nullptr;
  cq_attr.comp_vector   = 0;
  cq_attr.flags         = 0;
  cq_attr.comp_mask     = IBV_CQ_INIT_ATTR_MASK_PD;
  cq_attr.parent_domain = pd_parent;

  for (int i = 0; i < qps.size(); i++) {
#ifdef GDA_IONIC
    cq_attr.parent_domain = pd_uxdma[((i + 1) / 2) & 1];
#endif

    cq_ex = ibv_create_cq_ex(context, &cq_attr);
    CHECK_NNULL(cq_ex, "ibv_create_cq_ex");

    cqs[i] = ibv_cq_ex_to_cq(cq_ex);
    CHECK_NNULL(cqs[i], "ibv_cq_ex_to_cq");
  }
}

void GDABackend::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));

#ifdef GDA_IONIC
  ionic_dv_ctx dvctx;
  ionic_dv_get_ctx(&dvctx, context);

  void* gpu_db_page = nullptr;
  rocm_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id);

  uint64_t *db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
  uint64_t *gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);

  uint64_t *gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

  gpu_db_page = gpu_db_page;
  gpu_db_cq = &gpu_db_ptr[dvctx.cq_qtype];
  gpu_db_sq = &gpu_db_ptr[dvctx.sq_qtype];

  uint8_t udma_idx = ionic_dv_qp_get_udma_idx(qps[conn_num]);

  ionic_dv_cq dvcq;
  ionic_dv_get_cq(&dvcq, cqs[conn_num], udma_idx);

  gpu_qp->cq_dbreg = gpu_db_cq;
  gpu_qp->cq_dbval = dvcq.q.db_val;
  gpu_qp->cq_mask = dvcq.q.mask;

  gpu_qp->cq_buf = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);

  ionic_dv_qp dvqp;
  ionic_dv_get_qp(&dvqp, qps[conn_num]);

  gpu_qp->sq_dbreg = gpu_db_sq;
  gpu_qp->sq_dbval = dvqp.sq.db_val;
  gpu_qp->sq_mask = dvqp.sq.mask;
  gpu_qp->sq_buf = reinterpret_cast<ionic_v1_wqe *>(dvqp.sq.ptr);

  gpu_qp->qp_num = qps[conn_num]->qp_num;
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey = heap_rkey[conn_num % num_pes];
  gpu_qp->inline_threshold = 32;
#else // !GDA_IONIC
  mlx5dv_cq cq_out;
  mlx5dv_obj mlx_obj;
  mlx_obj.cq.in = cqs[conn_num];
  mlx_obj.cq.out = &cq_out;
  mlx5dv_ftable_.init_obj(&mlx_obj, MLX5DV_OBJ_CQ);
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
  mlx5dv_ftable_.init_obj(&mlx_obj, MLX5DV_OBJ_QP);
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
  gpu_qp->inline_threshold = inline_threshold;
  // The 2 in qp_out.bf.size * 2 below facilitates the switching between blue flame registers
  void* gpu_ptr{nullptr};
  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr, hip_dev_id);
  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);
#endif // !GDA_IONIC
}

void GDABackend::create_qps(int sq_length) {
  struct ibv_qp_init_attr_ex attr;

  memset(&attr, 0, sizeof(struct ibv_qp_init_attr_ex));
  attr.cap.max_send_wr     = sq_length;
  attr.cap.max_send_sge    = 1;
  attr.cap.max_inline_data = inline_threshold;
#ifdef GDA_IONIC
  attr.cap.max_recv_sge    = 1; // TODO allow zero sges in the driver
#endif
  attr.sq_sig_all          = 0;
  attr.qp_type             = IBV_QPT_RC;
  attr.comp_mask           = IBV_QP_INIT_ATTR_PD;
  attr.pd                  = pd_parent;

  for (int i = 0; i < qps.size(); i++) {
#ifdef GDA_IONIC
    attr.pd      = pd_uxdma[((i + 1) / 2) & 1];
#endif
    attr.send_cq = cqs[i];
    attr.recv_cq = cqs[i];

    qps[i] = ibv_create_qp_ex(context, &attr);
    CHECK_NNULL(qps[i], "ibv_create_qp_ex");
  }
}
#endif

void GDABackend::select_gid_index() {
  struct ibv_gid_entry *gid_entries;
  struct ibv_gid_entry *gid_entry;
  union ibv_gid current_gid;
  union ibv_gid selected_gid;
  uint32_t gid_type;
  int err;

  const uint8_t local_gid_prefix[2] = {0xFE, 0x80};
  uint32_t selected_gid_type        = IBV_GID_TYPE_ROCE_V1;
  int selected_gid_index            = -1;
  ssize_t gid_tbl_entries           = 0;

  int gid_tbl_len         = portinfo.gid_tbl_len;

  gid_entries = (struct ibv_gid_entry*) calloc(gid_tbl_len, sizeof(struct ibv_gid_entry));

  gid_tbl_entries = ibv_query_gid_table(context, gid_entries, gid_tbl_len, 0);
  if (gid_tbl_entries < 0) {
    fprintf(stderr, "[Warning] ibv_query_gid_table failed. No available GIDs\n");
    free(gid_entries);
    return;
  }

  for (int i = 0; i < gid_tbl_entries; i++) {
    gid_type = gid_entries[i].gid_type;

    /* rocSHMEM does not use GIDs for IB mode */
    if (gid_type == IBV_GID_TYPE_IB) {
      break;
    }

    current_gid = gid_entries[i].gid;

    err = ibv_query_gid(context, port, i, &current_gid);
    CHECK_ZERO(err, "ibv_query_gid");

    /* We don't want local GIDs */
    if (memcmp(current_gid.raw, &local_gid_prefix, 2) == 0) {
      continue;
    }

    /* Initialize using first available GID */
    if (selected_gid_index == -1) {
      selected_gid_index = i;
      selected_gid_type  = gid_type;
      selected_gid       = current_gid;
    }
    /* Choose RoCEv2 over RoCEv1 */
    else  if (gid_type > selected_gid_type) {
      selected_gid_index = i;
      selected_gid_type  = gid_type;
      selected_gid       = current_gid;
    }
  }

  gid_index = selected_gid_index;
  gid       = selected_gid;

  free(gid_entries);
}

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

}  // namespace rocshmem
