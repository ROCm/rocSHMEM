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
                           (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE +
                            ROCSHMEM_ATA_MAX_WRKDATA_SIZE);

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


  pAta_pool = reinterpret_cast<void *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(double) * ROCSHMEM_ATA_MAX_WRKDATA_SIZE
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

static void dump_ibv_context(struct ibv_context *x);
static void dump_ibv_device(struct ibv_device *x);
static void dump_ibv_pd(struct ibv_pd *x);
static void dump_ibv_port_attr(struct ibv_port_attr *x);
static void dump_ibv_qp(struct ibv_qp *qp, int conn_num);
static void dump_mlx5dv_qp(struct mlx5dv_qp *qp_dv, int conn_num);
static void dump_mlx5dv_cq(struct mlx5dv_cq *cq_dv, int conn_num);

void GDABackend::setup_ibv() {
  dest_info.resize(num_pes * (maximum_num_contexts_ + 1));
  int ib_devices{0};
  dev_list = ibv_get_device_list(&ib_devices);
  CHECK_NNULL(dev_list, "ibv_get_device");
  struct ibv_device* ib_dev = dev_list[0]; //TODO default to HIP selected device?
  if (requested_dev) {
    for (int i = 0; i < ib_devices; i++) {
      const char* select_dev{ibv_get_device_name(dev_list[i])};
      CHECK_NNULL(select_dev, "ibv_get_device_name");
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
  for (int i = 0; i < maximum_num_contexts_ + 1; i++) {
    if (backend_comm != MPI_COMM_NULL) {
      MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t), MPI_CHAR, dinfo + i * npes, sizeof(dest_info_t), MPI_CHAR, backend_comm);
    } else {
      Alltoall_char_inplace(reinterpret_cast<char*>(dinfo + i * npes), sizeof(dest_info_t), ROCSHMEM_TEAM_WORLD);
    }
  }

  for (int i = 0; i < qps.size(); i++) {
    change_status_rtr(qps[i], &dest_info[i], port);
  }
  rte_barrier();
  for (int i = 0; i < qps.size(); i++) {
    change_status_rts(qps[i], &dest_info[i]);
    dump_ibv_qp(qps[i], i);
  }
  rte_barrier();
}

void GDABackend::cleanup_ibv() {
  ibv_free_device_list(dev_list);

  delete ib_state;
  if (requested_dev != nullptr)
    free(requested_dev);
}


void GDABackend::setup_heap_memory_rkey() {
  auto *base_heap = heap.get_local_heap_base();
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  heap_mr = ibv_reg_mr(ib_state->pd_orig, base_heap, heap.get_size(), access);
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
    QueuePair qp(ib_state->pd_orig);
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
#ifndef GDA_BNXT
void GDABackend::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  ib_state = new ib_state_t;
  CHECK_NNULL(ib_state, "ib_state object create");

  ib_state->context = ibv_open_device(ib_dev);
  CHECK_NNULL(ib_state->context, "ib open device");
  dump_ibv_context(ib_state->context);
  dump_ibv_device(ib_state->context->device);

  ib_state->pd_orig = ibv_alloc_pd(ib_state->context);
  CHECK_NNULL(ib_state->pd_orig, "ib allocate pd");
  dump_ibv_pd(ib_state->pd_orig);

  ibv_parent_domain_init_attr pattr{};
  init_parent_domain_attr(&pattr);
  ib_state->pd_parent = ibv_alloc_parent_domain(ib_state->context, &pattr);
  CHECK_NNULL(ib_state->pd_parent, "ibv_alloc_parent_domain");
  dump_ibv_pd(ib_state->pd_parent);

#ifdef GDA_IONIC
  ionic_dv_pd_set_sqcmb(ib_state->pd_parent, false, false, false);
  ionic_dv_pd_set_rqcmb(ib_state->pd_parent, false, false, false);

  for (int uxdma_i = 0; uxdma_i < 2; ++uxdma_i) {
    ib_state->pd_uxdma[uxdma_i] = ibv_alloc_parent_domain(ib_state->context, &pattr);
    CHECK_NNULL(ib_state->pd_uxdma[uxdma_i], "ibv_alloc_parent_domain (uxdma)");

    ionic_dv_pd_set_sqcmb(ib_state->pd_uxdma[uxdma_i], false, false, false);
    ionic_dv_pd_set_rqcmb(ib_state->pd_uxdma[uxdma_i], false, false, false);
    ionic_dv_pd_set_udma_mask(ib_state->pd_uxdma[uxdma_i], 1u << uxdma_i);
  }
#endif

  int err = ibv_query_port(ib_state->context, port, &ib_state->portinfo);
  CHECK_ZERO(err, "ibv_query_port");
  dump_ibv_port_attr(&ib_state->portinfo);

  /* Must init after querying port */
  init_gid_index(port);

#ifdef GDA_IONIC
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
void GDABackend::try_to_modify_qp(ibv_qp* qp, StateType state) {
  int err = ibv_modify_qp(qp, &state.exp_qp_attr, state.exp_attr_mask);
  CHECK_ZERO(err, "ibv_modify_qp");
}

void GDABackend::init_qp_status(ibv_qp* qp, uint8_t port) {
  try_to_modify_qp<InitQPState>(qp, initqp(port));
}

void GDABackend::change_status_rtr(ibv_qp* qp, dest_info_t* dest, uint8_t port) {
  try_to_modify_qp<RtrState>(qp, rtr(dest, port));
}

void GDABackend::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  try_to_modify_qp<RtsState>(qp, rts(dest));
}

void GDABackend::create_qps(uint8_t port, ibv_port_attr* ib_port_att) {
  ibv_qp_cap cap{};
  cap.max_send_wr = sq_size;
  cap.max_send_sge = 1;
  cap.max_inline_data = 0;
#ifdef GDA_IONIC
  // TODO allow zero sges in the driver
  cap.max_recv_sge = 1;
#endif
  QPInitAttr qp_init_attr{qpattr(cap)};
  cqs.resize((maximum_num_contexts_ + 1) * num_pes);
  qps.resize((maximum_num_contexts_ + 1) * num_pes);
  int max_num_cqe = qp_init_attr.attr.cap.max_send_wr;
  for (int i = 0; i < qps.size(); i++) {
#ifdef GDA_IONIC
    int uxdma_i = ((i + 1) / 2) & 1;
    cqs[i] = create_cq(ib_state->context, ib_state->pd_uxdma[uxdma_i], max_num_cqe << 1);
    CHECK_NNULL(cqs[i], "create_cq");
    qps[i] = create_qp(ib_state->pd_uxdma[uxdma_i], ib_state->context, &qp_init_attr.attr, cqs[i]);
#else
    cqs[i] = create_cq(ib_state->context, ib_state->pd_parent, max_num_cqe);
    CHECK_NNULL(cqs[i], "create_cq");
    qps[i] = create_qp(ib_state->pd_parent, ib_state->context, &qp_init_attr.attr, cqs[i]);
#endif
    CHECK_NNULL(qps[i], "create_qp");
    init_qp_status(qps[i], port);
    dest_info[i].lid = ib_port_att->lid;
    dest_info[i].qpn = qps[i]->qp_num;
    dest_info[i].psn = 0;
    dest_info[i].gid = gid;
  }
}

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

void GDABackend::init_parent_domain_attr(ibv_parent_domain_init_attr* attr1) {
  attr1->pd = ib_state->pd_orig;
  attr1->td = nullptr;
  attr1->comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  attr1->alloc = GDABackend::pd_alloc;
  attr1->free = GDABackend::pd_release;
  attr1->pd_context = nullptr;
}

ibv_cq* GDABackend::create_cq(ibv_context* context, ibv_pd* pd, int cqe) {
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
  CHECK_NNULL(cq_ex, "ibv_create_cq_ex");
  ibv_cq *cq = ibv_cq_ex_to_cq(cq_ex);
  CHECK_NNULL(cq, "ibv_cq_ex_to_cq");
  return cq;
}

void GDABackend::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));

#ifdef GDA_IONIC
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
#else // !GDA_IONIC
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
#endif // !GDA_IONIC
}

ibv_qp* GDABackend::create_qp(ibv_pd* pd, ibv_context* context, ibv_qp_init_attr_ex* qp_attr, ibv_cq* cq) {
  ibv_qp* qp{nullptr};
  assert(pd);
  assert(context);
  assert(qp_attr);
  qp_attr->send_cq = cq;
  qp_attr->recv_cq = cq;
  qp_attr->pd = pd;
  qp_attr->comp_mask = IBV_QP_INIT_ATTR_PD;
  qp = ibv_create_qp_ex(context, qp_attr);
  CHECK_NNULL(qp, "ibv_create_qp_ex");
  return qp;
}

GDABackend::InitQPState GDABackend::initqp(uint8_t port) {
  InitQPState init{};
  init.exp_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  init.exp_qp_attr.port_num = port;
  init.exp_attr_mask |= IBV_QP_ACCESS_FLAGS;
  return init;
}

GDABackend::RtrState GDABackend::rtr(dest_info_t* dest, uint8_t port) {
  RtrState rtr{};
  rtr.exp_qp_attr.dest_qp_num = dest->qpn;
  rtr.exp_qp_attr.rq_psn = dest->psn;
  rtr.exp_qp_attr.ah_attr.port_num = port;
  rtr.exp_qp_attr.path_mtu = ib_state->portinfo.active_mtu;
  if (ib_state->portinfo.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    rtr.exp_qp_attr.ah_attr.dlid = dest->lid;
  } else {
    rtr.exp_qp_attr.ah_attr.is_global = 1;
    rtr.exp_qp_attr.ah_attr.grh.dgid = dest->gid;
    rtr.exp_qp_attr.ah_attr.grh.sgid_index = gid_index;
    rtr.exp_qp_attr.ah_attr.grh.hop_limit = 1;
  }
  rtr.exp_attr_mask |= IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  return rtr;
}

GDABackend::RtsState GDABackend::rts(dest_info_t* dest) {
  RtsState rts{};
  rts.exp_qp_attr.sq_psn = dest->psn;
  rts.exp_attr_mask |= IBV_QP_SQ_PSN;
  return rts;
}

GDABackend::QPInitAttr GDABackend::qpattr(ibv_qp_cap cap) {
  QPInitAttr qpattr(cap);
  qpattr.attr.qp_type = IBV_QPT_RC;
  return qpattr;
}
#endif

void GDABackend::init_gid_index(uint8_t port_num) {
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

  int gid_tbl_len         = ib_state->portinfo.gid_tbl_len;
  struct ibv_context *ctx = ib_state->context;

  gid_entries = (struct ibv_gid_entry*) calloc(gid_tbl_len, sizeof(struct ibv_gid_entry));

  gid_tbl_entries = ibv_query_gid_table(ctx, gid_entries, gid_tbl_len, 0);
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

    err = ibv_query_gid(ctx, port_num, i, &current_gid);
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

#if !defined(GDA_IONIC) && !defined(GDA_BNXT)
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
#endif // !GDA_IONIC

}  // namespace rocshmem
