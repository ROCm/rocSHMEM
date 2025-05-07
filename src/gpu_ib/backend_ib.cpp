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

#include "backend_ib.hpp"

#include <cstdio>
#include <cstdlib>
#include <endian.h>
#include <mpi.h>
#include <mutex>
#include <rocshmem/rocshmem.hpp>
#include <unistd.h>

#include "context_incl.hpp"
#include "gpuib_macros.inl"
#include "gpu_ib_team.hpp"
#include "host/host.hpp"
#include "queue_pair.hpp"

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

GPUIBBackend::GPUIBBackend(MPI_Comm comm) : heap{comm} {
  CHECK_HIP(hipMalloc(&print_lock, sizeof(*print_lock)));
  *print_lock = 0;
  int* print_lock_addr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&print_lock_addr), HIP_SYMBOL(print_lock)));
  CHECK_HIP(hipMemcpy(print_lock_addr, &print_lock, sizeof(print_lock), hipMemcpyDefault));
  int* device_backend_proxy_addr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&device_backend_proxy_addr), HIP_SYMBOL(device_backend_proxy)));
  GPUIBBackend* this_temp_addr{this};
  CHECK_HIP(hipMemcpy(device_backend_proxy_addr, &this_temp_addr, sizeof(this), hipMemcpyDefault));
  if (auto maximum_num_contexts_str = getenv("ROCSHMEM_MAX_NUM_CONTEXTS")) {
    std::stringstream sstream(maximum_num_contexts_str);
    sstream >> maximum_num_contexts_;
  }
  init_mpi_once(comm);
  NET_CHECK(MPI_Comm_size(backend_comm, &num_pes));
  NET_CHECK(MPI_Comm_rank(backend_comm, &my_pe));
  host_interface = new HostInterface(backend_comm, &heap);
  setup_default_host_ctx();
  setup_team_world();
  rocshmem_collective_init();
  teams_init();
  NET_CHECK(MPI_Barrier(backend_comm));
  initialize_network();
  setup_ctxs();
//  setup_default_ctx();
}

__device__ bool GPUIBBackend::create_ctx(rocshmem_ctx_t *ctx) {
  GPUIBContext *ctx_;
  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;
  return true;
}

void GPUIBBackend::ctx_create(void **ctx) {
  GPUIBHostContext *new_ctx = nullptr;
  new_ctx = new GPUIBHostContext(this);
  *ctx = new_ctx;
}

GPUIBHostContext *get_internal_gpu_ib_ctx(Context *ctx) {
  return reinterpret_cast<GPUIBHostContext*>(ctx);
}

void GPUIBBackend::ctx_destroy(Context *ctx) {
  GPUIBHostContext *gpu_ib_host_ctx = get_internal_gpu_ib_ctx(ctx);
  delete gpu_ib_host_ctx;
}

__device__ void GPUIBBackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<GPUIBContext*>(ctx->ctx_opaque));
}

__host__ void GPUIBBackend::global_exit(int status) {
  MPI_Abort(backend_comm, status);
}

void GPUIBBackend::create_new_team([[maybe_unused]] Team *parent_team, TeamInfo *team_info_wrt_parent, TeamInfo *team_info_wrt_world, int num_pes,
                                   int my_pe_in_new_team, MPI_Comm team_comm, rocshmem_team_t *new_team) {
  NET_CHECK(MPI_Allreduce(pool_bitmask_, reduced_bitmask_, bitmask_size_, MPI_CHAR, MPI_BAND, team_comm));
  auto max_num_teams{team_tracker.get_max_num_teams()};
  int common_index = get_ls_non_zero_bit(reduced_bitmask_, max_num_teams);
  if (common_index < 0) { abort(); }
  int byte = common_index / CHAR_BIT;
  pool_bitmask_[byte] &= ~(1 << (common_index % CHAR_BIT));
  GPUIBTeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(GPUIBTeam)));
  new (new_team_obj) GPUIBTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes, my_pe_in_new_team, team_comm, common_index);
  *new_team = get_external_team(new_team_obj);
}

void GPUIBBackend::team_destroy(rocshmem_team_t team) {
  GPUIBTeam *team_obj = get_internal_gpu_ib_team(team);
  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);
  team_obj->~GPUIBTeam();
  CHECK_HIP(hipFree(team_obj));
}

void GPUIBBackend::initialize_network() {
  networkImpl.networkHostSetup(this);
}

void GPUIBBackend::setup_default_host_ctx() {
  default_host_ctx_ = new GPUIBHostContext(this);
  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx_;
}

void GPUIBBackend::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(GPUIBContext) * maximum_num_contexts_));
  for (int i = 0; i < maximum_num_contexts_; i++) {
    new (&ctx_array[i]) GPUIBContext(this, i);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

void GPUIBBackend::setup_default_ctx() {
  CHECK_HIP(hipMalloc(&default_ctx_, sizeof(GPUIBContext)));
  new (default_ctx_) GPUIBContext(this, 0);
  int *symbol_address;
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_address), HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT)));
  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;
  rocshmem_ctx_t ctx_default_host{default_ctx_, tinfo};
  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(symbol_address, &ctx_default_host, sizeof(rocshmem_ctx_t), hipMemcpyDefault, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));
}

void GPUIBBackend::setup_team_world() {
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));
  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);
  MPI_Comm team_world_comm;
  NET_CHECK(MPI_Comm_dup(backend_comm, &team_world_comm));
  GPUIBTeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(GPUIBTeam)));
  new (team_world) GPUIBTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes, my_pe, team_world_comm, 0);
  team_tracker.set_team_world(team_world);
  ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
}

void GPUIBBackend::init_mpi_once(MPI_Comm comm) {
  static std::mutex init_mutex;
  const std::lock_guard<std::mutex> lock(init_mutex);
  int init_done{0};
  NET_CHECK(MPI_Initialized(&init_done));
  if (init_done == 0) {
    int provided;
    NET_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
  }
  NET_CHECK(MPI_Comm_dup(comm, &backend_comm));
}

void GPUIBBackend::teams_init() {
  auto max_num_teams{team_tracker.get_max_num_teams()};
  barrier_pSync_pool = reinterpret_cast<long*>(rocshmem_malloc(sizeof(long) * ROCSHMEM_BARRIER_SYNC_SIZE * max_num_teams));
  long *barrier_pSync;
  for (int team_i{0}; team_i < max_num_teams; team_i++) {
    barrier_pSync = reinterpret_cast<long*>(&barrier_pSync_pool[team_i * ROCSHMEM_BARRIER_SYNC_SIZE]);
    for (int i{0}; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
      barrier_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
  }

  /*
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
  bitmask_size_ = (max_num_teams % CHAR_BIT) ? (max_num_teams / CHAR_BIT + 1) : (max_num_teams / CHAR_BIT);
  pool_bitmask_ = reinterpret_cast<char*>(malloc(bitmask_size_));
  reduced_bitmask_ = reinterpret_cast<char*>(malloc(bitmask_size_));

  memset(pool_bitmask_, 0, bitmask_size_);
  memset(reduced_bitmask_, 0, bitmask_size_);
  for (int bit_i{1}; bit_i < max_num_teams; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    pool_bitmask_[byte_i] |= 1 << (bit_i % CHAR_BIT);
  }
  NET_CHECK(MPI_Barrier(backend_comm));
}

void GPUIBBackend::teams_destroy() {
  rocshmem_free(barrier_pSync_pool);
  free(pool_bitmask_);
  free(reduced_bitmask_);
}

void GPUIBBackend::rocshmem_collective_init() {
  size_t one_sync_size_bytes {sizeof(*barrier_sync)};
  size_t total_sync_elems {
    ROCSHMEM_BARRIER_SYNC_SIZE * (maximum_num_contexts_ + 1)};
  size_t sync_size_bytes {one_sync_size_bytes * total_sync_elems};

  heap.malloc(reinterpret_cast<void**>(&barrier_sync), sync_size_bytes);
  for (int i{0}; i < total_sync_elems; i++) {
    barrier_sync[i] = ROCSHMEM_SYNC_VALUE;
  }
  NET_CHECK(MPI_Barrier(backend_comm));
}

void GPUIBBackend::track_ctx(Context* ctx) {
  list_of_ctxs.push_back(ctx);
}

void GPUIBBackend::untrack_ctx(Context* ctx) {
  std::vector<Context*>::iterator it = std::find(list_of_ctxs.begin(), list_of_ctxs.end(), ctx);
  assert(it != list_of_ctxs.end());
  list_of_ctxs.erase(it);
}

void GPUIBBackend::destroy_remaining_ctxs() {
  while (!list_of_ctxs.empty()) {
    ctx_destroy(list_of_ctxs.back());
    list_of_ctxs.pop_back();
  }
}

GPUIBBackend::~GPUIBBackend() {
  CHECK_HIP(hipFree(print_lock));
  teams_destroy();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));
  delete default_host_ctx_;
  NET_CHECK(MPI_Comm_free(&backend_comm));
  if (default_ctx_) {
    CHECK_HIP(hipFree(default_ctx_->device_qp_proxy));
    CHECK_HIP(hipFree(default_ctx_));
    default_ctx_ = nullptr;
  }
  delete host_interface;
  host_interface = nullptr;
  networkImpl.networkHostFinalize();
  CHECK_HIP(hipFree(ctx_array));
}

}  // namespace rocshmem
