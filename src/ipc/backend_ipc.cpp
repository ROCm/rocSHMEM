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

#include "backend_ipc.hpp"

namespace rocshmem {

#define NET_CHECK(cmd)                                       \
  {                                                          \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort() ;                                              \
    }                                                        \
  }

extern roc_shmem_ctx_t ROC_SHMEM_HOST_CTX_DEFAULT;

IPCBackend::IPCBackend(MPI_Comm comm)
    :  Backend() {
  type = BackendType::IPC_BACKEND;
    
  if (auto maximum_num_contexts_str = getenv("ROC_SHMEM_MAX_NUM_CONTEXTS")) {
    std::stringstream sstream(maximum_num_contexts_str);
    sstream >> maximum_num_contexts_;
  }

  init_mpi_once(comm);
    
  initIPC();
    
  auto *bp{ipc_backend_proxy.get()};
    
  bp->heap_ptr = &heap;

  /* Initialize the host interface */
  host_interface =
      new HostInterface(hdp_proxy_.get(), thread_comm, &heap);
  //free host interface

  default_host_ctx = std::make_unique<IPCHostContext>(this, 0);

  ROC_SHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();

  init_g_ret(&heap, thread_comm, MAX_NUM_BLOCKS, &bp->g_ret);

  allocate_atomic_region(&bp->atomic_ret, MAX_NUM_BLOCKS);

  default_context_proxy_ = IPCDefaultContextProxyT(this);

  setup_ctxs();
  
}

IPCBackend::~IPCBackend() {
  ipc_net_free_runtime();
  CHECK_HIP(hipFree(ctx_array));
}

void IPCBackend::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(IPCContext) * maximum_num_contexts_));
  for (int i = 0; i < maximum_num_contexts_; i++) {
    new (&ctx_array[i]) IPCContext(this);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

__device__ bool IPCBackend::create_ctx(int64_t options, roc_shmem_ctx_t *ctx) {
  IPCContext *ctx_{nullptr};

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;
  return true;
}

__device__ void IPCBackend::destroy_ctx(roc_shmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<IPCContext *>(ctx->ctx_opaque));
}

void IPCBackend::init_mpi_once(MPI_Comm comm) {
  int init_done{};
  NET_CHECK(MPI_Initialized(&init_done));

  int provided{};
  if (!init_done) {
    NET_CHECK(MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided));
    if (provided != MPI_THREAD_MULTIPLE) {
      std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
    }
  }
  if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;

  NET_CHECK(MPI_Comm_dup(comm, &thread_comm));
  NET_CHECK(MPI_Comm_size(thread_comm, &num_pes));
  NET_CHECK(MPI_Comm_rank(thread_comm, &my_pe));
}

void IPCBackend::team_destroy(roc_shmem_team_t team) {
  assert(false);
}

void IPCBackend::create_new_team(Team *parent_team,
                                TeamInfo *team_info_wrt_parent,
                                TeamInfo *team_info_wrt_world, int num_pes,
                                int my_pe_in_new_team, MPI_Comm team_comm,
                                roc_shmem_team_t *new_team) {
  assert(false);
}

void IPCBackend::ctx_create(int64_t options, void **ctx) {
  IPCHostContext *new_ctx{nullptr};
  new_ctx = new IPCHostContext(this, options);
  *ctx = new_ctx;
}

IPCHostContext *get_internal_ipc_net_ctx(Context *ctx) {
  return reinterpret_cast<IPCHostContext *>(ctx);
}

void IPCBackend::ctx_destroy(Context *ctx) {
  IPCHostContext *ro_net_host_ctx{get_internal_ipc_net_ctx(ctx)};
  delete ro_net_host_ctx;
}

void IPCBackend::reset_backend_stats() {
  assert(false);
}

void IPCBackend::dump_backend_stats() {
  assert(false);
}

void IPCBackend::initIPC() {
  const auto &heap_bases{heap.get_heap_bases()};

  ipcImpl.ipcHostInit(my_pe, heap_bases,
                      thread_comm);
}

void IPCBackend::ipc_net_free_runtime() {
  /*
   * Validate that a handle was passed that is not a nullptr.
   */
  auto *bp{ipc_backend_proxy.get()};
  assert(bp);

  /*
   * Free the atomic_ret array.
   */
  CHECK_HIP(hipFree(bp->atomic_ret->atomic_base_ptr));
  
  // TODO(Avinash) Free g_ret
}

void IPCBackend::global_exit(int status) {
  assert(false);
}


}  // namespace rocshmem