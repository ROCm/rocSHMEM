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

#include "host.hpp"

#include <mpi.h>

#include "host_helpers.hpp"
#include "memory/window_info.hpp"

namespace rocshmem {

HostContextWindowInfo::HostContextWindowInfo(MPI_Comm comm_world, SymmetricHeap* heap) {
  window_info_ = new WindowInfo(comm_world, heap->get_local_heap_base(), heap->get_size());
}

HostContextWindowInfo::~HostContextWindowInfo() {
  delete window_info_;
}

WindowInfo* HostInterface::acquire_window_context() {
  auto index{find_avail_pool_entry()};
  HostContextWindowInfo* acquired_win_info = host_window_context_pool_[index];
  acquired_win_info->mark_unavail();
  return acquired_win_info->get();
}

void HostInterface::release_window_context(WindowInfo* window_info) {
  auto index{find_win_info_in_pool(window_info)};
  host_window_context_pool_[index]->mark_avail();
}

int HostInterface::find_avail_pool_entry() {
  for (int i{0}; i < max_num_ctxs_; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      return i;
    }
  }
  assert(false);
  return -1;
}

int HostInterface::find_win_info_in_pool(WindowInfo* window_info) {
  for (int i{0}; i < max_num_ctxs_; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      continue;
    }
    if (window_info == host_window_context_pool_[i]->get()) {
      return i;
    }
  }
  assert(false);
  return -1;
}

HostInterface::HostInterface(MPI_Comm rocshmem_comm, SymmetricHeap* heap) {
  MPI_Comm_dup(rocshmem_comm, &host_comm_world_);
  MPI_Comm_rank(host_comm_world_, &my_pe_);
  MPI_Comm_rank(host_comm_world_, &num_pes_);
  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_HOST_CONTEXTS"))) {
    max_num_ctxs_ = atoi(value);
  }
  size_t pool_size = max_num_ctxs_ * sizeof(HostContextWindowInfo*);
  host_window_context_pool_ = reinterpret_cast<HostContextWindowInfo**>(malloc(pool_size));
  for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
    host_window_context_pool_[ctx_i] = new HostContextWindowInfo(host_comm_world_, heap);
  }
}

HostInterface::~HostInterface() {
  for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
    delete host_window_context_pool_[ctx_i];
  }
  free(host_window_context_pool_);
  MPI_Comm_free(&host_comm_world_);
}

void HostInterface::putmem_nbi(void* dest, const void* source, size_t nelems, int pe, WindowInfo* window_info) {
  initiate_put(dest, source, nelems, pe, window_info);
}

void HostInterface::putmem(void* dest, const void* source, size_t nelems, int pe, WindowInfo* window_info) {
  initiate_put(dest, source, nelems, pe, window_info);
  MPI_Win_flush_local(pe, window_info->get_win());
}

void HostInterface::fence(WindowInfo* window_info) {
  complete_all(window_info->get_win());
  return;
}

void HostInterface::quiet(WindowInfo* window_info) {
  complete_all(window_info->get_win());
  return;
}

void HostInterface::sync_all(WindowInfo* window_info) {
  MPI_Win_sync(window_info->get_win());
  MPI_Barrier(host_comm_world_);
  return;
}

void HostInterface::barrier_all(WindowInfo* window_info) {
  complete_all(window_info->get_win());
  MPI_Barrier(host_comm_world_);
}

void HostInterface::barrier_for_sync() {
  MPI_Barrier(host_comm_world_);
}

}  // namespace rocshmem
