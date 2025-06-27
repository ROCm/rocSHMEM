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

#include "host.hpp"

#include <mpi.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "host_helpers.hpp"
#include "../memory/window_info.hpp"
#include "../util.hpp"

#include <cassert>

namespace rocshmem {

__host__ HostContextWindowInfo::HostContextWindowInfo(MPI_Comm comm_world,
                                                      SymmetricHeap* heap) {
  window_info_ =
      new WindowInfoMPI(comm_world, heap->get_local_heap_base(), heap->get_size());
}

__host__ HostContextWindowInfo::HostContextWindowInfo(SymmetricHeap* heap) {
  window_info_ =
      new WindowInfo(heap->get_local_heap_base(), heap->get_size());
}

__host__ HostContextWindowInfo::~HostContextWindowInfo() {
  delete window_info_;
}

WindowInfo* HostInterface::acquire_window_context() {
  auto index{find_avail_pool_entry()};
  /* Entry should have been available; consider this as an error. */
  assert(index >= 0);

  HostContextWindowInfo* acquired_win_info = host_window_context_pool_[index];

  acquired_win_info->mark_unavail();

  return acquired_win_info->get();
}

__host__ void HostInterface::release_window_context(WindowInfo* window_info) {
  auto index{find_win_info_in_pool(window_info)};
  /* Entry should have been present; consider this as an error. */
  assert(index >= 0);

  host_window_context_pool_[index]->mark_avail();
}

int HostInterface::find_avail_pool_entry() {
  for (int i{0}; i < max_num_ctxs_; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      return i;
    }
  }
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
  return -1;
}

__host__ HostInterface::HostInterface(MPI_Comm rocshmem_comm,
                                      SymmetricHeap* heap) {
  /*
   * Duplicate a communicator from roc_shem's comm
   * world for the host interface
   */
  MPI_Comm_dup(rocshmem_comm, &host_comm_world_);
  MPI_Comm_rank(host_comm_world_, &my_pe_);
  MPI_Comm_rank(host_comm_world_, &num_pes_);

  /*
   * Allocate and initialize pool of windows for contexts
   */
  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_HOST_CONTEXTS"))) {
    max_num_ctxs_ = atoi(value);
  }

  size_t pool_size = max_num_ctxs_ * sizeof(HostContextWindowInfo*);
  host_window_context_pool_ =
      reinterpret_cast<HostContextWindowInfo**>(malloc(pool_size));

  for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
    host_window_context_pool_[ctx_i] =
        new HostContextWindowInfo(host_comm_world_, heap);
  }
}

__host__ HostInterface::HostInterface(TcpBootstrap *bootstr,
                                      SymmetricHeap* heap) {
  host_bootstrap_ = bootstr;
  my_pe_ = bootstr->getRank();
  num_pes_ = bootstr->getNranks();

  /*
   * Allocate and initialize pool of windows for contexts
   */
  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_HOST_CONTEXTS"))) {
    max_num_ctxs_ = atoi(value);
  }

  size_t pool_size = max_num_ctxs_ * sizeof(HostContextWindowInfo*);
  host_window_context_pool_ =
      reinterpret_cast<HostContextWindowInfo**>(malloc(pool_size));

  for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
    host_window_context_pool_[ctx_i] =
        new HostContextWindowInfo(heap);
  }
}

__host__ HostInterface::~HostInterface() {
  /* Detroy the pool of contexts */

  if (host_window_context_pool_ != nullptr) {
    for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
      delete host_window_context_pool_[ctx_i];
    }
    free(host_window_context_pool_);
  }

  if (host_comm_world_ != MPI_COMM_NULL) {
    MPI_Comm_free(&host_comm_world_);
  }
}

__host__ void HostInterface::putmem_nbi(void* dest, const void* source,
                                        size_t nelems, int pe,
                                        WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  initiate_put(dest, source, nelems, pe, window_info_mpi);
}

__host__ void HostInterface::putmem(void* dest, const void* source,
                                    size_t nelems, int pe,
                                    WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  initiate_put(dest, source, nelems, pe, window_info_mpi);

  MPI_Win_flush_local(pe, window_info_mpi->get_win());
}

__host__ void HostInterface::quiet(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  complete_all(window_info_mpi->get_win());

  return;
}

__host__ void HostInterface::sync_all(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    MPI_Win_sync(window_info_mpi->get_win());
    MPI_Barrier(host_comm_world_);
  } else {
    host_bootstrap_->barrier();
  }

  return;
}

__host__ void HostInterface::barrier_all(WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (window_info_mpi) {
    complete_all(window_info_mpi->get_win());
    MPI_Barrier(host_comm_world_);
  } else {
    host_bootstrap_->barrier();
  }

  return;
}

__host__ void HostInterface::barrier_for_sync() {
  if (host_comm_world_ != MPI_COMM_NULL) {
    MPI_Barrier(host_comm_world_);
  } else {
    host_bootstrap_->barrier();
  }
}

}  // namespace rocshmem
