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

#ifndef LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP_
#define LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP_

#include <mpi.h>
#include <vector>

#include <rocshmem/rocshmem.hpp>
#include "rocshmem_config.h"
#include "context_incl.hpp"
#include "containers/free_list_impl.hpp"
#include "memory/hip_allocator.hpp"
#include "memory/symmetric_heap.hpp"
#include "network_policy.hpp"
#include "team_tracker.hpp"

namespace rocshmem {

class HostInterface;
class Team;
class TeamInfo;

/**
 * @class GPUIBBackend backend.hpp
 * @brief InfiniBand specific backend.
 *
 * @brief Container class for the persistent state used by the library.
 *
 * GPUIBBackend is populated by host-side initialization and allocation calls.
 * It uses this state to populate Context objects which the GPU may use to
 * perform networking operations.
 *
 * The rocshmem.cpp implementation file wraps many the GPUIBBackend public
 * members to implement the library's public API.
 *
 * The InfiniBand (GPUIB) backend enables the device to enqueue network
 * requests to InfiniBand queues (with minimal host intervention). The setup
 * requires some effort from the host, but the device is able to craft
 * InfiniBand requests and send them on its own.
 */
class GPUIBBackend {
 public:
  explicit GPUIBBackend(MPI_Comm comm);

  ~GPUIBBackend();

  /**
   * @brief Abort the application.
   *
   * @param[in] status Exit code.
   *
   * @return void.
   *
   * @note This routine terminates the entire application.
   */
  void global_exit(int status);

  /**
   * @brief Create a new team object and initialize it.
   *
   * @param[in] parent_team Pointer to the parrent team object.
   * @param[in] team_info_wrt_parent TeamInfo object wrt parent team.
   * @param[in] team_info_wrt_world TeamInfo object wrt TEAM_WORLD.
   * @param[in] num_pes Number of PEs in this team.
   * @param[in] my_pe_in_new_team Index of this PE in the new team.
   * @param[in] team_comm MPI communicator for this team.
   *
   * @param[out] new_team pointer to the new team.
   */
  void create_new_team(Team* parent_team,
                       TeamInfo* team_info_wrt_parent,
                       TeamInfo* team_info_wrt_world, int num_pes,
                       int my_pe_in_new_team, MPI_Comm team_comm,
                       rocshmem_team_t* new_team);

  /**
   * @brief Destruct a team
   *
   * @param[in] team Handle to the team to destroy.
   */
  void team_destroy(rocshmem_team_t team);

  /**
   * @brief Reports processing element number id.
   *
   * @return Unique numeric identifier for each processing element.
   */
  __host__ __device__ int getMyPE() const { return my_pe; }

  /**
   * @brief Reports number of processing elements.
   *
   * @return Number of active processing elements tracked by library.
   */
  __host__ __device__ int getNumPEs() const { return num_pes; }

  /**
   * @brief Creates a new OpenSHMEM context.
   *
   * @param[in] ctx     Address of the pointer to the new context
   *
   * @return Zero on success, nonzero otherwise.
   */
  void ctx_create(void** ctx);

  __device__ bool create_ctx(rocshmem_ctx_t *ctx);

  /**
   * @brief Destroys a context.
   *
   * @param[in] ctx Context handle.
   *
   * @return void.
   */
  void ctx_destroy(Context* ctx);

  __device__ void destroy_ctx(rocshmem_ctx_t* ctx);

  /**
   * @brief Remove all ctxs from the list of user-created ctxs
   */
  void destroy_remaining_ctxs();

  /**
   * @brief Add ctx from the list of user-created ctxs
   */
  void track_ctx(Context* ctx);

  /**
   * @brief Remove ctx from the list of user-created ctxs
   */
  void untrack_ctx(Context* ctx);

  /**
   * @brief initialize MPI.
   *
   * GPUIB relies on MPI just to exchange the connection information.
   */
  void init_mpi_once(MPI_Comm comm);

  /**
   * @brief init the network support
   */
  void initialize_network();

  /**
   * @brief Allocate and initialize the ROCSHMEM_CTX_DEFAULT variable.
   */
  void setup_default_ctx();
  void setup_ctxs();

  /**
   * @brief Allocate and initialize the default context for host
   * operations.
   */
  void setup_default_host_ctx();

  /**
   * @brief Allocate and initialize team world.
   */
  void setup_team_world();

  /**
   * @brief Initialize the resources required to support teams
   */
  void teams_init();

  /**
   * @brief Destruct the resources required to support teams
   */
  void teams_destroy();

  /**
   * @brief Allocate and initialize barrier operation addresses on
   * symmetric heap.
   *
   * When this method completes, the barrier_sync member will be available
   * for use.
   */
  void rocshmem_collective_init();

  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the GPUIBBackend
   */
  HostInterface *host_interface{nullptr};

  /**
   * @brief Handle for raw memory for barrier sync
   */
  long *barrier_pSync_pool{nullptr};

  /**
   * @brief rocSHMEM's copy of MPI_COMM_WORLD (for interoperability
   * with orthogonal MPI usage in an MPI+rocSHMEM program).
   */
  MPI_Comm gpu_ib_comm_world{};
  MPI_Comm backend_comm{};

  /**
   * @brief Holds number of blocks used in library
   */
  size_t num_blocks_{1};

  /**
   * @brief Scratchpad for the internal barrier algorithms.
   */
  int64_t *barrier_sync{nullptr};

  /**
   * @brief Compile-time configuration policy for network (IB)
   *
   *
   * The configuration option "USE_SINGLE_NODE" can be enabled to not build
   * with network support.
   */
  NetworkImpl networkImpl{};

  /**
   * @brief An array of @ref ROContexts that backs the context FreeList.
   */
  GPUIBContext *ctx_array{nullptr};

  /**
   * @brief A free-list containing contexts.
   */
  FreeListProxy<HIPAllocator, GPUIBContext *> ctx_free_list{};

  /**
   * @brief Holds maximum number of contexts used in library
   */
  size_t maximum_num_contexts_{1024};

  /**
   * @brief The bitmask representing the availability of teams in the pool
   */
  char *pool_bitmask_{nullptr};

  /**
   * @brief Bitmask to store the reduced result of bitmasks on pariticipating
   * PEs
   *
   * With no thread-safety for this bitmask, multithreaded creation of teams is
   * not supported.
   */
  char *reduced_bitmask_{nullptr};

  /**
   * @brief Size of the bitmask
   */
  int bitmask_size_{-1};

  /**
   * @brief Holds a copy of the default context (see OpenSHMEM
   * specification).
   */
  GPUIBContext *default_ctx_{nullptr};

  /**
   * @brief Holds a copy of the default context for host functions
   */
  GPUIBHostContext *default_host_ctx_{nullptr};

  /**
   * @brief Number of processing elements running in job.
   */
  int num_pes{0};

  /**
   * @brief Unique numeric identifier ranging from 0 (inclusive) to
   * num_pes (exclusive) [0 ... num_pes).
   */
  int my_pe{-1};

  /**
   * @brief indicate when init is done on the CPU. Non-blocking init is only
   * available with GPU-IB
   */
  uint8_t* done_init{nullptr};

  MPI_Comm thread_comm{};

  /**
   * @brief Object contains the interface and internal data structures
   * needed to allocate/free memory on the symmetric heap.
   */
  SymmetricHeap heap{};

  /**
   * @brief Maintains information about teams
   */
  TeamTracker team_tracker{};

  /**
   * @brief List of ctxs created by the user.
   */
  std::vector<Context*> list_of_ctxs{};
};

/**
 * @brief Global handle used by the device to access the backend.
 */
extern __constant__ GPUIBBackend* device_backend_proxy;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP_
