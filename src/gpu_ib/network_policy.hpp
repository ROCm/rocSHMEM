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

#ifndef LIBRARY_SRC_GPU_IB_NETWORK_POLICY_HPP_
#define LIBRARY_SRC_GPU_IB_NETWORK_POLICY_HPP_

#include <hip/hip_runtime.h>
#include <mpi.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "connection_policy.hpp"
#include "queue_pair.hpp"
#include "../memory/symmetric_heap.hpp"
#include "../util.hpp"

struct ibv_mr;

namespace rocshmem {

struct atomic_ret_t;
class GPUIBBackend;
class GPUIBContext;
class GPUIBHostContext;
class Connection;

class NetworkOnImpl {
 public:
  /**
   * @brief setup the network resources and initialization for the
   * GPUIBBackend
   */
  __host__ void networkHostSetup(GPUIBBackend *B);

  /**
   * @brief deallocate and close the network resources
   */
  __host__ void networkHostFinalize();

  /**
   * @brief initialize the network resources for each context
   */
  __host__ void networkHostInit(GPUIBContext *ctx, int buffer_id);

  /**
   * @brief initialize the network resources for each context on GPU side
   */
  __device__ void networkGpuInit(GPUIBContext *ctx, int buffer_id);

  /**
   * @brief returns the QP for the targeted pe
   */
  __device__ __host__ QueuePair *getQueuePair(QueuePair *qp, int pe);

  /**
   * @brief returns the numbers of QPs used per the calling PE
   */
  __device__ __host__ int getNumQueuePairs();

  /**
   * @brief returns the number of PEs accessible via network
   */
  __device__ __host__ int getNumDest() { return num_pes; }

  static uint32_t externSharedBytes(int num_pes) {
    int remote_conn{1};
#ifndef USE_DC
    remote_conn = num_pes;
#endif
    return remote_conn * sizeof(QueuePair);
  }

 protected:
  /**
   * @brief flag to indicated that the helper thread reach this milestone
   */
  volatile bool network_init_done{false};

  void heap_memory_rkey(char *local_heap_base, size_t heap_size,
                        MPI_Comm thread_comm, bool is_managed);

  /**
   * @brief Allocate and initialize the atomic region.
   *
   * The atomic region is used by the atomic operations which have return
   * values. The library user does not need to provide an address for the
   * return value so we are forced to do it on their behalf.
   *
   * The atomic_ret member is initialized upon completion of this method.
   */
  void setup_atomic_region();

  /**
   * @brief Allocate and initialize device-side queue pair objects.
   *
   * Upon completion, the gpu_qps member will be initialized.
   */
  void setup_gpu_qps(GPUIBBackend *B);

  /**
   * @brief Allocate and initialize device-side memory that will be used for
   * the return of g shmem ops (eg: shmem_int_g)
   */
  void rocshmem_g_init(SymmetricHeap *heap_handle, MPI_Comm thread_comm);

  /**
   * @brief The backend delegates some InfiniBand connection setup to
   * the Connection class.
   */
  Connection *connection{nullptr};

 public:
  /**
   * @brief Number of PEs. Get directly from the GPUIBBackend.
   */
  int num_pes{0};

  /**
   * @brief This PE's rank.
   */
  int my_pe{-1};

  /**
   * @brief Number of WG that will be performing communication
   */
  int num_blocks{0};

  /**
   * @brief Set of QueuePairs used by device to do networking.
   *
   * The member is used during Context creation.
   *
   * @todo What we really need here is a collection of Contexts that can
   * either be copied into LDS or used directly by the GPU depending on
   * what type of context it is (shareable, serialized, or private).
   * No need to pool up QueuePairs, they can just be managed by their
   * owning Context. Should then consider pushing into base class since
   * it's not gpu-ib specific.
   */
  QueuePair *gpu_qps{nullptr};

  /**
   * @brief C-array of symmetric heap base pointers.
   *
   * A C-array of char* pointers corresponding to the heap base pointers
   * virtual address for each processing element that we can communicate
   * with.
   */
  uint32_t *heap_rkey{nullptr};

  /**
   * @brief Handle for the symmetric heap memory region.
   */
  ibv_mr *heap_mr{nullptr};

  /**
   * @brief Local key for the symmetric heap memory region.
   */
  uint32_t lkey{0};

  /**
   * @brief Control struct for atomic memory region.
   *
   * The atomic region is used by the atomic operations which have return
   * values. The library user does not need to provide an address for the
   * return value so we are forced to do it on their behalf.
   */
  atomic_ret_t *atomic_ret{nullptr};

  /**
   * @brief Handle for the atomic memory region.
   *
   * @todo Provide more descriptive variable name.
   */
  ibv_mr *mr{nullptr};

  /**
   * @brief Buffer used to store the results of a *_g operation.
   *
   * These operations do not provide a destination buffer so the runtime
   * must manage one.
   */
  char *g_ret{nullptr};

  /**
   * @brief Compile-time configuration policy for InfiniBand connections.
   *
   * The configuration option "USE_DC" can be enabled to create
   * Dynamic connection types. By default, Reliable connections are
   * created.
   */
  ConnectionImpl *connection_policy{nullptr};
};

// clang-format off
NOWARN(-Wunused-parameter,
class NetworkOffImpl {
 public:
  __host__ void networkHostSetup(GPUIBBackend *B);

  __host__ void networkHostFinalize();

  __host__ void networkHostInit(GPUIBContext *ctx, int buffer_id) {}

  __device__ void networkGpuInit(GPUIBContext *ctx, int buffer_id) {}

  __device__ __host__ QueuePair *getQueuePair(QueuePair *qp, int pe) {
    return nullptr;
  }

  __device__ __host__ int getNumQueuePairs() { return 0; }

  __device__ __host__ int getNumDest() { return 0; }

  static uint32_t externSharedBytes(int num_pes) { return 0; }

 public:
  int num_pes{0};

  int my_pe{-1};

  int num_blocks{0};

  QueuePair *gpu_qps{nullptr};

  uint32_t *heap_rkey{nullptr};

  ibv_mr *heap_mr{nullptr};

  uint32_t lkey{0};

  atomic_ret_t *atomic_ret{nullptr};

  ibv_mr *mr{nullptr};

  char *g_ret{nullptr};

  ConnectionImpl *connection_policy{nullptr};
};
)
// clang-format on

/*
 * Select which one of our IPC policies to use at compile time.
 */
#ifdef USE_SINGLE_NODE
typedef NetworkOffImpl NetworkImpl;
#else
typedef NetworkOnImpl NetworkImpl;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_NETWORK_POLICY_HPP_
