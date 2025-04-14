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

#include "network_policy.hpp"

#include <mpi.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "atomic_return.hpp"
#include "context_incl.hpp"
#include "backend_ib.hpp"
#include "connection.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

void NetworkImpl::setup_atomic_region() {
  allocate_atomic_region(&atomic_ret, num_contexts);
  connection->reg_mr(atomic_ret->atomic_base_ptr, sizeof(uint64_t) * max_nb_atomic * num_contexts, &mr);
  atomic_ret->atomic_lkey = htobe32(mr->lkey);
}

void NetworkImpl::heap_memory_rkey(char *local_heap_base, size_t heap_size, MPI_Comm thread_comm) {
  /*
   * Allocate host-side memory to hold remote keys for all processing elements.
   */
  const size_t rkeys_size = sizeof(uint32_t) * num_pes;
  uint32_t *host_rkey_cpy = reinterpret_cast<uint32_t *>(malloc(rkeys_size));
  if (host_rkey_cpy == nullptr) { abort(); }

  /*
   * Using the Connection class, register the symmetric heap with the
   * InfiniBand network.
   */
  void *base_heap = local_heap_base;
  connection->reg_mr(base_heap, heap_size, &heap_mr);

  /*
   * Using the memory region from the prior heap memory registration,
   * allocate and initialize some device-side memory to hold the remote
   * keys for the symmetric heap base.
   *
   * Only the device-side memory entry for this processing element will be
   * updated with the key for the heap memory region.
   */
  connection->initialize_rkey_handle(&heap_rkey, heap_mr);

  /*
   * Copy the device-side heap base remote key array to the host-side
   * heap base remote key array.
   */
  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(host_rkey_cpy, heap_rkey, rkeys_size, hipMemcpyDeviceToHost, stream));
  CHECK_HIP(hipStreamSynchronize(stream));

  /*
   * Do all-to-all exchange of symmetric heap base remote key between the
   * processing elements.
   */
  MPI_Allgather(MPI_IN_PLACE, sizeof(uint32_t), MPI_CHAR, host_rkey_cpy, sizeof(uint32_t), MPI_CHAR, thread_comm);

  /*
   * Copy the recently updated host-side heap base remote key array back
   * to the device-side memory.
   */
  CHECK_HIP(hipMemcpyAsync(heap_rkey, host_rkey_cpy, rkeys_size, hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));

  /*
   * Free the host-side resources used to do the processing element
   * exchange of keys and addresses for the symmetric heap base.
   */
  free(host_rkey_cpy);

  /*
   * Initialize this member variable to hold the InfiniBand memory
   * region's local key.
   */
  lkey = heap_mr->lkey;
}

void NetworkImpl::setup_gpu_qps(GPUIBBackend *backend) {
  int connections = connection->total_number_connections();
  CHECK_HIP(hipMalloc(&gpu_qps, sizeof(QueuePair) * connections));
  for (int i{0}; i < connections; i++) {
    new (&gpu_qps[i]) QueuePair(backend);
    connection->init_gpu_qp_from_connection(&gpu_qps[i], i);
  }
}

void NetworkImpl::networkHostSetup(GPUIBBackend *backend) {
  num_pes = backend->num_pes;
  my_pe = backend->my_pe;
  num_contexts = backend->maximum_num_contexts_;
  connection = new Connection(backend);
  connection->initialize(num_contexts);
  const auto &heap_bases{backend->heap.get_heap_bases()};
  heap_memory_rkey(heap_bases[my_pe], backend->heap.get_size(), backend->thread_comm);
  setup_atomic_region();
  setup_gpu_qps(backend);
}

void NetworkImpl::networkHostFinalize() {
  CHECK_HIP(hipFree(atomic_ret));
  atomic_ret = nullptr;
  CHECK_HIP(hipFree(gpu_qps));
  gpu_qps = nullptr;
  connection->free_rkey_handle(heap_rkey);
  connection->finalize();
  delete connection;
  connection = nullptr;
}

void NetworkImpl::networkHostInit(GPUIBContext *ctx, int buffer_id) {
  CHECK_HIP(hipMalloc(&ctx->device_qp_proxy, num_pes * sizeof(QueuePair)));
  for (int i{0}; i < num_pes; i++) {
    int offset = num_contexts * i + buffer_id;
    new (ctx->getQueuePair(i)) QueuePair(gpu_qps[offset]);
    auto *qp = ctx->getQueuePair(i);
    qp->global_qp = &gpu_qps[offset];
    qp->num_cqs = num_pes;
    qp->atomic_ret.atomic_base_ptr = &atomic_ret->atomic_base_ptr[max_nb_atomic * buffer_id];
    qp->base_heap = ctx->base_heap;
  }
}

__device__ void NetworkImpl::networkGpuInit(GPUIBContext *ctx, int buffer_id) {
  for (int i{0}; i < num_pes; i++) {
    int offset = num_contexts * i + buffer_id;
    auto *qp = ctx->getQueuePair(i);
    new (qp) QueuePair(gpu_qps[offset]);
    qp->global_qp = &gpu_qps[offset];
    qp->num_cqs = num_pes;
    qp->atomic_ret.atomic_base_ptr = &atomic_ret->atomic_base_ptr[max_nb_atomic * buffer_id];
    qp->base_heap = ctx->base_heap;
  }
}

__device__ __host__ QueuePair *NetworkImpl::getQueuePair(QueuePair *qp_handle, int pe) {
  return &qp_handle[pe];
}

}  // namespace rocshmem
