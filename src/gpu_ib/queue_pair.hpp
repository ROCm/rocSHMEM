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

#ifndef LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP_
#define LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP_

/**
 * @file queue_pair.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <infiniband/mlx5dv.h>

#include "containers/free_list.hpp"
#include "memory/hip_allocator.hpp"

namespace rocshmem {

class GDADevice;

typedef union db_reg {
  uint64_t *ptr;
  uintptr_t uint;
} db_reg_t;

class QueuePair {
 public:
  friend GDADevice;

  /**
   * @brief Constructor.
   */
  explicit QueuePair(struct ibv_pd* pd);

  /**
   * @brief Create and enqueue a non-blocking put work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] pe Destination processing element of data transmission.
   */
  __device__ void put_nbi(void *dest, const void *source, size_t nelems, int pe);

  /**
   * @brief Create and enqueue a non-blocking put work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] pe Destination processing element of data transmission.
   */
  __device__ void put_nbi_wave(void *dest, const void *source, size_t nelems, int pe);

  /**
   * @brief Empty all completions from the completion queue.
   */
  __device__ void quiet();

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] atomic_op The atomic operation to perform.
   *
   * @return An atomic value
   */
  __device__ int64_t atomic_fetch(void *dest, int64_t value, int64_t cond, int pe, uint8_t atomic_op);

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] atomic_op The atomic operation to perform.
   */
  __device__ void atomic_nofetch(void *dest, int64_t value, int64_t cond, int pe, uint8_t atomic_op);

  char *const *base_heap{nullptr};

 private:
  /**
   * @brief Helper method to build work requests for the send queue.
   *
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] size Size in bytes of data transmission.
   * @param[in] laddr Local address.
   * @param[in] raddr Remote address.
   * @param[in] opcode Operation to be performed.
   */
  __device__ __attribute__((noinline)) uint64_t post_wqe_amo(int pe, int32_t size, uintptr_t *raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetch);

  /**
   * @brief Helper method to build work requests for the send queue.
   *
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] size Size in bytes of data transmission.
   * @param[in] laddr Local address.
   * @param[in] raddr Remote address.
   * @param[in] opcode Operation to be performed.
   */
  __device__ __attribute__((noinline)) void post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode);

  /**
   * @brief Helper method to ring the doorbell
   *
   * @param[in] db_val Doorbell value is written by method.
   */
  __device__ void ring_doorbell(uint64_t db_val, uint64_t my_sq_counter);

  db_reg_t db{};

  uint64_t cq_consumer{0};
  uint64_t quiet_posted{0};
  uint64_t quiet_active{0};
  uint64_t quiet_completed{0};

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
  mlx5_cqe64 *cq_buf{nullptr};
  volatile uint32_t *cq_dbrec{nullptr};
  uint32_t cq_cnt{0};
  uint32_t cq_log_cnt{0};

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
  volatile uint32_t *dbrec{nullptr};
  uint64_t *sq_buf{nullptr};
  uint16_t sq_wqe_cnt{0};
  uint64_t sq_posted{0};
  uint64_t sq_db_touched{0};
  uint64_t sq_sunk{0};

  static constexpr size_t OUTSTANDING_TABLE_SIZE = 65536;
  uint64_t outstanding_wqes[OUTSTANDING_TABLE_SIZE]{0};

  uint32_t qp_num{0};
  uint32_t rkey{0};
  uint32_t lkey{0};

  uint64_t* nonfetching_atomic{nullptr};
  uint32_t nonfetching_atomic_lkey{0};

  uint64_t* fetching_atomic{nullptr};
  uint32_t fetching_atomic_lkey{0};

  static const uint32_t FETCHING_ATOMIC_CNT{1024};
  static_assert(FETCHING_ATOMIC_CNT % __AMDGCN_WAVEFRONT_SIZE == 0);
  using FreeListT = FreeList<uint64_t*, HIPAllocator>;
  FreeListT* fetching_atomic_freelist{nullptr};

  HIPAllocator allocator{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP_
