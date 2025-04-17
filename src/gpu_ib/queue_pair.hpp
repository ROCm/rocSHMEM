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

#include "atomic_return.hpp"

namespace rocshmem {

class GPUIBBackend;
class Connection;

typedef union db_reg {
  uint64_t *ptr;
  uintptr_t uint;
} db_reg_t;

class QueuePair {
 public:
  friend Connection;

  /**
   * @brief Constructor.
   *
   * @param[in] backend GPUIBBackend needed for member access.
   */
  explicit QueuePair(GPUIBBackend *backend);

  /**
   * @brief Inspect completion queue and possibly wait for free space.
   *
   * @param[in] num_msgs Number of entries needing space in completion queue.
   */
  __device__ void waitCQSpace(int num_msgs);

  /**
   * @brief Inspect send queue and possibly wait for free space.
   *
   * @param[in] num_msgs Number of entries needing space in send queue.
   */
  __device__ void waitSQSpace(int num_msgs);

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
   * @brief Consume a completion queue entry from this queue pair's
   * completion queue.
   */
  __device__ void quiet_single();

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

  atomic_ret_t atomic_ret{};

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
   * @param[in] atomic_data An atomic data value to be used.
   * @param[in] atomic_cmp An atomic comparison operation to be performed.
   * @param[in] atomic_ret_pos Index into atomic return structure.
   */
  __device__ __attribute__((noinline)) void post_wqe_amo(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode,
                                                         int64_t atomic_data, int64_t atomic_cmp, uint64_t atomic_ret_pos);

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
   * @brief Helper method to drain completion queue entries.
   */
  __device__ __attribute__((noinline)) void quiet_internal();

  /**
   * @brief Helper method to ring the doorbell
   *
   * @param[in] db_val Doorbell value is written by method.
   */
  __device__ void ring_doorbell(uint64_t db_val);

  /**
   * @brief Helper method to extract syndrome field from cqe.
   *
   * @param[in] cq_entry Completion queue entry.
   */
  __device__ uint8_t get_cq_error_syndrome(mlx5_cqe64 *cq_entry);

  db_reg_t db{};

  uint32_t sq_counter{0};
  uint32_t local_sq_cnt{0};
  uint32_t cq_consumer_counter{0};
  uint32_t quiet_counter{0};

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
  mlx5_cqe64 *cq_buf_head{nullptr};
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
  volatile uint32_t *sq_dbrec{nullptr};
  uint64_t *sq_buf{nullptr};
  uint64_t *sq_buf_head{nullptr};
  uint16_t sq_wqe_cnt{0};

  uint32_t qp_num{0};
  uint32_t rkey{0};
  uint32_t lkey{0};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP_
