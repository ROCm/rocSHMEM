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
   * @param[in] db_ring Denotes whether send queue door bell should be rung.
   */
  __device__ void put_nbi(void *dest, const void *source, size_t nelems, int pe, bool db_ring);

  /**
   * @brief Create and enqueue a non-blocking put work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] source Source address for data transmission.
   * @param[in] nelems Size in bytes of data transmission.
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] db_ring Denotes whether send queue door bell should be rung.
   */
  __device__ void put_nbi_wave(void *dest, const void *source, size_t nelems, int pe, bool db_ring);

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
   * @param[in] db_ring Denotes whether send queue door bell should be rung.
   * @param[in] atomic_op The atomic operation to perform.
   *
   * @return An atomic value
   */
  __device__ int64_t atomic_fetch(void *dest, int64_t value, int64_t cond, int pe, bool db_ring, uint8_t atomic_op);

  /**
   * @brief Create and enqueue an atomic fetch work queue entry (wqe).
   *
   * @param[in] dest Destination address for data transmission.
   * @param[in] value Data value for the atomic operation.
   * @param[in] cond Used in atomic comparisons.
   * @param[in] pe Destination processing element of data transmission.
   * @param[in] db_ring Denotes whether send queue door bell should be rung.
   * @param[in] atomic_op The atomic operation to perform.
   */
  __device__ void atomic_nofetch(void *dest, int64_t value, int64_t cond, int pe, bool db_ring, uint8_t atomic_op);

  /**
   * @brief Helper method to set the doorbell's value.
   *
   * @param[in] val Desired value for the doorbell.
   */
  void setDBval(uint64_t val);

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
   * @param[in] ring_db Boolean denoting if doorbell should be rung.
   * @param[in] atomic_ret_pos Index into atomic return structure.
   */
  __device__ __attribute__((noinline)) void update_posted_wqe_generic(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode,
      int64_t atomic_data, int64_t atomic_cmp, bool ring_db, uint64_t atomic_ret_pos);

  /**
   * @brief Helper method to drain completion queue entries.
   */
  __device__ __attribute__((noinline)) void quiet_internal();

  /**
   * @brief Helper method to compute doorbell value opcode which is used to
   * ring the doorbell.
   *
   * @param[in,out] db_val
   * @param[in] dbrec_val
   * @param[in] opcode
   */
  __device__ void compute_db_val_opcode(uint64_t *db_val, uint16_t dbrec_val, uint8_t opcode);

  /**
   * @brief Helper method that sets the field in a work queue entry to
   * generate a completion entry in the completion queue.
   *
   * @param num_wqes Number of work entries this completion entry represents.
   */
  __device__ void set_completion_flag_on_wqe(int num_wqes);

  /**
   * @brief Helper method to update fields for the work queue entry.
   */
  __device__ void update_wqe_ce(int num_wqes);

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

  uint32_t cq_lock = 0;
  uint32_t sq_lock = 0;

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


  uint32_t ctrl_qp_sq{0};
  uint64_t ctrl_sig{0};
  uint32_t rkey{0};
  uint32_t lkey{0};

  uint64_t db_val{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP_
