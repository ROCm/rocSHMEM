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

#include "queue_pair.hpp"

#include <hip/hip_runtime.h>

#include "backend_ib.hpp"
#include "endian.hpp"
#include "segment_builder.hpp"
#include "util.hpp"

namespace rocshmem {

QueuePair::QueuePair(GPUIBBackend *backend) {
  atomic_ret.atomic_lkey = backend->networkImpl.atomic_ret->atomic_lkey;
  atomic_ret.atomic_counter = 0;
}

__device__ uint8_t QueuePair::get_cq_error_syndrome(mlx5_cqe64 *cqe_entry) {
  mlx5_err_cqe *cqe_err = reinterpret_cast<mlx5_err_cqe*>(cqe_entry);
  return cqe_err->syndrome;
}

__device__ void QueuePair::ring_doorbell(uint64_t db_val, uint32_t my_sq_counter) {
  swap_endian_store(const_cast<uint32_t*>(dbrec), my_sq_counter);
  __threadfence_system();

  __hip_atomic_store(db.ptr, db_val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t db_uint = __hip_atomic_load(&db.uint, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  db_uint ^= 0x100;
  __hip_atomic_store(&db.uint, db_uint, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::quiet() {
  constexpr size_t BROADCAST_SIZE = 1024 / __AMDGCN_WAVEFRONT_SIZE;
  __shared__ uint32_t wqe_broadcast[BROADCAST_SIZE];
  uint8_t wavefront_id = get_flat_block_id() / __AMDGCN_WAVEFRONT_SIZE;
  wqe_broadcast[wavefront_id] = 0;

  uint64_t activemask = __ballot(1);
  uint8_t num_active_lanes = __popcll(activemask);
  uint8_t my_logical_lane_id = __popcll(activemask & __lanemask_lt());
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = __ffsll((unsigned long long)activemask) - 1;

  while (true) {
    bool done{false};
    uint64_t quiet_amount{0};
    uint32_t wave_cq_consumer{0};
    do {
      uint32_t posted = __hip_atomic_load(&quiet_posted, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
      uint32_t active = __hip_atomic_load(&quiet_active, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
      uint32_t completed = __hip_atomic_load(&quiet_completed, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
      if (!(posted - completed)) {
        return;
      }
      uint64_t quiet_val = posted - active;
      if (!quiet_val) {
        continue;
      }
      quiet_amount = min(num_active_lanes, quiet_val);
      if (is_leader) {
        done = __hip_atomic_compare_exchange_strong(&quiet_active, &active, active + quiet_amount, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
        if (done) {
          wave_cq_consumer = __hip_atomic_fetch_add(&cq_consumer, quiet_amount, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      done = __shfl(done, leader_phys_lane_id);
    } while (!done);
    wave_cq_consumer = __shfl(wave_cq_consumer, leader_phys_lane_id);
    uint64_t my_cq_consumer = wave_cq_consumer + my_logical_lane_id;
    uint64_t my_cq_index = my_cq_consumer % cq_cnt;

    if (my_logical_lane_id < quiet_amount) {
      volatile mlx5_cqe64 *cqe_entry = &cq_buf[my_cq_index];
      bool vote_failed{true};
      uint16_t be_wqe_counter{0};
      uint8_t op_own{0};
      uint8_t owner_bit = (my_cq_consumer >> cq_log_cnt) & 1;
      do {
        op_own = *((volatile uint8_t*)&cqe_entry->op_own);
	bool my_ownership_vote = (op_own & 1) == owner_bit;
        bool my_opcode_vote = (op_own >> 4) != MLX5_CQE_INVALID;
        uint64_t votes = __ballot(my_ownership_vote && my_opcode_vote);
        vote_failed = __popcll(votes) < quiet_amount;
        if (!vote_failed) {
          be_wqe_counter = *((volatile uint16_t*)&cqe_entry->wqe_counter);
	}
      } while (vote_failed);

      uint16_t wqe_counter;
      swap_endian_store(const_cast<uint16_t*>(&wqe_counter), reinterpret_cast<uint16_t>(be_wqe_counter));
      uint32_t wqe_id =  outstanding_wqes[wqe_counter];
      __hip_atomic_fetch_max(&wqe_broadcast[wavefront_id], wqe_id, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_WORKGROUP);
      uint8_t mlx5_invld_bits = MLX5_CQE_INVALID << 4 | owner_bit;
      *((volatile uint8_t*)&cqe_entry->op_own) = mlx5_invld_bits;
      __threadfence_system();
    }
    if (is_leader) {
      uint64_t posted {0};
      do {
        posted = __hip_atomic_load(&quiet_completed, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
      } while (posted != wave_cq_consumer);

      swap_endian_store(const_cast<uint32_t*>(cq_dbrec), (uint32_t)(wave_cq_consumer + quiet_amount));
      __threadfence_system();

      uint32_t sunk_wqe_id = wqe_broadcast[wavefront_id];
      __hip_atomic_store(&sq_counter_sunk, sunk_wqe_id, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&quiet_completed, quiet_amount, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  constexpr size_t SQ_BROADCAST_SIZE = 1024 / __AMDGCN_WAVEFRONT_SIZE;
  constexpr uint64_t ALL_ONES_MASK = -1;
  __shared__ uint64_t sq_wave_broadcast[SQ_BROADCAST_SIZE];
  uint64_t active_thread_mask = __ballot(1);
  uint8_t num_active_lanes = __popcll(active_thread_mask);
  uint8_t my_physical_lane_id = __lane_id();
  uint64_t lane_mask{ALL_ONES_MASK << my_physical_lane_id};
  uint64_t inverted_mask{~lane_mask};
  uint64_t lower_active_lanes{active_thread_mask & inverted_mask};
  uint8_t my_logical_lane_id = __popcll(lower_active_lanes);
  bool is_lowest_active_lane{my_logical_lane_id == 0};
  uint8_t num_wqes{num_active_lanes};
  uint64_t wave_sq_counter{0};

  if (is_lowest_active_lane) {
    wave_sq_counter = __hip_atomic_fetch_add(&sq_counter, num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  }
  uint8_t wavefront_id = get_flat_block_id() / __AMDGCN_WAVEFRONT_SIZE;
  if (is_lowest_active_lane) {
    sq_wave_broadcast[wavefront_id] = wave_sq_counter;
    __threadfence_block();
  }
  wave_sq_counter = sq_wave_broadcast[wavefront_id];
  uint64_t my_sq_counter = wave_sq_counter + my_logical_lane_id;
  uint64_t my_sq_index = my_sq_counter % sq_wqe_cnt;

  do {
    uint64_t posted = __hip_atomic_load(&sq_counter_db_posted, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t sunk = __hip_atomic_load(&sq_counter_sunk, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = posted - sunk;
    uint64_t num_free_entries = min(sq_wqe_cnt, cq_cnt) - num_active_sq_entries;
    uint64_t num_entries_until_wave_last_entry = wave_sq_counter + num_active_lanes - posted;
    if (num_free_entries > num_entries_until_wave_last_entry) {
      break;
    }
    quiet();
  } while (true);

  outstanding_wqes[my_sq_counter % 65536] = my_sq_counter;

  SegmentBuilder seg_build(my_sq_index, sq_buf);
  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num, MLX5_WQE_CTRL_CQ_UPDATE, 3, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);
  seg_build.update_data_seg(laddr, size, lkey);
  __threadfence_system();

  uint8_t *base_ptr = reinterpret_cast<uint8_t*>(sq_buf);
  if (is_lowest_active_lane) {
    uint64_t posted {0};
    do {
      posted = __hip_atomic_load(&sq_counter_db_posted, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    } while (posted != wave_sq_counter);

    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * ((wave_sq_counter + num_wqes - 1) % sq_wqe_cnt)]);
    ring_doorbell(*ctrl_wqe_8B_for_db, wave_sq_counter + num_wqes);

    __hip_atomic_fetch_add(&quiet_posted, num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&sq_counter_db_posted, wave_sq_counter + num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ void QueuePair::post_wqe_amo(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode,
                                                 int64_t atomic_data, int64_t atomic_cmp, uint64_t atomic_ret_pos) {
  uint32_t num_wqes = 1;

  uint64_t my_sq_counter = atomicAdd(&sq_counter, num_wqes);
  uint64_t my_sq_index = my_sq_counter % sq_wqe_cnt;

  uint32_t lkey_in_stack_frame = lkey;
  uint32_t rkey_in_stack_frame = rkey;

  SegmentBuilder seg_build(my_sq_index, sq_buf);
//  seg_build.update_ctrl_seg(opcode, my_sq_counter, ctrl_qp_sq_in_stack_frame, ctrl_sig_in_stack_frame);
//  seg_build.update_raddr_seg(raddr, rkey_in_stack_frame);

//  if (opcode == MLX5_OPCODE_ATOMIC_FA || opcode == MLX5_OPCODE_ATOMIC_CS) {
//    seg_build.update_atomic_seg(atomic_data, atomic_cmp);
//    size = 8;
//    lkey_in_stack_frame = atomic_ret.atomic_lkey;
//    laddr = &atomic_ret.atomic_base_ptr[atomic_ret_pos];
//  }

//  seg_build.update_data_seg(laddr, size, lkey_in_stack_frame);

//  uint16_t be_sq_counter;
//  uint16_t sq_counter_u16 = my_sq_counter;
//  swap_endian_store(&be_sq_counter, sq_counter_u16);

//  if (ring_db) {
//    uint64_t db_val = sq_buf[8 * ((be_sq_counter - num_wqes) % sq_wqe_cnt)];
//    update_wqe_ce(num_wqes);
//    ring_doorbell(db_val);
//  }
}

/******************************************************************************
 ****************************** SHMEM INTERFACE *******************************
 *****************************************************************************/
__device__ void QueuePair::put_nbi(void *dest, const void *source, size_t nelems, int pe) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, src, dst, MLX5_OPCODE_RDMA_WRITE);
}

__device__ void QueuePair::put_nbi_wave(void *dest, const void *source, size_t nelems, int pe) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, src, dst, MLX5_OPCODE_RDMA_WRITE);
}

__device__ int64_t QueuePair::atomic_fetch(void *dest, int64_t value, int64_t cond, int pe, uint8_t atomic_op) {
  uint64_t pos = atomicAdd(&atomic_ret.atomic_counter, 1);
  pos = pos % max_nb_atomic;
  int64_t *atomic_base_ptr = reinterpret_cast<int64_t*>(atomic_ret.atomic_base_ptr);
  int64_t *load_address = &atomic_base_ptr[pos];
  *load_address = -100;
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_amo(pe, sizeof(int64_t), nullptr, dst, atomic_op, value, cond, pos);
  quiet();
  while (uncached_load(load_address) == -100) { }
  int64_t ret = *load_address;
  __threadfence();
  return ret;
}

__device__ void QueuePair::atomic_nofetch(void *dest, int64_t value, int64_t cond, int pe, uint8_t atomic_op) {
  uint64_t pos = atomicAdd(&atomic_ret.atomic_counter, 1);
  pos = pos % max_nb_atomic;
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_amo(pe, sizeof(int64_t), nullptr, dst, atomic_op, value, cond, pos);
  quiet();
}

}  // namespace rocshmem
