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

void QueuePair::dump() {
  DPRINTF("\n"
         "===============================================\n"
         "           HOST DUMPING SHMEM INTERNAL QP\n"
         "===============================================\n"
         "  (char*const*)        base_heap           = %p\n"
         "  (uint32_t)           sq_counter          = %u\n"
         "  (uint32_t)           cq_consumer_counter = %u\n"
         "  (mlx5_cqe64*)        cq_buf_head         = %p\n"
         "  (mlx5_cqe64*)        cq_buf              = %p\n"
         "  (volatile uint32_t*) cq_dbrec            = %p\n"
         "  (uint32_t)           cq_cnt              = %u\n"
         "  (uint32_t)           cq_log_cnt          = 0x%x\n"
         "  (volatile uint32_t*) dbrec               = %p\n"
         "  (uint64_t*)          sq_buf              = %p\n"
         "  (uint64_t*)          sq_buf_head         = %p\n"
         "  (uint16_t)           sq_wqe_cnt          = %u\n"
         "  (uint32_t)           qp_num              = 0x%x\n"
         "  (uint32_t)           rkey                = 0x%x\n"
         "  (uint32_t)           lkey                = 0x%x\n",
         base_heap, sq_counter, cq_consumer_counter, cq_buf_head,
         cq_buf, cq_dbrec, cq_cnt, cq_log_cnt, dbrec, sq_buf, sq_buf_head, sq_wqe_cnt, qp_num, rkey, lkey);
}

__device__ void QueuePair::dump() {
  GPU_DPRINTF("\n"
	      "===============================================\n"
              "        DEVICE DUMPING SHMEM INTERNAL QP\n"
              "===============================================\n"
              "  (char*const*)        base_heap           = %p\n"
	      "  (uint32_t)           sq_counter          = %u\n"
	      "  (uint32_t)           cq_consumer_counter = %u\n"
	      "  (mlx5_cqe64*)        cq_buf_head         = %p\n"
	      "  (mlx5_cqe64*)        cq_buf              = %p\n"
	      "  (volatile uint32_t*) cq_dbrec            = %p\n"
	      "  (uint32_t)           cq_cnt              = %u\n"
	      "  (uint32_t)           cq_log_cnt          = 0x%x\n"
	      "  (volatile uint32_t*) dbrec               = %p\n"
	      "  (uint64_t*)          sq_buf              = %p\n"
	      "  (uint64_t*)          sq_buf_head         = %p\n"
	      "  (uint16_t)           sq_wqe_cnt          = %u\n"
	      "  (uint32_t)           qp_num              = 0x%x\n"
	      "  (uint32_t)           rkey                = 0x%x\n"
	      "  (uint32_t)           lkey                = 0x%x\n",
	      base_heap, sq_counter, cq_consumer_counter, cq_buf_head,
	      cq_buf, cq_dbrec, cq_cnt, cq_log_cnt, dbrec, sq_buf, sq_buf_head, sq_wqe_cnt, qp_num, rkey, lkey);
}

__device__ void QueuePair::ring_doorbell(uint64_t db_val, uint32_t my_sq_counter) {
  dump();

  swap_endian_store(const_cast<uint32_t*>(dbrec), my_sq_counter);
  __threadfence_system();

  uint8_t *db_u8p = reinterpret_cast<uint8_t*>(&db_val);
  GPU_DPRINTF("storing db_val %02x %02x %02x %02x %02x %02x %02x %02x (%lx) to db.ptr %p\n",
	      db_u8p[0], db_u8p[1], db_u8p[2], db_u8p[3], db_u8p[4], db_u8p[5], db_u8p[6], db_u8p[7], db_val, db.ptr);

  __hip_atomic_store(db.ptr, db_val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t db_uint = __hip_atomic_load(&db.uint, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  db_uint ^= 0x100;
  __hip_atomic_store(&db.uint, db_uint, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::quiet() {
  constexpr size_t BROADCAST_SIZE = 1024 / 64;
  constexpr uint64_t ALL_ONES_MASK = -1;
  __shared__ uint64_t cq_wave_broadcast[BROADCAST_SIZE];
  __shared__ uint32_t wqe_broadcast[BROADCAST_SIZE];
  __shared__ bool done_broadcast;

  uint64_t ballot = __ballot(1);
  uint8_t num_active_lanes = __popcll(ballot);
  uint8_t my_physical_lane_id = __lane_id();
  uint64_t lane_mask{ALL_ONES_MASK << my_physical_lane_id};
  uint64_t inverted_mask{~lane_mask};
  uint64_t lower_active_lanes{ballot & inverted_mask};
  uint8_t my_logical_lane_id = __popcll(lower_active_lanes);
  bool is_lowest_active_lane{my_logical_lane_id == 0};
  uint8_t wavefront_id = get_flat_block_id() / 64;

  cq_wave_broadcast[wavefront_id] = 0;
  wqe_broadcast[wavefront_id] = 0;
  done_broadcast = false;

  while (true) {
    if (is_lowest_active_lane) {
      done_broadcast = false;
      __threadfence_block();
    }

    bool done{false};
    uint64_t quiet_amount{0};
    uint32_t wave_cq_consumer_counter{0};
    do {
      if (is_lowest_active_lane) {
        gpu_dprintf("quiet_counter_hard %u quiet_counter_soft %u\n", quiet_counter_hard, quiet_counter_soft);
      }
      if (!__hip_atomic_load(&quiet_counter_hard, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT)) {
        return;
      }
      uint32_t quiet_val = __hip_atomic_load(&quiet_counter_soft, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      if (!quiet_val) {
        continue;
      }
      quiet_amount = min(num_active_lanes, quiet_val);
      if (is_lowest_active_lane) {
        done_broadcast = __hip_atomic_compare_exchange_strong(&quiet_counter_soft, &quiet_val, quiet_val - quiet_amount, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
        if (done_broadcast) {
          gpu_dprintf("succeeded on quiet_counter_soft CAS quiet_val %d quiet_amount %d\n", quiet_val, quiet_amount);
          wave_cq_consumer_counter = __hip_atomic_fetch_add(&cq_consumer_counter, quiet_amount, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
          cq_wave_broadcast[wavefront_id] = wave_cq_consumer_counter;
        }
        __threadfence_block();
      }
      done = done_broadcast;
    } while (!done);
    wave_cq_consumer_counter = cq_wave_broadcast[wavefront_id];
    uint64_t my_cq_consumer_counter = wave_cq_consumer_counter + my_logical_lane_id;
    uint64_t my_cq_index = my_cq_consumer_counter % cq_cnt;

    if (my_logical_lane_id < quiet_amount) {
      volatile mlx5_cqe64 *cqe_entry = &cq_buf[my_cq_index];
      const volatile uint8_t *d = reinterpret_cast<const volatile uint8_t*>(cqe_entry);
      bool vote_failed{true};
      uint16_t be_wqe_counter{0};
      uint8_t op_own{0};
      uint8_t owner_bit = (my_cq_consumer_counter >> cq_log_cnt) & 1;
      do {
        gpu_dprintf(
         "Observing CQE at address %p at index %u\n"
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x "
         "%02x %02x %02x %02x %02x %02x %02x %02x\n",
         cqe_entry, my_cq_index,
          d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],
          d[8],  d[9], d[10], d[11], d[12], d[13], d[14], d[15],
         d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
         d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
         d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
         d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
         d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
         d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);

        op_own = __hip_atomic_load(&cqe_entry->op_own, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
	bool my_ownership_vote = (op_own & 1) == owner_bit;
        bool my_opcode_vote = (op_own >> 4) != MLX5_CQE_INVALID;
        uint64_t votes = __ballot(my_ownership_vote && my_opcode_vote);
        vote_failed = __popcll(votes) < quiet_amount;
        if (!vote_failed) {
	  be_wqe_counter = __hip_atomic_load(&cqe_entry->wqe_counter, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
	}
      } while (vote_failed);

      uint16_t wqe_counter;
      swap_endian_store(const_cast<uint16_t*>(&wqe_counter), reinterpret_cast<uint16_t>(be_wqe_counter));
      uint32_t wqe_id =  outstanding_wqes[wqe_counter];
      __hip_atomic_fetch_max(&wqe_broadcast[wavefront_id], wqe_id, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_WORKGROUP);
      uint8_t mlx5_invld_bits = MLX5_CQE_INVALID << 4 | owner_bit;
      __hip_atomic_store(&cqe_entry->op_own, mlx5_invld_bits, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
      GPU_DPRINTF(
       "Clearing CQE at address %p at index %lu\n"
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x "
       "%02x %02x %02x %02x %02x %02x %02x %02x\n",
       cqe_entry, my_cq_index,
        d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],
        d[8],  d[9], d[10], d[11], d[12], d[13], d[14], d[15],
       d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
       d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
       d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
       d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
       d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
       d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
    }
    if (is_lowest_active_lane) {
      swap_endian_store(const_cast<uint32_t*>(cq_dbrec), cq_consumer_counter);
      __threadfence_system();

      uint32_t sunk_wqe_id = wqe_broadcast[wavefront_id];
      __hip_atomic_store(&sq_counter_sunk, sunk_wqe_id, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&quiet_counter_hard, -quiet_amount, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  constexpr size_t SQ_BROADCAST_SIZE = 1024 / 64;
  constexpr uint64_t ALL_ONES_MASK = -1;
  __shared__ uint64_t sq_wave_broadcast[SQ_BROADCAST_SIZE];
  uint64_t ballot = __ballot(1);
  uint8_t num_active_lanes = __popcll(ballot);
  uint8_t my_physical_lane_id = __lane_id();
  uint64_t lane_mask{ALL_ONES_MASK << my_physical_lane_id};
  uint64_t inverted_mask{~lane_mask};
  uint64_t lower_active_lanes{ballot & inverted_mask};
  uint8_t my_logical_lane_id = __popcll(lower_active_lanes);
  bool is_lowest_active_lane{my_logical_lane_id == 0};
  uint8_t num_wqes{num_active_lanes};
  uint64_t wave_sq_counter{0};

  if (is_lowest_active_lane) {
    wave_sq_counter = __hip_atomic_fetch_add(&sq_counter, num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  }
  uint8_t wavefront_id = get_flat_block_id() / 64;
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
    uint64_t num_free_entries = sq_wqe_cnt - num_active_sq_entries;
    uint64_t num_entries_until_wave_last_entry = wave_sq_counter + num_active_lanes - sunk;
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
  const uint8_t *d = reinterpret_cast<const uint8_t*>(&base_ptr[16 * 4 * my_sq_index]);
  GPU_DPRINTF(
   "WQE post to address %p at index %lu\n"
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x\n",
   sq_buf, my_sq_index,
    d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],
    d[8],  d[9], d[10], d[11], d[12], d[13], d[14], d[15],
   d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
   d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
   d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
   d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
   d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
   d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);

  if (is_lowest_active_lane) {
    uint64_t posted {0};
    do {
      posted = __hip_atomic_load(&sq_counter_db_posted, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    } while (posted != wave_sq_counter);

    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * ((wave_sq_counter + num_wqes - 1) % sq_wqe_cnt)]);

    uint8_t *db_u8p = reinterpret_cast<uint8_t*>(ctrl_wqe_8B_for_db);
    GPU_DPRINTF("post_wqe_rma::ctrl_wqe_8B_for_db %02x %02x %02x %02x %02x %02x %02x %02x (%lx)\n",
                 db_u8p[0], db_u8p[1], db_u8p[2], db_u8p[3], db_u8p[4], db_u8p[5], db_u8p[6], db_u8p[7], *ctrl_wqe_8B_for_db);

    GPU_DPRINTF("ringing doorbell for sq_counter_db_posted %d\n", posted);
    ring_doorbell(*ctrl_wqe_8B_for_db, wave_sq_counter + num_wqes);

    __hip_atomic_fetch_add(&quiet_counter_soft, num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_fetch_add(&quiet_counter_hard, num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
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
