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

__device__ void QueuePair::ring_doorbell(uint64_t db_val) {
  swap_endian_store(const_cast<uint32_t*>(sq_dbrec), reinterpret_cast<uint32_t>(sq_counter));
  STORE(db.ptr, db_val);
  db.uint ^= 256;
}

__device__ void QueuePair::set_completion_flag_on_wqe(int num_wqes) {
  uint64_t *wqe = &sq_buf[8 * ((sq_counter - num_wqes) % sq_wqe_cnt)];
  uint8_t *wqe_ce = reinterpret_cast<uint8_t*>(wqe) + 11;
  *wqe_ce = 8;
}

__device__ void QueuePair::update_wqe_ce(int num_wqes) {
  set_completion_flag_on_wqe(num_wqes);
  atomicAdd(&quiet_counter, 1);
}

__device__ void QueuePair::compute_db_val_opcode(uint64_t *db_val, uint16_t dbrec_val, uint8_t opcode) {
  uint64_t opcode64 = opcode;
  opcode64 = opcode64 << 24 & 0x000000FFFF000000;
  uint64_t dbrec = dbrec_val << 8;
  dbrec = dbrec & 0x0000000000FFFF00;
  uint64_t val = *db_val;
  val = val & 0xFFFFFFFFFF0000FF;
  *db_val = val | dbrec | opcode64;
}

__device__ void QueuePair::quiet_internal() {
  uint32_t quiet_val = quiet_counter;
  if (!quiet_val) {
    return;
  }

  cq_consumer_counter = cq_consumer_counter + quiet_val - 1;
  uint32_t index = (cq_consumer_counter % cq_cnt);
  mlx5_cqe64 *cqe_entry = &cq_buf[index];

  int val_ld = uncached_load_ubyte(&(cqe_entry->op_own));
  uint8_t val_op_own = val_ld;

  while (!((val_op_own & 0x1) == ((cq_consumer_counter >> cq_log_cnt) & 1)) || ((val_op_own) >> 4) == 0xF) {
    val_ld = uncached_load_ubyte(&(cqe_entry->op_own));
    val_op_own = val_ld;
  }

  uint8_t opcode = val_op_own >> 4;
  if (opcode != 0) {
    uint8_t syndrome = get_cq_error_syndrome(cqe_entry);
    mlx5_err_cqe *cqe_err = reinterpret_cast<mlx5_err_cqe*>(cqe_entry);
    GPU_DPRINTF("QUIET ERROR: signature %d opcode_qpn %llx wqe_cnt %llx \n", syndrome, cqe_err->s_wqe_opcode_qpn, cqe_err->wqe_counter);
  }

  quiet_counter -= quiet_val;

  cq_consumer_counter++;
  swap_endian_store(const_cast<uint32_t*>(cq_dbrec), cq_consumer_counter);
}

__device__ void QueuePair::quiet_single() {
  int thread_id = get_flat_block_id();
  if (thread_id % WF_SIZE == 0) {
    quiet_internal();
    __threadfence();
  }
}

__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode, bool ring_db) {

  // determine active threads
  // reserve space for all active threads in sq
  // generate per-thread index modulo sq.wqe_cnt
  // wait for cq to drain enough for space to be available in sq
  // write segments at per-thread index
  // threadfence_system
  // protect sq.dbrec by updating with cmpswp when sq.dbrec with condition of value equals starting location and swap in that location + offset
  // in one thread:
  //   ring blue-flame doorbell


  // need sq_counter
  // need cq_counter
  // will signal every completion

  uint32_t num_wqes = 1;

  uint64_t my_sq_counter = atomicAdd(&sq_counter, num_wqes);
  uint64_t my_sq_index = my_sq_counter % sq_wqe_cnt;

  SegmentBuilder seg_build(my_sq_index, sq_buf);
//  seg_build.update_ctrl_seg(opcode, my_sq_counter, ctrl_qp_sq_in_stack_frame, ctrl_sig_in_stack_frame);
//  seg_build.update_raddr_seg(raddr, rkey);
//  seg_build.update_data_seg(laddr, size, lkey);

//  uint16_t be_sq_counter;
//  uint16_t sq_counter_u16 = my_sq_counter;
//  swap_endian_store(&be_sq_counter, sq_counter_u16);

//  if (ring_db) {
//    uint64_t db_val = sq_buf[8 * ((be_sq_counter - num_wqes) % sq_wqe_cnt)];
//    update_wqe_ce(num_wqes);
//    ring_doorbell(db_val);
//  }
}

__device__ void QueuePair::post_wqe_amo(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode,
                                                 int64_t atomic_data, int64_t atomic_cmp, bool ring_db, uint64_t atomic_ret_pos) {
  uint32_t num_wqes = 1;

  uint64_t my_sq_counter = atomicAdd(&sq_counter, num_wqes);
  uint64_t my_sq_index = my_sq_counter % sq_wqe_cnt;

  uint32_t lkey_in_stack_frame = lkey;
  uint32_t rkey_in_stack_frame = rkey;
  uint32_t ctrl_qp_sq_in_stack_frame = ctrl_qp_sq;
  uint64_t ctrl_sig_in_stack_frame = ctrl_sig;

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
__device__ void QueuePair::put_nbi(void *dest, const void *source, size_t nelems, int pe, bool db_ring) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, src, dst, MLX5_OPCODE_RDMA_WRITE, db_ring);
}

__device__ void QueuePair::put_nbi_wave(void *dest, const void *source, size_t nelems, int pe, bool db_ring) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, src, dst, MLX5_OPCODE_RDMA_WRITE, db_ring);
}

__device__ int64_t QueuePair::atomic_fetch(void *dest, int64_t value, int64_t cond, int pe, bool db_ring, uint8_t atomic_op) {
  uint64_t pos = atomicAdd(&atomic_ret.atomic_counter, 1);
  pos = pos % max_nb_atomic;
  int64_t *atomic_base_ptr = reinterpret_cast<int64_t*>(atomic_ret.atomic_base_ptr);
  int64_t *load_address = &atomic_base_ptr[pos];
  *load_address = -100;
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_amo(pe, sizeof(int64_t), nullptr, dst, atomic_op, value, cond, db_ring, pos);
  quiet_single();
  while (uncached_load(load_address) == -100) { }
  int64_t ret = *load_address;
  __threadfence();
  return ret;
}

__device__ void QueuePair::atomic_nofetch(void *dest, int64_t value, int64_t cond, int pe, bool db_ring, uint8_t atomic_op) {
  uint64_t pos = atomicAdd(&atomic_ret.atomic_counter, 1);
  pos = pos % max_nb_atomic;
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_amo(pe, sizeof(int64_t), nullptr, dst, atomic_op, value, cond, db_ring, pos);
  quiet_single();
}

__device__ void QueuePair::waitCQSpace(int num_msgs) {
  if ((quiet_counter + num_msgs) >= cq_cnt) {
    quiet_single();
  }
}

__device__ void QueuePair::waitSQSpace(int num_msgs) {
  local_sq_cnt += num_msgs;
  int div = local_sq_cnt / sq_wqe_cnt;
  if (div > 0) {
    quiet_single();
    local_sq_cnt = local_sq_cnt % sq_wqe_cnt;
  }
}

void QueuePair::setDBval(uint64_t val) {
  db_val = val;
}

}  // namespace rocshmem
