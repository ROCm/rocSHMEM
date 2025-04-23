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
  doorbell_mutex = backend->mutex_dobj_.get();
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
         "  (uint32_t)           local_sq_cnt        = %u\n"
         "  (uint32_t)           cq_consumer_counter = %u\n"
         "  (uint32_t)           quiet_counter       = %u\n"
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
         base_heap, sq_counter, local_sq_cnt, cq_consumer_counter, quiet_counter, cq_buf_head,
         cq_buf, cq_dbrec, cq_cnt, cq_log_cnt, dbrec, sq_buf, sq_buf_head, sq_wqe_cnt, qp_num, rkey, lkey);
}

__device__ void QueuePair::dump() {
  GPU_DPRINTF("\n"
	      "===============================================\n"
              "        DEVICE DUMPING SHMEM INTERNAL QP\n"
              "===============================================\n"
              "  (char*const*)        base_heap           = %p\n"
	      "  (uint32_t)           sq_counter          = %u\n"
	      "  (uint32_t)           local_sq_cnt        = %u\n"
	      "  (uint32_t)           cq_consumer_counter = %u\n"
	      "  (uint32_t)           quiet_counter       = %u\n"
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
	      base_heap, sq_counter, local_sq_cnt, cq_consumer_counter, quiet_counter, cq_buf_head,
	      cq_buf, cq_dbrec, cq_cnt, cq_log_cnt, dbrec, sq_buf, sq_buf_head, sq_wqe_cnt, qp_num, rkey, lkey);
}

__device__ void QueuePair::ring_doorbell(uint64_t db_val, uint32_t my_sq_counter) {
  dump();
  __threadfence_system();
  uint32_t be_sq_counter;
  swap_endian_store(const_cast<uint32_t*>(&be_sq_counter), reinterpret_cast<uint32_t>(my_sq_counter));
  uint8_t *dbrec_u8p = reinterpret_cast<uint8_t*>(&be_sq_counter);
  GPU_DPRINTF("storing (__be32) be_sq_counter %02x %02x %02x %02x (%lx) to dbrec %p\n", dbrec_u8p[0], dbrec_u8p[1], dbrec_u8p[2], dbrec_u8p[3], be_sq_counter, dbrec);
  __hip_atomic_store(dbrec, be_sq_counter, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_load(dbrec, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  __threadfence_system();
  uint8_t *db_u8p = reinterpret_cast<uint8_t*>(&db_val);
  GPU_DPRINTF("storing db_val %02x %02x %02x %02x %02x %02x %02x %02x (%lx) to db.ptr %p\n",
	      db_u8p[0], db_u8p[1], db_u8p[2], db_u8p[3], db_u8p[4], db_u8p[5], db_u8p[6], db_u8p[7], db_val, db.ptr);
  STORE(db.ptr, db_val);
  db.uint ^= 0x100;
}

__device__ void QueuePair::quiet_internal() {
  uint32_t quiet_val = quiet_counter;
  if (!quiet_val) {
    return;
  }

  cq_consumer_counter = cq_consumer_counter + quiet_val - 1;
  uint32_t index = (cq_consumer_counter % cq_cnt);
  mlx5_cqe64 *cqe_entry = &cq_buf[index];

  const uint8_t *d = reinterpret_cast<const uint8_t*>(cqe_entry);
  GPU_DPRINTF(
   "Observing CQE at address %p at index %lu\n"
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x "
   "%02x %02x %02x %02x %02x %02x %02x %02x\n",
   cqe_entry, index,
    d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],
    d[8],  d[9], d[10], d[11], d[12], d[13], d[14], d[15],
   d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
   d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
   d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
   d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
   d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
   d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);

  while ((*((volatile uint8_t*)&cqe_entry->op_own) >> 4) == MLX5_CQE_INVALID) {
    GPU_DPRINTF(
     "Observing CQE at address %p at index %u\n"
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x "
     "%02x %02x %02x %02x %02x %02x %02x %02x\n",
     cqe_entry, index,
      d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],
      d[8],  d[9], d[10], d[11], d[12], d[13], d[14], d[15],
     d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
     d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
     d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
     d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
     d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
     d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  }

  if ((*((volatile uint8_t*)&cqe_entry->op_own) >> 4) != 0) {
    uint8_t syndrome = get_cq_error_syndrome(cqe_entry);
    mlx5_err_cqe *cqe_err = reinterpret_cast<mlx5_err_cqe*>(cqe_entry);
    printf("QUIET ERROR: signature %d opcode_qpn %x wqe_cnt %hx \n", syndrome, cqe_err->s_wqe_opcode_qpn, cqe_err->wqe_counter);
  }

  *((volatile uint8_t*)&cqe_entry->op_own) = (uint8_t)0xF0;

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
   cqe_entry, index,
    d[0],  d[1],  d[2],  d[3],  d[4],  d[5],  d[6],  d[7],
    d[8],  d[9], d[10], d[11], d[12], d[13], d[14], d[15],
   d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
   d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
   d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
   d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
   d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
   d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);

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

__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  // determine active threads
  // reserve space for all active threads in sq
  // generate per-thread index modulo sq.wqe_cnt
  // wait for cq to drain enough for space to be available in sq
  // write segments at per-thread index
  // threadfence_system
  // protect dbrec by updating with cmpswp when dbrec with condition of value equals starting location and swap in that location + offset
  // threadfence_system
  // in one thread:
  //   ring blue-flame doorbell


  // need sq_counter
  // need cq_counter
  // will signal every completion

  uint32_t num_wqes{1};
  uint32_t my_sq_counter = atomicAdd(&sq_counter, num_wqes);
  uint32_t my_sq_index = my_sq_counter % sq_wqe_cnt;
  atomicAdd(&quiet_counter, 1);

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

  uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * my_sq_index]);
  uint8_t *db_u8p = reinterpret_cast<uint8_t*>(ctrl_wqe_8B_for_db);
  GPU_DPRINTF("post_wqe_rma::ctrl_wqe_8B_for_db %02x %02x %02x %02x %02x %02x %02x %02x (%lx)\n",
	      db_u8p[0], db_u8p[1], db_u8p[2], db_u8p[3], db_u8p[4], db_u8p[5], db_u8p[6], db_u8p[7], *ctrl_wqe_8B_for_db);

  uint64_t ticket{0};
//  ticket = doorbell_mutex->lock();
  ring_doorbell(*ctrl_wqe_8B_for_db, my_sq_counter + num_wqes);
//  doorbell_mutex->unlock(ticket);
  __threadfence_system();

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
  quiet_single();
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

}  // namespace rocshmem
