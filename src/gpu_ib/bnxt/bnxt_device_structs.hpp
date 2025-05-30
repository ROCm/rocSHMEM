/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

// NOTE: Most of this code is from rdma-example/src/client.h
//       Should we provide a different copyright?
//       There was no license in the original header

#ifndef LIBRARY_SRC_GPU_IB_BNXT_BNXT_DEVICE_STRUCTS_HPP_
#define LIBRARY_SRC_GPU_IB_BNXT_BNXT_DEVICE_STRUCTS_HPP_

struct d_bnxt_spinlock {
  uint32_t lock; /* was pthread_spinlock_t lock; */
  int in_use;
  int need_lock;
};

struct d_bnxt_re_queue {
  struct d_bnxt_spinlock qlock;
  uint32_t flags;
  uint32_t *dbtail;
  void *va;
  uint32_t head;
  uint32_t depth; /* no. of entries */
  void *pad; /* to hold the padding area */
  uint32_t pad_stride_log2;
  uint32_t tail;
  uint32_t max_slots;
  /* Represents the difference between the real queue depth allocated in
   * HW and the user requested queue depth and is used to correctly flag
   * queue full condition based on user supplied queue depth.
   * This value can vary depending on the type of queue and any HW
   * requirements that mandate keeping a fixed gap between the producer
   * and the consumer indices in the queue
   */
  uint32_t diff;
  uint32_t stride;
  uint32_t msn;
  uint32_t msn_tbl_sz;
  /*
   * This flag is set when CQ is resized. It will be cleared after the
   * first CQE is received on the newly resized CQ
   */
  bool cq_resized;

  /* this tracks the 'head' index on the old CQ before resizing */
  uint32_t old_head;
};

struct d_bnxt_re_qp {
  uint8_t qptyp;
  uint16_t mtu;
  uint32_t sq_psn;
  uint32_t qpid;
  uint64_t *udpi_dbpage;
  struct d_bnxt_re_queue *sq;
};

struct thor2_control_block {
  struct d_bnxt_re_qp d_qp;
  struct d_bnxt_re_qp *qp;
  struct d_bnxt_re_queue *d_sq;
  int slots;
};

#endif  // LIBRARY_SRC_GPU_IB_BNXT_BNXT_DEVICE_STRUCTS_HPP_
