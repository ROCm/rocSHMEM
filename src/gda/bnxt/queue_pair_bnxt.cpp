/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "gda/queue_pair.hpp"
#include "util.hpp"

namespace rocshmem {

static const __device__ char bnxt_re_wc_error_strings[12][14] = {
  "OK",
  "BAD_RESP",
  "LOC_LEN",
  "LOC_QP_OP",
  "PROT",
  "MEM_OP",
  "REM_INVAL",
  "REM_ACC",
  "REM_OP",
  "RNR_NAK_XCED",
  "TRNSP_XCED",
  "WR_FLUSH",
};

__device__ static inline void bnxt_re_init_db_hdr(struct bnxt_re_db_hdr *hdr,
                                                  uint32_t indx, uint32_t toggle,
                                                  uint32_t qid, uint32_t typ) {
  uint64_t key_lo;
  uint64_t key_hi;

  key_lo = (indx | toggle);

  key_hi = (qid & BNXT_RE_DB_QID_MASK)
         | ((typ & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT)
         | (0x1UL << BNXT_RE_DB_VALID_SHIFT);

  hdr->typ_qid_indx = (key_lo | (key_hi << 32));
}

__device__ static inline struct bnxt_re_msns* bnxt_re_pull_psn_buff(struct bnxt_device_sq *sq) {
  return (struct bnxt_re_msns*)(((char *) sq->msntbl) + ((sq->msn) << sq->psn_sz_log2));
}

__device__ static inline uint64_t bnxt_re_update_msn_tbl(uint32_t st_idx, uint32_t npsn,
                                                         uint32_t start_psn) {
   return ((((uint64_t)(st_idx) << BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK) |
                       (((uint64_t)(npsn) << BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK) |
                       (((start_psn) << BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK));
}

__device__ static inline void bnxt_re_fill_psns_for_msntbl(struct bnxt_device_sq *sq,
                                                           uint32_t msg_len) {
   uint32_t npsn = 0, start_psn = 0, next_psn = 0;
   struct bnxt_re_msns msns;
   uint64_t *msns_ptr;
   uint32_t pkt_cnt = 0;
   /* Start slot index of the WQE */
   uint32_t st_idx = sq->tail; // * BNXT_RE_STATIC_WQE_SIZE_SLOTS; Do we need this?
   // Get the MSN table address
   msns_ptr = (uint64_t *)bnxt_re_pull_psn_buff(sq);
   // Start PSN is the last recorded PSN
   // Calculate the packet count based on the len of the WQE/MTU
   msns.start_idx_next_psn_start_psn = 0;
   start_psn = sq->psn;
   pkt_cnt = (msg_len / sq->mtu);

   if (msg_len % sq->mtu)
       pkt_cnt++;

   /* Increment the psn even for 0 len packets
    * e.g. for opcode rdma-write-with-imm-data
    * with length field = 0
    */
   if (msg_len == 0)
       pkt_cnt = 1;

   /* make it 24 bit */
   next_psn = sq->psn + pkt_cnt;
   npsn = next_psn;
   sq->psn = next_psn;
   msns.start_idx_next_psn_start_psn |= bnxt_re_update_msn_tbl(st_idx, npsn, start_psn);
   sq->msn++;
   sq->msn %= sq->msn_tbl_sz;

   memcpy(msns_ptr, &msns, sizeof(uint64_t));
}

__device__ static inline void bnxt_re_incr_tail(struct bnxt_device_sq *sq, uint8_t cnt)
{
  sq->tail += cnt;
  if (sq->tail >= sq->depth) {
    sq->tail %= sq->depth;
    /* Rolled over, Toggle Tail bit in epoch flags */
    sq->flags ^= 1UL << BNXT_RE_FLAG_EPOCH_TAIL_SHIFT;
  }
}

__device__ static inline void* bnxt_re_get_hwqe(struct bnxt_device_sq *sq, uint32_t idx)
{
  idx += sq->tail;
  if (idx >= sq->depth)
    idx -= sq->depth;
  return (void *)((char*)sq->buf + (idx << 4));
}

__device__ static inline void bnxt_re_incr_head(struct bnxt_device_cq *cq, uint8_t cnt)
{
  cq->head += cnt;
  if (cq->head >= cq->depth) {
    cq->head %= cq->depth;
    /* Rolled over, Toggle HEAD bit in epoch flags */
    cq->flags ^= 1UL << BNXT_RE_FLAG_EPOCH_HEAD_SHIFT;
  }
}

__device__ static inline void bnxt_re_change_cq_phase(struct bnxt_device_cq *cq)
{
  if (!cq->head) {
    cq->phase = !(cq->phase & BNXT_RE_BCQE_PH_MASK);
  }
}

__device__ static inline void aquire_lock(uint32_t *lock) {
  uint32_t expected;

  do {
    expected = 0;
  } while (0 == __hip_atomic_compare_exchange_strong(lock, &expected, 1,
                                                     __ATOMIC_SEQ_CST,
                                                     __ATOMIC_SEQ_CST,
                                                     __HIP_MEMORY_SCOPE_SYSTEM));
}

__device__ static inline void release_lock(uint32_t *lock) {
  *lock = 0;
}

__device__ void QueuePair::ring_cq_doorbell(uint32_t slot_idx) {
  struct bnxt_re_db_hdr hdr;
  uint32_t epoch;

  epoch = (cq.flags & BNXT_RE_FLAG_EPOCH_HEAD_MASK) << BNXT_RE_DB_EPOCH_HEAD_SHIFT;

  bnxt_re_init_db_hdr(&hdr, (slot_idx | epoch), 0, cq.flags, BNXT_RE_QUE_TYPE_CQ);

  __threadfence_system();
  __hip_atomic_store(dbr, hdr.typ_qid_indx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ void QueuePair::ring_sq_doorbell(uint32_t slot_idx) {
  struct bnxt_re_db_hdr hdr;
  uint32_t epoch;

  epoch = (sq.flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  bnxt_re_init_db_hdr(&hdr, (slot_idx | epoch), 0, sq.id, BNXT_RE_QUE_TYPE_SQ);

  __threadfence_system();
  __hip_atomic_store(dbr, hdr.typ_qid_indx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ int QueuePair::poll_cq() {
  struct bnxt_re_bcqe *hdr;
  void *cqe;
  uint32_t flg_val;
  int type;
  uint8_t status;

  cqe = (void*) ((char*) cq.buf + (cq.head * bnxt_re_get_cqe_sz()));
  hdr = (struct bnxt_re_bcqe*) ((char*)cqe + sizeof(struct bnxt_re_req_cqe));

  flg_val = hdr->flg_st_typ_ph;

  __threadfence_system();

  if (bnxt_re_is_cqe_valid(flg_val, cq.phase)) {
    // Is the CQE valid?
    status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT)
           & BNXT_RE_BCQE_STATUS_MASK;

    if (status != BNXT_RE_REQ_ST_OK) {
      printf("CQ Error %s (%x)\n", bnxt_re_wc_error_strings[status], status);
      abort();
      return -1;
    }

    /* Update the CQ Ptr */
    bnxt_re_incr_head(&cq, 1);
    bnxt_re_change_cq_phase(&cq);

    /* Ring Doorbell */
    ring_cq_doorbell(cq.head);

    __hip_atomic_fetch_sub(&sq.posted, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);

    return 1;
  }

  return 0;
}

__device__ void QueuePair::quiet() {
  uint64_t active_lane_mask;
  uint8_t active_lane_id;

  active_lane_mask  = get_active_lane_mask();
  active_lane_id    = get_active_lane_num(active_lane_mask);

  if (0 == active_lane_id) {
    aquire_lock(&cq.lock);
    while (__hip_atomic_load(&sq.posted, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT)) {
      poll_cq();
    }
    release_lock(&cq.lock);
  }
}

__device__ void QueuePair::post_wqe_rma(int pe, int32_t length, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  uint64_t active_lane_mask;
  uint8_t active_lane_count;
  uint8_t active_lane_id;

  active_lane_mask  = get_active_lane_mask();
  active_lane_count = get_active_lane_count(active_lane_mask);
  active_lane_id    = get_active_lane_num(active_lane_mask);

  if (0 == active_lane_id) {
    aquire_lock(&sq.lock);
  }

  for (int i = 0; i < active_lane_count; i++) {
    if (i == active_lane_id) {
      struct bnxt_re_bsqe hdr;
      struct bnxt_re_rdma rdma;
      struct bnxt_re_sge sge;
      struct bnxt_re_bsqe *hdr_ptr;
      struct bnxt_re_rdma *rdma_ptr;
      struct bnxt_re_sge *sge_ptr;
      uint32_t wqe_size;
      uint32_t wqe_type;
      uint32_t hdr_flags;
      uint32_t inline_msg;

      uint32_t rma_slots  = 3; // (Three slots: hdr, rdma, sge)

      inline_msg = length <= inline_threshold &&
                   opcode == gda_op_rdma_write;

      hdr_ptr  = (struct bnxt_re_bsqe*) bnxt_re_get_hwqe(&sq, 0);
      rdma_ptr = (struct bnxt_re_rdma*) bnxt_re_get_hwqe(&sq, 1);
      sge_ptr  = (struct bnxt_re_sge*)  bnxt_re_get_hwqe(&sq, 2);

      /* Populate Header Segment */
      wqe_type  = BNXT_RE_HDR_WT_MASK & opcode;
      wqe_size  = BNXT_RE_HDR_WS_MASK & rma_slots;
      hdr_flags = ((uint32_t) BNXT_RE_HDR_FLAGS_MASK)
                & ((uint32_t) BNXT_RE_WR_FLAGS_SIGNALED);

      if (inline_msg) {
        hdr_flags |= ((uint32_t) BNXT_RE_WR_FLAGS_INLINE);
      }

      hdr.rsv_ws_fl_wt  = (wqe_size  << BNXT_RE_HDR_WS_SHIFT)
                        | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT)
                        | wqe_type;
      hdr.key_immd      = 0;
      hdr.lhdr.qkey_len = length;

      /* Populate RDMA Segment */
      rdma.rva  = (uint64_t) raddr;
      rdma.rkey = rkey;

      if (!inline_msg) {
        /* Populate SG Segment */
        sge.pa     = (uint64_t) laddr;
        sge.lkey   = lkey;
        sge.length = length;
      }

      /* Write WQE to SQ */
      memcpy(hdr_ptr,  &hdr,  sizeof(struct bnxt_re_bsqe));
      memcpy(rdma_ptr, &rdma, sizeof(struct bnxt_re_rdma));

      if (inline_msg) {
        memcpy(sge_ptr,  laddr,  length);
      } else {
        memcpy(sge_ptr,  &sge,  sizeof(struct bnxt_re_sge));
      }

      /* Populate MSN Table */
      bnxt_re_fill_psns_for_msntbl(&sq, length);

      /* Update SQ Pointer */
      bnxt_re_incr_tail(&sq, rma_slots);

      /* Ring Doorbell */
      ring_sq_doorbell(sq.tail);

      __hip_atomic_fetch_add(&sq.posted, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);

    }
    __threadfence_system();
    quiet();
  }

  if (0 == active_lane_id) {
    release_lock(&sq.lock);
  }
}

__device__ uint64_t QueuePair::post_wqe_amo(int pe, int32_t length, uintptr_t *raddr, uint8_t opcode,
                                            int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t active_lane_mask;
  uint8_t active_lane_count;
  uint8_t active_lane_id;
  uint32_t atomic_idx = 0;

  active_lane_mask  = get_active_lane_mask();
  active_lane_count = get_active_lane_count(active_lane_mask);
  active_lane_id    = get_active_lane_num(active_lane_mask);

  if (0 == active_lane_id) {
    aquire_lock(&sq.lock);
  }

  for (int i = 0; i < active_lane_count; i++) {
    if (i == active_lane_id) {
      struct bnxt_re_bsqe hdr;
      struct bnxt_re_atomic amo;
      struct bnxt_re_sge sge;
      struct bnxt_re_bsqe *hdr_ptr;
      struct bnxt_re_atomic *amo_ptr;
      struct bnxt_re_sge *sge_ptr;
      uint32_t wqe_size;
      uint32_t wqe_type;
      uint32_t hdr_flags;
      uint32_t amo_slots = 3; // (Three slots: hdr, amo, sge)

      hdr_ptr = (struct bnxt_re_bsqe*)   bnxt_re_get_hwqe(&sq, 0);
      amo_ptr = (struct bnxt_re_atomic*) bnxt_re_get_hwqe(&sq, 1);
      sge_ptr = (struct bnxt_re_sge*)    bnxt_re_get_hwqe(&sq, 2);

      /* Populate Header Segment */
      wqe_size  = BNXT_RE_HDR_WS_MASK    & amo_slots;
      hdr_flags = ((uint32_t) BNXT_RE_HDR_FLAGS_MASK)
                & ((uint32_t) BNXT_RE_WR_FLAGS_SIGNALED);
      wqe_type  = BNXT_RE_HDR_WT_MASK    & opcode;

      hdr.rsv_ws_fl_wt  = (wqe_size  << BNXT_RE_HDR_WS_SHIFT)
                        | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT)
                        | wqe_type;
      hdr.key_immd = rkey;
      hdr.lhdr.rva = (uint64_t) raddr;

      /* Populate AMO Segment */
      amo.swp_dt = atomic_data;

      /* Populate SG Segment - (Return address of atomic) */
      if (fetching) {
        atomic_idx = fetching_atomic_idx++ % FETCHING_ATOMIC_CNT;
        sge.pa     = (uint64_t) &fetching_atomic[atomic_idx];
        sge.lkey   = fetching_atomic_lkey;
      } else {
        sge.pa     = (uint64_t) nonfetching_atomic;
        sge.lkey   = nonfetching_atomic_lkey;
      }
      sge.length = length;

      /* Write WQE to SQ */
      memcpy(hdr_ptr, &hdr, sizeof(struct bnxt_re_bsqe));
      memcpy(amo_ptr, &amo, sizeof(struct bnxt_re_atomic));
      memcpy(sge_ptr, &sge, sizeof(struct bnxt_re_sge));

      /* Populate MSN Table */
      bnxt_re_fill_psns_for_msntbl(&sq, length);

      /* Update SQ Pointer */
      bnxt_re_incr_tail(&sq, amo_slots);

      /* Ring Doorbell */
      ring_sq_doorbell(sq.tail);

      __hip_atomic_fetch_add(&sq.posted, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence_system();
    quiet();
  }

  if (0 == active_lane_id) {
    release_lock(&sq.lock);
  }

  if (fetching) {
    return fetching_atomic[atomic_idx];
  }

  return 0;
}

}  // namespace rocshmem
