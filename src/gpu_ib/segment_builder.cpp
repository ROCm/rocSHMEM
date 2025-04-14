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

#include "segment_builder.hpp"

#include "util.hpp"
#include "endian.hpp"

namespace rocshmem {

__device__ SegmentBuilder::SegmentBuilder(uint64_t wqe_idx, void *base) {
  mlx5_segment *base_ptr = static_cast<mlx5_segment *>(base);
  size_t segment_offset = SEGMENTS_PER_WQE * wqe_idx;
  seg_ptr = &base_ptr[segment_offset];
}

/*
 * Control segment - contains some control information for the current WQE.
 *
 * Output:
 *      seg       - control segment to be filled
 * Input:
 *      pi        - WQEBB number of the first block of this WQE.
 *                  This number should wrap at 0xffff, regardless of
 *                  size of the WQ.
 *      opcode    - Opcode of this WQE. Encodes the type of operation
 *                  to be executed on the QP.
 *      opmod     - Opcode modifier.
 *      qp_num    - QP/SQ number this WQE is posted to.
 *      fm_ce_se  - FM (fence mode), CE (completion and event mode)
 *                  and SE (solicited event).
 *      ds        - WQE size in octowords (16-byte units). DS accounts for all
 *                  the segments in the WQE as summarized in WQE construction.
 *      signature - WQE signature.
 *      imm       - Immediate data/Invalidation key/UMR mkey.
 */
/*
 * static MLX5DV_ALWAYS_INLINE
 * void mlx5dv_set_ctrl_seg(struct mlx5_wqe_ctrl_seg *seg, uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm)
 * {
 *   seg->opmod_idx_opcode   = htobe32(((uint32_t)opmod << 24) | ((uint32_t)pi << 8) | opcode);
 *   seg->qpn_ds             = htobe32((qp_num << 8) | ds);
 *   seg->fm_ce_se           = fm_ce_se;
 *   seg->signature          = signature;
 *   // The caller should prepare "imm" in advance based on WR opcode.
 *   // For IBV_WR_SEND_WITH_IMM and IBV_WR_RDMA_WRITE_WITH_IMM,
 *   // the "imm" should be assigned as is.
 *   // For the IBV_WR_SEND_WITH_INV, it should be htobe32(imm).
 *   seg->imm                = imm;
 * }
 */
__device__ void SegmentBuilder::update_cntrl_seg(uint8_t opcode, uint16_t wqe_idx, uint32_t ctrl_qp_sq, uint64_t ctrl_sig, bool zero_byte_rd) {
  mlx5_wqe_ctrl_seg ctrl_seg;
  ctrl_seg.opmod_idx_opcode = (opcode << 24) | (wqe_idx << 8);
  uint32_t DS = 2;
  if (zero_byte_rd == false) {
    DS = (opcode == MLX5_OPCODE_RDMA_WRITE || opcode == MLX5_OPCODE_RDMA_READ) ? 3 : 4;
  }
  ctrl_seg.qpn_ds = (DS << 24) | ctrl_qp_sq;
  ctrl_seg.signature = ctrl_sig;
  ctrl_seg.fm_ce_se = ctrl_sig >> 24;
  ctrl_seg.imm = ctrl_sig >> 32;
  memcpy(&seg_ptr->ctrl_seg, &ctrl_seg, sizeof(mlx5_wqe_ctrl_seg));
  seg_ptr++;
}

__device__ void SegmentBuilder::update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp) {
  mlx5_wqe_atomic_seg atomic_seg;
  swap_endian_store(reinterpret_cast<uint64_t*>(&atomic_seg.swap_add), atomic_data);
  swap_endian_store(reinterpret_cast<uint64_t*>(&atomic_seg.compare), atomic_cmp);
  memcpy(&seg_ptr->atomic_seg, &atomic_seg, sizeof(mlx5_wqe_atomic_seg));
  seg_ptr++;
}

__device__ void SegmentBuilder::update_rdma_seg(uintptr_t *raddr, uint32_t rkey) {
  mlx5_wqe_raddr_seg raddr_seg;
  raddr_seg.rkey = rkey;
  swap_endian_store(reinterpret_cast<uint64_t*>(&raddr_seg.raddr), reinterpret_cast<uint64_t>(raddr));
  memcpy(&seg_ptr->raddr_seg, &raddr_seg, sizeof(mlx5_wqe_raddr_seg));
  seg_ptr++;
}

/*
 * Data Segments - contain pointers and a byte count for the scatter/gather list.
 * They can optionally contain data, which will save a memory read access for
 * gather Work Requests.
 */
/*
 * static MLX5DV_ALWAYS_INLINE
 * void mlx5dv_set_data_seg(struct mlx5_wqe_data_seg *seg, uint32_t length, uint32_t lkey, uintptr_t address) {
 *   seg->byte_count = htobe32(length);
 *   seg->lkey       = htobe32(lkey);
 *   seg->addr       = htobe64(address);
 * }
 */
__device__ void SegmentBuilder::update_data_seg(uintptr_t *laddr, int32_t size, uint32_t lkey) {
  if (laddr == nullptr) { return; }
  mlx5_wqe_data_seg data_seg;
  swap_endian_store(&data_seg.byte_count, size & 0x7FFFFFFFU);
  data_seg.lkey = lkey;
  swap_endian_store(reinterpret_cast<uint64_t*>(&data_seg.addr), reinterpret_cast<uint64_t>(laddr));
  memcpy(&seg_ptr->data_seg, &data_seg, sizeof(mlx5_wqe_data_seg));
  seg_ptr++;
}

__device__ void SegmentBuilder::update_inl_data_seg(uintptr_t *laddr, int32_t size) {
  mlx5_wqe_inl_data_seg inl_data_seg;
  swap_endian_store(&inl_data_seg.byte_count, (size & 0x3FF) | 0x80000000);
  size_t field_size{sizeof(mlx5_wqe_inl_data_seg)};
  if (!laddr) {
    uint8_t flush_val = 1;
    memcpy(&inl_data_seg + 1, &flush_val, sizeof(flush_val));
    field_size += sizeof(flush_val);
  } else {
    memcpy(&inl_data_seg + 1, laddr, size);
    field_size += size;
  }
  memcpy(&seg_ptr->inl_data_seg, &inl_data_seg, field_size);
  seg_ptr++;
}

}  // namespace rocshmem
