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

#ifndef LIBRARY_SRC_GPU_IB_SEGMENT_BUILDER_HPP_
#define LIBRARY_SRC_GPU_IB_SEGMENT_BUILDER_HPP_

#include <infiniband/mlx5dv.h>

#include "infiniband_structs.hpp"
#include "util.hpp"

namespace rocshmem {

class SegmentBuilder {
 public:
  __device__ SegmentBuilder(uint64_t wqe_idx, void *base);

  /*
   * struct mlx5_wqe_ctrl_seg {
   *   __be32 opmod_idx_opcode;
   *   __be32 qpn_ds;
   *   uint8_t signature;
   *   __be16 dci_stream_channel_id;
   *   uint8_t fm_ce_se;
   *   __be32 imm;
   * } __attribute__((__packed__)) __attribute__((__aligned__(4)));
   */
  __device__ void update_cntrl_seg(uint8_t opcode, uint16_t wqe_idx, uint32_t ctrl_qp_sq, uint64_t ctrl_sig, bool zero_byte_rd);

  /*
   * struct mlx5_wqe_atomic_seg {
   *   __be64 swap_add;
   *   __be64 compare;
   * };
   */
  __device__ void update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp);

  /*
   * struct mlx5_wqe_raddr_seg {
   *   __be64          raddr;
   *   __be32          rkey;
   *   __be32          reserved;
   * };
   */
  __device__ void update_rdma_seg(uintptr_t *raddr, uint32_t rkey);

  /*
   * struct mlx5_wqe_inl_data_seg {
   *   uint32_t        byte_count;
   * };
   */
  __device__ void update_inl_data_seg(uintptr_t *laddr, int32_t size);

  /*
   * struct mlx5_wqe_data_seg {
   * __be32 byte_count;
   * __be32 lkey;
   * __be64 addr;
   * };
   */
  __device__ void update_data_seg(uintptr_t *laddr, int32_t size, uint32_t lkey);

 private:
  const int SEGMENTS_PER_WQE = 4;

  mlx5_segment *seg_ptr;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_SEGMENT_BUILDER_HPP_
