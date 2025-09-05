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

#ifndef LIBRARY_SRC_GDA_SEGMENT_BUILDER_HPP_
#define LIBRARY_SRC_GDA_SEGMENT_BUILDER_HPP_

#include <infiniband/mlx5dv.h>

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
  __device__ void update_ctrl_seg(uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm);

  /*
   * struct mlx5_wqe_raddr_seg {
   *   __be64 raddr;
   *   __be32 rkey;
   *   __be32 reserved;
   * };
   */
  __device__ void update_raddr_seg(uint64_t *raddr, uint32_t rkey);

  /*
   * struct mlx5_wqe_data_seg {
   * __be32 byte_count;
   * __be32 lkey;
   * __be64 addr;
   * };
   */
  __device__ void update_data_seg(uint64_t *laddr, uint32_t size, uint32_t lkey);

  /*
   * struct mlx5_wqe_atomic_seg {
   *   __be64 swap_add;
   *   __be64 compare;
   * };
   */
  __device__ void update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp);

 private:
  const int SEGMENTS_PER_WQE = 4;

  union mlx5_segment {
    mlx5_wqe_ctrl_seg ctrl_seg;
    mlx5_wqe_raddr_seg raddr_seg;
    mlx5_wqe_data_seg data_seg;
    mlx5_wqe_atomic_seg atomic_seg;
  };

  mlx5_segment *segp;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_SEGMENT_BUILDER_HPP_
