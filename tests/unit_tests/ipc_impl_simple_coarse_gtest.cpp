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

#include "ipc_impl_simple_coarse_gtest.hpp"

using namespace rocshmem;

TEST_F(IPCImplSimpleCoarseTestFixture, ptr_check) {
    ASSERT_NE(heap_mem_.get_ptr(), nullptr);
}

TEST_F(IPCImplSimpleCoarseTestFixture, MPI_num_pes) {
    ASSERT_EQ(mpi_.num_pes(), 2);
}

TEST_F(IPCImplSimpleCoarseTestFixture, IPC_bases) {
  ASSERT_NE(ipc_impl_.ipc_bases, nullptr);
  for(int i{0}; i < mpi_.num_pes(); i++) {
    ASSERT_NE(ipc_impl_.ipc_bases[i], nullptr);
  }
}

TEST_F(IPCImplSimpleCoarseTestFixture, golden_1048576_int) {
    iota_golden(1048576);
    validate_golden(1048576);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_1024x1x1_32_int) {
    dim3 grid {1,1,1};
    dim3 block {1024,1,1};
    write_wg(grid, block, 32);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_1024x1x1_32_int) {
    dim3 grid {1,1,1};
    dim3 block {1024,1,1};
    read_wg(grid, block, 32);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_1x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_2x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {2,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_4x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {4,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_8x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {8,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_16x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {16,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_32x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {32,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_64x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_128x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {128,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_256x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {256,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_512x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {512,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_768x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {768,1,1};
    write_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wg_1x1x1_1024x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1024,1,1};
    write_wg(grid, block, 1048576);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_1x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_2x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {2,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_4x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {4,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_8x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {8,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_16x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {16,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_32x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {32,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_64x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_128x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {128,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_256x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {256,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_512x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {512,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_768x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {768,1,1};
    read_wg(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wg_1x1x1_1024x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1024,1,1};
    read_wg(grid, block, 1048576);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_64x1x1_1_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    write_wave(grid, block, 1);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_64x1x1_1_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    read_wave(grid, block, 1);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_64x1x1_32_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    write_wave(grid, block, 32);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_64x1x1_32_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    read_wave(grid, block, 32);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_1x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_2x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {2,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_3x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {3,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_4x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {4,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_5x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {5,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_6x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {6,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_7x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {7,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_8x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {8,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_9x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {9,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_10x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {10,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_11x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {11,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_12x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {12,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_13x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {13,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_14x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {14,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_15x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {15,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_16x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {16,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_17x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {17,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_18x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {18,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_19x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {19,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_20x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {20,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_21x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {21,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_22x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {22,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_23x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {23,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_24x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {24,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_25x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {25,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_26x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {26,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_27x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {27,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_28x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {28,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_29x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {29,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_30x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {30,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_31x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {31,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_32x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {32,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_33x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {33,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_34x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {34,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_35x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {35,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_36x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {36,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_37x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {37,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_38x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {38,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_39x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {39,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_40x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {40,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_41x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {41,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_42x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {42,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_43x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {43,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_44x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {44,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_45x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {45,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_46x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {46,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_47x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {47,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_48x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {48,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_49x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {49,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_50x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {50,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_51x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {51,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_52x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {52,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_53x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {53,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_54x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {54,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_55x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {55,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_56x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {56,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_57x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {57,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_58x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {58,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_59x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {59,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_60x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {60,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_61x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {61,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_62x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {62,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_63x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {63,1,1};
    write_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, write_wave_1x1x1_64x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    write_wave(grid, block, 1048576);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_1x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_2x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {2,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_3x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {3,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_4x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {4,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_5x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {5,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_6x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {6,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_7x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {7,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_8x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {8,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_9x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {9,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_10x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {10,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_11x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {11,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_12x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {12,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_13x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {13,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_14x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {14,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_15x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {15,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_16x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {16,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_17x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {17,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_18x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {18,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_19x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {19,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_20x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {20,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_21x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {21,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_22x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {22,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_23x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {23,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_24x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {24,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_25x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {25,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_26x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {26,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_27x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {27,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_28x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {28,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_29x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {29,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_30x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {30,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_31x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {31,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_32x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {32,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_33x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {33,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_34x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {34,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_35x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {35,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_36x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {36,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_37x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {37,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_38x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {38,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_39x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {39,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_40x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {40,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_41x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {41,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_42x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {42,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_43x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {43,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_44x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {44,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_45x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {45,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_46x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {46,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_47x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {47,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_48x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {48,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_49x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {49,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_50x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {50,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_51x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {51,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_52x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {52,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_53x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {53,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_54x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {54,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_55x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {55,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_56x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {56,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_57x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {57,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_58x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {58,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_59x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {59,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_60x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {60,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_61x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {61,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_62x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {62,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_63x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {63,1,1};
    read_wave(grid, block, 1048576);
}

TEST_F(IPCImplSimpleCoarseTestFixture, read_wave_1x1x1_64x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {64,1,1};
    read_wave(grid, block, 1048576);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, write_1x1x1_1x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1,1,1};
    write(grid, block, 1048576);
}

//=============================================================================

TEST_F(IPCImplSimpleCoarseTestFixture, read_1x1x1_1x1x1_1048576_int) {
    dim3 grid {1,1,1};
    dim3 block {1,1,1};
    read(grid, block, 1048576);
}

