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

#ifndef LIBRARY_SRC_UTIL_HPP_
#define LIBRARY_SRC_UTIL_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>

namespace rocshmem {

#define CHECK_HIP(cmd)                                                        \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#ifdef DEBUG
#define DPRINTF(...)     \
  do {                   \
    printf(__VA_ARGS__); \
  } while (0);
#else
#define DPRINTF(...) \
  do {               \
  } while (0);
#endif

#ifdef DEBUG
#define GPU_DPRINTF(...)      \
  do {                        \
    gpu_dprintf(__VA_ARGS__); \
  } while (0);
#else
#define GPU_DPRINTF(...) \
  do {                   \
  } while (0);
#endif

extern const int gpu_clock_freq_mhz;

/* Device-side internal functions */
__device__ __forceinline__ uint32_t lowerID() {
  return __ffsll(__ballot(1)) - 1;
}

__device__ __forceinline__ int wave_SZ() { return __popcll(__ballot(1)); }

/*
 * Returns true if the caller's thread index is (0, 0, 0) in its block.
 */
__device__ __forceinline__ bool is_thread_zero_in_block() {
  return hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0;
}

/*
 * Returns true if the caller's block index is (0, 0, 0) in its grid.  All
 * threads in the same block will return the same answer.
 */
__device__ __forceinline__ bool is_block_zero_in_grid() {
  return hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0;
}

/*
 * Returns the number of threads in the caller's flattened thread block.
 */
__device__ __forceinline__ int get_flat_block_size() {
  return hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
}

/*
 * Returns the number of blocks in the caller's flattened grid.
 */
__device__ __forceinline__ int get_flat_grid_size() {
  return hipGridDim_x * hipGridDim_y * hipGridDim_z;
}

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
__device__ __forceinline__ int get_flat_block_id() {
  return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
         hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
}

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
__device__ __forceinline__ int get_flat_grid_id() {
  return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
         hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
}

/*
 * Returns the flattened thread index of the calling thread within the grid.
 */
__device__ __forceinline__ int get_flat_id() {
    return get_flat_grid_id() * (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z) + get_flat_block_id();
}

/*
 * Returns true if the caller's thread flad_id is 0 in its wave.
 */
__device__ __forceinline__ bool is_thread_zero_in_wave() {
  return (get_flat_block_id() % __AMDGCN_WAVEFRONT_SIZE) == 0;
}

extern __constant__ int* print_lock;

template <typename... Args>
__device__ void gpu_dprintf(const char* fmt, const Args&... args) {
  for (int i{0}; i < __AMDGCN_WAVEFRONT_SIZE; i++) {
    if ((get_flat_block_id() % __AMDGCN_WAVEFRONT_SIZE) == i) {
      /*
       * GPU-wide global lock that ensures that both prints are executed
       * by a single thread atomically.  We deliberately break control
       * flow so that only a single thread in a WF accesses the lock at a
       * time.  If multiple threads in the same WF attempt to gain the
       * lock at the same time, you have a classic GPU control flow
       * deadlock caused by threads in the same WF waiting on each other.
       */
      while (atomicCAS(print_lock, 0, 1) == 1) {
      }

      printf("WG (%u, %u, %u) TH (%u, %u, %u) ", hipBlockIdx_x,
             hipBlockIdx_y, hipBlockIdx_z, hipThreadIdx_x, hipThreadIdx_y,
             hipThreadIdx_z);
      printf(fmt, args...);

      *print_lock = 0;
    }
  }
}

int rocm_init();

void rocm_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr, int gpu_id);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_UTIL_HPP_
