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

#include "fused_gemm_allreduce_push_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/

// used for tile-contiguous data mapping (needed for put_signal transfer)
// assume matrix is split among pes in x dimension
__device__ __forceinline__ int get_1d_idx(int pe_idx, int tile_idx_x, int tile_idx_y, int thread_idx_x, int thread_idx_y, int num_pes, int x, int y, int tile_x, int tile_y) {
  int tile_cols_per_pe = x/num_pes/tile_x; // number of tile columns that will be mapped to each PE
  return pe_idx*(x*y/num_pes) + (tile_idx_y*tile_cols_per_pe + tile_idx_x)*tile_x*tile_y + thread_idx_x*tile_y + thread_idx_y;
}

// used for 2d data mapping (more natural for GEMM access)
// assume matrix is split among pes in x dimension
__device__ __forceinline__ int get_2d_idx(int pe_idx, int tile_idx_x, int tile_idx_y, int thread_idx_x, int thread_idx_y, int num_pes, int x, int y, int tile_x, int tile_y) {
  int tile_cols_per_pe = x/num_pes/tile_x; // number of tile columns that will be mapped to each PE
  return (tile_idx_y*tile_y + thread_idx_y)*x + (pe_idx*tile_cols_per_pe + tile_idx_x)*tile_x + thread_idx_x;
}

__global__ void FusedGemmAllreducePushTest(int loop, int skip, long long int *start_time,
                             long long int *end_time,
                             uint64_t *prod_sigs, uint64_t *con_sigs,
			                       int num_pes, ShmemContextType ctx_type,
                             __half2 **act_buffers, __half2 *weights, __half2 *bias, int n_act_buffers,
                             int m, int n, int k, int tile_m, int tile_n, int tile_k,
                             bool disable_gemm, bool disable_rs) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id    = blockIdx.y * gridDim.x + blockIdx.x;
  int num_wgs  = gridDim.x * gridDim.y;
  int tile_size = tile_m * tile_n;

  if (!disable_rs) {
    rocshmem_wg_ctx_create(ctx_type, &ctx);
  }

  int my_pe = rocshmem_my_pe();

  // tile_idx within the pe partition is constant for each wg for the duration of the kernel
  int tile_idx_m = blockIdx.y;
  int tile_idx_n = blockIdx.x;

  for (int i = 0; i < loop + skip; i++) {
    // used to synchronize between producer and consumer in each iteration
    uint64_t sig_value = i;
    __half2 *mat_a = act_buffers[(i*3) % n_act_buffers];
    __half2 *mat_b = weights;
    __half2 *mat_c = act_buffers[(i*3+1) % n_act_buffers];
    __half2 *xfer_buffer = act_buffers[(i*3+2) % n_act_buffers];
    __half2 *mat_a2 = act_buffers[(i*3+3) % n_act_buffers];
    
    if (i == skip) {
      if (is_thread_zero_in_block()) {
        start_time[wg_id] = wall_clock64();
      }
      __syncthreads();
    }

    // each WG produces <num_pe> output tiles and sends each
    // one to a different PE, starting with my_pe+1.
    // Later, each WG will consume <num_pe> tiles sent from different PEs 
    for (int pe_idx = 0; pe_idx < num_pes; pe_idx++) {
      int dst_pe = (my_pe + pe_idx + 1) % num_pes;

      
      // Perform tile matrix multiplication C[m,n]=A[m,k]*B^T[n,k], assuming:
      // - inputs are partitioned among pes in K dimension
      // - we must index into my_pe's partition of input A, but input B
      //   is sharded (no need to consider pe indexing)
      // - M dimension will be partitioned in next iteration, so
      //   each WG computes one tile (from B, to C) for each partition (dst_pe)
      // - weights is stored transposed as B^T[col][k] (col-major in k),
      //   so the k loop is stride-1 per thread instead of stride-n
      if (!disable_gemm)
      {
        // Column base for the B tile owned by this (dst_pe, tile_idx_n) pair.
        // With B^T[col][k] layout: element [col][k] is at offset col*tile_k + k.
        int b_col_base = get_2d_idx(dst_pe, tile_idx_n, 0, 0, 0, num_pes, n, tile_k, tile_n, tile_k);

        for (int thread_idx = threadIdx.x; thread_idx < tile_size; thread_idx += blockDim.x) {
          // Use float accumulation for mixed precision (reduce roundoff error)
          float sum0 = 0.0f;
          float sum1 = 0.0f;
          int thread_idx_m = thread_idx / tile_n;
          int thread_idx_n = thread_idx % tile_n;
          int c_idx = get_1d_idx(dst_pe, tile_idx_n, tile_idx_m, thread_idx_n, thread_idx_m, num_pes, n, m, tile_n, tile_m);
          int a_base = get_2d_idx(my_pe, 0, tile_idx_m, 0, thread_idx_m, num_pes, k, m, tile_k, tile_m);
          // B^T[col][k]: base for this thread's column; k loop is stride-1
          int b_base_T = (b_col_base + thread_idx_n) * tile_k;

          // Unroll k loop by 4 for better performance
          int kidx = 0;
          for (; kidx + 3 < tile_k; kidx += 4) {
            // Accumulate 4 products using __hfma2, then convert to float
            __half2 block_sum = __half2half2(__float2half(0.0f));
            block_sum = __hfma2(mat_a[a_base + kidx],     mat_b[b_base_T + kidx],     block_sum);
            block_sum = __hfma2(mat_a[a_base + kidx + 1], mat_b[b_base_T + kidx + 1], block_sum);
            block_sum = __hfma2(mat_a[a_base + kidx + 2], mat_b[b_base_T + kidx + 2], block_sum);
            block_sum = __hfma2(mat_a[a_base + kidx + 3], mat_b[b_base_T + kidx + 3], block_sum);

            // Convert to float and accumulate to reduce roundoff error
            sum0 += __half2float(block_sum.x);
            sum1 += __half2float(block_sum.y);
          }

          // Handle remaining iterations
          for (; kidx < tile_k; kidx++) {
            __half2 prod = __hmul2(mat_a[a_base + kidx], mat_b[b_base_T + kidx]);
            sum0 += __half2float(prod.x);
            sum1 += __half2float(prod.y);
          }

          // Convert final result back to __half2
          mat_c[c_idx] = __halves2half2(__float2half(sum0), __float2half(sum1));
        }
      }
      __syncthreads();

      // use put_signal to copy result to associated PE's a matrix, place in partition associated with my_pe
      if (!disable_rs) {
        int src_c_idx_base = get_1d_idx(dst_pe, tile_idx_n, tile_idx_m, 0, 0, num_pes, n, m, tile_n, tile_m);
        if (dst_pe != my_pe) {
        int con_sig_idx = dst_pe*num_wgs+wg_id;
        int prod_sig_idx = my_pe*num_wgs+wg_id;
        // if this is not the first iteration, check that the consumer has completed the last iter
        if (i > 0) {
          if (is_thread_zero_in_block()) {
            rocshmem_uint64_wait_until(&con_sigs[con_sig_idx], ROCSHMEM_CMP_EQ, sig_value-1);
          }
          __builtin_amdgcn_s_barrier();
        }
        int dst_a_1d_idx_base = get_1d_idx(my_pe, tile_idx_n, tile_idx_m, 0, 0, num_pes, n, m, tile_n, tile_m);
          rocshmem_ctx_float_put_nbi_wg(ctx, reinterpret_cast<float*>(&xfer_buffer[dst_a_1d_idx_base]), reinterpret_cast<float*>(&mat_c[src_c_idx_base]), tile_size, dst_pe);
          __builtin_amdgcn_s_barrier();
          if (is_thread_zero_in_block()) {
            rocshmem_ctx_fence(ctx, dst_pe);
            rocshmem_ctx_uint64_atomic_set(ctx, &prod_sigs[prod_sig_idx], sig_value,
                                           dst_pe);
          }
          __builtin_amdgcn_s_barrier();
        }
      }
    }

    if (disable_rs) {
      break;
    }

    // accumulate partial results from each pe in my partition of <a>
    // first, copy my 1d-idexed partition of c to my 2d-indexed partition of <a>. then
    // wait for peers to complete their put_signals to <a>, and accumulate their 1d-indexed
    // result in my 2d-indexed partition of <a> to complete reduce-scatter operation.
    // iterate over src PEs in reverse order to match order of send.
    for (int pe_idx = 0; pe_idx < num_pes; pe_idx++) {
      int src_pe = (my_pe + num_pes - pe_idx) % num_pes;
      int src_a_1d_idx_base = get_1d_idx(src_pe, tile_idx_n, tile_idx_m, 0, 0, num_pes, n, m, tile_n, tile_m);
      if (src_pe == my_pe) {
        // indexing should always start with my_pe, so we initialize the mat_a partition (no need to add)
        for (int thread_idx = threadIdx.x; thread_idx < tile_size; thread_idx+=blockDim.x) {
          int thread_idx_m = thread_idx / tile_n;
          int thread_idx_n = thread_idx % tile_n;
          int dst_a_2d_idx = get_2d_idx(my_pe, tile_idx_n, tile_idx_m, thread_idx_n, thread_idx_m, num_pes, n, m, tile_n, tile_m);
          // initialize mat_a2 with my partition of mat_c plus bias
          mat_a2[dst_a_2d_idx] = mat_c[src_a_1d_idx_base+thread_idx] + bias[dst_a_2d_idx];
        }
      }
      else {
        int con_sig_idx = my_pe*num_wgs+wg_id;
        int prod_sig_idx = src_pe*num_wgs+wg_id;
        if (is_thread_zero_in_block()) {
          rocshmem_uint64_wait_until(&prod_sigs[prod_sig_idx], ROCSHMEM_CMP_EQ, sig_value);
	}
        __builtin_amdgcn_s_barrier();
        for (int thread_idx = threadIdx.x; thread_idx < tile_size; thread_idx+=blockDim.x) {
          int thread_idx_m = thread_idx / tile_n;
          int thread_idx_n = thread_idx % tile_n;
          int dst_a_2d_idx = get_2d_idx(my_pe, tile_idx_n, tile_idx_m, thread_idx_n, thread_idx_m, num_pes, n, m, tile_n, tile_m);
          mat_a2[dst_a_2d_idx] = __hadd2(mat_a2[dst_a_2d_idx], uncached_load(&xfer_buffer[src_a_1d_idx_base+thread_idx]));
        }
        if (is_thread_zero_in_block()) {
          rocshmem_ctx_uint64_atomic_set(ctx, &con_sigs[con_sig_idx], sig_value, src_pe);
	      }
        __builtin_amdgcn_s_barrier();
      }
    }
  }

  if (is_thread_zero_in_block()) {
    end_time[wg_id] = wall_clock64();
    if (!disable_rs) {
      rocshmem_ctx_quiet(ctx);
    }
  }
  __syncthreads();

  if (!disable_rs) {
    rocshmem_wg_ctx_destroy(&ctx);
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
 #define MIN_BLOCK_SIZE 64
 #define MAX_BLOCK_SIZE 1024
FusedGemmAllreducePushTester::FusedGemmAllreducePushTester(TesterArguments args) : Tester(args) {
  // Initialize GEMM and tile dimensions for allocation
  m = args.gemm_m;
  n = args.gemm_max_n;
  k = args.gemm_max_n; // force k = n to ensure output can be used as input for next iter
  tile_k = k/args.numprocs; // partition k dimension across PEs; resulting partial sums will be transferred and reduced
  tile_n = max(n/(args.num_wgs*args.numprocs), min(n/args.numprocs, args.wg_size)); // accesses will be coalesced in n dimension, so tile_n should be >= wg size if possible
  tile_n = min(tile_n, MAX_BLOCK_SIZE); // cap so tile_n*tile_m can't exceed MAX_BLOCK_SIZE
  int numtiles_n = n/(args.numprocs*tile_n); // each WG accesses numprocs tiles of width tile_n
  tile_m = args.wg_size / tile_n; // derive tile_m so tile_n*tile_m == wg_size exactly
  int numtiles_m = m / tile_m;
  int num_wgs = numtiles_n * numtiles_m;
  assert(numtiles_n > 0);
  assert(tile_m > 0);
  assert(tile_n > 0);
  assert(tile_k > 0);

  // resize base-class timer arrays to match actual num_wgs
  CHECK_HIP(hipFree(timer));
  CHECK_HIP(hipFree(start_time));
  CHECK_HIP(hipFree(end_time));
  num_timers = num_wgs;
  CHECK_HIP(hipMalloc((void**)&timer,      sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&start_time, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&end_time,   sizeof(long long int) * num_timers));

  int ar_elems = args.max_msg_size*args.numprocs*num_wgs/sizeof(__half2);;
  int sig_elems = args.numprocs*num_wgs;
  n_act_buffers = 3;

  size_t timer_bytes = sizeof(long long int) * static_cast<size_t>(num_timers);
  size_t sig_bytes = sizeof(uint64_t) * static_cast<size_t>(sig_elems);
  size_t act_ptr_bytes = sizeof(__half2*) * static_cast<size_t>(n_act_buffers);
  size_t act_buf_bytes = sizeof(__half2) * static_cast<size_t>(m) * static_cast<size_t>(n);
  size_t weights_bytes = sizeof(__half2) * static_cast<size_t>(k/args.numprocs) * static_cast<size_t>(n);
  size_t bias_bytes = sizeof(__half2) * static_cast<size_t>(m) * static_cast<size_t>(n);

  // Allocate matrices for producer and consumer signals
  prod_sigs = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * sig_elems);
  con_sigs = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * sig_elems);
  // Allocate matrices for GEMM using __half2
  act_buffers = (__half2**)rocshmem_malloc(sizeof(__half2*) * n_act_buffers);
  for (int i = 0; i < n_act_buffers; i++) {
    act_buffers[i] = (__half2*)rocshmem_malloc(sizeof(__half2)*m*n);
    assert(act_buffers[i] != nullptr);
    if (!act_buffers[i]) { fprintf(stderr, "act allocation failed\n"); abort(); }
  }

  weights = (__half2*)rocshmem_malloc(sizeof(__half2) * k/args.numprocs * n);
  bias = (__half2*)rocshmem_malloc(sizeof(__half2)*m*n);
  assert(weights != nullptr);
  if (!weights) { fprintf(stderr, "weights allocation failed\n"); abort(); }
  if (!bias) { fprintf(stderr, "bias allocation failed\n"); abort(); }

}


FusedGemmAllreducePushTester::~FusedGemmAllreducePushTester() {
  rocshmem_free(prod_sigs);
  rocshmem_free(con_sigs);

  for (int i = 0; i < n_act_buffers; i++) {
    rocshmem_free(act_buffers[i]);
  }
  rocshmem_free(act_buffers);
  rocshmem_free(weights);
  rocshmem_free(bias);
}

#define TILE_N 64

void FusedGemmAllreducePushTester::resetBuffers(size_t size) {
  // Initialize GEMM and tile dimensions for this test size
  m = args.gemm_m;
  n = size;
  k = n; // force k = n to ensure output can be used as input for next iter
  tile_k = k/args.numprocs; // partition k dimension across PEs; resulting partial sums will be transferred and reduced
  int num_wg = m*n/(args.numprocs*args.wg_size);
  tile_n = max(n/(num_wg*args.numprocs), min(TILE_N, args.wg_size)); // accesses will be coalesced in n dimension, so tile_n should be >= wg size if possible
  tile_n = min(tile_n, MAX_BLOCK_SIZE);
  int numtiles_n = n/(args.numprocs*tile_n); // each WG accesses numprocs tiles of width tile_n
  tile_m = args.wg_size / tile_n; // derive tile_m so tile_n*tile_m == wg_size exactly
  int numtiles_m = m / tile_m;
  int num_wgs = numtiles_n * numtiles_m;
  if (numtiles_n <= 0) { fprintf(stderr, "numtiles_n must be > 0 (tile_n too large)\n"); abort(); }
  if (tile_m <= 0) { fprintf(stderr, "tile_m must be > 0\n"); abort(); }
  if (tile_n <= 0) { fprintf(stderr, "tile_n must be > 0\n"); abort(); }
  if (tile_k <= 0) { fprintf(stderr, "tile_k must be > 0\n"); abort(); }

  __half2 one_half2 = __half2half2(__float2half(1.0f));
  __half2 negone_half2 = __half2half2(__float2half(-1.0f));
  __half2 zero_half2 = __half2half2(__float2half(0.0f));

  // Initialize weights in transposed layout B^T[col][k]: element [col][k] at offset col*tile_k + k.
  // Alternating +1/-1 across the k dimension so each dot product sums to zero, then bias adds 1.
  for (int col = 0; col < n; col++) {
    for (int kidx = 0; kidx < tile_k; kidx++) {
      weights[col * tile_k + kidx] = (kidx % 2 == 0) ? one_half2 : negone_half2;
    }
  }

  // only need to init the first buffer - the rest will be populated by the kernel
  for (int i = 0; i < m * n; i++) {
    act_buffers[0][i] = one_half2;
    bias[i] = one_half2;
  }
  
  int sig_elems = args.numprocs * num_wgs;
  for (int i = 0; i < sig_elems; i++) {
    prod_sigs[i] = -1;
    con_sigs[i] = -1;
  }
}

BandwidthMetrics FusedGemmAllreducePushTester::computeBandwidth(uint64_t size, double time_s) {
  int local_k = size / args.numprocs;
  long a_bytes = (long)args.gemm_m * local_k * sizeof(__half2);
  long b_bytes = (long)local_k * size * sizeof(__half2);
  long c_bytes = (long)args.gemm_m * size * sizeof(__half2);

  // Network BW: each PE sends (numprocs-1) tiles of a_bytes per iteration.
  // Only applies when reduce-scatter is enabled.
  double nw_gbs = 0.0;
  if (!args.disable_rs) {
    nw_gbs = static_cast<double>(num_timed_msgs) * (args.numprocs - 1) * a_bytes
             / time_s / pow(2, 30);
  }

  // Memory BW: GEMM reads A + B and writes C; RS reads C for transfer,
  // writes xfer_buf on send, reads xfer_buf on receive, and reads+writes mat_a2.
  long gemm_mem = args.disable_gemm ? 0L : (a_bytes + b_bytes + c_bytes);
  long rs_mem   = args.disable_rs   ? 0L : (a_bytes + c_bytes * 3);
  double mem_gbs = 0.0;
  if (gemm_mem + rs_mem > 0) {
    mem_gbs = static_cast<double>(num_timed_msgs) * (gemm_mem + rs_mem)
              / time_s / pow(2, 30);
  }

  // Compute BW: __half2 packs 2 values; each MAC counts as 2 FLOPs; bias add is 2 FLOPs/element.
  // Only applies when GEMM is enabled.
  double comp_gflops = 0.0;
  if (!args.disable_gemm) {
    long gemm_flops = 2L * (2L * args.gemm_m * size * local_k) + 2L * args.gemm_m * size;
    comp_gflops = static_cast<double>(num_timed_msgs) * gemm_flops
                  / time_s / pow(2, 30);
  }

  return {nw_gbs, mem_gbs, comp_gflops};
}

void FusedGemmAllreducePushTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  size_t size) {
  size_t shared_bytes = 0;

  const dim3 kernelBlock(tile_n, tile_m, 1);
  const int numtiles_n = n / (args.numprocs * tile_n);
  const int numtiles_m = m / tile_m;
  const dim3 kernelGrid(numtiles_n, numtiles_m, 1);

  int per_wg_elems = size/sizeof(__half2);
  if (per_wg_elems < 1) { fprintf(stderr, "per_wg_elems must be >= 1\n"); abort(); }
  assert(tile_n * tile_m <= MAX_BLOCK_SIZE);

  if (args.disable_rs) {
    // GEMM-only mode: launch once per iteration so the host loop controls skip/timing.
    // Warm-up launches (skip iterations).
    for (int i = 0; i < args.skip; i++) {
      hipLaunchKernelGGL(FusedGemmAllreducePushTest, kernelGrid, kernelBlock, shared_bytes, stream,
                         1, 0, start_time, end_time,
                         prod_sigs, con_sigs,
                         args.numprocs, _shmem_context,
                         act_buffers, weights, bias, n_act_buffers,
                         m, n, k, tile_m, tile_n, tile_k,
                         args.disable_gemm, args.disable_rs);
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipStreamSynchronize(stream));
    }
    // Timed launches (loop iterations). start_time/end_time are recorded per-launch.
    for (int i = 0; i < loop; i++) {
      hipLaunchKernelGGL(FusedGemmAllreducePushTest, kernelGrid, kernelBlock, shared_bytes, stream,
                         1, 0, start_time, end_time,
                         prod_sigs, con_sigs,
                         args.numprocs, _shmem_context,
                         act_buffers, weights, bias, n_act_buffers,
                         m, n, k, tile_m, tile_n, tile_k,
                         args.disable_gemm, args.disable_rs);
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipStreamSynchronize(stream));
    }
  } else {
    hipLaunchKernelGGL(FusedGemmAllreducePushTest, kernelGrid, kernelBlock, shared_bytes, stream,
                       loop, args.skip, start_time, end_time,
                       prod_sigs, con_sigs,
                       args.numprocs, _shmem_context,
                       act_buffers, weights, bias, n_act_buffers,
                       m, n, k, tile_m, tile_n, tile_k,
                       args.disable_gemm, args.disable_rs);
    CHECK_HIP(hipGetLastError());
  }

  num_msgs = (loop + args.skip);
  num_timed_msgs = loop;
}

void FusedGemmAllreducePushTester::verifyResults(size_t size) {
  // After iterations, mat_a should contain accumulated results
  // Initial values: mat_a = 1, mat_b = 1
  // Each iteration: C = A * B where each element is sum of tile_k products
  // After reduce-scatter: A gets num_pes * tile_k accumulated results = k
  // So after each iteration, A is multiplied by k
  // After num_msgs iterations: A should equal k^num_msgs
  int my_pe = rocshmem_my_pe();
  __half2 *mat_a = act_buffers[(num_msgs*3) % n_act_buffers]; // mat_a and mat_c buffers are reused, so we need to index based on iteration number
  
  // Calculate expected value: 1.0
  // activation, weight, and bias values are set so each iteration should
  // result in a matrix of only 1.0
  float expected_val = 1.0f;

  
  // Copy mat_a back to host for verification
  __half2 *h_mat_a = new __half2[m * k];
  CHECK_HIP(hipMemcpy(h_mat_a, mat_a, sizeof(__half2) * m * k, hipMemcpyDeviceToHost));
  
  // Verify each PE's partition of mat_a
  // Each PE owns k/num_pes columns
  int my_k_start = my_pe * (k / args.numprocs);
  int my_k_end = my_k_start + (k / args.numprocs);
  
  bool all_correct = true;
  int errors = 0;
  const int max_errors_to_print = 10;
  
  for (int i = 0; i < m; i++) {
    for (int j = my_k_start; j < my_k_end; j++) {
      __half2 val = h_mat_a[i * k + j];
      float val0 = __half2float(val.x);
      float val1 = __half2float(val.y);
      
      // Allow some tolerance for floating point errors
      float tolerance = expected_val * 0.01f; // 1% tolerance
      
      if (fabs(val0 - expected_val) > tolerance || fabs(val1 - expected_val) > tolerance) {
        all_correct = false;
        if (errors < max_errors_to_print) {
          printf("ERROR pe%d: mat_a[%d,%d] = (%f, %f), expected %f\n", 
                 my_pe, i, j, val0, val1, expected_val);
        }
        errors++;
      }
    }
  }
  
  if (!all_correct) {
    printf("ERROR pe%d: Verification failed! %d errors found (showing first %d)\n", 
           my_pe, errors, max_errors_to_print);
    exit(-1);
  }
  
  delete[] h_mat_a;
}
