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

#include "allreduce_push_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void AllreducePushTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, int *buf1, int *buf2,
                             int nelems, uint64_t *prod_sigs, uint64_t *con_sigs, int num_pes,
			     ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int my_pe = rocshmem_ctx_my_pe(ctx);


  int *src_buf, *dst_buf;
  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      if (is_thread_zero_in_block()) {
        start_time[wg_id] = wall_clock64();
      }
      __syncthreads();
    }

    if (i % 2 == 0) {
      src_buf = buf1;
      dst_buf = buf2;
    }
    else {
      src_buf = buf2;
      dst_buf = buf1;
    }
    uint64_t sig_value = i;
    int buf_idx = (my_pe*gridDim.x+blockIdx.x)*nelems;
    // use put to copy my data to a reserved dst buffer slot on peer pes
    //int wf_id = get_flat_block_id() / WF_SIZE;
    //int wf_count = (int) ceil((double)get_flat_block_size() / (double)WF_SIZE);
    for (int pe_idx = 0; pe_idx < num_pes; pe_idx++) {
      // start with my_pe + wg_id + 1. this indexing avoids a bottleneck
      // on any queue pair from concurrent wgs. the consumer will
      // wait on remote PEs in a matching order.
      int dst_pe = (my_pe + (pe_idx+wg_id+1)) % num_pes;
      if (dst_pe != my_pe) {
        int con_sig_idx = dst_pe*gridDim.x+blockIdx.x;
        int prod_sig_idx = my_pe*gridDim.x+blockIdx.x;
        // if this is not the first iteration, check that the consumer has completed the last iter
        if (i > 0) {

          //if (is_thread_zero_in_wave()) {
          if (is_thread_zero_in_block()) {
            rocshmem_uint64_wait_until(&con_sigs[con_sig_idx], ROCSHMEM_CMP_EQ, sig_value-1);
	  }
        }
        rocshmem_ctx_int_put_nbi_wg(ctx, &dst_buf[buf_idx], &src_buf[buf_idx], nelems, dst_pe);
        __builtin_amdgcn_s_barrier();
        if (is_thread_zero_in_block()) {
          rocshmem_ctx_fence(ctx, dst_pe);
          rocshmem_ctx_uint64_atomic_set(ctx, &prod_sigs[prod_sig_idx], sig_value,
                                         dst_pe);
        }
        __builtin_amdgcn_s_barrier();
      }
    }
    // before polling on completion, copy from this pe's src to local dst buffer slot
    for (int offset = threadIdx.x; offset < nelems; offset+=blockDim.x) {
      dst_buf[buf_idx+offset] = src_buf[buf_idx+offset];
    }    

    //__builtin_amdgcn_s_barrier(); // barrier may not be needed here, same threads store data above and load it below
    // wait for peers to complete their puts, then accumulate in local
    // destination buffer slot
    for (int pe_idx = 0; pe_idx < num_pes; pe_idx++) {
      // start with my_pe and go backwards. this should match the order
      // on the sender side.
      int src_pe = (my_pe + num_pes*gridDim.x - (pe_idx+wg_id)) % num_pes;
      if (src_pe != my_pe) {
        int con_sig_idx = my_pe*gridDim.x+blockIdx.x;
        int prod_sig_idx = src_pe*gridDim.x+blockIdx.x;
        int in_buf_idx = (src_pe*gridDim.x+hipBlockIdx_x)*nelems;
        int out_buf_idx = (my_pe*gridDim.x+hipBlockIdx_x)*nelems;
        if (is_thread_zero_in_block()) {
          rocshmem_uint64_wait_until(&prod_sigs[prod_sig_idx], ROCSHMEM_CMP_EQ, sig_value);
	}
        __builtin_amdgcn_s_barrier();

        for (int offset = threadIdx.x; offset < nelems; offset+=blockDim.x) {
          dst_buf[out_buf_idx+offset] += uncached_load(&dst_buf[in_buf_idx+offset]);
        }
        if (is_thread_zero_in_block()) {
          rocshmem_ctx_uint64_atomic_set(ctx, &con_sigs[con_sig_idx], sig_value, src_pe);
	      }
        __syncthreads();
      }
    }
  }
  if (is_thread_zero_in_block()) {
    end_time[wg_id] = wall_clock64();
    rocshmem_ctx_quiet(ctx);
  }
  __syncthreads();

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
AllreducePushTester::AllreducePushTester(TesterArguments args) : Tester(args) {
  int buf_elems = args.max_msg_size*args.numprocs*args.num_wgs/sizeof(int);
  int sig_elems = args.numprocs*args.num_wgs;
  size_t buf_bytes = sizeof(int) * static_cast<size_t>(buf_elems);
  size_t sig_bytes = sizeof(uint64_t) * static_cast<size_t>(sig_elems);
  buf1 = (int *)rocshmem_malloc(sizeof(int) * buf_elems);
  buf2 = (int *)rocshmem_malloc(sizeof(int) * buf_elems);
  if (!buf1 || !buf2) {
    printf("rocshmem_malloc failed - tried to allocate 2x%zuB\n", buf_bytes);
    exit(-1);
  }
  prod_sigs = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * sig_elems);
  con_sigs = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * sig_elems);
  if (!prod_sigs || !con_sigs) {
    printf("rocshmem_malloc failed - tried to allocate 2x%zuB\n", sig_bytes);
    exit(-1);
  }
}

AllreducePushTester::~AllreducePushTester() {
  rocshmem_free(buf1);
  rocshmem_free(buf2);
  rocshmem_free(prod_sigs);
  rocshmem_free(con_sigs);
}

void AllreducePushTester::resetBuffers(size_t size) {
  int buf_elems = size*args.numprocs*args.num_wgs/sizeof(int);
  int sig_elems = args.numprocs*args.num_wgs;
  for (int i=0; i<buf_elems; i++) {
    buf1[i] = 1;
    buf2[i] = 0;
  }
  for (int i=0; i<sig_elems; i++) {
    prod_sigs[i] = -1;
    con_sigs[i] = -1;
  }
}

void AllreducePushTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  size_t size) {
  size_t shared_bytes = 0;

  int my_pe = rocshmem_my_pe();
  int per_wg_elems = size/sizeof(int);
  assert(per_wg_elems >= 1);
  hipLaunchKernelGGL(AllreducePushTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, buf1, buf2, per_wg_elems,
		     prod_sigs, con_sigs,
		     args.numprocs, _shmem_context);

  num_msgs = (loop + args.skip);
  num_timed_msgs = loop;
}

void AllreducePushTester::verifyResults(size_t size) {
  int correct_val = 1;
  for (int i=0; i<num_msgs; i++) {
    correct_val = (correct_val*args.numprocs);
  }
  int nelems = args.num_wgs*size/sizeof(int);
  // we accumulate in the local pe slice - only check results there
  int my_pe = rocshmem_my_pe();
  int localstart = nelems*my_pe;
  int localstop = localstart + nelems;
  for (int i=localstart; i<localstop; i++) {
    if (buf1[i] != correct_val && buf2[i] != correct_val) {
      printf("ERROR in verifyResults: buf1[%d]=%d buf2[%d]=%d != expected %d\n", i, buf1[i], i, buf2[i], correct_val);
      exit(-1);
    }
  }
}
