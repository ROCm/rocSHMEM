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

#include "primitive_tester.hpp"

#include <roc_shmem/roc_shmem.hpp>
#include <debug.hpp>

#include <unistd.h>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void
PrimitiveTest(int loop,
              int *flag,
              char *s_buf,
              char *r_buf,
              int size,
              int my_pe,
              ShmemContextType ctx_type)
{
    __shared__ roc_shmem_ctx_t ctx;
    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);

    int block_id = hipBlockIdx_x;
    for(int i =0; i< loop; i++){
        roc_shmemx_ctx_putmem_nbi_wg(ctx, &r_buf[my_pe*size], &s_buf[block_id * size], size, block_id);
        if(hipThreadIdx_x==0){
            //roc_shmem_ctx_quiet(ctx);
            //roc_shmem_ctx_threadfence_system(ctx);
            roc_shmem_ctx_int_p(ctx, &flag[my_pe], i+1, block_id);
            //roc_shmem_ctx_quiet(ctx);
            roc_shmem_int_wait_until(&flag[block_id], ROC_SHMEM_CMP_EQ, i+1);

        }
        __syncthreads();
    }

    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PrimitiveTester::PrimitiveTester(TesterArguments args)
    : Tester(args)
{
    flag = (int*) roc_shmem_malloc(args.numprocs);
    memset(flag, 0, args.numprocs*sizeof(int));
   // s_buf = (char *)roc_shmem_malloc(args.max_msg_size * args.wg_size);
   // r_buf = (char *)roc_shmem_malloc(args.max_msg_size * args.wg_size);
}

PrimitiveTester::~PrimitiveTester()
{
    roc_shmem_free(s_buf);
    roc_shmem_free(r_buf);
}

void
PrimitiveTester::resetBuffers(uint64_t size)
{
    memset(s_buf, '0', size * args.numprocs);
    memset(r_buf, '1', size * args.numprocs);
}

void
PrimitiveTester::launchKernel(dim3 gridSize,
                              dim3 blockSize,
                              int loop,
                              uint64_t size,
                              int nproc, int my_pe)
{

    void* sendBuf = malloc(64);
    void* recvBuf = malloc(64 * nproc);

    s_buf = (char *)roc_shmem_malloc(size * nproc);
    r_buf = (char *)roc_shmem_malloc(size * nproc);
    resetBuffers(size);

    MPI_Allgather(sendBuf, 64, MPI_CHAR,
                  recvBuf, 64, MPI_CHAR,
                  MPI_COMM_WORLD);

    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(PrimitiveTest,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       flag,
                       s_buf,
                       r_buf,
                       size,
                       my_pe,
                       _shmem_context);

    //num_msgs = (loop + args.skip) * gridSize.x;
    num_timed_msgs = loop ;
}

void
PrimitiveTester::verifyResults(uint64_t size)
{
    int check_id =0;
    if (args.myid == check_id) {
        for (int i = 0; i < size*args.numprocs; i++) {
            if (r_buf[i] != '0') {
                fprintf(stderr, "Data validation error at idx %d\n", i);
                fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
                exit(-1);
            }
        }
    }
}
