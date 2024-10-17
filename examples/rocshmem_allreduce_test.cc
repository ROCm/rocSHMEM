/*
** hipcc -c -fgpu-rdc -x hip rocshmem_allreduce_test.cc -I/opt/rocm/include 
**       -I$ROCHSMEM_INSTALL_DIR/include -I$OPENMPI_UCX_INSTALL_DIR/include/
** hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test 
**       $ROCHSMEM_INSTALL_DIR/lib/librocshmem.a $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so 
**       -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64
**
** ROC_SHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 8 ./rocshmem_allreduce_test
*/

#include <iostream>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <roc_shmem.hpp>

#define CHECK_HIP(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

using namespace rocshmem;

__global__ void allreduce_test(int *source, int *dest, int* pWork, long *pSync, size_t nelem)
{
    __shared__ roc_shmem_ctx_t ctx;
    int64_t ctx_type = 0;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);
    int num_pes = roc_shmem_ctx_n_pes(ctx);

    roc_shmem_ctx_int_sum_wg_to_all(ctx, dest, source, nelem, 0, 0, num_pes, pWork, pSync);

    roc_shmem_ctx_quiet(ctx);
    __syncthreads();

    roc_shmem_wg_ctx_destroy(&ctx);
    roc_shmem_wg_finalize();
}

static void init_sendbuf (int *sendbuf, int count, int mynode)
{
    for (int i = 0; i < count; i++) {
        sendbuf[i] = mynode + i%9;
    }
}

static bool check_recvbuf(int *recvbuf, int nprocs, int rank, int count)
{
    bool res=true;
    int expected = nprocs * (nprocs -1) / 2;

    for (int i=0; i<count; i++) {
        int result = expected + nprocs * (i%9);
        if (recvbuf[i] != result) {
            res = false;
#ifdef VERBOSE
            printf("recvbuf[%d] = %d expected %d \n", i, recvbuf[i], result);
#endif
        }
    }

    return res;
}

#define MAX_ELEM 256

int main (int argc, char **argv)
{
    int rank = roc_shmem_my_pe();
    int ndevices, my_device = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    my_device = rank % ndevices;
    CHECK_HIP(hipSetDevice(my_device));
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    roc_shmem_init();
    int npes =  roc_shmem_n_pes();
    int *source = (int *)roc_shmem_malloc(nelem * sizeof(int));
    int *result = (int *)roc_shmem_malloc(nelem * sizeof(int));
    if (NULL == source || NULL == result) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        roc_shmem_global_exit(1);
    }

    init_sendbuf(source, nelem, rank);
    for (int i=0; i<nelem; i++) {
        result[i] = -1;
    }

    size_t p_wrk_size = ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE;
    int *pWrk = (int *)roc_shmem_malloc(p_wrk_size * sizeof(int));

    size_t p_sync_size = ROC_SHMEM_REDUCE_SYNC_SIZE;
    long *pSync = (long *)roc_shmem_malloc(p_sync_size * sizeof(long));
    for (int i = 0; i < p_sync_size; i++) {
        pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    allreduce_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(source, result, pWrk, pSync, nelem);
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = check_recvbuf(result, npes, rank, nelem);

    printf("Test %s \t nelem %d %s\n", argv[0], nelem, pass ? "[PASS]" : "[FAIL]");
    
    roc_shmem_free(source);
    roc_shmem_free(result);
    roc_shmem_free(pWrk);
    roc_shmem_free(pSync);

    roc_shmem_finalize();
    return 0;
}
