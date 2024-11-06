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

__global__ void allreduce_test(int *source, int *dest, size_t nelem,
        roc_shmem_team_t team) {
    __shared__ roc_shmem_ctx_t ctx;
    int64_t ctx_type = 0;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);
    int num_pes = roc_shmem_ctx_n_pes(ctx);

    roc_shmem_ctx_int_sum_wg_to_all(ctx, team, dest, source, nelem);

    roc_shmem_ctx_quiet(ctx);
    __syncthreads();

    roc_shmem_wg_ctx_destroy(&ctx);
    roc_shmem_wg_finalize();
}

static void init_sendbuf (int *source, int nelem, int my_pe)
{
    for (int i = 0; i < nelem; i++) {
        source[i] = my_pe + i%9;
    }
}

static bool check_recvbuf(int *dest, int nelem, int my_pe, int npes)
{
    bool res=true;
    int expected = npes * (npes -1) / 2;

    for (int i = 0; i < nelem; i++) {
        int result = expected + npes * (i%9);
        if (dest[i] != result) {
            res = false;
#ifdef VERBOSE
            printf("recvbuf[%d] = %d expected %d \n", i, dest[i], result);
#endif
        }
    }

    return res;
}

#define MAX_ELEM 256

int main (int argc, char **argv)
{
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    int my_pe = roc_shmem_my_pe();
    int npes =  roc_shmem_n_pes();

    int ndevices, my_device = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    my_device = my_pe % ndevices;
    CHECK_HIP(hipSetDevice(my_device));

    roc_shmem_init();

    int *source = (int *)roc_shmem_malloc(nelem * sizeof(int));
    int *dest = (int *)roc_shmem_malloc(nelem * sizeof(int));
    if (NULL == source || NULL == dest) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        roc_shmem_global_exit(1);
    }

    init_sendbuf(source, nelem, my_pe);
    for (int i=0; i<nelem; i++) {
        dest[i] = -1;
    }

    roc_shmem_team_t team_reduce_world_dup;
    team_reduce_world_dup = ROC_SHMEM_TEAM_INVALID;
    roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0,
                               &team_reduce_world_dup);

    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    allreduce_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(source, dest,
                        nelem, team_reduce_world_dup);
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = check_recvbuf(dest, nelem, my_pe, npes);
    printf("Test %s \t nelem %d %s\n", argv[0], nelem, pass ? "[PASS]" : "[FAIL]");
    
    roc_shmem_free(source);
    roc_shmem_free(dest);

    roc_shmem_finalize();
    return 0;
}
