/*
hipcc -c -fgpu-rdc -x hip rocshmem_getmem_test.cc \
  -I/opt/rocm/include \
  -I$ROCSHMEM_SRC_DIR/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link rocshmem_getmem_test.o -o rocshmem_getmem_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

ROC_SHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_getmem_test
*/

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <roc_shmem/roc_shmem.hpp>

#define CHECK_HIP(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

using namespace rocshmem;

__global__ void simple_getmem_test(int *src, int *dst, size_t nelem)
{
    roc_shmem_wg_init();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId == 0) {
        int rank = roc_shmem_my_pe();
        int peer =  rank ? 0 : 1;
        roc_shmem_getmem(dst, src, nelem * sizeof(int), peer);
        roc_shmem_quiet();
    }

    __syncthreads();
    roc_shmem_wg_finalize();
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
    int *src = (int *)roc_shmem_malloc(nelem * sizeof(int));
    int *dst = (int *)roc_shmem_malloc(nelem * sizeof(int));
    if (NULL == src || NULL == dst) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        std::cout << "source: " << src << ", dest: " << dst << ", size: "
          << sizeof(int) * nelem << std::endl;
        roc_shmem_global_exit(1);
    }

    for (int i=0; i<nelem; i++) {
        src[i] = 0;
        dst[i] = 1;
    }
    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    simple_getmem_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(src, dst, nelem);
    roc_shmem_barrier_all();
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = true;
    for (int i=0; i<nelem; i++) {
        if (dst[i] != 0) {
            pass = false;
#if VERBOSE
            printf("[%d] Error in element %d expected 0 got %d\n", rank, i, dst[i]);
#endif
        }
    }
    printf("Test %s \t %s\n", argv[0], pass ? "[PASS]" : "[FAIL]");

    roc_shmem_free(src);
    roc_shmem_free(dst);
    roc_shmem_finalize();
    return 0;
}
