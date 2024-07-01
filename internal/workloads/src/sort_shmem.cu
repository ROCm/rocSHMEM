#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <roc_shmem/roc_shmem.hpp>
#include <unistd.h>
using namespace std;
using namespace rocshmem;

#include "common.h"
#include "sort.h"

//#define TIME_PERF
#ifdef TIME_PERF
#define TIMERS 10
__device__ uint64_t timers[TIMERS] = {0};
__device__ uint64_t time_start;
#define TIMERS_START() \
    if(threadIdx.x == 0) {\
        time_start = roc_shmem_timer();\
    }

#define TIME(TIMER_NUM) \
    if(threadIdx.x == 0) {\
        timers[TIMER_NUM] = roc_shmem_timer() - time_start;\
        time_start = roc_shmem_timer();\
    }

#define OUTPUT_TIME() \
    if(threadIdx.x == 0 && my_pe == 0) { \
        uint64_t sum = 0; \
        for(int i = 0; i < TIMERS; ++i) { \
            sum += timers[i]; \
        } \
        for(int i = 0; i < TIMERS; ++i) { \
            printf("%d: %f\n", i, (double)timers[i] / (double)sum); \
        } \
    }
#else
#define TIMERS_START()
#define TIME(x)
#define OUTPUT_TIME() 
#endif

__device__ __inline__ void alltoall(roc_shmem_ctx_t &ctx, 
                                    roc_shmem_team_t team, 
                                    int *dst, int *src) {
    // Perform alltoall
    roc_shmem_ctx_int_wg_alltoall(ctx,
                team,
                dst,    // T* dest
                src,  // const T* source
                1);       // int nelement
}

__global__ void sort(volatile int *keys, int *keyBuffer1,
                     int *keyBuffer2, int *sendCount, 
                     int *recvCount, int *sendOffset,
                     int *recvOffset, int *outputKeys, 
                     size_t size, roc_shmem_team_t team, 
                     int max_iters) {
    __shared__ roc_shmem_ctx_t ctx;
    __shared__ int bucketCounter[MAX_PES];
    __shared__ int bucketPtr[MAX_PES];
    __shared__ int total_size;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ROC_SHMEM_CTX_WG_PRIVATE, &ctx);

    int n_pes = roc_shmem_ctx_n_pes(ctx);
    int my_pe = roc_shmem_my_pe();
    int buckets = n_pes;

    int tid = threadIdx.x; // + blockDim.x * blockIdx.x;
    const int K_PER_BUCK = (MAX_KEY / buckets);

    for(int iter = 0; iter < max_iters; ++iter) {
        // Reset
        for(int i = threadIdx.x; i < buckets; i += blockDim.x) {
            bucketCounter[i] = 0;
            bucketPtr[i] = 0;
        }
        __syncthreads();
        TIMERS_START()
        // Count size of each bucket
        for(int i = tid; i < size; i += blockDim.x) {
            atomicAdd(&bucketCounter[keys[i] / K_PER_BUCK], 1);
        }
        __syncthreads();
        TIME(0)
        // Update in global memory
        for(int i = tid; i < buckets; i += blockDim.x) {
            sendCount[i] = bucketPtr[i] = bucketCounter[i];
        }
        __syncthreads();
        TIME(1)
        // Perform local scan to get ptrs set
        for(int shift = 1; shift < buckets; shift *= 2) {
            int temp = 0;
            if(threadIdx.x >= shift && threadIdx.x < buckets) {
                temp = bucketPtr[threadIdx.x - shift];
            }
            __syncthreads();
            if(threadIdx.x < buckets) {
                bucketPtr[threadIdx.x] += temp;
            }
            __syncthreads();
        }
        __syncthreads();
        TIME(2)
        // Find offsets of where we're sending
        for(int i = threadIdx.x; i < buckets; i += blockDim.x) {
            sendOffset[i] = bucketPtr[i] - sendCount[i];
        }
        // Sort keys into buckets
        for(int i = threadIdx.x; i < size; i += blockDim.x) {
            int loc = atomicAdd(&bucketPtr[keys[i] / K_PER_BUCK], -1) - 1;
            keyBuffer1[loc] = keys[i];
        }
        roc_shmem_ctx_threadfence_system(ctx);
        // Force sync to wait for all PEs to update bucket sizes
        roc_shmem_ctx_wg_team_sync(ctx, team);
        TIME(3)
        // Let all PEs know how many keys you wish to send
        alltoall(ctx, team, recvCount, sendCount);
        // Let all PEs know where the offsets are of the keys
        alltoall(ctx, team, recvOffset, sendOffset);
        __syncthreads();
        TIME(4)
        if(threadIdx.x == 0) {
            total_size = 0;
            for(int i = 0; i < buckets; ++i) {
                roc_shmem_int_get_nbi(&keyBuffer2[total_size], 
                    &keyBuffer1[recvOffset[i]], recvCount[i], i);
                total_size += recvCount[i];
            }
            roc_shmem_quiet();
        }
        for(int i = threadIdx.x; i < K_PER_BUCK; i += blockDim.x)
            outputKeys[i] = 0;
        __syncthreads();
        TIME(5)
        int min_key_val = my_pe * K_PER_BUCK;
        int max_key_val = (my_pe + 1) * K_PER_BUCK - 1;

        int *key_buff_ptr = outputKeys - min_key_val;
        for(int i = threadIdx.x; i < total_size; i += blockDim.x) {
            atomicAdd(&key_buff_ptr[keyBuffer2[i]], 1);
        }
        __syncthreads();
        TIME(6)
        // Perform local scan on keys
        for(int shift = 1; shift < K_PER_BUCK; shift *= 2) {
            int temp = 0;
            if(threadIdx.x >= shift && threadIdx.x < K_PER_BUCK) {
                temp = outputKeys[threadIdx.x - shift];
            }
            __syncthreads();
            if(threadIdx.x < K_PER_BUCK) {
                outputKeys[threadIdx.x] += temp;
            }
            __syncthreads();
        }
        TIME(7)
    }
    OUTPUT_TIME()
    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

bool verify(int *outputKeys, int *keyBuffer2, size_t size)
{   
    int num_pes = roc_shmem_n_pes();
    int my_pe = roc_shmem_my_pe();

    MPI_Status  status;
    MPI_Request request;

    int min_key_val = my_pe * (MAX_KEY / num_pes);
    int max_key_val = (my_pe + 1) * (MAX_KEY / num_pes) - 1;

    int *key_array = new int[size];
    // Perform final untimed sort on keys
    for(int i = 0; i < size; ++i)
        if(outputKeys[keyBuffer2[i] - min_key_val] > 0)
            key_array[--outputKeys[keyBuffer2[i] - min_key_val]] = keyBuffer2[i];
        else {
            fprintf(stderr, "%d: Found wrong key %d at %d with %d\n", my_pe, keyBuffer2[i], i, outputKeys[keyBuffer2[i]]);
            return false;
        }

    if(size < 1)
        size = 1;

    int k;
    const int MPI_TAG = 1000;
    // Check if largest key is smaller than next processor's
    if(my_pe > 0)
        MPI_Irecv(&k, 1, MPI_INT, my_pe - 1, MPI_TAG, MPI_COMM_WORLD,
                  &request);                   
    if(my_pe < num_pes - 1)
        MPI_Send(&key_array[size - 1], 1, MPI_INT, my_pe + 1, MPI_TAG,
                 MPI_COMM_WORLD );
    if(my_pe > 0)
        MPI_Wait(&request, &status);

    // Check if it is smaller
    int j = 0;
    if( my_pe > 0 && size > 1 )
        if( k > key_array[0] )
            j++;

    // Check if keys correctly sorted
    for(int i = 1; i < size; i++)
        if(key_array[i - 1] > key_array[i])
            j++;

    delete[] key_array;

    if(j != 0) {
        fprintf(stderr, "Processor %d:  Full_verify: number of keys out of sort: %d\n",
                my_pe, j );
        return false;
    }
    return true;
}

void initGPU() 
{
    // Calculation for local rank, taken from rccl-tests
    int localRank = 0;
    int proc = roc_shmem_my_pe();
    int nProcs = roc_shmem_n_pes();
    char hostname[1024];
    gethostname(hostname, 1024);
    for (int i=0; i< 1024; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            break;
        }
    }
    uint64_t hostHashs[nProcs];
    hostHashs[proc] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int p=0; p<nProcs; p++) {
      if (p == proc) break;
      if (hostHashs[p] == hostHashs[proc]) localRank++;
    }
    
    /***
     * Select a GPU
     */    
    int ndevices, my_device=0;
    hipGetDeviceCount (&ndevices);
    my_device = localRank % ndevices;
    hipSetDevice(my_device);

    printf("Rank %d: Device %d, Host %s\n", proc, my_device, hostname);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    // Init roc_shmem stuff
    initGPU();
    roc_shmem_init(NUM_WGS);
    int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);
    roc_shmem_team_t team_world_dup = ROC_SHMEM_TEAM_INVALID;
    roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD,
                                 0,
                                 1,
                                 n_pes,
                                 nullptr,
                                 0,
                                 &team_world_dup);

    int iterations = 1000;
    if(argc > 1)
        iterations = atoi(argv[1]);
    
    int num_pes = roc_shmem_n_pes();
    int my_pe = roc_shmem_my_pe();

    // Configure input and outputs
    size_t size = 1024; //atoi(argv[2]);
    int *keys, *outputKeys;
    hipMalloc((void**)&keys, sizeof(int) * size);
    hipMalloc((void**)&outputKeys, sizeof(int) * WG_SIZE);

/*  Generate random number sequence and subsequent keys on all procs */
    create_seq( find_my_seed( my_pe, 
                              num_pes, 
                              4*(long)size*num_pes,
                              314159265.00,      /* Random number gen seed */
                              1220703125.00 ),   /* Random number gen mult */
                1220703125.00, keys, size );     /* Random number gen mult */


    // Init buffers
    int *keyBuffer1, *keyBuffer2;
    keyBuffer1 = (int*)roc_shmem_malloc(sizeof(int) * size);
    keyBuffer2 = (int*)roc_shmem_malloc(sizeof(int) * size * 4);
    
    int *sendCount, *recvCount, *sendOffset, *recvOffset;
    sendCount = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);
    recvCount = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);
    sendOffset = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);
    recvOffset = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);

    // Untimed run
    roc_shmem_barrier_all();
    sort<<<1, WG_SIZE>>>((int*)keys, keyBuffer1, keyBuffer2, 
        sendCount, recvCount, sendOffset, recvOffset, 
        outputKeys, size, team_world_dup, 1);
    hipDeviceSynchronize();

    // Verify correctness
    if(!verify(outputKeys, keyBuffer2, outputKeys[MAX_KEY / num_pes - 1])) {
        fprintf(stderr, "Wrong output\n");
        return -1;
    }

    // Timed run
    roc_shmem_barrier_all();
    auto time_start = TIME_NOW;
    sort<<<1, WG_SIZE>>>((int*)keys, keyBuffer1, keyBuffer2, 
        sendCount, recvCount, sendOffset, recvOffset, 
        outputKeys, size, team_world_dup, iterations);
    hipDeviceSynchronize();
    double tot_time = (double)TIME_DIFF(TIME_NOW, time_start);
    
    double all_time = 0;
    MPI_Allreduce(&tot_time, &all_time, 1,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(my_pe == 0) {
        printf("Avg time:\t%f\tus\n", all_time / 
                (double)(1000.0 * iterations * num_pes));
    }

    // Verify correctness
    if(!verify(outputKeys, keyBuffer2, outputKeys[MAX_KEY / num_pes - 1])) {
        fprintf(stderr, "Wrong output\n");
        return -1;
    }

    // Clean up
    hipFree(keys);
    hipFree(outputKeys);
    roc_shmem_free(keyBuffer1);
    roc_shmem_free(keyBuffer2);
    roc_shmem_free(sendCount);
    roc_shmem_free(recvCount);
    roc_shmem_free(sendOffset);
    roc_shmem_free(recvOffset);
    roc_shmem_finalize();
    return 0;
}