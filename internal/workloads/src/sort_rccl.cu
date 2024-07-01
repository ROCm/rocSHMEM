#include "rccl.h"
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

__global__ void sort1(volatile int *keys, int *keyBuffer1,
                     int *keyBuffer2, int *sendCount, 
                     int *recvCount, int *sendOffset,
                     int *recvOffset, int *outputKeys, 
                     size_t size, int n_pes, int my_pe) {
    __shared__ int bucketCounter[MAX_PES];
    __shared__ int bucketPtr[MAX_PES];
    __shared__ int total_size;

    int buckets = n_pes;

    int tid = threadIdx.x; // + blockDim.x * blockIdx.x;
    const int K_PER_BUCK = (MAX_KEY / buckets);

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
    TIME(3)
    OUTPUT_TIME()
}

__global__ void sort2(volatile int *keys, int *keyBuffer1,
                     int *keyBuffer2, int *sendCount, 
                     int *recvCount, int *sendOffset,
                     int *recvOffset, int *outputKeys, 
                     size_t size, int n_pes, int my_pe) {
    __shared__ int total_size;

    int buckets = n_pes;

    int tid = threadIdx.x; // + blockDim.x * blockIdx.x;
    const int K_PER_BUCK = (MAX_KEY / buckets);

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
    OUTPUT_TIME()
}

void sort(volatile int *keys, int *keyBuffer1,
            int *keyBuffer2, int *sendCount, 
            int *recvCount, int *sendOffset,
            int *recvOffset, int *outputKeys, 
            size_t size, int max_iters, ncclComm_t comm) {
    int nProcs, my_pe;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
    
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    for(int iter = 0; iter < max_iters; ++iter) {
        //fprintf(stderr, "%d: %d %d %p %p\n", my_pe, iter, max_iters, sendCount, recvCount);
        sort1<<<1, WG_SIZE, 0, stream>>>(keys, keyBuffer1,
                     keyBuffer2, sendCount, recvCount, sendOffset,
                     recvOffset, outputKeys, size, nProcs, my_pe);
        NCCLCHECK(ncclAllToAll(sendCount, recvCount, 1, 
                               ncclInt, comm, stream));
        NCCLCHECK(ncclAllToAll(sendOffset, recvOffset, 1, 
                               ncclInt, comm, stream));
        HIPCHECK(hipStreamSynchronize(stream));
        NCCLCHECK(ncclGroupStart());
        int total_size = 0;
        for(int i = 0; i < nProcs; ++i) {
            ncclSend(&keyBuffer1[sendOffset[i]], sendCount[i], 
                     ncclInt, i, comm, stream);
            ncclRecv(&keyBuffer2[total_size], recvCount[i], 
                     ncclInt, i, comm, stream);
            total_size += recvCount[i];
        }
        NCCLCHECK(ncclGroupEnd());
        HIPCHECK(hipStreamSynchronize(stream));
        sort2<<<1, WG_SIZE, 0, stream>>>(keys, keyBuffer1,
                     keyBuffer2, sendCount, recvCount, sendOffset,
                     recvOffset, outputKeys, size, nProcs, my_pe);
        HIPCHECK(hipStreamSynchronize(stream));
    }
}

bool verify(int *outputKeys, int *keyBuffer2, size_t size)
{   
    int num_pes, my_pe;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);

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

void initGPU(ncclComm_t &comms) 
{
    // Calculation for local rank, taken from rccl-tests
    int localRank = 0;
    int nProcs, proc;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
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
    
    ncclUniqueId ncclId;
    if (proc == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef RCCL_MULTIRANKPERGPU
	NCCLCHECK(ncclCommInitRankMulti(&comms, nProcs, ncclId, proc, proc));
#else
	NCCLCHECK(ncclCommInitRank(&comms, nProcs, ncclId, proc));
#endif

    printf("Rank %d: Device %d, Host %s\n", proc, my_device, hostname);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
}

void *roc_shmem_malloc(size_t size)
{
    void *v;
    hipMalloc((void **)&v, size);
    return v;
}

int roc_shmem_free(void *v)
{
    return hipFree(v);
}

int main(int argc, char *argv[])
{
    if(argc < 1) {
        printf("Format: %s [iterations]\n", argv[0]);
        return -1;
    }

    // Init stuff
    MPI_Init(&argc, &argv);
    ncclComm_t comms;
    initGPU(comms);

    int iterations = 1000;
    if(argc > 1)
        iterations = atoi(argv[1]);
    
    int num_pes, my_pe;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);

    // Configure input and outputs
    size_t size = 1024; //atoi(argv[1]);
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
    
    int *sendCount = 0, *recvCount = 0, *sendOffset = 0, *recvOffset = 0;
    sendCount = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);
    recvCount = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);
    sendOffset = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);
    recvOffset = (int*)roc_shmem_malloc(sizeof(int) * MAX_PES);

    printf("Begin untimed run\n");
    // Untimed run
    MPI_Barrier(MPI_COMM_WORLD);
    sort((int*)keys, keyBuffer1, keyBuffer2, 
        sendCount, recvCount, sendOffset, recvOffset, 
        outputKeys, size, 1, comms);
    hipDeviceSynchronize();

    printf("Verify untimed run\n");
    // Verify correctness
    if(!verify(outputKeys, keyBuffer2, outputKeys[MAX_KEY / num_pes - 1])) {
        fprintf(stderr, "Wrong output\n");
        return -1;
    }

    printf("Begin timed run\n");
    // Timed run
    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = TIME_NOW;
    sort((int*)keys, keyBuffer1, keyBuffer2, 
        sendCount, recvCount, sendOffset, recvOffset, 
        outputKeys, size, iterations, comms);
    hipDeviceSynchronize();
    double tot_time = (double)TIME_DIFF(TIME_NOW, time_start);
    
    double all_time = 0;
    MPI_Allreduce(&tot_time, &all_time, 1,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(my_pe == 0) {
        printf("Avg time:\t%.3f\tus\n", all_time / (double)(1000.0 * iterations * num_pes));
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
    ncclCommDestroy(comms);
    MPI_Finalize();
    return 0;
}