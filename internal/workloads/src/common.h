#include <chrono>
#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
using namespace std;

#define TIME_NOW std::chrono::steady_clock::now()
#define TIME_DIFF(a, b) std::chrono::duration_cast<std::chrono::nanoseconds>(a - b).count()

#define HIPCHECK(cmd) do {                         \
  hipError_t e = cmd;                              \
  if( e != hipSuccess ) {                          \
    printf("Failed: Hip error %s:%d '%s'\n",             \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Copied from rccl-tests, used to hash hostname
static uint64_t getHash(const char* string, size_t n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash, strlen(hostHash));
}
