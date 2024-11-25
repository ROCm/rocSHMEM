/*
 *  This test program is derived from a unit test created by Nick Park.
 *  The original unit test is a work of the U.S. Government and is not subject
 *  to copyright protection in the United States.  Foreign copyrights may
 *  apply.
 *
 *  Copyright (c) 2017 Intel Corporation. All rights reserved.
 *  This software is available to you under the BSD license below:
 *
 *      Redistribution and use in source and binary forms, with or
 *      without modification, are permitted provided that the following
 *      conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

enum op {
  INC = 0,
  ATOMIC_INC,
  CTX_ATOMIC_INC,
  FINC,
  ATOMIC_FETCH_INC,
  CTX_ATOMIC_FETCH_INC,
  ATOMIC_FETCH_INC_NBI,
  CTX_ATOMIC_FETCH_INC_NBI
};

#ifdef ENABLE_DEPRECATED_TESTS
#define DEPRECATED_INC(TYPENAME, ...) rocshmem_##TYPENAME##_inc(__VA_ARGS__)
#define DEPRECATED_FINC(TYPENAME, ...) rocshmem_##TYPENAME##_finc(__VA_ARGS__)
#else
#define DEPRECATED_INC(TYPENAME, ...) \
  rocshmem_##TYPENAME##_atomic_inc(__VA_ARGS__)
#define DEPRECATED_FINC(TYPENAME, ...) \
  rocshmem_##TYPENAME##_atomic_fetch_inc(__VA_ARGS__)
#endif /* ENABLE_DEPRECATED_TESTS */

#define SHMEM_NBI_OPS_CASES(OP, TYPE, TYPENAME)                               \
  case ATOMIC_FETCH_INC_NBI:                                                  \
    rocshmem_##TYPENAME##_atomic_fetch_inc_nbi(&old, remote, i);              \
    rocshmem_quiet();                                                         \
    if (old > (TYPE)npes) {                                                   \
      printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP,   \
             #TYPE);                                                          \
      rc = EXIT_FAILURE;                                                      \
    }                                                                         \
    break;                                                                    \
  case CTX_ATOMIC_FETCH_INC_NBI:                                              \
    rocshmem_ctx_##TYPENAME##_atomic_fetch_inc_nbi(ROCSHMEM_CTX_DEFAULT,      \
                                                    &old, remote, i);         \
    rocshmem_quiet();                                                         \
    if (old > (TYPE)npes) {                                                   \
      printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP,   \
             #TYPE);                                                          \
      rc = EXIT_FAILURE;                                                      \
    }                                                                         \
    break;

#define TEST_SHMEM_INC(OP, TYPE, TYPENAME)                                    \
  do {                                                                        \
    TYPE *remote;                                                             \
    TYPE old;                                                                 \
    const int mype = rocshmem_my_pe();                                        \
    const int npes = rocshmem_n_pes();                                        \
    remote = (TYPE *)rocshmem_malloc(sizeof(TYPE));                           \
    *remote = (TYPE)0;                                                        \
    rocshmem_barrier_all();                                                   \
    for (int i = 0; i < npes; i++) switch (OP) {                              \
        case INC:                                                             \
          DEPRECATED_INC(TYPENAME, remote, i);                                \
          break;                                                              \
        case ATOMIC_INC:                                                      \
          rocshmem_##TYPENAME##_atomic_inc(remote, i);                        \
          break;                                                              \
        case CTX_ATOMIC_INC:                                                  \
          rocshmem_ctx_##TYPENAME##_atomic_inc(ROCSHMEM_CTX_DEFAULT, remote,  \
                                                i);                           \
          break;                                                              \
        case FINC:                                                            \
          old = DEPRECATED_FINC(TYPENAME, remote, i);                         \
          if (old > (TYPE)npes) {                                             \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype,  \
                   #OP, #TYPE);                                               \
            rc = EXIT_FAILURE;                                                \
          }                                                                   \
          break;                                                              \
        case ATOMIC_FETCH_INC:                                                \
          old = rocshmem_##TYPENAME##_atomic_fetch_inc(remote, i);            \
          if (old > (TYPE)npes) {                                             \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype,  \
                   #OP, #TYPE);                                               \
            rc = EXIT_FAILURE;                                                \
          }                                                                   \
          break;                                                              \
        case CTX_ATOMIC_FETCH_INC:                                            \
          old = rocshmem_ctx_##TYPENAME##_atomic_fetch_inc(                   \
              ROCSHMEM_CTX_DEFAULT, remote, i);                               \
          if (old > (TYPE)npes) {                                             \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype,  \
                   #OP, #TYPE);                                               \
            rc = EXIT_FAILURE;                                                \
          }                                                                   \
          break;                                                              \
        /*SHMEM_NBI_OPS_CASES(OP, TYPE, TYPENAME)*/                           \
        default:                                                              \
          printf("Invalid operation (%d)\n", OP);                             \
          rocshmem_global_exit(1);                                            \
      }                                                                       \
    rocshmem_barrier_all();                                                   \
    if ((*remote) != (TYPE)npes) {                                            \
      printf("PE %i observed error with TEST_SHMEM_INC(%s, %s)\n", mype, #OP, \
             #TYPE);                                                          \
      rc = EXIT_FAILURE;                                                      \
    }                                                                         \
    rocshmem_free(remote);                                                    \
    if (rc == EXIT_FAILURE) rocshmem_global_exit(1);                          \
  } while (false)

int main(int argc, char *argv[]) {
  rocshmem_init();

  int rc = EXIT_SUCCESS;

#ifdef ENABLE_DEPRECATED_TESTS
  TEST_SHMEM_INC(INC, int, int);
  TEST_SHMEM_INC(INC, long, long);
  TEST_SHMEM_INC(INC, long long, longlong);
  TEST_SHMEM_INC(INC, unsigned int, uint);
  TEST_SHMEM_INC(INC, unsigned long, ulong);
  TEST_SHMEM_INC(INC, unsigned long long, ulonglong);
  TEST_SHMEM_INC(INC, int32_t, int32);
  TEST_SHMEM_INC(INC, int64_t, int64);
  TEST_SHMEM_INC(INC, uint32_t, uint32);
  TEST_SHMEM_INC(INC, uint64_t, uint64);
  TEST_SHMEM_INC(INC, size_t, size);
  TEST_SHMEM_INC(INC, ptrdiff_t, ptrdiff);

  TEST_SHMEM_INC(FINC, int, int);
  TEST_SHMEM_INC(FINC, long, long);
  TEST_SHMEM_INC(FINC, long long, longlong);
  TEST_SHMEM_INC(FINC, unsigned int, uint);
  TEST_SHMEM_INC(FINC, unsigned long, ulong);
  TEST_SHMEM_INC(FINC, unsigned long long, ulonglong);
  TEST_SHMEM_INC(FINC, int32_t, int32);
  TEST_SHMEM_INC(FINC, int64_t, int64);
  TEST_SHMEM_INC(FINC, uint32_t, uint32);
  TEST_SHMEM_INC(FINC, uint64_t, uint64);
  TEST_SHMEM_INC(FINC, size_t, size);
  TEST_SHMEM_INC(FINC, ptrdiff_t, ptrdiff);
#endif /* ENABLE_DEPRECATED_TESTS */

  // TEST_SHMEM_INC(ATOMIC_INC, int, int);
  // TEST_SHMEM_INC(ATOMIC_INC, long, long);
  // TEST_SHMEM_INC(ATOMIC_INC, long long, longlong);
  // TEST_SHMEM_INC(ATOMIC_INC, unsigned int, uint);
  // TEST_SHMEM_INC(ATOMIC_INC, unsigned long, ulong);
  // TEST_SHMEM_INC(ATOMIC_INC, unsigned long long, ulonglong);
  // TEST_SHMEM_INC(ATOMIC_INC, int32_t, int32);
  TEST_SHMEM_INC(ATOMIC_INC, int64_t, int64);
  // TEST_SHMEM_INC(ATOMIC_INC, uint32_t, uint32);
  TEST_SHMEM_INC(ATOMIC_INC, uint64_t, uint64);
  // TEST_SHMEM_INC(ATOMIC_INC, size_t, size);
  // TEST_SHMEM_INC(ATOMIC_INC, ptrdiff_t, ptrdiff);

  // TEST_SHMEM_INC(CTX_ATOMIC_INC, int, int);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, long, long);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, long long, longlong);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, unsigned int, uint);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, unsigned long, ulong);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, unsigned long long, ulonglong);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, int32_t, int32);
  TEST_SHMEM_INC(CTX_ATOMIC_INC, int64_t, int64);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, uint32_t, uint32);
  TEST_SHMEM_INC(CTX_ATOMIC_INC, uint64_t, uint64);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, size_t, size);
  // TEST_SHMEM_INC(CTX_ATOMIC_INC, ptrdiff_t, ptrdiff);

  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, int, int);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, long, long);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, long long, longlong);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, unsigned int, uint);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, unsigned long, ulong);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, unsigned long long, ulonglong);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, int32_t, int32);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC, int64_t, int64);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, uint32_t, uint32);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC, uint64_t, uint64);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, size_t, size);
  // TEST_SHMEM_INC(ATOMIC_FETCH_INC, ptrdiff_t, ptrdiff);

  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, int, int);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, long, long);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, long long, longlong);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, unsigned int, uint);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, unsigned long, ulong);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, unsigned long long, ulonglong);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, int32_t, int32);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, int64_t, int64);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, uint32_t, uint32);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, uint64_t, uint64);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, size_t, size);
  // TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC, ptrdiff_t, ptrdiff);

  /*
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, int, int);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, long, long);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, long long, longlong);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, unsigned int, uint);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, unsigned long, ulong);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, unsigned long long, ulonglong);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, int32_t, int32);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, int64_t, int64);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, uint32_t, uint32);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, uint64_t, uint64);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, size_t, size);
  TEST_SHMEM_INC(ATOMIC_FETCH_INC_NBI, ptrdiff_t, ptrdiff);

  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, int, int);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, long, long);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, long long, longlong);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, unsigned int, uint);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, unsigned long, ulong);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, unsigned long long, ulonglong);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, int32_t, int32);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, int64_t, int64);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, uint32_t, uint32);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, uint64_t, uint64);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, size_t, size);
  TEST_SHMEM_INC(CTX_ATOMIC_FETCH_INC_NBI, ptrdiff_t, ptrdiff);
  */

  rocshmem_finalize();
  return rc;
}
