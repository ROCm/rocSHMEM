/*
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

#include <stdio.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define NELEM 10

int main(void) {
  int i, me, npes;
  int errors = 0;
  long *src, *dst_max, *dst_min;
  long *min_psync, *max_psync;
  long *min_pwrk, *max_pwrk;

  rocshmem_init();

  me = rocshmem_my_pe();
  npes = rocshmem_n_pes();

  src = (long *)rocshmem_malloc(NELEM * sizeof(long));
  dst_max = (long *)rocshmem_malloc(NELEM * sizeof(long));
  dst_min = (long *)rocshmem_malloc(NELEM * sizeof(long));

  for (i = 0; i < NELEM; i++) {
    src[i] = me;
    dst_max[i] = -1;
    dst_min[i] = -1;
  }

  max_psync =
      (long *)rocshmem_malloc(ROCSHMEM_REDUCE_SYNC_SIZE * sizeof(long));
  min_psync =
      (long *)rocshmem_malloc(ROCSHMEM_REDUCE_SYNC_SIZE * sizeof(long));
  for (i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i++) {
    max_psync[i] = ROCSHMEM_SYNC_VALUE;
    min_psync[i] = ROCSHMEM_SYNC_VALUE;
  }

  max_pwrk = (long *)rocshmem_malloc(
      (NELEM / 2 + ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE) * sizeof(long));
  min_pwrk = (long *)rocshmem_malloc(
      (NELEM / 2 + ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE) * sizeof(long));

  if (me == 0) printf("Shrinking active set test\n");

  rocshmem_barrier_all();

  /* A total of npes tests are performed, where the active set in each test
   * includes PEs i..npes-1 */
  for (i = 0; i <= me; i++) {
    int j;

    if (me == i)
      printf(" + PE_start=%d, logPE_stride=0, PE_size=%d\n", i, npes - i);

    rocshmem_ctx_long_max_to_all(ROCSHMEM_CTX_DEFAULT, dst_max, src, NELEM, i,
                                  0, npes - i, max_pwrk, max_psync);

    /* Validate reduced data */
    for (j = 0; j < NELEM; j++) {
      long expected = npes - 1;
      if (dst_max[j] != expected) {
        printf(
            "%d: Max expected dst_max[%d] = %ld, got dst_max[%d] = %ld, "
            "iteration %d\n",
            me, j, expected, j, dst_max[j], i);
        errors++;
      }
    }

    rocshmem_ctx_long_min_to_all(ROCSHMEM_CTX_DEFAULT, dst_min, src, NELEM, i,
                                  0, npes - i, min_pwrk, min_psync);

    /* Validate reduced data */
    for (j = 0; j < NELEM; j++) {
      long expected = i;
      if (dst_min[j] != expected) {
        printf(
            "%d: Min expected dst_min[%d] = %ld, got dst_min[%d] = %ld, "
            "iteration %d\n",
            me, j, expected, j, dst_min[j], i);
        errors++;
      }
    }
  }

  rocshmem_free(src);
  rocshmem_free(dst_max);
  rocshmem_free(dst_min);

  rocshmem_free(max_psync);
  rocshmem_free(min_psync);

  rocshmem_free(max_pwrk);
  rocshmem_free(min_pwrk);

  rocshmem_finalize();

  return errors != 0;
}
