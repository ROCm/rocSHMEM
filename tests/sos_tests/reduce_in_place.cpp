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
  int me, npes;
  int errors = 0;
  long *psync, *pwrk, *src;

  rocshmem_init();

  me = rocshmem_my_pe();
  npes = rocshmem_n_pes();

  src = (long *)rocshmem_malloc(NELEM * sizeof(long));
  for (int i = 0; i < NELEM; i++) src[i] = me;

  psync = (long *)rocshmem_malloc(ROCSHMEM_REDUCE_SYNC_SIZE * sizeof(long));
  for (int i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i++)
    psync[i] = ROCSHMEM_SYNC_VALUE;

  pwrk = (long *)rocshmem_malloc(
      (NELEM / 2 + ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE) * sizeof(long));

  rocshmem_barrier_all();

  rocshmem_ctx_long_max_to_all(ROCSHMEM_CTX_DEFAULT, src, src, NELEM, 0, 0,
                                npes, pwrk, psync);

  /* Validate reduced data */
  for (int j = 0; j < NELEM; j++) {
    long expected = npes - 1;
    if (src[j] != expected) {
      printf("%d: Expected src[%d] = %ld, got src[%d] = %ld\n", me, j, expected,
             j, src[j]);
      errors++;
    }
  }

  rocshmem_free(src);
  rocshmem_free(psync);
  rocshmem_free(pwrk);

  rocshmem_finalize();

  return errors != 0;
}
