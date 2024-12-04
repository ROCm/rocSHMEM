/*
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
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

/*
 * reduce [0...num_pes]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define N 3

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define WRK_SIZE MAX(N / 2 + 1, ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE)

int main(int argc, char *argv[]) {
  int i, Verbose = 0;
  char *pgm;
  long *pSync, *pWrk;
  long *src, *dst;

  if ((pgm = strrchr(argv[0], '/'))) {
    pgm++;
  } else {
    pgm = argv[0];
  }

  if (argc > 1) {
    if (strncmp(argv[1], "-v", 3) == 0) {
      Verbose = 1;
    } else if (strncmp(argv[1], "-h", 3) == 0) {
      fprintf(stderr, "usage: %s {v(verbose)|h(help)}\n", pgm);
      rocshmem_finalize();
      exit(1);
    }
  }

  rocshmem_init();

  src = (long *)rocshmem_malloc(N * sizeof(long));
  for (i = 0; i < N; i += 1) {
    src[i] = rocshmem_my_pe() + i;
  }

  dst = (long *)rocshmem_malloc(N * sizeof(long));

  pSync = (long *)rocshmem_malloc(ROCSHMEM_REDUCE_SYNC_SIZE * sizeof(long));
  for (i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i += 1) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }

  pWrk = (long *)rocshmem_malloc(WRK_SIZE * sizeof(long));

  rocshmem_barrier_all();

  rocshmem_ctx_long_max_to_all(ROCSHMEM_CTX_DEFAULT, dst, src, N, 0, 0,
                                rocshmem_n_pes(), pWrk, pSync);

  if (Verbose) {
    printf("%d/%d\tdst =", rocshmem_my_pe(), rocshmem_n_pes());
    for (i = 0; i < N; i += 1) {
      printf(" %ld", dst[i]);
    }
    printf("\n");
  }

  for (i = 0; i < N; i += 1) {
    if (dst[i] != rocshmem_n_pes() - 1 + i) {
      printf("[%3d] Error: dst[%d] == %ld, expected %ld\n", rocshmem_my_pe(),
             i, dst[i], rocshmem_n_pes() - 1 + (long)i);
      rocshmem_global_exit(1);
    }
  }

  rocshmem_free(dst);
  rocshmem_free(src);
  rocshmem_free(pSync);
  rocshmem_free(pWrk);

  rocshmem_finalize();

  return 0;
}
