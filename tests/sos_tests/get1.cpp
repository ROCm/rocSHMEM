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

#include <stdio.h>
#include <string.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

static long target[10];

int main(int argc, char *argv[]) {
  int i;
  long *source;

  rocshmem_init();

  if (rocshmem_n_pes() == 1) {
    printf("%s: Requires number of PEs > 1\n", argv[0]);
    rocshmem_finalize();
    return 0;
  }

  source = (long *)rocshmem_malloc(10 * sizeof(long));
  for (i = 0; i < 10; i++) {
    source[i] = i + 1;
  }

  rocshmem_barrier_all(); /* sync sender and receiver */

  if (rocshmem_my_pe() == 0) {
    memset(target, 0, sizeof(target));
    /* put 10 elements into target on PE 1 */
    rocshmem_long_get(target, source, 10, 1);
  }

  rocshmem_barrier_all(); /* sync sender and receiver */

  if (rocshmem_my_pe() == 0) {
    if (0 != memcmp(source, target, sizeof(long) * 10)) {
      fprintf(stderr, "[%d] Src & Target mismatch?\n", rocshmem_my_pe());
      for (i = 0; i < 10; ++i) {
        printf("%ld,%ld ", source[i], target[i]);
      }
      printf("\n");
      rocshmem_global_exit(1);
    }
  }

  rocshmem_free(source);

  rocshmem_finalize();

  return 0;
}
