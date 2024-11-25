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
#include <stdlib.h>

#include <rocshmem/rocshmem.hpp>

#define NUM_CTX 32

using namespace rocshmem;

int main(int argc, char **argv) {
  int me, npes, i;
  int errors = 0;
  rocshmem_ctx_t ctx[NUM_CTX];

  rocshmem_init();

  me = rocshmem_my_pe();
  npes = rocshmem_n_pes();

  int64_t *data = (int64_t *)rocshmem_malloc(sizeof(int64_t));

  /* Initialize the counter */
  memset(data, 0, sizeof(int64_t));
  rocshmem_barrier_all();

  for (i = 0; i < NUM_CTX; i++) {
    int err = rocshmem_ctx_create(0, &ctx[i]);

    if (err) {
      printf("%d: Warning, could not create context %d (%d)\n", me, i, err);
      ctx[i] = ROCSHMEM_CTX_DEFAULT;
    }
  }

  for (i = 0; i < NUM_CTX; i++)
    rocshmem_ctx_int64_atomic_inc(ctx[i], data, (me + 1) % npes);

  for (i = 0; i < NUM_CTX; i++) rocshmem_ctx_quiet(ctx[i]);

  rocshmem_sync_all();

  if ((*data) != NUM_CTX) {
    printf("%d: error expected %d, got %ld\n", me, NUM_CTX, (*data));
    ++errors;
  }

  rocshmem_free(data);

  rocshmem_finalize();
  return errors;
}
