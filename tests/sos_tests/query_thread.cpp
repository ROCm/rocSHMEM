/*
 *  Copyright (c) 2018 Intel Corporation. All rights reserved.
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

int main(int argc, char* argv[]) {
  int provided;

  int tl, ret;
  rocshmem_init_thread(ROCSHMEM_THREAD_FUNNELED, &tl);

  if (tl < ROCSHMEM_THREAD_FUNNELED || ret != 0) {
    printf("Init failed (requested thread level %d, got %d)\n",
           ROCSHMEM_THREAD_FUNNELED, tl);
    rocshmem_global_exit(1);
  }

  rocshmem_query_thread(&provided);
  printf("%d: Query result for thread level %d\n", rocshmem_my_pe(), provided);

  if (provided < ROCSHMEM_THREAD_FUNNELED) {
    printf("Error: thread support changed to an invalid level after init\n");
    rocshmem_global_exit(1);
  }

  rocshmem_finalize();
  return 0;
}
