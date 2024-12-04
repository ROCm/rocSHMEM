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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define NUM_POINTS 10000

int main(int argc, char *argv[], char *envp[]) {
  int me, myrocshmem_n_pes;
  long long *inside, *total;

  /*
  ** Starts/Initializes SHMEM/OpenSHMEM
  */
  rocshmem_init();
  /*
  ** Fetch the number or processes
  ** Some implementations use num_pes();
  */
  myrocshmem_n_pes = rocshmem_n_pes();
  /*
  ** Assign my process ID to me
  */
  me = rocshmem_my_pe();

  inside = (long long *)rocshmem_malloc(sizeof(long long));
  total = (long long *)rocshmem_malloc(sizeof(long long));
  *inside = *total = 0;

  srand(1 + me);

  for ((*total) = 0; (*total) < NUM_POINTS; ++(*total)) {
    double x, y;
    x = rand() / (double)RAND_MAX;
    y = rand() / (double)RAND_MAX;

    if (x * x + y * y < 1) {
      ++(*inside);
    }
  }

  rocshmem_barrier_all();

  int errors = 0;

  if (me == 0) {
    for (int i = 1; i < myrocshmem_n_pes; ++i) {
      long long remoteInside, remoteTotal;
      rocshmem_longlong_get(&remoteInside, inside, 1, i);
      rocshmem_longlong_get(&remoteTotal, total, 1, i);
      (*total) += remoteTotal;
      (*inside) += remoteInside;
    }

    double approx_pi = 4.0 * (*inside) / (double)(*total);

    if (fabs(M_PI - approx_pi) > 0.1) {
      ++errors;
    }

    if (NULL == getenv("MAKELEVEL")) {
      printf("Pi from %llu points on %d PEs: %lf\n", *total, myrocshmem_n_pes,
             approx_pi);
    }
  }

  rocshmem_free(inside);
  rocshmem_free(total);

  rocshmem_finalize();

  return errors;
}
