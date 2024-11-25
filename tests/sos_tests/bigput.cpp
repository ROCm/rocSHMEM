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
 * (big puts) each PE puts N elements (1MB) to ((my_pe()+1) mod num_pes()).
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define NUM_ELEMENTS 4194304  // 32 MB by longs
//#define DFLT_LOOPS 10000  // reset when Portals4 can achieve this.
#define DFLT_LOOPS 100

int Verbose;
int Sync;
int Track;
int elements = NUM_ELEMENTS;
double sum_time, time_taken;

static int atoi_scaled(char *s) {
  long val;
  char *e;

  val = strtol(s, &e, 0);
  if (e == NULL || *e == '\0') return (int)val;

  if (*e == 'k' || *e == 'K')
    val *= 1024;
  else if (*e == 'm' || *e == 'M')
    val *= 1024 * 1024;
  else if (*e == 'g' || *e == 'G')
    val *= 1024 * 1024 * 1024;

  return (int)val;
}

static void usage(char *pgm) {
  fprintf(
      stderr,
      "usage: %s -{hvclst}\n"
      "  where: (big puts)\n"
      "    -v              be verbose, multiple 'v' more verbose\n"
      "    -e element-cnt (%d)  # of int sized elements to put\n"
      "    -l loops (%d)  loop count.\n"
      "    -s             synchronize: barrier after each rocshmem_put()\n"
      "    -t             track: output '.' for every 200 rocshmem_put()s\n",
      pgm, NUM_ELEMENTS, DFLT_LOOPS);
}

static inline double shmem_wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (double)((tv.tv_usec / 1000000.0) + tv.tv_sec);
}

int main(int argc, char **argv) {
  int loops = DFLT_LOOPS;
  char *pgm;
  int *Target;
  int *Source;
  int i, me, npes;
  int target_PE;
  long bytes;
  double start_time, *total_time;
  long *pSync;
  double *pWrk;

  rocshmem_init();
  me = rocshmem_my_pe();
  npes = rocshmem_n_pes();

  if ((pgm = strrchr(argv[0], '/')))
    pgm++;
  else
    pgm = argv[0];

  while ((i = getopt(argc, argv, "hve:l:st")) != EOF) {
    switch (i) {
      case 'v':
        Verbose++;
        break;
      case 'e':
        if ((elements = atoi_scaled(optarg)) <= 0) {
          fprintf(stderr, "ERR: Bad elements count %d\n", elements);
          rocshmem_finalize();
          return 1;
        }
        break;
      case 'l':
        if ((loops = atoi_scaled(optarg)) <= 0) {
          fprintf(stderr, "ERR: Bad loop count %d\n", loops);
          rocshmem_finalize();
          return 1;
        }
        break;
      case 's':
        Sync++;
        break;
      case 't':
        Track++;
        break;
      case 'h':
        if (me == 0) usage(pgm);
        return 0;
      default:
        if (me == 0) {
          fprintf(stderr, "%s: unknown switch '-%c'?\n", pgm, i);
          usage(pgm);
        }
        rocshmem_finalize();
        return 1;
    }
  }

  pSync = (long *)rocshmem_malloc(ROCSHMEM_BCAST_SYNC_SIZE);
  if (!pSync) {
    fprintf(stderr, "ERR: bad pSync rocshmem_malloc(%ld)\n",
            ROCSHMEM_BCAST_SYNC_SIZE);
    rocshmem_global_exit(1);
  }

  for (i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i++)
    pSync[i] = ROCSHMEM_SYNC_VALUE;

  pWrk = (double *)rocshmem_malloc(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
  if (!pWrk) {
    fprintf(stderr, "ERR: bad pWrk rocshmem_malloc(%ld)\n",
            ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
    rocshmem_free(pSync);
    rocshmem_global_exit(1);
  }

  target_PE = (me + 1) % npes;

  total_time = (double *)rocshmem_malloc(npes * sizeof(double));
  if (!total_time) {
    fprintf(stderr, "ERR: bad total_time rocshmem_malloc(%ld)\n",
            (elements * sizeof(double)));
    rocshmem_free(pSync);
    rocshmem_free(pWrk);
    rocshmem_global_exit(1);
  }
  for (i = 0; i < npes; i++) total_time[i] = -1.0;

  Source = (int *)rocshmem_malloc(elements * sizeof(*Source));
  if (!Source) {
    fprintf(stderr, "ERR: bad Source rocshmem_malloc(%ld)\n",
            (elements * sizeof(*Target)));
    rocshmem_free(pSync);
    rocshmem_free(pWrk);
    rocshmem_free(total_time);
    rocshmem_global_exit(1);
  }

  Target = (int *)rocshmem_malloc(elements * sizeof(*Target));
  if (!Target) {
    fprintf(stderr, "ERR: bad Target rocshmem_malloc(%ld)\n",
            (elements * sizeof(*Target)));
    rocshmem_free(pSync);
    rocshmem_free(pWrk);
    rocshmem_free(Source);
    rocshmem_free(total_time);
    rocshmem_global_exit(1);
  }

  for (i = 0; i < elements; i++) {
    Target[i] = -90;
    Source[i] = i + 1;
  }

  bytes = loops * sizeof(int) * elements;

  if (Verbose && me == 0) {
    fprintf(stderr,
            "%s: INFO - %d loops, put %d (int) elements to PE+1 Max put ??\n",
            pgm, loops, elements);
  }
  rocshmem_barrier_all();

  for (i = 0; i < loops; i++) {
    start_time = shmem_wtime();

    rocshmem_int_put(Target, Source, elements, target_PE);

    time_taken += (shmem_wtime() - start_time);

    if (me == 0) {
      if (Track && i > 0 && ((i % 200) == 0)) fprintf(stderr, ".%d", i);
    }
    if (Sync) rocshmem_barrier_all();
  }

  // collect time per node.
  rocshmem_double_put(&total_time[me], &time_taken, 1, 0);
  rocshmem_ctx_double_sum_to_all(ROCSHMEM_CTX_DEFAULT, &sum_time, &time_taken,
                                  1, 0, 0, npes, pWrk, pSync);

  rocshmem_barrier_all();

  for (i = 0; i < elements; i++) {
    if (Target[i] != i + 1) {
      printf("%d: Error Target[%d] = %d, expected %d\n", me, i, Target[i],
             i + 1);
      rocshmem_global_exit(1);
    }
  }

  if (Track && me == 0) fprintf(stderr, "\n");

  if (Verbose && me == 0) {
    double rate, comp_time;

    if (Verbose > 1) fprintf(stdout, "Individule PE times: (seconds)\n");
    for (i = 0, comp_time = 0.0; i < npes; i++) {
      comp_time += total_time[i];
      if (Verbose > 1) fprintf(stdout, "  PE[%d] %8.6f\n", i, total_time[i]);
    }

    sum_time /= (double)npes;
    comp_time /= (double)npes;
    if (sum_time != comp_time)
      printf("%s: computed_time %7.5f != sum_to_all_time %7.5f)\n", pgm,
             comp_time, sum_time);

    rate = ((double)bytes / (1024.0 * 1024.0)) / comp_time;
    printf("%s: rocshmem_int_put() %7.4f MB/sec (bytes %ld secs %7.4f)\n", pgm,
           rate, bytes, sum_time);
  }

  rocshmem_free(pSync);
  rocshmem_free(pWrk);
  rocshmem_free(total_time);
  rocshmem_free(Target);
  rocshmem_free(Source);

  rocshmem_finalize();

  return 0;
}
