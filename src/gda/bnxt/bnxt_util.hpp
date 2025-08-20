/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_BNXT_BNXT_UTIL_HPP_
#define LIBRARY_SRC_GDA_BNXT_BNXT_UTIL_HPP_

#include "gda/gda_macros.inl"
#include "util.hpp"

#include <sys/utsname.h> // utsname

/*
 * Finds the next power of two
 * TODO: find a faster solution
 */
static inline int next_pow(int value, int base) {
  int new_value = 1;
  while (new_value <= value) {
    new_value = new_value * base;
  }
  return new_value;
}

static inline bool rocm_has_dmabuf_support() {
  // Note: The contents of the function is from perftest. Specificly ./src/rocm_memory.c
  int dmabuf_supported = 0;
  const char kernel_opt1[] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
  const char kernel_opt2[] = "CONFIG_PCI_P2PDMA=y";
  int found_opt1           = 0;
  int found_opt2           = 0;
  FILE *fp;
  struct utsname utsname;
  char kernel_conf_file[128];
  char buf[256];

  if (uname(&utsname) == -1) {
    fprintf(stderr, "Could not get kernel name.\n");
    return false;
  }

  snprintf(kernel_conf_file, sizeof(kernel_conf_file),
           "/boot/config-%s", utsname.release);
  fp = fopen(kernel_conf_file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Could not open kernel conf file %s error: %m\n",
            kernel_conf_file);
    return false;
  }

  while (fgets(buf, sizeof(buf), fp) != NULL) {
    if (strstr(buf, kernel_opt1) != NULL) {
      found_opt1 = 1;
    }
    if (strstr(buf, kernel_opt2) != NULL) {
      found_opt2 = 1;
    }
    if (found_opt1 && found_opt2) {
      dmabuf_supported = 1;
      break;
    }
  }
  fclose(fp);

  if (dmabuf_supported == 0) {
    return false;
  }

  return true;
}

static inline struct ibv_mr *bnxt_re_dv_reg_mr(struct ibv_pd *pd, void *addr,
                                               size_t length, int access) {
  struct ibv_mr *mr;
  hsa_status_t status;
  int dmabuf_fd = 0;
  uint64_t offset = 0;

  if (false == rocm_has_dmabuf_support()) {
    fprintf(stderr, "DMABUF not supported on this machine.\n");
    return nullptr;
  }

  status = hsa_amd_portable_export_dmabuf(addr, length, &dmabuf_fd, &offset);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to export dmabuf handle for addr %p / %zu\n", addr, length);
    abort();
  }

  mr = ibv_reg_dmabuf_mr(pd, offset, length, (uint64_t) addr, dmabuf_fd, access);
  GDA_CHECK_NNULL(mr, "ibv_reg_dmabuf_mr");
  return mr;
}

#endif  // LIBRARY_SRC_GDA_BNXT_BNXT_UTIL_HPP_
