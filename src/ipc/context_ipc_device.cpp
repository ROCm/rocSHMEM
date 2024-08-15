/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "context_ipc_device.hpp"
#include "context_ipc_tmpl_device.hpp"

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "config.h"  // NOLINT(build/include_subdir)
#include "roc_shmem/roc_shmem.hpp"

namespace rocshmem {

__host__ IPCContext::IPCContext(Backend *b)
    : Context(b, false) {
}

__device__ void IPCContext::threadfence_system() {
}

__device__ void IPCContext::ctx_create() {
}

__device__ void IPCContext::ctx_destroy(){
}

__device__ void IPCContext::putmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
}

__device__ void IPCContext::getmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
}

__device__ void IPCContext::putmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
}

__device__ void IPCContext::getmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
}

__device__ void IPCContext::fence() {
}

__device__ void IPCContext::fence(int pe) {
}

__device__ void IPCContext::quiet() {
}

__device__ void *IPCContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  return ret;
}

__device__ void IPCContext::barrier_all() {
  __syncthreads();
}

__device__ void IPCContext::sync_all() {
  __syncthreads();
}

__device__ void IPCContext::sync(roc_shmem_team_t team) {
  __syncthreads();
}

__device__ void IPCContext::putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  __syncthreads();
}

__device__ void IPCContext::getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  __syncthreads();
}

__device__ void IPCContext::putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  __syncthreads();
}

__device__ void IPCContext::getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  __syncthreads();
}

__device__ void IPCContext::putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
}

__device__ void IPCContext::getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
}

__device__ void IPCContext::putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
}

__device__ void IPCContext::getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
}

}  // namespace rocshmem
