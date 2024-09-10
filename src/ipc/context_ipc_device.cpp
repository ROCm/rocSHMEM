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
#include "backend_ipc.hpp"

namespace rocshmem {

__host__ IPCContext::IPCContext(Backend *b)
    : Context(b, false) {
  IPCBackend *backend{static_cast<IPCBackend *>(b)};
  ipcImpl_.ipc_bases = b->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = b->ipcImpl.shm_size;

  auto *bp{backend->ipc_backend_proxy.get()};

  barrier_sync = backend->barrier_sync;
  g_ret = bp->g_ret;
  atomic_base_ptr = bp->atomic_ret->atomic_base_ptr;
}

__device__ void IPCContext::threadfence_system() {
}

__device__ void IPCContext::ctx_create() {
}

__device__ void IPCContext::ctx_destroy(){
}

__device__ void IPCContext::putmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                   const_cast<void *>(source), nelems);
}

__device__ void IPCContext::getmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset =
      const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[pe] + L_offset, nelems);
}

__device__ void IPCContext::putmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  putmem(dest, source, nelems, pe);
}

__device__ void IPCContext::getmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  getmem(dest, source, nelems, pe);
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

__device__ void IPCContext::putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[pe] + L_offset,
                      const_cast<void *>(source), nelems);
  __syncthreads();
}

__device__ void IPCContext::getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset =
      const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[pe] + L_offset, nelems);
  __syncthreads();
}

__device__ void IPCContext::putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  putmem_wg(dest, source, nelems, pe);
}

__device__ void IPCContext::getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  getmem_wg(dest, source, nelems, pe);
}

__device__ void IPCContext::putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[pe] + L_offset,
                        const_cast<void *>(source), nelems);
}

__device__ void IPCContext::getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  uint64_t L_offset =
      const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[pe] + L_offset,
                        nelems);
}

__device__ void IPCContext::putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  putmem_wave(dest, source, nelems, pe);
}

__device__ void IPCContext::getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  getmem_wave(dest, source, nelems, pe);
}

}  // namespace rocshmem
