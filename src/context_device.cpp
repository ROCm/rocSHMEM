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

#include "context_incl.hpp"
#include "gpu_ib/backend_ib.hpp"

namespace rocshmem {

__device__
Context::Context(GPUIBBackend* handle)
    : num_pes(handle->getNumPEs()), my_pe(handle->getMyPE()) {
  __syncthreads();
}

/******************************************************************************
 ************************** CONTEXT IMPLEMENTATIONS ***************************
 *****************************************************************************/

__device__
void Context::ctx_create() {
  static_cast<GPUIBContext*>(this)->ctx_create();
}

__device__
void Context::ctx_destroy() {
  static_cast<GPUIBContext*>(this)->ctx_destroy();
}

__device__
void Context::fence() {
  static_cast<GPUIBContext*>(this)->fence();
}

__device__
void Context::putmem(void* dest, const void* source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->putmem(dest, source, nelems, pe);
}

__device__
void Context::putmem_nbi(void* dest, const void* source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->putmem_nbi(dest, source, nelems, pe);
}

__device__
void Context::quiet() {
  static_cast<GPUIBContext*>(this)->quiet();
}

__device__
void* Context::shmem_ptr(const void* dest, int pe) {
  void *ret_val = static_cast<GPUIBContext*>(this)->shmem_ptr(dest, pe);
  return ret_val;
}

__device__
void Context::barrier_all() {
  static_cast<GPUIBContext*>(this)->barrier_all();
}

__device__
void Context::barrier(rocshmem_team_t team) {
  static_cast<GPUIBContext*>(this)->barrier(team);
}

__device__
void Context::sync_all() {
  static_cast<GPUIBContext*>(this)->sync_all();
}

__device__
void Context::sync(rocshmem_team_t team) {
  static_cast<GPUIBContext*>(this)->sync(team);
}

__device__
void Context::putmem_wave(void* dest, const void* source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->putmem_wave(dest, source, nelems, pe);
}

__device__
void Context::putmem_nbi_wave(void* dest, const void* source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBContext*>(this)->putmem_nbi_wave(dest, source, nelems, pe);
}

}  // namespace rocshmem
