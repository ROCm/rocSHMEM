/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "context_incl.hpp"
#include "gpu_ib/gda_device.hpp"

namespace rocshmem {

__host__ Context::Context(GDADevice* device)
    : num_pes(device->num_pes), my_pe(device->my_pe) {
}

/******************************************************************************
 ********************** CONTEXT DISPATCH IMPLEMENTATIONS **********************
 *****************************************************************************/

__host__
void Context::putmem(void* dest, const void* source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->putmem(dest, source, nelems, pe);
}

__host__
void Context::putmem_nbi(void* dest, const void* source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }
  static_cast<GPUIBHostContext*>(this)->putmem_nbi(dest, source, nelems, pe);
}

__host__
void Context::quiet() {
  static_cast<GPUIBHostContext*>(this)->quiet();
}

__host__
void Context::sync_all() {
  static_cast<GPUIBHostContext*>(this)->sync_all();
}

__host__
void Context::barrier_all() {
  static_cast<GPUIBHostContext*>(this)->barrier_all();
}

}  // namespace rocshmem
