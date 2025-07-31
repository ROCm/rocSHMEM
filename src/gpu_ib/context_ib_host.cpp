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

#include "context_ib_host.hpp"

#include <mpi.h>

#include "context_incl.hpp"
#include "host/host.hpp"
#include "gda_device.hpp"

namespace rocshmem {

GPUIBHostContext::GPUIBHostContext(GDADevice *device)
    : Context(device) {
  host_interface = device->host_interface;
  context_window_info = host_interface->acquire_window_context();
}

GPUIBHostContext::~GPUIBHostContext() {
  host_interface->release_window_context(context_window_info);
}

void GPUIBHostContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
  host_interface->putmem_nbi(dest, source, nelems, pe, context_window_info);
}

void GPUIBHostContext::putmem(void *dest, const void *source, size_t nelems, int pe) {
  host_interface->putmem(dest, source, nelems, pe, context_window_info);
}

void GPUIBHostContext::quiet() {
  host_interface->quiet(context_window_info);
}

void GPUIBHostContext::sync_all() {
  host_interface->sync_all(context_window_info);
}

void GPUIBHostContext::barrier_all() {
  host_interface->barrier_all(context_window_info);
}

}  // namespace rocshmem
