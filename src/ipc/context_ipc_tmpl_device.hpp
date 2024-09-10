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

#ifndef LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_

#include "config.h"  // NOLINT(build/include_subdir)
#include "roc_shmem/roc_shmem.hpp"
#include "context_ipc_device.hpp"
#include "../util.hpp"
#include "ipc_team.hpp"

namespace rocshmem {

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <typename T>
__device__ void IPCContext::p(T *dest, T value, int pe) {
  putmem_nbi(dest, &value, sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put(T *dest, const T *source, size_t nelems,
				int pe) {
  putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put_nbi(T *dest, const T *source, size_t nelems,
				    int pe) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ T IPCContext::g(const T *source, int pe) {
  T ret;
  return ret;
}

template <typename T>
__device__ void IPCContext::get(T *dest, const T *source, size_t nelems,
				int pe) {
  getmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void IPCContext::get_nbi(T *dest, const T *source, size_t nelems,
				    int pe) {
  getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

// Atomics
template <typename T>
__device__ void IPCContext::amo_add(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOAdd(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ void IPCContext::amo_set(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOSet(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_swap(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ T IPCContext::amo_fetch_and(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void IPCContext::amo_and(void *dst, T value, int pe) {
  assert(false);
}

template <typename T>
__device__ T IPCContext::amo_fetch_or(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void IPCContext::amo_or(void *dst, T value, int pe) {
  assert(false);
}

template <typename T>
__device__ T IPCContext::amo_fetch_xor(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void IPCContext::amo_xor(void *dst, T value, int pe) {
  assert(false);
}

template <typename T>
__device__ void IPCContext::amo_cas(void *dest, T value, T cond, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOCas(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), cond,
      value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_add(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchAdd(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_cas(void *dest, T value, T cond, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchCas(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), cond,
      value);
}
  
// Collectives
template <typename T, ROC_SHMEM_OP Op>
__device__ void IPCContext::to_all(roc_shmem_team_t team, T *dest,
				   const T *source, int nreduce) {
  //to_all<T, Op>(dest, source, nreduce, pe_start, log_pe_stride, pe_size, pWrk,
  //              p_sync);
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void IPCContext::to_all(T *dest, const T *source, int nreduce,
				   int PE_start, int logPE_stride,
				   int PE_size, T *pWrk,
				   long *pSync) {  // NOLINT(runtime/int)
}

template <typename T>
__device__ void IPCContext::internal_put_broadcast(
    T *dst, const T *src, int nelems, int pe_root, int pe_start,
    int log_pe_stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int stride = 1 << log_pe_stride;
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (i != my_pe) {
        put_nbi_wg(dst, src, nelems, i);
      }
    }
  }
}

template <typename T>
__device__ void IPCContext::internal_get_broadcast(
  T *dst, const T *src, int nelems, int pe_root) {  // NOLINT(runtime/int)
  if (my_pe != pe_root) {
    get_wg(dst, src, nelems, pe_root);
  }
}

template <typename T>
__device__ void IPCContext::broadcast(roc_shmem_team_t team, T *dst,
				      const T *src, int nelems, int pe_root) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  /**
   * Ensure that the stride is a multiple of 2 .
   */
  int log_pe_stride = static_cast<int>(team_obj->tinfo_wrt_world->log_stride);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  broadcast<T>(dst, src, nelems, pe_root_world, pe_start, log_pe_stride,
               pe_size, p_sync);
}

template <typename T>
__device__ void IPCContext::broadcast(T *dst, const T *src, int nelems,
				      int pe_root, int pe_start,
				      int log_pe_stride, int pe_size,
				      long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) {
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, log_pe_stride,
                           pe_size);
  } else {
    internal_get_broadcast(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync(my_pe, pe_start, (1 << log_pe_stride), pe_size, p_sync);
}

template <typename T>
__device__ void IPCContext::alltoall(roc_shmem_team_t team, T *dst,
				     const T *src, int nelems) {
  alltoall_linear(team, dst, src, nelems);
}

template <typename T>
__device__ void IPCContext::alltoall_linear(roc_shmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  /**
   * Ensure that the stride is a multiple of 2
   */
  int log_pe_stride = static_cast<int>(team_obj->tinfo_wrt_world->log_stride);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wg(&dst[my_pe_in_team * nelems], &src[j * nelems], nelems, dest_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void IPCContext::fcollect(roc_shmem_team_t team, T *dst,
				     const T *src, int nelems) {
  fcollect_linear(team, dst, src, nelems);
}

template <typename T>
__device__ void IPCContext::fcollect_linear(roc_shmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  /**
   * Ensure that the stride is a multiple of 2.
   */
  int log_pe_stride = static_cast<int>(team_obj->tinfo_wrt_world->log_stride);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wg(&dst[my_pe_in_team * nelems], src, nelems, dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync(my_pe, pe_start, stride, pe_size, pSync);
}

// Block/wave functions
template <typename T>
__device__ void IPCContext::put_wg(T *dest, const T *source, size_t nelems,
				   int pe) {
  putmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put_nbi_wg(T *dest, const T *source,
				       size_t nelems, int pe) {
  putmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

  template <typename T>
__device__ void IPCContext::put_wave(T *dest, const T *source, size_t nelems,
				     int pe) {
  putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put_nbi_wave(T *dest, const T *source,
					 size_t nelems, int pe) {
  putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_wg(T *dest, const T *source, size_t nelems,
				   int pe) {
  getmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_nbi_wg(T *dest, const T *source,
				       size_t nelems, int pe) {
  getmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_wave(T *dest, const T *source, size_t nelems,
				     int pe) {
  getmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_nbi_wave(T *dest, const T *source,
					 size_t nelems, int pe) {
  getmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_
