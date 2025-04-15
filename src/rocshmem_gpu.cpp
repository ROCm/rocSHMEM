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

/**
 * @file rocshmem.cpp
 * @brief Public header for rocSHMEM device and host libraries.
 */

#include <rocshmem/rocshmem.hpp>

#include <cstdlib>
#include <hip/hip_runtime.h>

#include "context_incl.hpp"
#include "gpu_ib/backend_ib.hpp"
#include "team.hpp"
#include "templates.hpp"
#include "util.hpp"

/******************************************************************************
 **************************** Device Vars And Init ****************************
 *****************************************************************************/

namespace rocshmem {

__device__ __constant__ rocshmem_ctx_t ROCSHMEM_CTX_DEFAULT{};

__constant__ GPUIBBackend *device_backend_proxy;

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

__device__
void rocshmem_putmem(void *dest, const void *source, size_t nelems, int pe) {
  rocshmem_ctx_putmem(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ 
void rocshmem_put(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ 
void rocshmem_p(T *dest, T value, int pe) {
  rocshmem_p(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

__device__ 
void rocshmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ 
void rocshmem_put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put_nbi(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ 
void rocshmem_quiet() {
  rocshmem_ctx_quiet(ROCSHMEM_CTX_DEFAULT);
}

template <typename T>
__device__ 
T rocshmem_atomic_fetch_add(T *dest, T val, int pe) {
  return rocshmem_atomic_fetch_add(ROCSHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ 
T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe) {
  return rocshmem_atomic_compare_swap(ROCSHMEM_CTX_DEFAULT, dest, cond, val, pe);
}

template <typename T>
__device__ 
void rocshmem_atomic_add(T *dest, T val, int pe) {
  rocshmem_atomic_add(ROCSHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ 
void rocshmem_atomic_set(T *dest, T value, int pe) {
  rocshmem_atomic_set(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ 
T rocshmem_atomic_swap(T *dest, T value, int pe) {
  return rocshmem_atomic_swap(ROCSHMEM_CTX_DEFAULT, dest, value, pe);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__device__ 
int translate_pe(rocshmem_ctx_t ctx, int pe) {
  if (ctx.team_opaque) {
    TeamInfo *tinfo = reinterpret_cast<TeamInfo*>(ctx.team_opaque);
    return (tinfo->pe_start + tinfo->stride * pe);
  } else {
    return pe;
  }
}

__host__ 
void set_internal_ctx(rocshmem_ctx_t *ctx) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT), ctx, sizeof(rocshmem_ctx_t), 0, hipMemcpyHostToDevice));
}

__device__ 
Context *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<Context*>(ctx.ctx_opaque);
}

__device__ 
int rocshmem_wg_ctx_create(rocshmem_ctx_t *ctx) {
  bool result{true};
  if (get_flat_block_id() == 0) {
    ctx->team_opaque = reinterpret_cast<TeamInfo*>(ROCSHMEM_CTX_DEFAULT.team_opaque);
    result = device_backend_proxy->create_ctx(ctx);
  }
  __syncthreads();
  return result == true ? 0 : -1;
}

__device__ 
int rocshmem_wg_team_create_ctx(rocshmem_team_t team, rocshmem_ctx_t *ctx) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  }

  bool result{true};
  if (get_flat_block_id() == 0) {
    Team *team_obj{get_internal_team(team)};
    TeamInfo *info_wrt_world = team_obj->tinfo_wrt_world;
    ctx->team_opaque = info_wrt_world;
    result = device_backend_proxy->create_ctx(ctx);
  }
  __syncthreads();

  return result == true ? 0 : -1;
}

__device__ 
void rocshmem_wg_ctx_destroy([[maybe_unused]] rocshmem_ctx_t *ctx) {
  if (get_flat_block_id() == 0) {
    device_backend_proxy->destroy_ctx(ctx);
  }
}

__device__ 
void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  get_internal_ctx(ctx)->putmem(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ 
void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  get_internal_ctx(ctx)->put(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ 
void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  get_internal_ctx(ctx)->p(dest, value, pe_in_world);
}

__device__
void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__
void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  int pe_in_world = translate_pe(ctx, pe);
  get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe_in_world);
}

__device__ 
void rocshmem_ctx_quiet(rocshmem_ctx_t ctx) {
  get_internal_ctx(ctx)->quiet();
}

__device__ 
void *rocshmem_ptr(const void *dest, int pe) {
  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->shmem_ptr(dest, pe);
}

__device__ 
void rocshmem_ctx_wg_barrier_all(rocshmem_ctx_t ctx) {
  get_internal_ctx(ctx)->barrier_all();
}

__device__ 
void rocshmem_wg_barrier_all() {
  rocshmem_ctx_wg_barrier_all(ROCSHMEM_CTX_DEFAULT);
}

__device__ 
void rocshmem_barrier(rocshmem_team_t team) {
  get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->barrier(team);
}

__device__ 
void rocshmem_ctx_wg_sync_all(rocshmem_ctx_t ctx) {
  get_internal_ctx(ctx)->sync_all();
}

__device__ 
void rocshmem_wg_sync_all() {
  rocshmem_ctx_wg_sync_all(ROCSHMEM_CTX_DEFAULT);
}

__device__ 
void rocshmem_ctx_wg_team_sync(rocshmem_ctx_t ctx, rocshmem_team_t team) {
  get_internal_ctx(ctx)->sync(team);
}

__device__ 
void rocshmem_wg_team_sync(rocshmem_team_t team) {
  rocshmem_ctx_wg_team_sync(ROCSHMEM_CTX_DEFAULT, team);
}

__device__ 
int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx) {
  TeamInfo *tinfo = reinterpret_cast<TeamInfo*>(ctx.team_opaque);
  return tinfo->size;
}

__device__ 
int rocshmem_n_pes() {
  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->num_pes;
}

__device__ 
int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx) {
  TeamInfo *tinfo = reinterpret_cast<TeamInfo*>(ctx.team_opaque);
  int my_pe{get_internal_ctx(ctx)->my_pe};
  int pe_start{tinfo->pe_start};
  int stride{tinfo->stride};
  int size{tinfo->size};

  int translated_pe = (my_pe - pe_start) / stride;
  if ((my_pe < pe_start) || ((my_pe - pe_start) % stride) || (translated_pe >= size)) {
    translated_pe = -1;
  }
  return translated_pe;
}

__device__ 
int rocshmem_my_pe() {
  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->my_pe;
}

template <typename T>
__device__ 
T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, val, pe);
}

template <typename T>
__device__ 
T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond, T val, int pe) {
  return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T>
__device__ 
void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  get_internal_ctx(ctx)->amo_add<T>(dest, val, pe);
}

template <typename T>
__device__ 
void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  get_internal_ctx(ctx)->amo_set(dest, val, pe);
}

template <typename T>
__device__ 
T rocshmem_atomic_swap(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  return get_internal_ctx(ctx)->amo_swap(dest, val, pe);
}

/******************************************************************************
 ******************** SHMEM X RMA API for WG and Wave level *******************
 *****************************************************************************/

__device__ 
void rocshmem_ctx_putmem_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->putmem_wave(dest, source, nelems, pe);
}

__device__ 
void rocshmem_ctx_putmem_nbi_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->putmem_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ 
void rocshmem_put_wave(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->put_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ 
void rocshmem_put_nbi_wave(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->put_nbi_wave(dest, source, nelems, pe);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__device__ 
int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe, rocshmem_team_t dst_team) {
  return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************* Template Generation Macros *************************
 *****************************************************************************/

/*
 * Declare templates for the required datatypes (for the compiler)
 */
#define RMA_GEN(T)                                                                                                        \
  template __device__ void rocshmem_put<T>(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe);          \
  template __device__ void rocshmem_put_nbi<T>(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe);      \
  template __device__ void rocshmem_p<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);                                   \
  template __device__ void rocshmem_put<T>(T *dest, const T *source, size_t nelems, int pe);                              \
  template __device__ void rocshmem_put_nbi<T>(T *dest, const T *source, size_t nelems, int pe);                          \
  template __device__ void rocshmem_p<T>(T *dest, T value, int pe);                                                       \
  template __device__ void rocshmem_put_wave<T>(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe);     \
  template __device__ void rocshmem_put_wave<T>(T *dest, const T *source, size_t nelems, int pe);                         \
  template __device__ void rocshmem_put_nbi_wave<T>(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe); \
  template __device__ void rocshmem_put_nbi_wave<T>(T *dest, const T *source, size_t nelems, int pe);

/*
 * Declare templates for the standard amo types
 */
#define AMO_STANDARD_GEN(T)                                                                                    \
  template __device__ T rocshmem_atomic_compare_swap<T>(rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe); \
  template __device__ T rocshmem_atomic_compare_swap<T>(T *dest, T cond, T value, int pe);                     \
  template __device__ T rocshmem_atomic_fetch_add<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);            \
  template __device__ T rocshmem_atomic_fetch_add<T>(T *dest, T value, int pe);                                \
  template __device__ void rocshmem_atomic_add<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);               \
  template __device__ void rocshmem_atomic_add<T>(T *dest, T value, int pe);

/*
 * Declare templates for the extended amo types
 */
#define AMO_EXTENDED_GEN(T)                                                                      \
  template __device__ void rocshmem_atomic_set<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe); \
  template __device__ void rocshmem_atomic_set<T>(T *dest, T value, int pe);                     \
  template __device__ T rocshmem_atomic_swap<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);   \
  template __device__ T rocshmem_atomic_swap<T>(T *dest, T value, int pe);

/*
 * Declare templates for the bitwise amo types
 */
#define RMA_DEF_GEN(T, TNAME)                                                                                                \
  __device__ void rocshmem_ctx_##TNAME##_put(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {          \
    rocshmem_put<T>(ctx, dest, source, nelems, pe);                                                                          \
  }                                                                                                                          \
  __device__ void rocshmem_ctx_##TNAME##_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {      \
    rocshmem_put_nbi<T>(ctx, dest, source, nelems, pe);                                                                      \
  }                                                                                                                          \
  __device__ void rocshmem_ctx_##TNAME##_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {                                   \
    rocshmem_p<T>(ctx, dest, value, pe);                                                                                     \
  }                                                                                                                          \
  __device__ void rocshmem_##TNAME##_put(T *dest, const T *source, size_t nelems, int pe) {                                  \
    rocshmem_put<T>(dest, source, nelems, pe);                                                                               \
  }                                                                                                                          \
  __device__ void rocshmem_##TNAME##_put_nbi(T *dest, const T *source, size_t nelems, int pe) {                              \
    rocshmem_put_nbi<T>(dest, source, nelems, pe);                                                                           \
  }                                                                                                                          \
  __device__ void rocshmem_##TNAME##_p(T *dest, T value, int pe) {                                                           \
    rocshmem_p<T>(dest, value, pe);                                                                                          \
  }                                                                                                                          \
  __device__ void rocshmem_ctx_##TNAME##_put_wave(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {     \
    rocshmem_put_wave<T>(ctx, dest, source, nelems, pe);                                                                     \
  }                                                                                                                          \
  __device__ void rocshmem_##TNAME##_put_wave(T *dest, const T *source, size_t nelems, int pe) {                             \
    rocshmem_put_wave<T>(dest, source, nelems, pe);                                                                          \
  }                                                                                                                          \
  __device__ void rocshmem_ctx_##TNAME##_put_nbi_wave(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) { \
    rocshmem_put_nbi_wave<T>(ctx, dest, source, nelems, pe);                                                                 \
  }                                                                                                                          \
  __device__ void rocshmem_##TNAME##_put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {                         \
    rocshmem_put_nbi_wave<T>(dest, source, nelems, pe);                                                                      \
  }

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                                                            \
  __device__ T rocshmem_ctx_##TNAME##_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe) { \
    return rocshmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe);                                           \
  }                                                                                                               \
  __device__ T rocshmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, int pe) {                         \
    return rocshmem_atomic_compare_swap<T>(dest, cond, value, pe);                                                \
  }                                                                                                               \
  __device__ T rocshmem_ctx_##TNAME##_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T value, int pe) {            \
    return rocshmem_atomic_fetch_add<T>(ctx, dest, value, pe);                                                    \
  }                                                                                                               \
  __device__ T rocshmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) {                                    \
    return rocshmem_atomic_fetch_add<T>(dest, value, pe);                                                         \
  }                                                                                                               \
  __device__ void rocshmem_ctx_##TNAME##_atomic_add(rocshmem_ctx_t ctx, T *dest, T value, int pe) {               \
    rocshmem_atomic_add<T>(ctx, dest, value, pe);                                                                 \
  }                                                                                                               \
  __device__ void rocshmem_##TNAME##_atomic_add(T *dest, T value, int pe) {                                       \
    rocshmem_atomic_add<T>(dest, value, pe);                                                                      \
  }

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                                              \
  __device__ void rocshmem_ctx_##TNAME##_atomic_set(rocshmem_ctx_t ctx, T *dest, T value, int pe) { \
    rocshmem_atomic_set<T>(ctx, dest, value, pe);                                                   \
  }                                                                                                 \
  __device__ void rocshmem_##TNAME##_atomic_set(T *dest, T value, int pe) {                         \
    rocshmem_atomic_set<T>(dest, value, pe);                                                        \
  }                                                                                                 \
  __device__ T rocshmem_ctx_##TNAME##_atomic_swap(rocshmem_ctx_t ctx, T *dest, T value, int pe) {   \
    return rocshmem_atomic_swap<T>(ctx, dest, value, pe);                                           \
  }                                                                                                 \
  __device__ T rocshmem_##TNAME##_atomic_swap(T *dest, T value, int pe) {                           \
    return rocshmem_atomic_swap<T>(dest, value, pe);                                                \
  }

/******************************************************************************
 ************************* Macro Invocation Per Type **************************
 *****************************************************************************/

// clang-format off

RMA_GEN(float)
RMA_GEN(double)
// RMA_GEN(long double)
RMA_GEN(char)
RMA_GEN(signed char)
RMA_GEN(short)
RMA_GEN(int)
RMA_GEN(long)
RMA_GEN(long long)
RMA_GEN(unsigned char)
RMA_GEN(unsigned short)
RMA_GEN(unsigned int)
RMA_GEN(unsigned long)
RMA_GEN(unsigned long long)

AMO_STANDARD_GEN(int)
AMO_STANDARD_GEN(long)
AMO_STANDARD_GEN(long long)
AMO_STANDARD_GEN(unsigned int)
AMO_STANDARD_GEN(unsigned long)
AMO_STANDARD_GEN(unsigned long long)

AMO_EXTENDED_GEN(float)
AMO_EXTENDED_GEN(double)
AMO_EXTENDED_GEN(int)
AMO_EXTENDED_GEN(long)
AMO_EXTENDED_GEN(long long)
AMO_EXTENDED_GEN(unsigned int)
AMO_EXTENDED_GEN(unsigned long)
AMO_EXTENDED_GEN(unsigned long long)

/* Supported synchronization types */

RMA_DEF_GEN(float, float)
RMA_DEF_GEN(double, double)
RMA_DEF_GEN(char, char)
// RMA_DEF_GEN(long double, longdouble)
RMA_DEF_GEN(signed char, schar)
RMA_DEF_GEN(short, short)
RMA_DEF_GEN(int, int)
RMA_DEF_GEN(long, long)
RMA_DEF_GEN(long long, longlong)
RMA_DEF_GEN(unsigned char, uchar)
RMA_DEF_GEN(unsigned short, ushort)
RMA_DEF_GEN(unsigned int, uint)
RMA_DEF_GEN(unsigned long, ulong)
RMA_DEF_GEN(unsigned long long, ulonglong)
RMA_DEF_GEN(int8_t, int8)
RMA_DEF_GEN(int16_t, int16)
RMA_DEF_GEN(int32_t, int32)
RMA_DEF_GEN(int64_t, int64)
RMA_DEF_GEN(uint8_t, uint8)
RMA_DEF_GEN(uint16_t, uint16)
RMA_DEF_GEN(uint32_t, uint32)
RMA_DEF_GEN(uint64_t, uint64)
RMA_DEF_GEN(size_t, size)
RMA_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_STANDARD_DEF_GEN(int, int)
AMO_STANDARD_DEF_GEN(long, long)
AMO_STANDARD_DEF_GEN(long long, longlong)
AMO_STANDARD_DEF_GEN(unsigned int, uint)
AMO_STANDARD_DEF_GEN(unsigned long, ulong)
AMO_STANDARD_DEF_GEN(unsigned long long, ulonglong)
AMO_STANDARD_DEF_GEN(int32_t, int32)
AMO_STANDARD_DEF_GEN(int64_t, int64)
AMO_STANDARD_DEF_GEN(uint32_t, uint32)
AMO_STANDARD_DEF_GEN(uint64_t, uint64)
AMO_STANDARD_DEF_GEN(size_t, size)
AMO_STANDARD_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_EXTENDED_DEF_GEN(float, float)
AMO_EXTENDED_DEF_GEN(double, double)
AMO_EXTENDED_DEF_GEN(int, int)
AMO_EXTENDED_DEF_GEN(long, long)
AMO_EXTENDED_DEF_GEN(long long, longlong)
AMO_EXTENDED_DEF_GEN(unsigned int, uint)
AMO_EXTENDED_DEF_GEN(unsigned long, ulong)
AMO_EXTENDED_DEF_GEN(unsigned long long, ulonglong)
AMO_EXTENDED_DEF_GEN(int32_t, int32)
AMO_EXTENDED_DEF_GEN(int64_t, int64)
AMO_EXTENDED_DEF_GEN(uint32_t, uint32)
AMO_EXTENDED_DEF_GEN(uint64_t, uint64)
AMO_EXTENDED_DEF_GEN(size_t, size)
AMO_EXTENDED_DEF_GEN(ptrdiff_t, ptrdiff)

// clang-format on

}  // namespace rocshmem
