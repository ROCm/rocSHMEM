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
#include <cstring>
#include <functional>
#include <random>
#include <unistd.h>

#include "context_incl.hpp"
#include "gpu_ib/backend_ib.hpp"
#include "mpi_init_singleton.hpp"
#include "team.hpp"
#include "util.hpp"

namespace rocshmem {

#define VERIFY_BACKEND() {                                            \
    if (!backend) {                                                   \
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n", \
              "Call 'rocshmem_init'", __FILE__, __LINE__);            \
      abort();                                                        \
    }                                                                 \
  }

GPUIBBackend *backend = nullptr;

rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

[[maybe_unused]] void inline library_init(MPI_Comm comm) {
  assert(!backend);
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) { abort(); }
  if (count == 0) { abort(); }

  rocm_init();
  rocshmem_env_config_init();

  CHECK_HIP(hipHostMalloc(&backend, sizeof(GPUIBBackend)));
  backend = new (backend) GPUIBBackend(comm);
  if (!backend) { abort(); }
}

[[maybe_unused]] int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t *attr) {
  MPI_Comm comm = MPI_COMM_WORLD;

  if ((attr == nullptr) || 
      ((flags != ROCSHMEM_INIT_WITH_UNIQUEID) &&
       (flags != ROCSHMEM_INIT_WITH_MPI_COMM)) ) {
    fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
            "Call 'rocshmem_init_attr: invalid input argument'",
            __FILE__, __LINE__);
    return ROCSHMEM_ERROR;
  }

  if (flags == ROCSHMEM_INIT_WITH_MPI_COMM) {
    comm = *(static_cast<MPI_Comm*>(attr->mpi_comm));
  }

  library_init(comm);

  if (flags == ROCSHMEM_INIT_WITH_UNIQUEID) {
    int worldsize = backend->getNumPEs();
    if (worldsize != attr->nranks) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_init_attr: mismatch between world-team size and "
	      "attribute value'",  __FILE__, __LINE__);
      abort();
    }
  }
  return ROCSHMEM_SUCCESS;
}

[[maybe_unused]] int rocshmem_set_attr_uniqueid_args(int rank, int nranks, rocshmem_uniqueid_t *uid, rocshmem_init_attr_t *attr) {
  if (uid == nullptr || attr == nullptr) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_get_uniqueid: invalid input argument'",
	      __FILE__, __LINE__);
      return ROCSHMEM_ERROR;
  }
  attr->rank = rank;
  attr->nranks = nranks;
  attr->uid = *uid;
  attr->mpi_comm = nullptr;
  return ROCSHMEM_SUCCESS;
}

[[maybe_unused]] int rocshmem_get_uniqueid(rocshmem_uniqueid_t *uid) {
  if (uid == nullptr) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_get_uniqueid: invalid input argument'",
	      __FILE__, __LINE__);
      return ROCSHMEM_ERROR;
  }
  std::random_device dev;
  std::mt19937_64 rng(dev());
  std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
  char hostname[HOST_NAME_MAX+1];
  if (0 != gethostname(hostname, HOST_NAME_MAX)) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_get_uniqueid: could not get hostname'",
	      __FILE__, __LINE__);
      return ROCSHMEM_ERROR;
  }
  uid->random = dist(rng);
  std::memcpy(uid->hostname, hostname, ROCSHMEM_HOSTNAME_LEN);
  uid->pid = static_cast<uint32_t>(getpid());
  return ROCSHMEM_SUCCESS;
}

[[maybe_unused]] void rocshmem_init(MPI_Comm comm) {
  library_init(comm);
}

[[maybe_unused]] int rocshmem_my_pe() {
  MPIInitSingleton *s = s->GetInstance();
  return s->get_rank();
}

[[maybe_unused]] int rocshmem_n_pes() {
  MPIInitSingleton *s = s->GetInstance();
  return s->get_nprocs();
}

[[maybe_unused]] void *rocshmem_malloc(size_t size) {
  VERIFY_BACKEND();
  void *ptr;
  backend->heap.malloc(&ptr, size);
  rocshmem_barrier_all();
  return ptr;
}

[[maybe_unused]] void rocshmem_free(void *ptr) {
  VERIFY_BACKEND();
  rocshmem_barrier_all();
  backend->heap.free(ptr);
}

[[maybe_unused]] void rocshmem_finalize() {
  VERIFY_BACKEND();
  backend->destroy_remaining_ctxs();
  auto team_destroy{std::bind(&GPUIBBackend::team_destroy, backend, std::placeholders::_1)};
  backend->team_tracker.destroy_all(team_destroy);
  backend->~GPUIBBackend();
  CHECK_HIP(hipHostFree(backend));
  delete MPIInitSingleton::GetInstance();
}

void rocshmem_global_exit(int status) {
  VERIFY_BACKEND();
  backend->global_exit(status);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

int rocshmem_team_n_pes(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->num_pes;
  }
}

int rocshmem_team_my_pe(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->my_pe;
  }
}

inline int pe_in_active_set(int start, int stride, int size, int pe) {
  int translated_pe = (pe - start) / stride;
  if ((pe < start) || ((pe - start) % stride) || (translated_pe >= size)) {
    translated_pe = -1;
  }
  return translated_pe;
}

int rocshmem_team_split_strided(rocshmem_team_t parent_team, int start, int stride, int size,
    [[maybe_unused]] const rocshmem_team_config_t *config,
    [[maybe_unused]] long config_mask, rocshmem_team_t *new_team) {
  VERIFY_BACKEND();

  *new_team = ROCSHMEM_TEAM_INVALID;

  auto num_user_teams{backend->team_tracker.get_num_user_teams()};
  auto max_num_teams{backend->team_tracker.get_max_num_teams()};
  if (num_user_teams >= max_num_teams - 1) {
    return -1;
  }

  if (parent_team == ROCSHMEM_TEAM_INVALID) {
    return 0;
  }

  Team *parent_team_obj = get_internal_team(parent_team);

  if (start < 0 || start >= parent_team_obj->num_pes || size < 1 || size > parent_team_obj->num_pes || stride < 1) {
    return -1;
  }

  int pe_start_in_world = parent_team_obj->get_pe_in_world(start);
  int stride_in_world = stride * parent_team_obj->tinfo_wrt_world->stride;
  int pe_end_in_world = pe_start_in_world + stride_in_world * (size - 1);

  if (pe_end_in_world > backend->num_pes) {
    return -1;
  }

  int my_pe_in_world = backend->my_pe;
  int my_pe_in_new_team = pe_in_active_set(pe_start_in_world, stride_in_world, size, my_pe_in_world);

  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  new (team_info_wrt_parent) TeamInfo(parent_team_obj, start, stride, size);
  auto *team_world{backend->team_tracker.get_team_world()};
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));
  new (team_info_wrt_world) TeamInfo(team_world, pe_start_in_world, stride_in_world, size);

  int color;
  if (my_pe_in_new_team < 0) {
    color = MPI_UNDEFINED;
  } else {
    color = 1;
  }

  MPI_Comm team_comm;
  MPI_Comm_split(parent_team_obj->mpi_comm, color, my_pe_in_world, &team_comm);

  if (my_pe_in_new_team < 0) {
    *new_team = ROCSHMEM_TEAM_INVALID;
  } else {
    backend->create_new_team(parent_team_obj, team_info_wrt_parent, team_info_wrt_world, size, my_pe_in_new_team, team_comm, new_team);
    backend->team_tracker.track(*new_team);
  }
  return 0;
}

void rocshmem_team_destroy(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID || team == ROCSHMEM_TEAM_WORLD) {
    return;
  }
  backend->team_tracker.untrack(team);
  backend->team_destroy(team);
}

int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe, rocshmem_team_t dst_team) {
  return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

template <typename T>
void rocshmem_put(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

void rocshmem_putmem(void *dest, const void *source, size_t nelems, int pe) {
  rocshmem_ctx_putmem(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
void rocshmem_p(T *dest, T value, int pe) {
  rocshmem_p(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
void rocshmem_put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

void rocshmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
T rocshmem_atomic_fetch_add(T *dest, T val, int pe) {
  return rocshmem_atomic_fetch_add(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe) {
  return rocshmem_atomic_compare_swap(ROCSHMEM_HOST_CTX_DEFAULT, dest, cond, val, pe);
}

template <typename T>
void rocshmem_atomic_add(T *dest, T val, int pe) {
  rocshmem_atomic_add(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
void rocshmem_atomic_set(T *dest, T val, int pe) {
  rocshmem_atomic_set(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
T rocshmem_atomic_swap(T *dest, T value, int pe) {
  return rocshmem_atomic_swap(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

void rocshmem_fence() {
  rocshmem_ctx_fence(ROCSHMEM_HOST_CTX_DEFAULT);
}

void rocshmem_quiet() {
  rocshmem_ctx_quiet(ROCSHMEM_HOST_CTX_DEFAULT);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

Context *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

int rocshmem_ctx_create(rocshmem_ctx_t *ctx) {
  void *phys_ctx;
  backend->ctx_create(&phys_ctx);
  ctx->ctx_opaque = phys_ctx;
  ctx->team_opaque = nullptr;
  backend->track_ctx(reinterpret_cast<Context *>(phys_ctx));
  return 0;
}

void rocshmem_ctx_destroy(rocshmem_ctx_t ctx) {
  Context *phys_ctx = get_internal_ctx(ctx);
  backend->untrack_ctx(phys_ctx);
  backend->ctx_destroy(phys_ctx);
}

template <typename T>
void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->put(dest, source, nelems, pe);
}

void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->putmem(dest, source, nelems, pe);
}

template <typename T>
void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {
  get_internal_ctx(ctx)->p(dest, value, pe);
}

template <typename T>
void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe);
}

void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe) {
  get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe);
}

template <typename T>
T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, val, pe);
}

template <typename T>
T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond, T val, int pe) {
  return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T>
void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  get_internal_ctx(ctx)->amo_add<T>(dest, val, pe);
}

template <typename T>
void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  get_internal_ctx(ctx)->amo_set(dest, val, pe);
}

template <typename T>
T rocshmem_atomic_swap(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  return get_internal_ctx(ctx)->amo_swap(dest, val, pe);
}

void rocshmem_ctx_fence(rocshmem_ctx_t ctx) {
  get_internal_ctx(ctx)->fence();
}

void rocshmem_ctx_quiet(rocshmem_ctx_t ctx) {
  get_internal_ctx(ctx)->quiet();
}

void rocshmem_barrier_all() {
  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->barrier_all();
}

template <typename T>
void rocshmem_wait_until(T *ivars, int cmp, T val) {
  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until(ivars, cmp, val);
}

template <typename T>
void rocshmem_wait_until_all(T *ivars, size_t nelems, const int* status, int cmp, T val) {
  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_all(ivars, nelems, status, cmp, val);
}

template <typename T>
size_t rocshmem_wait_until_any(T *ivars, size_t nelems, const int* status, int cmp, T val) {
  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_any(ivars, nelems, status, cmp, val);
}

template <typename T>
size_t rocshmem_wait_until_some(T *ivars, size_t nelems, size_t* indices, const int* status, int cmp, T val) {
  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_some(ivars, nelems, indices, status, cmp, val);
}

template <typename T>
int rocshmem_test(T *ivars, int cmp, T val) {
  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->test(ivars, cmp, val);
}

/*
 * Declare templates for the required datatypes (for the compiler)
 */
#define RMA_GEN(T)                                                                                        \
  template void rocshmem_put<T>(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe);     \
  template void rocshmem_put_nbi<T>(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe); \
  template void rocshmem_p<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);                              \
  template void rocshmem_put<T>(T *dest, const T *source, size_t nelems, int pe);                         \
  template void rocshmem_put_nbi<T>(T *dest, const T *source, size_t nelems, int pe);                     \
  template void rocshmem_p<T>(T *dest, T value, int pe);

/*
 * Declare templates for the standard amo types
 */
#define AMO_STANDARD_GEN(T)                                                                         \
  template T rocshmem_atomic_compare_swap<T>(rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe); \
  template T rocshmem_atomic_compare_swap<T>(T *dest, T cond, T value, int pe);                     \
  template T rocshmem_atomic_fetch_add<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);            \
  template T rocshmem_atomic_fetch_add<T>(T *dest, T value, int pe);                                \
  template void rocshmem_atomic_add<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);               \
  template void rocshmem_atomic_add<T>(T *dest, T value, int pe);

/*
 * Declare templates for the extended amo types
 */
#define AMO_EXTENDED_GEN(T)                                                           \
  template void rocshmem_atomic_set<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe); \
  template void rocshmem_atomic_set<T>(T *dest, T value, int pe);                     \
  template T rocshmem_atomic_swap<T>(rocshmem_ctx_t ctx, T *dest, T value, int pe);   \
  template T rocshmem_atomic_swap<T>(T *dest, T value, int pe);

/*
 * Declare templates for the wait types
 */
#define WAIT_GEN(T)                                                                                                         \
  template void rocshmem_wait_until<T>(T *ivars, int cmp, T val);                                                           \
  template int rocshmem_test<T>(T *ivars, int cmp, T val);                                                                  \
  template void Context::wait_until<T>(T *ivars, int cmp, T val);                                                           \
  template size_t rocshmem_wait_until_any<T>(T *ivars, size_t nelems, const int* status, int cmp, T val);                   \
  template void rocshmem_wait_until_all<T>(T *ivars, size_t nelems, const int* status, int cmp, T val);                     \
  template size_t rocshmem_wait_until_some<T>(T *ivars, size_t nelems, size_t* indices, const int* status, int cmp, T val); \
  template int Context::test<T>(T *ivars, int cmp, T val);

/*
 * Define APIs to call the template functions
 */

#define RMA_DEF_GEN(T, TNAME)                                                                                \
  void rocshmem_ctx_##TNAME##_put(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {     \
    rocshmem_put<T>(ctx, dest, source, nelems, pe);                                                          \
  }                                                                                                          \
  void rocshmem_ctx_##TNAME##_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) { \
    rocshmem_put_nbi<T>(ctx, dest, source, nelems, pe);                                                      \
  }                                                                                                          \
  void rocshmem_ctx_##TNAME##_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {                              \
    rocshmem_p<T>(ctx, dest, value, pe);                                                                     \
  }                                                                                                          \
  void rocshmem_##TNAME##_put(T *dest, const T *source, size_t nelems, int pe) {                             \
    rocshmem_put<T>(dest, source, nelems, pe);                                                               \
  }                                                                                                          \
  void rocshmem_##TNAME##_put_nbi(T *dest, const T *source, size_t nelems, int pe) {                         \
    rocshmem_put_nbi<T>(dest, source, nelems, pe);                                                           \
  }                                                                                                          \
  void rocshmem_##TNAME##_p(T *dest, T value, int pe) {                                                      \
    rocshmem_p<T>(dest, value, pe);                                                                          \
  }

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                                                 \
  T rocshmem_ctx_##TNAME##_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe) { \
    return rocshmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe);                                \
  }                                                                                                    \
  T rocshmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, int pe) {                         \
    return rocshmem_atomic_compare_swap<T>(dest, cond, value, pe);                                     \
  }                                                                                                    \
  T rocshmem_ctx_##TNAME##_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T value, int pe) {            \
    return rocshmem_atomic_fetch_add<T>(ctx, dest, value, pe);                                         \
  }                                                                                                    \
  T rocshmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) {                                    \
    return rocshmem_atomic_fetch_add<T>(dest, value, pe);                                              \
  }                                                                                                    \
  void rocshmem_ctx_##TNAME##_atomic_add(rocshmem_ctx_t ctx, T *dest, T value, int pe) {               \
    rocshmem_atomic_add<T>(ctx, dest, value, pe);                                                      \
  }                                                                                                    \
  void rocshmem_##TNAME##_atomic_add(T *dest, T value, int pe) {                                       \
    rocshmem_atomic_add<T>(dest, value, pe);                                                           \
  }

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                                   \
  void rocshmem_ctx_##TNAME##_atomic_set(rocshmem_ctx_t ctx, T *dest, T value, int pe) { \
    rocshmem_atomic_set<T>(ctx, dest, value, pe);                                        \
  }                                                                                      \
  void rocshmem_##TNAME##_atomic_set(T *dest, T value, int pe) {                         \
    rocshmem_atomic_set<T>(dest, value, pe);                                             \
  }                                                                                      \
  T rocshmem_ctx_##TNAME##_atomic_swap(rocshmem_ctx_t ctx, T *dest, T value, int pe) {   \
    return rocshmem_atomic_swap<T>(ctx, dest, value, pe);                                \
  }                                                                                      \
  T rocshmem_##TNAME##_atomic_swap(T *dest, T value, int pe) {                           \
    return rocshmem_atomic_swap<T>(dest, value, pe);                                     \
  }

#define WAIT_DEF_GEN(T, TNAME)                                                                                             \
  void rocshmem_##TNAME##_wait_until(T *ivars, int cmp, T val) {                                                           \
    rocshmem_wait_until<T>(ivars, cmp, val);                                                                               \
  }                                                                                                                        \
  size_t rocshmem_##TNAME##_wait_until_any(T *ivars, size_t nelems, const int* status, int cmp, T val) {                   \
    return rocshmem_wait_until_any<T>(ivars, nelems, status, cmp, val);                                                    \
  }                                                                                                                        \
  void rocshmem_##TNAME##_wait_until_all(T *ivars, size_t nelems, const int* status, int cmp, T val) {                     \
    rocshmem_wait_until_all<T>(ivars, nelems, status, cmp, val);                                                           \
  }                                                                                                                        \
  size_t rocshmem_##TNAME##_wait_until_some(T *ivars, size_t nelems, size_t* indices, const int* status, int cmp, T val) { \
    return rocshmem_wait_until_some<T>(ivars, nelems, indices, status, cmp, val);                                          \
  }                                                                                                                        \
  int rocshmem_##TNAME##_test(T *ivars, int cmp, T val) {                                                                  \
    return rocshmem_test<T>(ivars, cmp, val);                                                                              \
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
WAIT_GEN(float)
WAIT_GEN(double)
// WAIT_GEN(long double)
WAIT_GEN(char)
WAIT_GEN(unsigned char)
WAIT_GEN(unsigned short)
WAIT_GEN(signed char)
WAIT_GEN(short)
WAIT_GEN(int)
WAIT_GEN(long)
WAIT_GEN(long long)
WAIT_GEN(unsigned int)
WAIT_GEN(unsigned long)
WAIT_GEN(unsigned long long)

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

WAIT_DEF_GEN(float, float)
WAIT_DEF_GEN(double, double)
// WAIT_DEF_GEN(long double, longdouble)
WAIT_DEF_GEN(char, char)
WAIT_DEF_GEN(signed char, schar)
WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
WAIT_DEF_GEN(unsigned char, uchar)
WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)
// clang-format on

}  // namespace rocshmem
