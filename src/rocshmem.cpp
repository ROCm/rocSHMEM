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

/**
 * @file rocshmem.cpp
 * @brief Public header for rocSHMEM device and host libraries.
 *
 * This is the implementation for the public rocshmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 */

#include "rocshmem/rocshmem.hpp"

#include "gpu_ib/gda_device.hpp"
#include "context_incl.hpp"
#include "mpi_init_singleton.hpp"
#include "team.hpp"
#include "templates_host.hpp"
#include "util.hpp"
#include "bootstrap/bootstrap.hpp"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <cassert>
#include <unistd.h>

namespace rocshmem {

#define VERIFY_DEVICE()                                                      \
  {                                                                           \
    if (!device) {                                                           \
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",         \
              "Call 'rocshmem_init'", __FILE__, __LINE__);                    \
      abort();                                                                \
    }                                                                         \
  }

GDADevice *device = nullptr;
MPIInitSingleton *mpi_instance = nullptr;
TcpBootstrap *bootstr = nullptr;
rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

/**
 * Begin Host Code
 **/

[[maybe_unused]] __host__ void inline library_init(MPI_Comm comm) {
  assert(!device);
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));

  if (count == 0) {
    printf("No GPU found! \n");
    abort();
  }

  rocm_init();

  mpi_instance = new MPIInitSingleton(comm);

  CHECK_HIP(hipHostMalloc(&device, sizeof(GDADevice)));
  device = new (device) GDADevice(comm);

  if (!device) {
    abort();
  }
}

[[maybe_unused]] __host__ static void inline library_init_subcomm(TcpBootstrap *bootstrap, int nranks, int rank) {
  int initialized;
  int world_size = -1;
  MPI_Initialized(&initialized);

  if (!initialized) {
    // This is an Open MPI specific solution to retrieve the number of
    // processes that have been started, value can be checked before MPI_Init
    char *value = getenv("OMPI_COMM_WORLD_SIZE");
    if (value != NULL) {
      world_size = atoi(value);
    }
    if (world_size != nranks) {
      // This solution will require MPI_Sessions. This is planned for the
      // future, but is not supported in the current version.
      fprintf (stderr, "Unsupported configuration to initialize rocSHMEM. Please "
               "initialize the MPI library using MPI_Init first, if you want to "
               "initialize rocSHMEM with a subset of the processes\n");
      abort();
    }
  } else {
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
  }

  if (world_size == nranks) {
    library_init(MPI_COMM_WORLD);
  } else {
    MPI_Group world_group;
    int world_rank;

    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    MPI_Comm_group (MPI_COMM_WORLD, &world_group);

    int *inc_ranks = new int[nranks];
    inc_ranks[rank] = world_rank;

    bootstr->allGather (inc_ranks, sizeof(int));

    MPI_Group sub_group;
    MPI_Comm sub_comm;
    MPI_Group_incl (world_group, nranks, inc_ranks, &sub_group);
    MPI_Comm_create_group (MPI_COMM_WORLD, sub_group, 1234, &sub_comm);

    library_init(sub_comm);

    MPI_Group_free (&sub_group);
    MPI_Group_free (&world_group);
    MPI_Comm_free (&sub_comm);
    delete[] inc_ranks;
  }
}

[[maybe_unused]] __host__ void inline library_init(TcpBootstrap *bootstrap) {
  assert(!device);
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));

  if (count == 0) {
    printf("No GPU found! \n");
    abort();
  }

  rocm_init();

  CHECK_HIP(hipHostMalloc(&device, sizeof(GDADevice)));
  device = new (device) GDADevice(bootstrap);

  if (!device) {
    abort();
  }
}

[[maybe_unused]] __host__ int rocshmem_init_attr(unsigned int flags,
                                                 rocshmem_init_attr_t *attr) {
  MPI_Comm comm = MPI_COMM_NULL;

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
    library_init(comm);
    return ROCSHMEM_SUCCESS;
  }

  if (flags == ROCSHMEM_INIT_WITH_UNIQUEID) {
    assert (attr->nranks > 0);
    assert (attr->rank >= 0);
    assert (attr->rank < attr->nranks);

    bootstr = new TcpBootstrap(attr->rank, attr->nranks);

    int timeout = 5;
    char *value;
    value = getenv("ROCSHMEM_BOOTSTRAP_TIMEOUT");
    if (value != nullptr) {
        timeout = atoi(value);
    }
    bootstr->initialize(attr->uid, timeout);

    bool uniqueid_with_mpi = false;
    value = getenv("ROCSHMEM_UNIQUEID_WITH_MPI");
    if (value != nullptr) {
      uniqueid_with_mpi = atoi(value);
    }
    if (uniqueid_with_mpi) {
      library_init_subcomm(bootstr, attr->nranks, attr->rank);
    } else {
      library_init (bootstr);
    }
  }

  return ROCSHMEM_SUCCESS;
}

[[maybe_unused]] __host__ int rocshmem_set_attr_uniqueid_args(int rank, int nranks,
							       rocshmem_uniqueid_t *uid,
							       rocshmem_init_attr_t *attr) {
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

// Note: this function will be called before rocshmem_init_*, so one
// cannot assume that a device is already set
[[maybe_unused]] __host__ int rocshmem_get_uniqueid(rocshmem_uniqueid_t *uid) {
  rocshmem_uniqueid_t tuid;
  if (uid == nullptr) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_get_uniqueid: invalid input argument'",
	      __FILE__, __LINE__);
      return ROCSHMEM_ERROR;
  }

  tuid = TcpBootstrap::createUniqueId();
  *uid = tuid;

  return ROCSHMEM_SUCCESS;
}

[[maybe_unused]] __host__ void rocshmem_init(MPI_Comm comm) {
  library_init(comm);
}

[[maybe_unused]] __host__ int rocshmem_my_pe() {
  if (device != nullptr) {
    return device->my_pe;
  }

  fprintf(stderr, "[WARNING] rocshmem_init() has not been called\n");
  return -1;
}

[[maybe_unused]] __host__ int rocshmem_n_pes() {
  if (device != nullptr) {
    return device->num_pes;
  }

  fprintf(stderr, "[WARNING] rocshmem_init() has not been called\n");
  return -1;
}

[[maybe_unused]] __host__ void *rocshmem_malloc(size_t size) {
  VERIFY_DEVICE();

  void *ptr;
  device->heap.malloc(&ptr, size);

  rocshmem_barrier_all();

  return ptr;
}

[[maybe_unused]] __host__ void rocshmem_free(void *ptr) {
  VERIFY_DEVICE();

  rocshmem_barrier_all();

  device->heap.free(ptr);
}

[[maybe_unused]] __host__ void rocshmem_reset_stats() {
  VERIFY_DEVICE();
  device->reset_stats();
}

[[maybe_unused]] __host__ void rocshmem_dump_stats() {
  /** TODO: Many stats are device independent! **/
  VERIFY_DEVICE();
  device->dump_stats();
}

[[maybe_unused]] __host__ void rocshmem_finalize() {
  VERIFY_DEVICE();

  /*
   * Destroy all the teams that the user
   * created but did not manually destroy
   */
  auto team_destroy{
      std::bind(&Device::team_destroy, device, std::placeholders::_1)};
  device->team_tracker.destroy_all(team_destroy);

  device->~GDADevice();
  CHECK_HIP(hipHostFree(device));

  if (bootstr == nullptr)
    delete mpi_instance;

  if (bootstr != nullptr)
    delete bootstr;
}

__host__ void rocshmem_query_thread(int *provided) {
  /*
   * Host-facing functions always support full
   * thread flexibility i.e. THREAD_MULTIPLE.
   */
  *provided = ROCSHMEM_THREAD_MULTIPLE;
}

__host__ void rocshmem_global_exit(int status) {
  VERIFY_DEVICE();
  device->global_exit(status);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__host__ int rocshmem_team_n_pes(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->num_pes;
  }
}

__host__ int rocshmem_team_my_pe(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->my_pe;
  }
}

__host__ inline int pe_in_active_set(int start, int stride, int size, int pe) {
  /* Active set triplet is described with respect to team world */

  int translated_pe = (pe - start) / stride;

  if ((pe < start) || ((pe - start) % stride) || (translated_pe >= size)) {
    translated_pe = -1;
  }

  return translated_pe;
}

__host__ int rocshmem_team_split_strided(
    rocshmem_team_t parent_team, int start, int stride, int size,
    [[maybe_unused]] const rocshmem_team_config_t *config,
    [[maybe_unused]] long config_mask, rocshmem_team_t *new_team) {
  VERIFY_DEVICE();

  *new_team = ROCSHMEM_TEAM_INVALID;

  auto num_user_teams{device->team_tracker.get_num_user_teams()};
  auto max_num_teams{device->team_tracker.get_max_num_teams()};
  if (num_user_teams >= max_num_teams - 1) {
    /* Exceeded maximum number of teams */
    return -1;
  }

  if (parent_team == ROCSHMEM_TEAM_INVALID) {
    return 0;
  }

  Team *parent_team_obj = get_internal_team(parent_team);

  /* Santity check inputs */
  if (start < 0 || start >= parent_team_obj->num_pes || size < 1 ||
      size > parent_team_obj->num_pes || stride < 1) {
    return -1;
  }

  /* Calculate pe_start, stride, and pe_end wrt team world */
  int pe_start_in_world = parent_team_obj->get_pe_in_world(start);
  int stride_in_world = stride * parent_team_obj->tinfo_wrt_world->stride;
  int pe_end_in_world = pe_start_in_world + stride_in_world * (size - 1);

  /* Check if size is out of bounds */
  if (pe_end_in_world > device->num_pes) {
    return -1;
  }

  /* Calculate my PE in the new team */
  int my_pe_in_world = device->my_pe;
  int my_pe_in_new_team = pe_in_active_set(pe_start_in_world, stride_in_world,
                                           size, my_pe_in_world);

  /* Create team infos */
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  new (team_info_wrt_parent) TeamInfo(parent_team_obj, start, stride, size);

  auto *team_world{device->team_tracker.get_team_world()};
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));
  new (team_info_wrt_world)
      TeamInfo(team_world, pe_start_in_world, stride_in_world, size);

  MPI_Comm team_comm{MPI_COMM_NULL};
  if (parent_team_obj->mpi_comm != MPI_COMM_NULL) {
    /* Create a new MPI communicator for this team */
    int color;
    if (my_pe_in_new_team < 0) {
      color = MPI_UNDEFINED;
    } else {
      color = 1;
    }

    MPI_Comm_split(parent_team_obj->mpi_comm, color, my_pe_in_world, &team_comm);
  }
  /**
   * Allocate new team for GPU-inittiated communication with device-specific
   * objects
   * TODO: are there any device specific objects?
   */

  if (my_pe_in_new_team < 0) {
    *new_team = ROCSHMEM_TEAM_INVALID;
  } else {
    device->create_team(parent_team_obj, team_info_wrt_parent,
			team_info_wrt_world, size, my_pe_in_new_team,
			team_comm, new_team);

    /* Track the newly created team to destroy it in finalize if the user does
     * not */
    device->team_tracker.track(*new_team);
  }

  if (team_comm != MPI_COMM_NULL) {
    MPI_Comm_free (&team_comm);
  }
  return 0;
}

__host__ void rocshmem_team_destroy(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID || team == ROCSHMEM_TEAM_WORLD) {
    /* Do nothing */
    return;
  }

  device->team_tracker.untrack(team);

  device->team_destroy(team);
}

__host__ int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe,
                                         rocshmem_team_t dst_team) {
  return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

template <typename T>
__host__ void rocshmem_put(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void rocshmem_putmem(void *dest, const void *source, size_t nelems,
                               int pe) {
  rocshmem_ctx_putmem(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ void rocshmem_p(T *dest, T value, int pe) {
  rocshmem_p(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ void rocshmem_put_nbi(T *dest, const T *source, size_t nelems,
                                int pe) {
  rocshmem_put_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void rocshmem_putmem_nbi(void *dest, const void *source,
                                   size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems,
                           pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_add(T *dest, T val, int pe) {
  return rocshmem_atomic_fetch_add(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe) {
  return rocshmem_atomic_compare_swap(ROCSHMEM_HOST_CTX_DEFAULT, dest, cond,
                                       val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_add(T *dest, T val, int pe) {
  rocshmem_atomic_add(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_set(T *dest, T val, int pe) {
  rocshmem_atomic_set(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_swap(T *dest, T value, int pe) {
  return rocshmem_atomic_swap(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

__host__ void rocshmem_quiet() {
  rocshmem_ctx_quiet(ROCSHMEM_HOST_CTX_DEFAULT);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__host__ Context *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

__host__ int rocshmem_ctx_create(int64_t options, rocshmem_ctx_t *ctx) {
  DPRINTF("Host function: rocshmem_ctx_create\n");

  void *phys_ctx;
  device->ctx_create(options, &phys_ctx);

  ctx->ctx_opaque = phys_ctx;
  /* This team in on TEAM_WORLD, no need for team info */
  ctx->team_opaque = nullptr;

  /* Track this context, if needed. */
  device->track_ctx(reinterpret_cast<Context *>(phys_ctx));

  return 0;
}

__host__ void rocshmem_ctx_destroy(rocshmem_ctx_t ctx) {
  DPRINTF("Host function: rocshmem_ctx_destroy\n");

  /* TODO: Implicit quiet on this context */

  Context *phys_ctx = get_internal_ctx(ctx);
  device->ctx_destroy(phys_ctx);
}

template <typename T>
__host__ void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_put\n");

  get_internal_ctx(ctx)->put(dest, source, nelems, pe);
}

__host__ void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest,
                                   const void *source, size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_ctx_putmem\n");

  get_internal_ctx(ctx)->putmem(dest, source, nelems, pe);
}

template <typename T>
__host__ void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {
  DPRINTF("Host function: rocshmem_p\n");

  get_internal_ctx(ctx)->p(dest, value, pe);
}

template <typename T>
__host__ void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_put_nbi\n");

  get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe);
}

__host__ void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                       const void *source, size_t nelems,
                                       int pe) {
  DPRINTF("Host function: rocshmem_ctx_putmem_nbi\n");

  get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val,
                                      int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch_add\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond,
                                         T val, int pe) {
  DPRINTF("Host function: rocshmem_atomic_compare_swap\n");

  return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T>
__host__ void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  DPRINTF("Host function: rocshmem_atomic_add\n");

  get_internal_ctx(ctx)->amo_add<T>(dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  DPRINTF("Host function: rocshmem_atomic_set\n");

  get_internal_ctx(ctx)->amo_set(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_swap(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  DPRINTF("Host function: rocshmem_atomic_set\n");

  return get_internal_ctx(ctx)->amo_swap(dest, val, pe);
}

__host__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx) {
  DPRINTF("Host function: rocshmem_ctx_quiet\n");

  get_internal_ctx(ctx)->quiet();
}

__host__ void rocshmem_barrier_all() {
  DPRINTF("Host function: rocshmem_barrier_all\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->barrier_all();
}

template <typename T>
__host__ void rocshmem_wait_until(T *ivars, int cmp, T val) {
  DPRINTF("Host function: rocshmem_wait_until\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until(ivars, cmp, val);
}

template <typename T>
__host__ void rocshmem_wait_until_all(T *ivars, size_t nelems, const int* status,
                                       int cmp, T val) {
  DPRINTF("Host function: rocshmem_wait_until_all\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_all(ivars,
      nelems, status, cmp, val);
}

template <typename T>
__host__ size_t rocshmem_wait_until_any(T *ivars, size_t nelems, const int* status,
                                       int cmp, T val) {
  DPRINTF("Host function: rocshmem_wait_until_any\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_any(ivars,
      nelems, status, cmp, val);
}

template <typename T>
__host__ size_t rocshmem_wait_until_some(T *ivars, size_t nelems, size_t* indices,
                                        const int* status, int cmp,
                                        T val) {
  DPRINTF("Host function: rocshmem_wait_until_some\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_some(ivars, nelems,
      indices, status, cmp, val);
}

template <typename T>
__host__ int rocshmem_test(T *ivars, int cmp, T val) {
  DPRINTF("Host function: rocshmem_testl\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->test(ivars, cmp, val);
}

/**
 * Declare templates for the required datatypes (for the compiler)
 **/
#define RMA_GEN(T)                                                            \
  template __host__ void rocshmem_put<T>(                                     \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __host__ void rocshmem_put_nbi<T>(                                 \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __host__ void rocshmem_p<T>(rocshmem_ctx_t ctx, T * dest,          \
                                        T value, int pe);                     \
  template __host__ void rocshmem_put<T>(T * dest, const T *source,           \
                                          size_t nelems, int pe);             \
  template __host__ void rocshmem_put_nbi<T>(T * dest, const T *source,       \
                                              size_t nelems, int pe);         \
  template __host__ void rocshmem_p<T>(T * dest, T value, int pe);            \

/**
 * Declare templates for the standard amo types
 */
#define AMO_STANDARD_GEN(T)                                                   \
  template __host__ T rocshmem_atomic_compare_swap<T>(                        \
      rocshmem_ctx_t ctx, T * dest, T cond, T value, int pe);                 \
  template __host__ T rocshmem_atomic_compare_swap<T>(T * dest, T cond,       \
                                                       T value, int pe);      \
  template __host__ T rocshmem_atomic_fetch_add<T>(                           \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __host__ T rocshmem_atomic_fetch_add<T>(T * dest, T value,         \
                                                    int pe);                  \
  template __host__ void rocshmem_atomic_add<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, T value, int pe);  \
  template __host__ void rocshmem_atomic_add<T>(T * dest, T value, int pe);

/**
 * Declare templates for the extended amo types
 */
#define AMO_EXTENDED_GEN(T)                                                   \
  template __host__ void rocshmem_atomic_set<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, T value, int pe);  \
  template __host__ void rocshmem_atomic_set<T>(T * dest, T value, int pe);   \
  template __host__ T rocshmem_atomic_swap<T>(rocshmem_ctx_t ctx, T * dest,   \
                                               T value, int pe);              \
  template __host__ T rocshmem_atomic_swap<T>(T * dest, T value, int pe);

/**
 * Declare templates for the wait types
 */
#define WAIT_GEN(T)                                                           \
  template __host__ void rocshmem_wait_until<T>(T *ivars, int cmp,            \
                                                 T val);                      \
  template __host__ int rocshmem_test<T>(T *ivars, int cmp, T val);           \
  template __host__ void Context::wait_until<T>(T *ivars, int cmp,            \
                                                T val);                       \
  template __host__ size_t rocshmem_wait_until_any<T>(T *ivars,               \
                                      size_t nelems, const int* status,       \
                                      int cmp, T val);                        \
  template __host__ void rocshmem_wait_until_all<T>(T *ivars,                 \
                                      size_t nelems, const int* status,       \
                                      int cmp, T val);                        \
  template __host__ size_t rocshmem_wait_until_some<T>(T *ivars, size_t nelems,\
                                      size_t* indices, const int* status,     \
                                      int cmp, T val);                        \
  template __host__ int Context::test<T>(T *ivars, int cmp, T val);

/**
 * Define APIs to call the template functions
 **/

#define RMA_DEF_GEN(T, TNAME)                                                 \
  __host__ void rocshmem_ctx_##TNAME##_put(                                   \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_put_nbi(                               \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_p(rocshmem_ctx_t ctx, T *dest,         \
                                          T value, int pe) {                  \
    rocshmem_p<T>(ctx, dest, value, pe);                                      \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_put(T *dest, const T *source,              \
                                        size_t nelems, int pe) {              \
    rocshmem_put<T>(dest, source, nelems, pe);                                \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_put_nbi(T *dest, const T *source,          \
                                            size_t nelems, int pe) {          \
    rocshmem_put_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_p(T *dest, T value, int pe) {              \
    rocshmem_p<T>(dest, value, pe);                                           \
  }                                                                           \

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                        \
  __host__ T rocshmem_ctx_##TNAME##_atomic_compare_swap(                      \
      rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe) {                 \
    return rocshmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe);       \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, \
                                                     int pe) {                \
    return rocshmem_atomic_compare_swap<T>(dest, cond, value, pe);            \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch_add(                         \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_add<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) {  \
    return rocshmem_atomic_fetch_add<T>(dest, value, pe);                     \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_add(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) { \
    rocshmem_atomic_add<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_add(T *dest, T value, int pe) {     \
    rocshmem_atomic_add<T>(dest, value, pe);                                  \
  }

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                        \
  __host__ void rocshmem_ctx_##TNAME##_atomic_set(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) {\
    rocshmem_atomic_set<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_set(T *dest, T value, int pe) {     \
    rocshmem_atomic_set<T>(dest, value, pe);                                  \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_swap(rocshmem_ctx_t ctx, T *dest,  \
                                                 T value, int pe) {           \
    return rocshmem_atomic_swap<T>(ctx, dest, value, pe);                     \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_swap(T *dest, T value, int pe) {       \
    return rocshmem_atomic_swap<T>(dest, value, pe);                          \
  }

#define WAIT_DEF_GEN(T, TNAME)                                                \
  __host__ void rocshmem_##TNAME##_wait_until(T *ivars, int cmp,              \
                                               T val) {                       \
    rocshmem_wait_until<T>(ivars, cmp, val);                                  \
  }                                                                           \
  __host__ size_t rocshmem_##TNAME##_wait_until_any(T *ivars, size_t nelems,  \
                                                     const int* status,       \
                                                     int cmp,                 \
                                                     T val) {                 \
    return rocshmem_wait_until_any<T>(ivars, nelems, status, cmp, val);       \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_wait_until_all(T *ivars, size_t nelems,    \
                                                   const int* status,         \
                                                   int cmp,                   \
                                                   T val) {                   \
    rocshmem_wait_until_all<T>(ivars, nelems, status, cmp, val);              \
  }                                                                           \
  __host__ size_t rocshmem_##TNAME##_wait_until_some(T *ivars, size_t nelems, \
                                                    size_t* indices,          \
                                                    const int* status,        \
                                                    int cmp,                  \
                                                    T val) {                  \
    return rocshmem_wait_until_some<T>(ivars, nelems, indices, status, cmp, val); \
  }                                                                           \

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
