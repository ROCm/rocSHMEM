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
#ifndef LIBRARY_SRC_HOST_HOST_TEMPLATES_HPP_
#define LIBRARY_SRC_HOST_HOST_TEMPLATES_HPP_

#include <utility>

#include "host_helpers.hpp"
#include "memory/window_info.hpp"
#include "team.hpp"

namespace rocshmem {

template <typename T>
void HostInterface::p(T* dest, T value, int pe, WindowInfo* window_info) {
  putmem(dest, &value, sizeof(T), pe, window_info);
}

template <typename T>
void HostInterface::put(T* dest, const T* source, size_t nelems, int pe, WindowInfo* window_info) {
  putmem(dest, source, sizeof(T) * nelems, pe, window_info);
}

template <typename T>
void HostInterface::put_nbi(T* dest, const T* source, size_t nelems, int pe, WindowInfo* window_info) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe, window_info);
}

inline MPI_Comm HostInterface::get_mpi_comm(int pe_start, int log_pe_stride, int pe_size) {
  MPI_Comm active_set_comm{};

  /*
   * First, check to see if the active set is the same as COMM_WORLD
   */
  int comm_world_size{-1};
  MPI_Comm_size(host_comm_world_, &comm_world_size);

  if (pe_start == 0 && log_pe_stride == 0 && pe_size == comm_world_size) {
    active_set_comm = host_comm_world_;
    return active_set_comm;
  }

  /*
   * Then, check to see if we had already created a communicator for
   * this active set
   */
  ActiveSetKey key(pe_start, log_pe_stride, pe_size);

  auto it{comm_map.find(key)};
  if (it != comm_map.end()) {
    return it->second;
  }

  /*
   * If there is not one cached, create a new one (expensive)
   */
  std::vector<int> active_set_ranks(pe_size);
  int stride{1 << log_pe_stride};
  active_set_ranks[0] = pe_start;

  for (int i{1}; i < pe_size; i++) {
    active_set_ranks[i] = active_set_ranks[i - 1] + stride;
  }

  MPI_Group comm_world_group{};
  MPI_Group active_set_group{};
  MPI_Comm_group(host_comm_world_, &comm_world_group);
  MPI_Group_incl(comm_world_group, pe_size, active_set_ranks.data(), &active_set_group);
  MPI_Comm_create_group(host_comm_world_, active_set_group, 0, &active_set_comm);

  comm_map.insert(std::pair<ActiveSetKey, MPI_Comm>(key, active_set_comm));

  return active_set_comm;
}

template <typename T>
inline MPI_Datatype HostInterface::get_mpi_type() {
  fprintf(stderr, "Unknown or unimplemented datatype \n");
}

#define GET_MPI_TYPE(T, MPI_T)                                    \
  template <>                                                     \
  inline MPI_Datatype HostInterface::get_mpi_type<T>() {          \
    return MPI_T;                                                 \
  }

GET_MPI_TYPE(int, MPI_INT)
GET_MPI_TYPE(unsigned int, MPI_UNSIGNED)
GET_MPI_TYPE(short, MPI_SHORT)
GET_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT)
GET_MPI_TYPE(long, MPI_LONG)
GET_MPI_TYPE(unsigned long, MPI_UNSIGNED_LONG)
GET_MPI_TYPE(long long, MPI_LONG_LONG)
GET_MPI_TYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG)
GET_MPI_TYPE(float, MPI_FLOAT)
GET_MPI_TYPE(double, MPI_DOUBLE)
GET_MPI_TYPE(char, MPI_CHAR)
GET_MPI_TYPE(signed char, MPI_SIGNED_CHAR)
GET_MPI_TYPE(unsigned char, MPI_UNSIGNED_CHAR)

template <typename T>
void HostInterface::amo_add(void* dst, T value, int pe, WindowInfo* window_info) {
  /*
   * Most MPI implementations tend to use active messages to implement
   * MPI_Accumulate. So, to eliminate the potential involvement of the
   * target PE, we instead use fetch_add and disregard the return value.
   */
  [[maybe_unused]] T ret{amo_fetch_add(dst, value, pe, window_info)};
}

template <typename T>
void HostInterface::amo_cas(void* dst, T value, T cond, int pe, WindowInfo* window_info) {
  /* Perform the compare and swap and disregard the return value */
  [[maybe_unused]] T ret{amo_fetch_cas(dst, value, cond, pe, window_info)};
}

template <typename T>
T HostInterface::amo_fetch_add(void* dst, T value, int pe, WindowInfo* window_info) {
  MPI_Aint offset{compute_offset(dst, window_info->get_start(), window_info->get_end())};
  MPI_Win win{window_info->get_win()};
  MPI_Datatype mpi_type{get_mpi_type<T>()};
  T ret{};
  MPI_Fetch_and_op(&value, &ret, mpi_type, pe, offset, MPI_SUM, win);
  MPI_Win_flush_local(pe, win);
  return ret;
}

template <typename T>
T HostInterface::amo_fetch_cas(void* dst, T value, T cond, int pe, WindowInfo* window_info) {
  MPI_Aint offset{compute_offset(dst, window_info->get_start(), window_info->get_end())};
  MPI_Win win{window_info->get_win()};
  MPI_Datatype mpi_type{get_mpi_type<T>()};
  T ret{};
  MPI_Compare_and_swap(&value, &cond, &ret, mpi_type, pe, offset, win);
  MPI_Win_flush_local(pe, win);
  return ret;
}

template <typename T>
inline int HostInterface::compare(int cmp, T input_val, T target_val) {
  int cond_satisfied{0};

  switch (cmp) {
    case ROCSHMEM_CMP_EQ:
      cond_satisfied = (input_val == target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_NE:
      cond_satisfied = (input_val != target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_GT:
      cond_satisfied = (input_val > target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_GE:
      cond_satisfied = (input_val >= target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_LT:
      cond_satisfied = (input_val < target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_LE:
      cond_satisfied = (input_val <= target_val) ? 1 : 0;
      break;
    default:
      assert(cmp >= ROCSHMEM_CMP_EQ && cmp <= ROCSHMEM_CMP_LE);
      break;
  }

  return cond_satisfied;
}

template <typename T>
inline int HostInterface::test_and_compare(MPI_Aint offset, MPI_Datatype mpi_type, int cmp, T val, MPI_Win win) {
  T fetched_val{};
  MPI_Fetch_and_op(nullptr /*no-op*/, &fetched_val, mpi_type, my_pe_, offset, MPI_NO_OP, win);
  MPI_Win_flush_local(my_pe_, win);
  return compare(cmp, fetched_val, val);
}

template <typename T>
void HostInterface::wait_until(T *ivars, int cmp, T val, WindowInfo* window_info) {
  MPI_Aint offset{compute_offset(ivars, window_info->get_start(), window_info->get_end())};
  MPI_Datatype mpi_type{get_mpi_type<T>()};
  MPI_Win win{window_info->get_win()};

  while (1) {
    int cond_satisfied{test_and_compare(offset, mpi_type, cmp, val, win)};
    if (cond_satisfied) { break; }
  }
}

inline size_t status_entry(size_t nelems, const int *status, bool* done_flags) {
  size_t i{0};
  size_t pos{SIZE_MAX};
  while (i < nelems) {
    if (status[i]) {
      done_flags[i] = 1;
    } else {
      pos = min(i, pos);
    }
    i++;
  }
  return pos;
}

inline size_t status_entry(size_t nelems, const int *status) {
  size_t i{0};
  while (i < nelems) {
    if (status[i] == 0) {
      return i;
    }
    i++;
  }
  return i;
}

template <typename T>
size_t HostInterface::wait_until_any(T* ivars, size_t nelems, const int *status, int cmp, T val, WindowInfo* window_info) {
  if (!nelems) { return SIZE_MAX; }
  size_t pos{status_entry(nelems, status)};
  if (pos == nelems) { return SIZE_MAX; }

  while (true) {
    for (size_t i{pos}; i < nelems; i++) {
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val, window_info)) {
        return i;
      }
    }
  }
}

template <typename T>
void HostInterface::wait_until_all(T* ivars, size_t nelems, const int *status, int cmp, T val, WindowInfo* window_info) {
  if (!nelems) { return; }
  size_t pos{status_entry(nelems, status)};
  if (pos == nelems) { return; }

  for (size_t i{pos}; i < nelems; i++) {
    if (status[i]) {
      continue;
    }
    while (!test(ivars + i, cmp, val, window_info)) {
    }
  }
}

template <typename T>
size_t HostInterface::wait_until_some(T* ivars, size_t nelems, size_t* indices, const int *status, int cmp, T val, WindowInfo* window_info) {
  if (!nelems) { return 0; }
  size_t pos{status_entry(nelems, status)};
  if (pos == nelems) { return 0; }

  bool done {false};
  size_t ncompleted {0};
  while (!done) {
    for (size_t i{pos}; i < nelems; i++) {
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val, window_info)) {
        done = true;
        indices[ncompleted] = i;
        ncompleted++;
      }
    }
  }
  return ncompleted;
}

template <typename T>
int HostInterface::test(T* ivars, int cmp, T val, WindowInfo* window_info) {
  MPI_Aint offset{compute_offset(ivars, window_info->get_start(), window_info->get_end())};
  MPI_Datatype mpi_type{get_mpi_type<T>()};
  return test_and_compare(offset, mpi_type, cmp, val, window_info->get_win());
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_HOST_HOST_TEMPLATES_HPP_
