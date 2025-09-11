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

#include "config.hpp"

#include <list>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace rocshmem {
namespace config {
  inline namespace _base {
    const var<bool> uniqueid_with_mpi("UNIQUEID_WITH_MPI", "", false);
  }  // inline namespace _base

  namespace bootstrap {
    const var<int64_t> timeout("TIMEOUT", "", 5);
    const var<std::string> hostid("HOSTID", "");
  }  // namespace bootstrap

  namespace ro {
    const var<bool> disable_ipc("DISABLE_IPC", "", false);
  }  // namespace ro

  namespace _detail {
    std::tuple<var_map_t&, std::mutex&> get_var_map() {
      // construct on first use idiom
      // allocate variable_map on heap to prevent static initialization order fiasco
      static auto variable_map = new var_map_t();
      static std::mutex map_mutex;
      // use std::tie to return a tuple of references
      return std::tie(*variable_map, map_mutex);
    }
  }  // namespace _detail
}  // namespace config
}  // namespace rocshmem
