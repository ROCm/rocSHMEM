###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# ADD ROCSHMEM TARGET FOR FILES IN CURRENT DIRECTORY
###############################################################################
foreach(ROCSHMEM_LIB_NAME IN LISTS ROCSHMEM_LIB_NAMES)
  target_sources(
    ${ROCSHMEM_LIB_NAME}
    PRIVATE
      atomic_return.cpp
      backend_bc.cpp
      context_host.cpp
      context_device.cpp
      mpi_init_singleton.cpp
      rocshmem_gpu.cpp
      rocshmem.cpp
      team.cpp
      team_tracker.cpp
      util.cpp
      wf_coal_policy.cpp
      ipc_policy.cpp
  )

  target_compile_options(
    ${ROCSHMEM_LIB_NAME}
    PUBLIC
      -fgpu-rdc
  )
endforeach()

###############################################################################
# ROCSHMEM TARGET FOR BACKENDS
###############################################################################
IF (USE_RO)
add_subdirectory(reverse_offload)
ELSE()
add_subdirectory(ipc)
ENDIF()
add_subdirectory(containers)
add_subdirectory(host)
add_subdirectory(memory)
add_subdirectory(sync)
add_subdirectory(bootstrap)
