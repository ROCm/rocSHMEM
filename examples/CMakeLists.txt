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
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "build type: Release, Debug, RelWithDebInfo, MinSizeRel")

add_compile_options($<$<CONFIG:Debug>:-ggdb>)
add_link_options($<$<CONFIG:Debug>:-ggdb>)

set(CMAKE_HIP_ARCHITECTURES OFF)

###############################################################################
# PROJECT
###############################################################################
project(rocshmem_examples VERSION 1.0.0 LANGUAGES HIP CXX)

###############################################################################
# SOURCES
###############################################################################
set(EXAMPLE_SOURCES
  rocshmem_allreduce_test.cc
  rocshmem_alltoall_test.cc
  rocshmem_broadcast_test.cc
  rocshmem_getmem_test.cc
  rocshmem_put_signal_test.cc
  rocshmem_init_attr_test.cc
)
set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)

find_package(MPI REQUIRED)
#find_package(hip REQUIRED HINTS ${ROCM_PATH} $ENV{ROCM_PATH} PATHS /opt/rocm)
if(NOT TARGET roc::rocshmem)
  find_package(rocshmem REQUIRED HINTS ${ROCM_PATH} $ENV{ROCM_PATH} PATHS /opt/rocm)
endif()

foreach(SOURCE_FILE IN LISTS EXAMPLE_SOURCES)
  get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

  add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

  set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_EXTENSIONS OFF
  )

  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
      MPI::MPI_CXX
      roc::rocshmem
  )
endforeach()
