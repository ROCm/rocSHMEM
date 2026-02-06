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

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# PROJECT
###############################################################################
include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)
project(rocshmem_relax_tests VERSION 1.0.0 LANGUAGES CXX)

###############################################################################
# SOURCES
###############################################################################
add_executable(${PROJECT_NAME} "")

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(
  ${PROJECT_NAME}
  PRIVATE
    test_driver.cpp
    tester.cpp
    tester_arguments.cpp
    workgroup_primitives.cpp
    wavefront_primitives.cpp
    allreduce_push_tester.cpp
    fused_gemm_allreduce_push_tester.cpp
)

###############################################################################
# ROCSHMEM
###############################################################################
if (BUILD_TESTS_ONLY)
  #TODO these find_packages should be performed as-needed in rocshmem-config.cmake
  find_package(hip REQUIRED PATHS /opt/rocm)
  find_package(MPI)
  find_package(Threads REQUIRED)
  find_package(rocshmem REQUIRED PATHS /opt/rocm)

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../src>
  )
endif()

find_package(PMIx)

target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE
    $<$<TARGET_EXISTS:PMIx::pmix>:HAVE_PMIX=1>
)

configure_file(../../scripts/relax_tests/driver.sh rocshmem_relax_driver.sh COPYONLY)
rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/rocshmem_relax_driver.sh COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)
rocm_install(TARGETS rocshmem_relax_tests COMPONENT tests)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    roc::rocshmem
    $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
    $<TARGET_NAME_IF_EXISTS:PMIx::pmix>
)
