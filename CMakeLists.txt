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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# AVOID IN SOURCE BUILD
###############################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND
   CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MSG "")
  message(STATUS "Warning! Building from the source directory is not recommended")
  message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
  message(STATUS "and build from a separate directory")
  message(FATAL_ERROR "In-source build")
endif()

###############################################################################
# CONFIGURATION OPTIONS
###############################################################################
option(DEBUG "Enable debug trace" OFF)
option(PROFILE "Enable statistics and timing support" OFF)
option(USE_RO "Enable RO conduit." OFF)
option(USE_IPC "Enable IPC support (using HIP)" OFF)
option(USE_THREADS "Enable workgroup threads to share network queues" OFF)
option(USE_WF_COAL "Enable wavefront message coalescing" OFF)
option(USE_COHERENT_HEAP "Enable support for coherent systems" ON)
option(USE_MANAGED_HEAP "Enable managed memory" OFF)
option(USE_HOST_HEAP "Enable host memory using malloc/free" OFF)
option(USE_HIP_HOST_HEAP "Enable host memory using hip api" OFF)
option(USE_ALLOC_DLMALLOC "Enable dlmalloc device memory allocator" ON)
option(USE_ALLOC_POW2BINS "Enable legacy Pow2Bins device memory allocator" OFF)
option(USE_FUNC_CALL "Force compiler to use function calls on library API" OFF)
option(USE_SHARED_CTX "Request support for shared ctx between WG" OFF)
option(USE_SINGLE_NODE "Enable single node support only." OFF)
option(USE_HOST_SIDE_HDP_FLUSH "Use a polling thread to flush the HDP cache on the host." OFF)

option(BUILD_FUNCTIONAL_TESTS "Build the functional tests" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_SOS_TESTS "Build the host-facing tests" OFF)
option(BUILD_UNIT_TESTS "Build the unit tests" ON)
option(BUILD_TESTS_ONLY "Build only tests. Used to link agains rocSHMEM in a ROCm Release" OFF)

option(BUILD_ROCSHMEM_IPC_LIB "Build librocshmem_ipc.a" ON)
option(BUILD_ROCSHMEM_RO "Build librocshmem_ro.a (for debug purposes)" OFF)
option(BUILD_ROCSHMEM_RO_IPC_LIB "Build librocshmem_ro_ipc.a" ON)

option(BUILD_LOCAL_GPU_TARGET_ONLY "Build only for GPUs detected on this machine" OFF)

configure_file(cmake/rocshmem_config.h.in rocshmem_config.h)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################


if ((NOT BUILD_ROCSHMEM_IPC_LIB) AND
    (NOT BUILD_ROCSHMEM_RO_LIB) AND
    (NOT BUILD_ROCSHMEM_RO_IPC_LIB))
  message(FATAL_ERROR "Neither IPC, RO, nor RO_IPC libraries were not selected.")
endif()

if (DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
else()
  set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()

if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

if (BUILD_TESTS_ONLY)
  if (DEFINED ENV{ROCSHMEM_HOME})
    set(ROCSHMEM_HOME "$ENV{ROCSHMEM_HOME}")
  else()
    message("Environment variable ROCSHMEM_HOME is not set.")
    message("Assuming that rocSHMEM is installed at ${ROCM_PATH}.")
    set(ROCSHMEM_HOME "${ROCM_PATH}")
  endif()
endif()

find_package(ROCM 0.8 REQUIRED PATHS ${ROCM_PATH})
set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)

include(cmake/rocm_local_targets.cmake)

###############################################################################
# PROJECT
###############################################################################
find_package(ROCmCMakeBuildTools)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)

rocm_setup_version(VERSION 2.0.0)
project(rocshmem VERSION 2.0.0 LANGUAGES CXX)

rocshmem_set_gpu_targets()

find_package(MPI REQUIRED)
find_package(hip REQUIRED)
find_package(hsa-runtime64 REQUIRED)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>               # rocshmem_config.h
  $<INSTALL_INTERFACE:include>
  ${MPI_CXX_HEADER_DIR}
)

link_libraries(
  Threads::Threads
  ${MPI_mpi_LIBRARY}
  ${MPI_mpicxx_LIBRARY}
  hip::device
  hip::host
  hsa-runtime64::hsa-runtime64
  -fgpu-rdc
)

add_compile_options("-fgpu-rdc")

if (BUILD_TESTS_ONLY)
  include_directories(
    $<BUILD_INTERFACE:${ROCSHMEM_HOME}/include>
  )
endif()

###############################################################################
# CREATE ROCSHMEM LIBRARY MACRO
###############################################################################

macro(rocshmem_add_library ROCSHMEM_LIB_NAME EXTRA_FLAGS)
  if (NOT BUILD_TESTS_ONLY)
    add_library(${ROCSHMEM_LIB_NAME})
    add_library(roc::${ROCSHMEM_LIB_NAME} ALIAS ${ROCSHMEM_LIB_NAME})

    rocm_install(TARGETS ${ROCSHMEM_LIB_NAME})

    rocm_export_targets(
      TARGETS roc::${ROCSHMEM_LIB_NAME}
      NAMESPACE roc::
    )
  endif()

  list(APPEND ROCSHMEM_LIB_NAMES ${ROCSHMEM_LIB_NAME})
endmacro()

###############################################################################
# CREATE ROCSHMEM LIBRARY
###############################################################################
if (BUILD_ROCSHMEM_IPC_LIB)
  rocshmem_add_library(rocshmem "-DUSE_IPC=ON -DUSE_SINGLE_NODE=ON")
endif()

if (BUILD_ROCSHMEM_RO_LIB)
    rocshmem_add_library(rocshmem_ro "-DUSE_RO=ON")
endif()

if (BUILD_ROCSHMEM_RO_IPC_LIB)
  rocshmem_add_library(rocshmem_ro_ipc "-DUSE_RO=ON -DUSE_IPC=ON")
endif()

if (NOT BUILD_TESTS_ONLY)
  add_subdirectory(src)
endif()

add_subdirectory(tests)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

###############################################################################
# INSTALL ROCSHMEM LIBRARY(S)
###############################################################################
if (NOT BUILD_TESTS_ONLY)
  rocm_install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  rocm_install(
    FILES "${CMAKE_BINARY_DIR}/rocshmem_config.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocshmem
  )

  rocm_package_add_dependencies(
    DEPENDS
      hsa-rocr
      hip-runtime-amd
      rocm-dev
  )

  rocm_create_package(
    NAME "rocSHMEM"
    DESCRIPTION "ROCm OpenSHMEM (rocSHMEM)"
    MAINTAINER "rocSHMEM Maintainer <rocshmem-maintainer@amd.com>"
  )
endif()
