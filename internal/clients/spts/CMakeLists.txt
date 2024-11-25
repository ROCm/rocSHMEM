###############################################################################
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# GLOBAL COMPILE FLAGS
###############################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE unspecified: generating Release build")

  set(
    CMAKE_BUILD_TYPE
    "Release"
    CACHE
      STRING
        "build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    FORCE
  )
endif()

###############################################################################
# PROJECT
###############################################################################
project(spts VERSION 1.1.0 LANGUAGES CXX)

###############################################################################
# CONFIGURATION OPTIONS
###############################################################################
option(USE_HIP "Build HIP version of the solver" OFF)
option(USE_ROCSHMEM "Build rocSHMEM enabled version of the solver" OFF)
option(ALL_ANALYZE "Build analyze and solve algorithm" OFF)
option(USE_DOUBLE "Use double precision floats for the data" OFF)
option(ALL_LEVELSET "Build levelset algorithm" OFF)
option(ALL_LEVELSYNC "Build levelsync algorithm" OFF)
option(ALL_SYNCFREE "Build syncfree algorithm" OFF)

configure_file(cmake/config.h.in config.h)

###############################################################################
# SOURCES
###############################################################################
add_executable(${PROJECT_NAME} "")

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>                 # CONFIG.H
)

target_sources(
  ${PROJECT_NAME}
  PRIVATE
    InputFlags.cpp
    Main.cpp
)

###############################################################################
# HIP / HIP + rocSHMEM
###############################################################################
if(USE_HIP)
  find_package(hip REQUIRED)

  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      HIPHelper.cpp
  )

  if(USE_ROCSHMEM)
    find_package(rocshmem CONFIG REQUIRED)

    target_include_directories(
      ${PROJECT_NAME}
      PRIVATE
        rocshmem::rocshmem
    )

    target_link_libraries(
      ${PROJECT_NAME}
      PRIVATE
        rocshmem::rocshmem
	hip::host
        -fgpu-rdc
    )
  endif()

###############################################################################
# OPENCL
###############################################################################
else()

  if(USE_ROCSHMEM)
    message(FATAL_ERROR "Cannot use rocSHMEM without USE_HIP")
  endif()

  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      OpenCLHelper.cpp
  )

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      /opt/rocm/opencl/include
  )

  target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
      -L/opt/rocm/opencl/lib/x86_64
      -lOpenCL
)

endif()
