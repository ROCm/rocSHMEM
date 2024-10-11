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
# AVOID IN SOURCE BUILD
###############################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(FATAL_ERROR "In-source build")
endif()

###############################################################################
# SYSTEM INFO
###############################################################################
cmake_host_system_information(RESULT LOCALHOST QUERY FQDN)
message(STATUS "Hostname: ${LOCALHOST}")

###############################################################################
# VERSIONING INFO DERIVED FROM FILE
###############################################################################
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
set(ROCSHMEM_FULL_VERSION "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" ROCSHMEM_VERSION
                     "${FULL_VERSION_STRING}")

###############################################################################
# CONFIGURATION OPTIONS
###############################################################################
option(DEBUG "Enable debug trace" OFF)
option(PROFILE "Enable statistics and timing support" OFF)
option(USE_GPU_IB "Enable GPU_IB conduit." ON)
option(USE_RO "Enable RO conduit." ON)
option(USE_DC "Enable IB dynamically connected transport (DC)" OFF)
option(USE_IPC "Enable IPC support (using HIP)" OFF)
option(USE_THREADS "Enable workgroup threads to share network queues" OFF)
option(USE_WF_COAL "Enable wavefront message coalescing" OFF)
option(USE_COHERENT_HEAP "Enable support for coherent systems" OFF)
option(USE_MANAGED_HEAP "Enable managed memory" OFF)
option(USE_HOST_HEAP "Enable host memory using malloc/free" OFF)
option(USE_HIP_HOST_HEAP "Enable host memory using hip api" OFF)
option(USE_FUNC_CALL "Force compiler to use function calls on library API" OFF)
option(USE_SHARED_CTX "Request support for shared ctx between WG" OFF)
option(USE_SINGLE_NODE "Enable single node support only." OFF)
option(USE_HOST_SIDE_HDP_FLUSH "Use a polling thread to flush the HDP cache on the host." OFF)
option(USE_COOPERATIVE_GROUPS "Use cooperative groups for internal syncronization" OFF)
option(BUILD_FUNCTIONAL_TESTS "Build the functional tests" ON)
option(BUILD_SOS_TESTS "Build the host-facing tests" OFF)
option(BUILD_UNIT_TESTS "Build the unit tests" ON)

set(ROCM_PATH "" CACHE PATH "ROCm path to use")

configure_file(cmake/config.h.in config.h)

###############################################################################
# Validate user passed options
###############################################################################
if(ROCM_PATH)
    set(HIPCONFIG_PATH "${ROCM_PATH}/bin/hipconfig")
    execute_process(
      COMMAND "${HIPCONFIG_PATH}" --version
      RESULT_VARIABLE HIPCONFIG_RETURN
      OUTPUT_VARIABLE ROCM_VERSION
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(HIPCONFIG_RETURN EQUAL 0)
      message(STATUS "Using ROCm with version: ${ROCM_VERSION}")
    else()
      message(FATAL_ERROR "ROCm at ${ROCM_PATH} is missing the hipconfig binary. "
        "Use -DROCM_PATH=/path/to/rocm or ensure ROCm's hipconfig is in your path.")
  endif()
else()
    set(HIPCONFIG_PATH "hipconfig")
    execute_process(
      COMMAND "${HIPCONFIG_PATH}" --rocmpath
      RESULT_VARIABLE HIPCONFIG_RETURN
      OUTPUT_VARIABLE ROCM_PATH
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(HIPCONFIG_RETURN EQUAL 0)
      message(STATUS "Found ROCm at: ${ROCM_PATH}")
    else()
      message(FATAL_ERROR "Ensure hipconfig is in your path or use -DROCM_PATH=/path/to/rocm.")
  endif()
endif()

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

###############################################################################
# MODULE SEARCH PATH
###############################################################################
set(
  CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake/Modules"
  "{ROCM_PATH}/hip/cmake"
  "{ROCM_PATH}/rocclr/lib/cmake/rocclr"
)

###############################################################################
# PROJECT
###############################################################################
  project(
    rocshmem
    VERSION ${ROCSHMEM_VERSION}
    LANGUAGES CXX
    DESCRIPTION "ROCSHMEM"
    HOMEPAGE_URL "https://github.com/ROCm-Developer-Tools/ROC_SHMEM")

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE
    "Release"
    CACHE
      STRING
        "build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    FORCE
  )
  message(STATUS "CMAKE_BUILD_TYPE unspecified: using ${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "/opt/lib/rocshmem"
        CACHE PATH "default install path" FORCE)
endif()
message(STATUS "Installation path: ${CMAKE_INSTALL_PREFIX}")

###############################################################################
# CREATE ROCSHMEM LIBRARY
###############################################################################
add_library(
  ${PROJECT_NAME}
  STATIC
  ""
)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

###############################################################################
# INCLUDE DIRECTORIES
###############################################################################
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>                 # CONFIG.H
    $<INSTALL_INTERFACE:include>
)

###############################################################################
# SPECIFY PUBLIC HEADER FILES
###############################################################################
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
    PUBLIC_HEADER
      "${CMAKE_BINARY_DIR}/config.h;${CMAKE_CURRENT_SOURCE_DIR}/include/roc_shmem/roc_shmem.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/roc_shmem/debug.hpp"
)

###############################################################################
# SUBDIRECTORY TARGETS
###############################################################################
add_subdirectory(src)
add_subdirectory(docs)
add_subdirectory(tests)

###############################################################################
# HIP
###############################################################################
find_package(hip REQUIRED)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    hip::device
    hip::host
)

###############################################################################
# HSA-RUNTIME64
###############################################################################
find_package(hsa-runtime64 REQUIRED)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    hsa-runtime64::hsa-runtime64
)

###############################################################################
# PTHREADS
###############################################################################
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    Threads::Threads
)

###############################################################################
# IBVERBS
###############################################################################
IF (USE_GPU_IB)
find_package(Ibverbs REQUIRED)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    ${IBVERBS_INCLUDE_DIRS}
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    ${IBVERBS_LIBRARIES}
)
ENDIF()

###############################################################################
# MPI
###############################################################################
find_package(MPI REQUIRED)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    ${MPI_CXX_HEADER_DIR}
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    ${MPI_mpi_LIBRARY}
    ${MPI_mpicxx_LIBRARY}
)

###############################################################################
# MODULEFILE
###############################################################################
set(MOD_INSTALL_PATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles"
    CACHE STRING "Install path for modulefile")
message(STATUS "Modulefile install path: ${MOD_INSTALL_PATH}")

set(moduleFileTemplate "${PROJECT_NAME}.lua.in")

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/${moduleFileTemplate}
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCSHMEM_FULL_VERSION}.lua
    @ONLY)

# Crusher
if(LOCALHOST MATCHES ".*\.crusher\.olcf\.ornl\.gov")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using crusher-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.crusher.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCSHMEM_FULL_VERSION}.lua
        ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# Frontier
if(LOCALHOST MATCHES ".*\.frontier\.olcf\.ornl\.gov")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using frontier-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.frontier.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCSHMEM_FULL_VERSION}.lua
        ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# hpcfund
if(LOCALHOST MATCHES ".*\.hpcfund")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using hpcfund-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.hpcfund.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCSHMEM_FULL_VERSION}.lua
        ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

###############################################################################
# INSTALL
###############################################################################
include(GNUInstallDirs)

# Specify layout within the build directory
set(
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(
  CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
)

# Offer alternatives for default installation subdirectories
set(
  INSTALL_LIBDIR
    ${CMAKE_INSTALL_LIBDIR} CACHE PATH
    "Installation directory for libraries"
)
set(
  INSTALL_BINDIR
    ${CMAKE_INSTALL_BINDIR} CACHE PATH
    "Installation directory for executables"
)
set(
  INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH
    "Installation directory for header files"
)
set(
  DEF_INSTALL_CMAKEDIR
  share/cmake/${PROJECT_NAME}
)
set(
  INSTALL_CMAKEDIR
    ${DEF_INSTALL_CMAKEDIR} CACHE PATH
    "Installation directory for CMake files"
)

install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    ARCHIVE
      DESTINATION ${INSTALL_LIBDIR}
      COMPONENT lib
    LIBRARY
      DESTINATION ${INSTALL_LIBDIR}
      COMPONENT lib
    PUBLIC_HEADER
      DESTINATION ${INSTALL_INCLUDEDIR}
      COMPONENT dev
    RUNTIME
      DESTINATION ${INSTALL_BINDIR}
      COMPONENT bin
)

install(
  EXPORT
    ${PROJECT_NAME}Targets
  FILE
    ${PROJECT_NAME}Targets.cmake
  NAMESPACE
    ${PROJECT_NAME}::
  DESTINATION
    ${INSTALL_CMAKEDIR}
  COMPONENT
    dev
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION
    ${INSTALL_CMAKEDIR}
)

install(
    FILES
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCSHMEM_FULL_VERSION}.lua
    DESTINATION ${MOD_INSTALL_PATH}/${PROJECT_NAME})

###############################################################################
# PACKAGE
###############################################################################
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION
    ${PROJECT_VERSION}
  COMPATIBILITY
    SameMajorVersion
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION
    ${INSTALL_CMAKEDIR}
)
