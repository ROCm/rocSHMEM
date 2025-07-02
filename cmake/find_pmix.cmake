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

# Find pmix installation.
# Two different scenarios need to be covered:
#  - pmix installed as part of Open MPI, i.e. it will be in the MPI installation directories
#  - pmix deployed with linux distros
#  - later: handle pmix deployed with slurm.

macro(check_pmix)
  set(prev_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  find_package(PMIx QUIET)
  if (PMIx_FOUND)
    message("-- Found pmix at ${PMIx_CONFIG} ${PMIx_INCLUDE_DIRS}")
  else()
    if(NOT PMIX_HEADER OR NOT PMIX_LIBRARY)
      message("-- Cound not find pmix using find_package")

      list(APPEND CMAKE_REQUIRED_INCLUDES "${MPI_CXX_HEADER_DIR}") #prefer Open MPI internal PMIx if any
      find_path(PMIX_HEADER pmix.h PATHS ${CMAKE_REQUIRED_INCLUDES})
      if (PMIX_HEADER)
        message("-- Found pmix.h at ${PMIX_HEADER}")
        get_filename_component(pmix_lib_dir ${PMIX_HEADER} DIRECTORY)
        find_library(PMIX_LIBRARY pmix PATHS ${pmix_lib_dir} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
      endif()
      if(PMIX_HEADER AND PMIX_LIBRARY)
        message("-- Found libpmix at ${PMIX_LIBRARY}")
      elseif(NOT PMIX_HEADER)
        message("-- Cound not find pmix.h")
      elseif(NOT PMIX_LIBRARY)
        message("-- Could not find libpmix.so")
      endif()
    endif()
  endif()
  set(CMAKE_REQUIRED_INCLUDES ${prev_CMAKE_REQUIRED_INCLUDES})
  if (PMIX_HEADER AND PMIX_LIBRARY)
    set(PMIX_INCLUDE_DIRECTORIES ${PMIX_HEADER})
    set(PMIX_LIBRARIES ${PMIX_LIBRARY})
    set(PMIx_FOUND TRUE)
  endif()
endmacro()
