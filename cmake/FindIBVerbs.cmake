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

find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
if (IBVerbs_ROOT )
  # We don't use IBVerbs_DIR as this is supposed to be used when finding hwloc-config.cmake only
  set(ENV{PKG_CONFIG_PATH} "${IBVerbs_ROOT}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()
pkg_check_modules(PC_IBVerbs QUIET libibverbs)
endif()

find_path(IBVerbs_INCLUDE_DIR infiniband/verbs.h
  HINTS ${PC_IBVerbs_INCLUDEDIR} ${PC_IBVerbs_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(IBVerbs_LIBRARY
  NAMES ibverbs libibverbs
  HINTS ${PC_IBVerbs_LIBDIR} ${PC_IBVerbs_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

if (GDA_IONIC)
find_library(IBVerbs_PROVIDER_LIBRARY
  NAMES ionic libionic
  HINTS ${PC_IBVerbs_LIBDIR} ${PC_IBVerbs_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(IBVerbs DEFAULT_MSG
  IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR IBVerbs_PROVIDER_LIBRARY
)
mark_as_advanced(IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR IBVerbs_PROVIDER_LIBRARY)

add_library(IBVerbs::verbs_provider UNKNOWN IMPORTED)
set_target_properties(IBVerbs::verbs_provider PROPERTIES
  IMPORTED_LOCATION "${IBVerbs_PROVIDER_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${IBVerbs_PROVIDER_INCLUDE_DIR}"
)
target_link_libraries(IBVerbs::verbs IBVerbs::verbs_provider)
endif()

find_package_handle_standard_args(IBVerbs DEFAULT_MSG
  IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR
)
mark_as_advanced(IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR)

if (IBVerbs_FOUND)
add_library(IBVerbs::verbs UNKNOWN IMPORTED)
set_target_properties(IBVerbs::verbs PROPERTIES
  IMPORTED_LOCATION "${IBVerbs_LIBRARY}"
  INTERFACE_COMPILE_OPTIONS "${PC_IBVerbs_CFLAGS_OTHER}"
  INTERFACE_INCLUDE_DIRECTORIES "${IBVerbs_INCLUDE_DIR}"
)

target_link_libraries(IBVerbs::verbs INTERFACE)

endif()
