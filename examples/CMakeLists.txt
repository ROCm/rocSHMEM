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

foreach(ROCSHMEM_LIB_NAME IN LISTS ROCSHMEM_LIB_NAMES)
  foreach(SOURCE_FILE IN LISTS EXAMPLE_SOURCES)
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    string(REPLACE "rocshmem" "${ROCSHMEM_LIB_NAME}" EXECUTABLE_NAME ${EXECUTABLE_NAME})

    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

    if (BUILD_TESTS_ONLY)
      target_link_libraries(
        ${EXECUTABLE_NAME}
        PRIVATE
          -L${ROCSHMEM_HOME}/lib
          -l${ROCSHMEM_LIB_NAME}
      )
    else()
      target_link_libraries(
        ${EXECUTABLE_NAME}
        PRIVATE
          roc::${ROCSHMEM_LIB_NAME}
      )
    endif()
  endforeach()
endforeach()
