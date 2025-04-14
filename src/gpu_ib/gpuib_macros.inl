/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GPUIB_MACROS_HPP_
#define LIBRARY_SRC_GPUIB_MACROS_HPP_

#define GPUIB_LIKELY(X)   __builtin_expect(X, 1)
#define GPUIB_UNLIKELY(X) __builtin_expect(X, 0)

/**
 * @name GPUIB_CHECK_NNULL
 * @brief Checks if value is not NULL. If it is NULL then it exits the program.
 *
 * @param[in] value    Value to check
 * @param[in] str      Error string to print
 *
 */
#define GPUIB_CHECK_NNULL(value, str)                  \
{                                                      \
  if (GPUIB_UNLIKELY(NULL == value)) {                 \
    fprintf(stderr,                                    \
            "[%s:%d] %s failed with errno %s (%d) \n", \
            __FILE__, __LINE__,                        \
            str, strerror(errno), errno);              \
    abort();                                           \
  }                                                    \
}

/**
 * @name GPUIB_CHECK_ZERO
 * @brief Checks if value is zero. If it is not zero then it exits the program.
 *
 * @param[in] value    Value to check
 * @param[in] str      Error string to print
 *
 */
#define GPUIB_CHECK_ZERO(value, str)                   \
{                                                      \
  if (GPUIB_UNLIKELY(0 != value)) {                    \
    fprintf(stderr,                                    \
            "[%s:%d] %s failed with errno %s (%d) \n", \
            __FILE__, __LINE__,                        \
            str, strerror(errno), errno);              \
    abort();                                           \
  }                                                    \
}

#endif // LIBRARY_SRC_GPUIB_MACROS_HPP_
