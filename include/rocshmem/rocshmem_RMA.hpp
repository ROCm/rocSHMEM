/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_INCLUDE_ROCSHMEM_RMA_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_RMA_HPP

namespace rocshmem {

/**
 * @name SHMEM_PUT
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put(
    float *dest, const float *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_float_put(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__host__ void rocshmem_float_put(float *dest,
    const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put(
    double *dest, const double *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_double_put(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__host__ void rocshmem_double_put(double *dest,
    const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put(
    char *dest, const char *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_char_put(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__host__ void rocshmem_char_put(char *dest,
    const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put(
    signed char *dest, const signed char *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_schar_put(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__host__ void rocshmem_schar_put(signed char *dest,
    const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put(
    short *dest, const short *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_short_put(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__host__ void rocshmem_short_put(short *dest,
    const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put(
    int *dest, const int *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_int_put(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__host__ void rocshmem_int_put(int *dest,
    const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put(
    long *dest, const long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_long_put(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__host__ void rocshmem_long_put(long *dest,
    const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put(
    long long *dest, const long long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_longlong_put(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__host__ void rocshmem_longlong_put(long long *dest,
    const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_uchar_put(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__host__ void rocshmem_uchar_put(unsigned char *dest,
    const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_ushort_put(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__host__ void rocshmem_ushort_put(unsigned short *dest,
    const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_uint_put(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__host__ void rocshmem_uint_put(unsigned int *dest,
    const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_ulong_put(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__host__ void rocshmem_ulong_put(unsigned long *dest,
    const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_ulonglong_put(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__host__ void rocshmem_ulonglong_put(unsigned long long *dest,
    const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem(rocshmem_ctx_t ctx,
                                                    void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem(void *dest, const void *source,
                                                size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into __host__ rocshmem_quiet() if remote completion is required.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest,
                                   const void *source, size_t nelems, int pe);

__host__ void rocshmem_putmem(void *dest, const void *source, size_t nelems,
                               int pe);


/**
 * @name SHMEM_P
 * @brief Writes a single value to \p dest at \p pe PE to \p dst at \p pe.
 * The caller must call into rocshmem_quiet() if remote completion is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] value  Value to write to dest at \p pe.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_p(
    rocshmem_ctx_t ctx, float *dest, float value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_p(
    float *dest, float value, int pe);
__host__ void rocshmem_ctx_float_p(
    rocshmem_ctx_t ctx, float *dest, float value,
    int pe);
__host__ void rocshmem_float_p(
    float *dest, float value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_p(
    rocshmem_ctx_t ctx, double *dest, double value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_p(
    double *dest, double value, int pe);
__host__ void rocshmem_ctx_double_p(
    rocshmem_ctx_t ctx, double *dest, double value,
    int pe);
__host__ void rocshmem_double_p(
    double *dest, double value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_p(
    rocshmem_ctx_t ctx, char *dest, char value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_p(
    char *dest, char value, int pe);
__host__ void rocshmem_ctx_char_p(
    rocshmem_ctx_t ctx, char *dest, char value,
    int pe);
__host__ void rocshmem_char_p(
    char *dest, char value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_p(
    rocshmem_ctx_t ctx, signed char *dest, signed char value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_p(
    signed char *dest, signed char value, int pe);
__host__ void rocshmem_ctx_schar_p(
    rocshmem_ctx_t ctx, signed char *dest, signed char value,
    int pe);
__host__ void rocshmem_schar_p(
    signed char *dest, signed char value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_p(
    rocshmem_ctx_t ctx, short *dest, short value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_p(
    short *dest, short value, int pe);
__host__ void rocshmem_ctx_short_p(
    rocshmem_ctx_t ctx, short *dest, short value,
    int pe);
__host__ void rocshmem_short_p(
    short *dest, short value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_p(
    rocshmem_ctx_t ctx, int *dest, int value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_p(
    int *dest, int value, int pe);
__host__ void rocshmem_ctx_int_p(
    rocshmem_ctx_t ctx, int *dest, int value,
    int pe);
__host__ void rocshmem_int_p(
    int *dest, int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_p(
    rocshmem_ctx_t ctx, long *dest, long value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_p(
    long *dest, long value, int pe);
__host__ void rocshmem_ctx_long_p(
    rocshmem_ctx_t ctx, long *dest, long value,
    int pe);
__host__ void rocshmem_long_p(
    long *dest, long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_p(
    rocshmem_ctx_t ctx, long long *dest, long long value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_p(
    long long *dest, long long value, int pe);
__host__ void rocshmem_ctx_longlong_p(
    rocshmem_ctx_t ctx, long long *dest, long long value,
    int pe);
__host__ void rocshmem_longlong_p(
    long long *dest, long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_p(
    rocshmem_ctx_t ctx, unsigned char *dest, unsigned char value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_p(
    unsigned char *dest, unsigned char value, int pe);
__host__ void rocshmem_ctx_uchar_p(
    rocshmem_ctx_t ctx, unsigned char *dest, unsigned char value,
    int pe);
__host__ void rocshmem_uchar_p(
    unsigned char *dest, unsigned char value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_p(
    rocshmem_ctx_t ctx, unsigned short *dest, unsigned short value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_p(
    unsigned short *dest, unsigned short value, int pe);
__host__ void rocshmem_ctx_ushort_p(
    rocshmem_ctx_t ctx, unsigned short *dest, unsigned short value,
    int pe);
__host__ void rocshmem_ushort_p(
    unsigned short *dest, unsigned short value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_p(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_p(
    unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_ctx_uint_p(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value,
    int pe);
__host__ void rocshmem_uint_p(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_p(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_p(
    unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ctx_ulong_p(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value,
    int pe);
__host__ void rocshmem_ulong_p(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_p(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value,
    int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_p(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ctx_ulonglong_p(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value,
    int pe);
__host__ void rocshmem_ulonglong_p(
    unsigned long long *dest, unsigned long long value, int pe);


/**
 * @name SHMEM_PUT_NBI
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_nbi(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_nbi(
    float *dest, const float *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_float_put_nbi(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__host__ void rocshmem_float_put_nbi(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_nbi(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_nbi(
    double *dest, const double *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_double_put_nbi(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__host__ void rocshmem_double_put_nbi(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_nbi(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_nbi(
    char *dest, const char *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_char_put_nbi(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__host__ void rocshmem_char_put_nbi(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_nbi(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_nbi(
    signed char *dest, const signed char *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_schar_put_nbi(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__host__ void rocshmem_schar_put_nbi(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_nbi(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_nbi(
    short *dest, const short *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_short_put_nbi(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__host__ void rocshmem_short_put_nbi(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_nbi(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_nbi(
    int *dest, const int *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_int_put_nbi(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__host__ void rocshmem_int_put_nbi(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_nbi(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_nbi(
    long *dest, const long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_long_put_nbi(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__host__ void rocshmem_long_put_nbi(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_nbi(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_nbi(
    long long *dest, const long long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_longlong_put_nbi(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__host__ void rocshmem_longlong_put_nbi(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_nbi(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_nbi(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_uchar_put_nbi(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__host__ void rocshmem_uchar_put_nbi(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_nbi(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_nbi(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_ushort_put_nbi(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__host__ void rocshmem_ushort_put_nbi(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_nbi(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_nbi(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_uint_put_nbi(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__host__ void rocshmem_uint_put_nbi(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_nbi(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_nbi(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_ulong_put_nbi(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__host__ void rocshmem_ulong_put_nbi(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_nbi(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_nbi(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);
__host__ void rocshmem_ctx_ulonglong_put_nbi(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__host__ void rocshmem_ulonglong_put_nbi(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx,
                                                        void *dest,
                                                        const void *source,
                                                        size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_nbi(void *dest,
                                                    const void *source,
                                                    size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * _host__ rocshmem_quiet() if completion notification is required.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                       const void *source, size_t nelems,
                                       int pe);

__host__ void rocshmem_putmem_nbi(void *dest, const void *source,
                                   size_t nelems, int pe);

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_RMA_HPP
