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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP

namespace rocshmem {

/**
 * @name SHMEM_ALLTOALL
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_wg_alltoall(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems);


/**
 * @name SHMEM_BROADCAST
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcase completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelement     Size of the buffer to participate in the broadcast.
 * @param[in] PE_root      Zero-based ordinal of the PE, with respect to the
                           active set, from which the data is copied
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pSync        Temporary sync buffer provided to ROCSHMEM. Must
                           be of size at least ROCSHMEM_REDUCE_SYNC_SIZE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_float_broadcast(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_float_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_double_broadcast(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_double_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_char_broadcast(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_char_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_schar_broadcast(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_schar_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_short_broadcast(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_short_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_int_broadcast(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_int_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_long_broadcast(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_long_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_longlong_broadcast(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_longlong_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_uchar_broadcast(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_uchar_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_ushort_broadcast(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_ushort_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_uint_broadcast(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_uint_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_ulong_broadcast(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_ulong_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_wg_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_ulonglong_broadcast(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_ulonglong_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems, int pe_root);


/**
 * @name SHMEM_FCOLLECT
 * @brief Concatenates blocks of data from multiple PEs to an array in every
 * PE participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks in source array.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_wg_fcollect(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems);


/**
 * @name SHMEM_REDUCTIONS
 * @brief Perform an allreduce between PEs in the active set. The caller
 * is blocked until the reduction completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nreduce      Size of the buffer to participate in the reduction.
 *
 * @return int (Zero on successful local completion. Nonzero otherwise.)
 */
__device__ ATTR_NO_INLINE int rocshmem_ctx_short_sum_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_min_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_max_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_prod_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_or_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_and_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_xor_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_sum_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_min_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_max_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_prod_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_or_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_and_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_xor_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_sum_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_min_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_max_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_prod_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_or_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_and_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_xor_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_sum_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_min_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_max_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_prod_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_or_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_and_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_xor_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_sum_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_min_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_max_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_prod_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_sum_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_min_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_max_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_prod_wg_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);


}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
