/********************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ********************************************************************************/

#include "GPUHelper.h"

#include <hip/hip_runtime.h>
#include <hip/math_functions.h>
#include <hip/device_functions.h>

#ifdef USE_ROCSHMEM
#include "rocshmem.hpp"
using namespace rocshmem;
#endif

#ifndef WF_PER_WG
#error "WF_PER_WG undefined!"
#endif

#ifndef WF_SIZE
#error "WF_SIZE undefind!"
#endif

#define as_uint (unsigned int)
#define as_ulong (unsigned long long)
#define as_float (float)

#ifdef USE_DOUBLE
typedef double FPTYPE;
#else
typedef float FPTYPE;
#endif

// GCN3 and below require slightly different inline asm than Vega
// v_add_u32 requires a "vcc" register output modifier on GCN3, but not on Vega
// global_load_ in Vega is required to be flat_load_ in GCN3 and below.
// Same for global_store_ and flat_store_.
// However, the global_ instructions require an "off" modifier.
#if defined(GCN3) || defined(GCN2)
#define VCC "vcc"
#define MEM_PREFIX "flat"
#define OFF_MODIFIER ""
#else
#define VCC ""
#define MEM_PREFIX "global"
#define OFF_MODIFIER "off"
#endif

#ifndef GCN2
#define LGKMCNT_0 0xc07f // GCN3 added more VMCNT bits at the upper end of the SIMM16
#define WAKEUP "s_wakeup"
#else
#define LGKMCNT_0 0x7f
#define WAKEUP "" // s_wakeup not supported on old GPUs
#endif

#define __builtin_amdgcn_ds_bpermute __hip_ds_bpermute
#define __builtin_amdgcn_ds_swizzle __hip_ds_swizzle
#define __builtin_amdgcn_mov_dpp __hip_move_dpp

#define HIP_ENABLE_PRINTF

// Internal functions to wrap atomics, depending on if we support 64-bit
// atomics or not. Helps keep the code clean in the other parts of the code.
// All of the 32-bit atomics are built assuming we're on a little endian architecture.
__device__
inline unsigned long spts_atomic_cmpxchg(unsigned long long *const ptr,
                                    const unsigned long long compare,
                                    const unsigned long long val)
{
#ifdef USE_DOUBLE
	return atomicCAS(ptr, compare, val);
#else
	return atomicCAS(ptr, compare, val);
#endif
}

__device__
void atomic_set (FPTYPE *ptr, FPTYPE temp)
{
#ifdef USE_DOUBLE
    unsigned long long newVal;
    unsigned long long prevVal;
    do
    {
        prevVal = as_ulong(*ptr);
        newVal = as_ulong(temp);
    } while (spts_atomic_cmpxchg((unsigned long long *)ptr, prevVal, newVal) != prevVal);

#else
    unsigned long long newVal;
    unsigned long long prevVal;
    do
    {
        prevVal = as_uint(*ptr);
        newVal = as_uint(temp);
    } while (spts_atomic_cmpxchg((unsigned long long *)ptr, prevVal, newVal) != prevVal);
#endif
}

__device__
inline void atomic_set_done(uint * done_array, uint row, uint val_to_set)
{
    atomicOr(&(done_array[row]), val_to_set);
}

__device__
inline unsigned int atomic_get_done(uint * done_array, uint val_to_check)
{
    return atomicOr(&(done_array[val_to_check]), 0x0);
}

// Use a traditional LDS-based reduction to have all of the threads in the wave
// add their values into OUTPUT_THREAD's variable.
__device__
FPTYPE lds_reduction(FPTYPE temp_sum, __shared__ FPTYPE *lds,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)
{
    const unsigned int lid = wg_lid % WF_SIZE;

    // Have all the threads in a workgroup reduce their data into a single
    // value that's then read by the lead thread
    // We start by calculating how many layers of reduction we actually need.
    // If this is a very short row (smaller than our wavefront size), then we don't need
    // to do all iterations of the below loop.
    unsigned int num_items = min(end_of_this_row - start_of_this_row - 1, (uint)WF_SIZE);
    // find next highest power of two. So if we have 5 things to reduce, we need to
    // do a reduction from 8 threads' values. The last 3 will be '0'
    num_items = 1 << (CHAR_BIT*(sizeof(unsigned int))-__clz(num_items-1));

    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        lds[wg_lid] = temp_sum;
        asm volatile ("s_waitcnt lgkmcnt(0)\n\t");

        if (lid < i)
            temp_sum += lds[wg_lid + i];
        asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
    }
    // at this point, thread 0's "temp_sum" contains the final useful value.
    return temp_sum;
}

// Use a traditional LDS-based reduction to have all of the threads in the wave
// add their values into OUTPUT_THREAD's variable.
// It hides the max work behind the same s_waitcnt on local memory,
// so it should be faster than calling the reduce function twice in a row.
__device__
FPTYPE lds_reduction_two(FPTYPE temp_sum, unsigned int row_max_depth,
        __shared__ FPTYPE *lds, __shared__ unsigned int *max_depth,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)
{
    const unsigned int lid = wg_lid % WF_SIZE;

    // Have all the threads in a workgroup reduce their data into a single
    // value that's then read by the lead thread
    // We start by calculating how many layers of reduction we actually need.
    // If this is a very short row (smaller than our wavefront size), then we don't need
    // to do all iterations of the below loop.
    unsigned int num_items = min(end_of_this_row - start_of_this_row - 1, (uint)WF_SIZE);
    // find next highest power of two. So if we have 5 things to reduce, we need to
    // do a reduction from 8 threads' values. The last 3 will be '0'
    num_items = 1 << (CHAR_BIT*(sizeof(unsigned int))-__clz(num_items-1));

    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        lds[wg_lid] = temp_sum;
        max_depth[wg_lid] = row_max_depth;
         asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
        if (lid < i)
        {
            temp_sum += lds[wg_lid + i];
            row_max_depth = max(row_max_depth, max_depth[wg_lid + i]);
        }
         asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
    }
    // at this point, max_depth[thread_0_within_each_wavefront]
    // contains the useful maximum depth for this row.
    max_depth[wg_lid] = row_max_depth;
    // at this point, thread 0's "temp_sum" contains the final useful value.
    return temp_sum;
}

// Use a traditional LDS-based reduction to have all of the threads in the wave
// add their values into OUTPUT_THREAD's variable.
// It hides the max work behind the same s_waitcnt on local memory,
// so it should be faster than calling the reduce function three times in a row.
__device__
FPTYPE lds_reduction_three(FPTYPE temp_sum, unsigned int row_max_depth,
        unsigned int spin_times, __shared__ FPTYPE *lds,
        __shared__ unsigned int *max_depth, __shared__ unsigned int *total_spins,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)
{
    const unsigned int lid = wg_lid % WF_SIZE;

    // Have all the threads in a workgroup reduce their data into a single
    // value that's then read by the lead thread
    // We start by calculating how many layers of reduction we actually need.
    // If this is a very short row (smaller than our wavefront size), then we don't need
    // to do all iterations of the below loop.
    unsigned int num_items = min(end_of_this_row - start_of_this_row - 1, (uint)WF_SIZE);
    // find next highest power of two. So if we have 5 things to reduce, we need to
    // do a reduction from 8 threads' values. The last 3 will be '0'
    num_items = 1 << (CHAR_BIT*(sizeof(unsigned int))-__clz(num_items-1));

    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        lds[wg_lid] = temp_sum;
        max_depth[wg_lid] = row_max_depth;
        total_spins[wg_lid] = spin_times;
         asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
        if (lid < i)
        {
            temp_sum += lds[wg_lid + i];
            row_max_depth = max(row_max_depth, max_depth[wg_lid + i]);
            spin_times += total_spins[wg_lid + i];
        }
         asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
    }
    // at this point, max_depth[thread_0_within_each_wavefront]
    // contains the useful maximum depth for this row.
    max_depth[wg_lid] = row_max_depth;
    // and total_spins[thread_0_within_each_wavefront] has its
    // total number of spin-loops.
    total_spins[wg_lid] = spin_times;
    // at this point, thread 0's "temp_sum" contains the final useful value.
    return temp_sum;
}

// Do a reduction using bpermute instructions.
// This is strictly worse than Swizzle-based reduction, since it is slower and
// only works on the same hardware as the swizzle instructions.
__device__
FPTYPE bpermute_reduction(FPTYPE temp_sum, unsigned int start_of_this_row,
        unsigned int end_of_this_row, unsigned int wg_lid)
{
    const unsigned int lid = wg_lid % WF_SIZE;

    // Have all the threads in a workgroup reduce their data into a single
    // value that's then read by the lead thread
    // We start by calculating how many layers of reduction we actually need.
    // If this is a very short row (smaller than our workgroup size), then we don't need
    // to do all iterations of the below loop.
    unsigned int num_items = min(end_of_this_row - start_of_this_row - 1, (uint)WF_SIZE);
    // find next highest power of two. So if we have 5 things to reduce, we need to
    // do a reduction from 8 threads' values. The last 3 will be '0'
    num_items = 1 << (CHAR_BIT*(sizeof(unsigned int))-__clz(num_items-1));

#ifdef USE_DOUBLE
    typedef union dbl_b32 {
        double val;
        uint2 b32;
    } dbl_b32_t;
    dbl_b32_t t_temp_sum;
    t_temp_sum.val = temp_sum;
    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        int pull_from = (lid + i) << 2;
        dbl_b32_t upper_sum;
        upper_sum.b32.x = __builtin_amdgcn_ds_bpermute(pull_from, t_temp_sum.b32.x);
        upper_sum.b32.y = __builtin_amdgcn_ds_bpermute(pull_from, t_temp_sum.b32.y);
        t_temp_sum.val += upper_sum.val;
    }
    temp_sum = t_temp_sum.val;
#else // !USE_DOUBLE
    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        uint pull_from = (lid + i) << 2;
        temp_sum += as_float(__builtin_amdgcn_ds_bpermute(pull_from, as_uint(temp_sum)));
    }
#endif // USE_DOUBLE
    return temp_sum;
}

// Do a reduction using bpermute instructions.
// This is strictly worse than Swizzle-based reduction, since it is slower and
// only works on the same hardware as the swizzle instructions.
// This version also does a max-reduce on the row_max_depth variable.
// It hides this bpermute instruction behind the same s_waitcnt on local memory,
// so it should be faster than calling the reduce function twice in a row.
__device__
FPTYPE bpermute_reduction_two(FPTYPE temp_sum, unsigned int *row_max_depth,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)
{
    const unsigned int lid = wg_lid % WF_SIZE;
    unsigned int max_depth = *row_max_depth;

    // Have all the threads in a workgroup reduce their data into a single
    // value that's then read by the lead thread
    // We start by calculating how many layers of reduction we actually need.
    // If this is a very short row (smaller than our workgroup size), then we don't need
    // to do all iterations of the below loop.
    unsigned int num_items = min(end_of_this_row - start_of_this_row - 1, (uint)WF_SIZE);
    // find next highest power of two. So if we have 5 things to reduce, we need to
    // do a reduction from 8 threads' values. The last 3 will be '0'
    num_items = 1 << (CHAR_BIT*(sizeof(unsigned int))-__clz(num_items-1));

#ifdef USE_DOUBLE
    typedef union dbl_b32 {
        double val;
        int2 b32;
    } dbl_b32_t;
    dbl_b32_t t_temp_sum;
    t_temp_sum.val = temp_sum;
    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        int pull_from = (lid + i) << 2;
        dbl_b32_t upper_sum;
        upper_sum.b32.x = __builtin_amdgcn_ds_bpermute(pull_from, t_temp_sum.b32.x);
        upper_sum.b32.y = __builtin_amdgcn_ds_bpermute(pull_from, t_temp_sum.b32.y);
        max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_bpermute(pull_from, max_depth)));
        t_temp_sum.val += upper_sum.val;
    }
    temp_sum = t_temp_sum.val;
#else // !USE_DOUBLE
    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        int pull_from = (lid + i) << 2;
        max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_bpermute(pull_from, max_depth)));
        temp_sum += as_float(__builtin_amdgcn_ds_bpermute(pull_from, as_uint(temp_sum)));
    }
#endif // USE_DOUBLE
    *row_max_depth = max_depth;
    return temp_sum;
}

// Do a reduction using bpermute instructions.
// This is strictly worse than Swizzle-based reduction, since it is slower and
// only works on the same hardware as the swizzle instructions.
// This version also does a max-reduce on the row_max_depth variable.
// This version also does a max-add on the spin-loops per thread variable.
// It hides this bpermute instruction behind the same s_waitcnt on local memory,
// so it should be faster than calling the reduce function thrice in a row.
__device__
FPTYPE bpermute_reduction_three(FPTYPE temp_sum, unsigned int *row_max_depth,
        unsigned int *spin_times, unsigned int start_of_this_row,
        unsigned int end_of_this_row, unsigned int wg_lid)
{
    const unsigned int lid = wg_lid % WF_SIZE;
    unsigned int max_depth = *row_max_depth;
    unsigned int spin_time = *spin_times;

    // Have all the threads in a workgroup reduce their data into a single
    // value that's then read by the lead thread
    // We start by calculating how many layers of reduction we actually need.
    // If this is a very short row (smaller than our workgroup size), then we don't need
    // to do all iterations of the below loop.
    unsigned int num_items = min(end_of_this_row - start_of_this_row - 1, (uint)WF_SIZE);
    // find next highest power of two. So if we have 5 things to reduce, we need to
    // do a reduction from 8 threads' values. The last 3 will be '0'
    num_items = 1 << (CHAR_BIT*(sizeof(unsigned int))-__clz(num_items-1));

#ifdef USE_DOUBLE
    typedef union dbl_b32 {
        double val;
        int2 b32;
    } dbl_b32_t;
    dbl_b32_t t_temp_sum;
    t_temp_sum.val = temp_sum;
    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        int pull_from = (lid + i) << 2;
        dbl_b32_t upper_sum;
        upper_sum.b32.x = __builtin_amdgcn_ds_bpermute(pull_from, t_temp_sum.b32.x);
        upper_sum.b32.y = __builtin_amdgcn_ds_bpermute(pull_from, t_temp_sum.b32.y);
        max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_bpermute(pull_from, max_depth)));
        spin_time += __builtin_amdgcn_ds_bpermute(pull_from, spin_time);
        t_temp_sum.val += upper_sum.val;
    }
    temp_sum = t_temp_sum.val;
#else // !USE_DOUBLE
    for (int i = num_items >> 1; i > 0; i >>= 1)
    {
        int pull_from = (lid + i) << 2;
        max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_bpermute(pull_from, max_depth)));
        spin_time += __builtin_amdgcn_ds_bpermute(pull_from, spin_time);
        temp_sum += as_float(__builtin_amdgcn_ds_bpermute(pull_from, as_uint(temp_sum)));
    }
#endif // USE_DOUBLE
    *row_max_depth = max_depth;
    *spin_times = spin_time;
    return temp_sum;
}

// Swizzle-based reduction; this will work on Sea Islands
/*
FPTYPE swizzle_reduction(FPTYPE temp_sum)
{
#ifdef USE_DOUBLE
    typedef union dbl_b32 {
        double val;
        int2 b32;
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;

    t_temp_sum.val = temp_sum;
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x80b1);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x80b1);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x804e);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x804e);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x101f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x101f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x201f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x201f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x401f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x401f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_readlane(t_temp_sum.b32.x, 32);
    upper_sum.b32.y = __builtin_amdgcn_readlane(t_temp_sum.b32.y, 32);
    t_temp_sum.val += upper_sum.val;
    temp_sum = t_temp_sum.val;
#else // Swizzle-based for SPFP
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x80b1));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x804e));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x101f));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x201f));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x401f));
    temp_sum += as_float(__builtin_amdgcn_readlane(as_uint(temp_sum), 32));
#endif // Single or double precision

    return temp_sum;
}

// Swizzle-based reduction; this will work on Sea Islands
// This version will also put in a max-reduction for row_max_depth behind
// the s_waitcnt instructions, making it faster than two sequential
// reductions back-to-back.
__device__
FPTYPE swizzle_reduction_two(FPTYPE temp_sum, unsigned int *row_max_depth)
{
#ifdef USE_DOUBLE
    typedef union dbl_b32 {
        double val;
        int2 b32;
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;

    t_temp_sum.val = temp_sum;
    unsigned int max_depth = *row_max_depth;
    unsigned int upper_max_depth;

    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x80b1)));
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x80b1);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x80b1);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x804e)));
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x804e);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x804e);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x101f)));
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x101f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x101f);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x201f)));
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x201f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x201f);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x401f)));
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x401f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x401f);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_readlane(max_depth, 32)));
    upper_sum.b32.x = __builtin_amdgcn_readlane(t_temp_sum.b32.x, 32);
    upper_sum.b32.y = __builtin_amdgcn_readlane(t_temp_sum.b32.y, 32);
    t_temp_sum.val += upper_sum.val;
    temp_sum = t_temp_sum.val;
#else // Swizzle-based for SPFP
    unsigned int max_depth = *row_max_depth;

    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x80b1));
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x80b1)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x804e));
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x804e)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x101f));
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x101f)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x201f));
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x201f)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x401f));
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x401f)));
    temp_sum += as_float(__builtin_amdgcn_readlane(as_uint(temp_sum), 32));
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_readlane(max_depth, 32)));
#endif // Single or double precision

#ifndef SYNCFREE_KERNEL
    *row_max_depth = max_depth;
#endif
    return temp_sum;
}

// Swizzle-based reduction; this will work on Sea Islands
// This version will also put in a max-reduction for row_max_depth
// add-reduction of the spin-loop counter behind the s_waitcnt
// instructions, making it faster than two sequential reductions
// back-to-back.
__device__
FPTYPE swizzle_reduction_three(FPTYPE temp_sum, unsigned int *row_max_depth, unsigned int *spin_times)
{
    unsigned int max_depth;
    unsigned int spins;

#ifdef USE_DOUBLE
    typedef union dbl_b32 {
        double val;
        int2 b32;
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;

    t_temp_sum.val = temp_sum;
    max_depth = *row_max_depth;
    spins = *spin_times;

    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x80b1)));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x80b1);
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x80b1);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x80b1);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x804e)));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x804e);
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x804e);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x804e);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x101f)));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x101f);
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x101f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x101f);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x201f)));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x201f);
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x201f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x201f);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x401f)));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x401f);
    upper_sum.b32.x = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.x, 0x401f);
    upper_sum.b32.y = __builtin_amdgcn_ds_swizzle(t_temp_sum.b32.y, 0x401f);
    t_temp_sum.val += upper_sum.val;
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_readlane(max_depth, 32)));
    spins += __builtin_amdgcn_readlane(spins, 32);
    upper_sum.b32.x = __builtin_amdgcn_readlane(t_temp_sum.b32.x, 32);
    upper_sum.b32.y = __builtin_amdgcn_readlane(t_temp_sum.b32.y, 32);
    t_temp_sum.val += upper_sum.val;
    temp_sum = t_temp_sum.val;

#else // Swizzle-based for SPFP
    max_depth = *row_max_depth;
    spins = *spin_times;

    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x80b1));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x80b1);
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x80b1)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x804e));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x804e);
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x804e)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x101f));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x101f);
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x101f)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x201f));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x201f);
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x201f)));
    temp_sum += as_float(__builtin_amdgcn_ds_swizzle(as_uint(temp_sum), 0x401f));
    spins += __builtin_amdgcn_ds_swizzle(spins, 0x401f);
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_ds_swizzle(max_depth, 0x401f)));
    temp_sum += as_float(__builtin_amdgcn_readlane(as_uint(temp_sum), 32));
    spins += __builtin_amdgcn_readlane(spins, 32);
    max_depth = max(max_depth, as_uint(__builtin_amdgcn_readlane(max_depth, 32)));
#endif // Single or double precision

    *row_max_depth = max_depth;
    *spin_times = spins;
    return temp_sum;
}
*/

// If we are in GCN3, then use DPP to further increase the performance of
// inter-lane reduction of the temp_sum variable.
__device__
FPTYPE dpp_reduction(FPTYPE temp_sum)
{
    // If we write the EXEC mask before the DPP op, we need 5 stall cycles.
    // So every one of these starts with an s_nop 4
    // We require an s_nop 1 at the end in case the compiler immediately uses
    // the last output value.
#ifndef GCN2
#ifdef USE_DOUBLE

    typedef struct b32_2 {
        int x;
        int y;
    } b32_t;

    typedef union dbl_b32 {
        double val;
        b32_t b32;
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;
    t_temp_sum.val = temp_sum;
    upper_sum.b32.x = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.x, 0x111, 0xf, 0xf, 0); // row_shr:1
    upper_sum.b32.y = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.y, 0x111, 0xf, 0xf, 0);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.x, 0x112, 0xf, 0xf, 0); // row_shr:2
    upper_sum.b32.y = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.y, 0x112, 0xf, 0xf, 0);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.x, 0x114, 0xf, 0xe, 0); // row_shr:4 bank_mask:0xe
    upper_sum.b32.y = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.y, 0x114, 0xf, 0xe, 0);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.x, 0x118, 0xf, 0xc, 0); // row_shr:8 bank_mask:0xc
    upper_sum.b32.y = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.y, 0x118, 0xf, 0xc, 0);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.x, 0x142, 0xa, 0xf, 0); // row_bcast:15 row_mask:0xa
    upper_sum.b32.y = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.y, 0x142, 0xa, 0xf, 0);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32.x = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.x, 0x143, 0xc, 0xf, 0); // row_bcast:31 row_maxk:0xc
    upper_sum.b32.y = __builtin_amdgcn_mov_dpp(t_temp_sum.b32.y, 0x143, 0xc, 0xf, 0);
    t_temp_sum.val += upper_sum.val;
    return t_temp_sum.val;
#else // USE_DOUBLE
    __asm__ volatile ("s_nop 4\n"
                      "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                      "s_nop 1\n"
                      "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                      "s_nop 1\n"
                      "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                      "s_nop 1\n"
                      "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                      "s_nop 1\n"
                      "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                      "s_nop 1\n"
                      "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                      "s_nop 1"
                      : "=v"(temp_sum)
                      : "0"(temp_sum));
    return temp_sum;
#endif // Single vs. Double
#else // We're in GCN2, so we will never enter this function
    return temp_sum;
#endif
}

// This version of the DPP reduction function also does a max-reduce on the
// row_max_depth variable. It fits these DPP functions into one of the NOP
// slots required by the DPP instructions, so it should be fast.
__device__
FPTYPE dpp_reduction_two(FPTYPE temp_sum, unsigned int *row_max_depth)
{
    // If we write the EXEC mask before the DPP op, we need 5 stall cycles.
    // So every one of these starts with an s_nop 4
    // We require an s_nop 1 at the end in case the compiler immediately uses
    // the last output value.
    unsigned int temp_max;
#ifdef USE_DOUBLE
    typedef struct b32_2 {
        int x;
        int y;
    } b32_t;

    typedef union dbl_b32 {
        double val;
        b32_t b32;
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;
    temp_max = *row_max_depth;
    t_temp_sum.val = temp_sum;
    __asm__ volatile ("s_nop 4\n"
                      "v_mov_b32 %0 %4 row_shr:1 bound_ctrl:0\n"
                      "v_mov_b32 %1 %5 row_shr:1 bound_ctrl:0\n"
                      "v_max_u32 %2 %2 %2 row_shr:1 bound_ctrl:0\n"
                      "s_nop 0\n"
                      "v_add_f64 %3 %7 %8\n"
                      "v_mov_b32 %0 %4 row_shr:2 bound_ctrl:0\n"
                      "v_mov_b32 %1 %5 row_shr:2 bound_ctrl:0\n"
                      "v_max_u32 %2 %2 %2 row_shr:2 bound_ctrl:0\n"
                      "s_nop 0\n"
                      "v_add_f64 %3 %7 %8\n"
                      "v_mov_b32 %0 %4 row_shr:4 bank_mask:0xe\n"
                      "v_mov_b32 %1 %5 row_shr:4 bank_mask:0xe\n"
                      "v_max_u32 %2 %2 %2 row_shr:4 bank_mask:0xe\n"
                      "s_nop 0\n"
                      "v_add_f64 %3 %7 %8\n"
                      "v_mov_b32 %0 %4 row_shr:8 bank_mask:0xc\n"
                      "v_mov_b32 %1 %5 row_shr:8 bank_mask:0xc\n"
                      "v_max_u32 %2 %2 %2 row_shr:8 bank_mask:0xc\n"
                      "s_nop 0\n"
                      "v_add_f64 %3 %7 %8\n"
                      "v_mov_b32 %0 %4 row_bcast:15 bank_mask:0xa\n"
                      "v_mov_b32 %1 %5 row_bcast:15 bank_mask:0xa\n"
                      "v_max_u32 %2 %2 %2 row_bcast:15 bank_mask:0xa\n"
                      "s_nop 0\n"
                      "v_add_f64 %3 %7 %8\n"
                      "v_mov_b32 %0 %4 row_bcast:31 row_mask:0xc\n"
                      "v_mov_b32 %1 %5 row_bcast:31 bank_mask:0xc\n"
                      "v_max_u32 %2 %2 %2 row_bcast:31 bank_mask:0xc\n"
                      "s_nop 0\n"
                      "v_add_f64 %3 %7 %8\n"
                      : "={v2}"(upper_sum.b32.x), "={v3}"(upper_sum.b32.y), "=v"(temp_max),  "=v"(t_temp_sum.val)
                      :  "v"(t_temp_sum.b32.x), "v"(t_temp_sum.b32.y), "2"(temp_max), "3"(t_temp_sum.val), "{v[2:3]}"(upper_sum.val));
    *row_max_depth = temp_max;
    return t_temp_sum.val;
#else
    temp_max = *row_max_depth;
    __asm__ volatile ("s_nop 4\n"
                      "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                      "v_max_u32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                      "s_nop 0\n"
                      "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                      "v_max_u32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                      "s_nop 0\n"
                      "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                      "v_max_u32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                      "s_nop 0\n"
                      "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                      "v_max_u32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                      "s_nop 0\n"
                      "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                      "v_max_u32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                      "s_nop 0\n"
                      "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                      "v_max_u32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                      "s_nop 1\n"
                      : "=v"(temp_sum), "=v"(temp_max)
                      : "0"(temp_sum), "1"(temp_max));
    *row_max_depth = temp_max;
    return temp_sum;
#endif // Single vs. Double
}

// This version of the DPP reduction function also does a max-reduce on the
// row_max_depth variable and max-add on the total spin variable.
// It fits these DPP functions into NOP slots required by the DPP
// instructions, so it should be fast.
__device__
FPTYPE dpp_reduction_three(FPTYPE temp_sum, unsigned int *row_max_depth, unsigned int *spin_times)
{
    // If we write the EXEC mask before the DPP op, we need 5 stall cycles.
    // So every one of these starts with an s_nop 4
    // We require an s_nop 1 at the end in case the compiler immediately uses
    // the last output value.
    unsigned int temp_max = *row_max_depth;
    unsigned int temp_spin = *spin_times;
#ifdef USE_DOUBLE
    typedef struct b32_2 {
        int x;
        int y;
    } b32_t;

    typedef union dbl_b32 {
        double val;
        b32_t b32;
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;
    temp_max = *row_max_depth;
    t_temp_sum.val = temp_sum;
    __asm__ volatile ("s_nop 4\n"
                      "v_mov_b32 %0 %5 row_shr:1 bound_ctrl:0\n"
                      "v_mov_b32 %1 %6 row_shr:1 bound_ctrl:0\n"
                      "v_max_u32 %2 %2 %2 row_shr:1 bound_ctrl:0\n"
                      "v_add_u32 %3 " VCC " %3 %3 row_shr:1 bound_ctrl:0\n"
                      "v_add_f64 %4 %9 %10\n"
                      "v_mov_b32 %0 %5 row_shr:2 bound_ctrl:0\n"
                      "v_mov_b32 %1 %6 row_shr:2 bound_ctrl:0\n"
                      "v_max_u32 %2 %2 %2 row_shr:2 bound_ctrl:0\n"
                      "v_add_u32 %3 " VCC " %3 %3 row_shr:2 bound_ctrl:0\n"
                      "v_add_f64 %4 %9 %10\n"
                      "v_mov_b32 %0 %5 row_shr:4 bank_mask:0xe\n"
                      "v_mov_b32 %1 %6 row_shr:4 bank_mask:0xe\n"
                      "v_max_u32 %2 %2 %2 row_shr:4 bank_mask:0xe\n"
                      "v_add_u32 %3 " VCC " %3 %3 row_shr:4 bank_mask:0xe\n"
                      "v_add_f64 %4 %9 %10\n"
                      "v_mov_b32 %0 %5 row_shr:8 bank_mask:0xc\n"
                      "v_mov_b32 %1 %6 row_shr:8 bank_mask:0xc\n"
                      "v_max_u32 %2 %2 %2 row_shr:8 bank_mask:0xc\n"
                      "v_add_u32 %3 " VCC " %3 %3 row_shr:8 bank_mask:0xc\n"
                      "v_add_f64 %4 %9 %10\n"
                      "v_mov_b32 %0 %5 row_bcast:15 row_mask:0xa\n"
                      "v_mov_b32 %1 %6 row_bcast:15 row_mask:0xa\n"
                      "v_max_u32 %2 %2 %2 row_bcast:15 row_mask:0xa\n"
                      "v_add_u32 %3 " VCC " %3 %3 row_bcast:15 row_mask:0xa\n"
                      "v_add_f64 %4 %9 %10\n"
                      "v_mov_b32 %0 %5 row_bcast:31 row_mask:0xc\n"
                      "v_mov_b32 %1 %6 row_bcast:31 row_mask:0xc\n"
                      "v_max_u32 %2 %2 %2 row_bcast:31 row_mask:0xc\n"
                      "v_add_u32 %3 " VCC " %3 %3 row_bcast:31 row_mask:0xc\n"
                      "v_add_f64 %4 %9 %10\n"
                      "s_nop 0\n"
                      : "={v2}"(upper_sum.b32.x), "={v3}"(upper_sum.b32.y), "=v"(temp_max), "=v"(temp_spin), "=v"(t_temp_sum.val)
                      : "v"(t_temp_sum.b32.x), "v"(t_temp_sum.b32.y), "2"(temp_max), "3"(temp_spin), "4"(t_temp_sum.val), "{v[2:3]}"(upper_sum.val));
    *row_max_depth = temp_max;
    *spin_times = temp_spin;
    return t_temp_sum.val;
#else
    __asm__ volatile ("s_nop 4\n"
                      "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                      "v_max_u32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                      "v_add_u32 %2 " VCC " %2 %2 row_shr:1 bound_ctrl:0\n"
                      "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                      "v_max_u32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                      "v_add_u32 %2 " VCC " %2 %2 row_shr:2 bound_ctrl:0\n"
                      "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                      "v_max_u32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                      "v_add_u32 %2 " VCC " %2 %2 row_shr:4 bank_mask:0xe\n"
                      "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                      "v_max_u32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                      "v_add_u32 %2 " VCC " %2 %2 row_shr:8 bank_mask:0xc\n"
                      "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                      "v_max_u32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                      "v_add_u32 %2 " VCC " %2 %2 row_bcast:15\n"
                      "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                      "v_max_u32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                      "v_add_u32 %2 " VCC " %2 %2 row_bcast:31\n"
                      "s_nop 1"
                      : "=v"(temp_sum), "=v"(temp_max), "=v"(temp_spin)
                      : "0"(temp_sum), "1"(temp_max), "2"(temp_spin));
    *row_max_depth = temp_max;
    *spin_times = temp_spin;
    return temp_sum;
#endif // Single vs. Double
}

// Possible reduction techniques:
//#define LDS_REDUCTION
//#define BPERMUTE_REDUCTION
//#define SWIZZLE_REDUCTION

//#define DPP_REDUCTION

#if defined(GCN2) && defined(DPP_REDUCTION)
#define SWIZZLE_REDUCTION
#undef DPP_REDUCTION
#endif

#ifdef DPP_REDUCTION
    #define OUTPUT_THREAD WF_SIZE-1
#else
    #define OUTPUT_THREAD 0
#endif

__device__
inline FPTYPE cross_lane_reduction(FPTYPE temp_sum, __shared__ FPTYPE *lds_ptr,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)
{
#ifdef LDS_REDUCTION
    FPTYPE temp_val = lds_reduction(temp_sum, lds_ptr, start_of_this_row,
            end_of_this_row, wg_lid);
    return temp_val;
#endif

#ifdef BPERMUTE_REDUCTION
    return bpermute_reduction(temp_sum, start_of_this_row, end_of_this_row,
            wg_lid);
#endif

#ifdef SWIZZLE_REDUCTION
    return swizzle_reduction(temp_sum);
#endif

#ifdef DPP_REDUCTION
    return dpp_reduction(temp_sum);
#endif
}

__device__
inline FPTYPE cross_lane_reduction_two(FPTYPE temp_sum, unsigned int *row_max_depth,
        __shared__ FPTYPE *lds_ptr, __shared__ unsigned int *max_depth_ptr,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)
{
#ifdef LDS_REDUCTION
    FPTYPE temp_val = lds_reduction_two(temp_sum, *row_max_depth, lds_ptr,
            max_depth_ptr, start_of_this_row, end_of_this_row, wg_lid);
    *row_max_depth = max_depth_ptr[wg_lid & (~(WF_SIZE-1))];
    return temp_val;
#endif

#ifdef BPERMUTE_REDUCTION
    return bpermute_reduction_two(temp_sum, row_max_depth, start_of_this_row,
            end_of_this_row, wg_lid);
#endif

#ifdef SWIZZLE_REDUCTION
    return swizzle_reduction_two(temp_sum, row_max_depth);
#endif

#ifdef DPP_REDUCTION
    return dpp_reduction_two(temp_sum, row_max_depth);
#endif
}

__device__
inline FPTYPE cross_lane_reduction_three(FPTYPE temp_sum, unsigned int *row_max_depth,
        unsigned int *spin_times, __shared__ FPTYPE *lds_ptr,
        __shared__ unsigned int *max_depth_ptr, __shared__ unsigned int *total_spins_ptr,
        unsigned int start_of_this_row, unsigned int end_of_this_row,
        unsigned int wg_lid)

{

#ifdef LDS_REDUCTION
    FPTYPE temp_val = lds_reduction_three(temp_sum, *row_max_depth, *spin_times,
            lds_ptr, max_depth_ptr, total_spins_ptr, start_of_this_row,
            end_of_this_row, wg_lid);
    *row_max_depth = max_depth_ptr[wg_lid & (~(WF_SIZE-1))];
    *spin_times = total_spins_ptr[wg_lid & (~(WF_SIZE-1))];
    return temp_val;
#endif

#ifdef BPERMUTE_REDUCTION
    return bpermute_reduction_three(temp_sum, row_max_depth, spin_times,
            start_of_this_row, end_of_this_row, wg_lid);
#endif

#ifdef SWIZZLE_REDUCTION
    return swizzle_reduction_three(temp_sum, row_max_depth, spin_times);
#endif

#ifdef DPP_REDUCTION
    return dpp_reduction_three(temp_sum, row_max_depth, spin_times);
#endif

	return temp_sum;
}

// The option below will, in the analyze and syncfree kernels, attempt to
// spin-loop on flags in the LDS for rows that are being solved by wavefronts
// earlier in the same workgroup. This should relieve global memory pressure.
// We found that, with careful control of branching for this logic, this yields
// an average of 20% better performance than global spin-looping.
#define USE_LDS_SPINLOOP

// The option below will, in the levelsync kernel, attempt to spin-loop on
// flags in the LDS for rows that are being solved for wavefronts earlier in
// the same workgroup. This is beneficial if levels have very few rows in them,
// as workgroups are likely to have multiple levels and thus require spinning.
// However, knowing what rows are in the LDS entry is more difficult for the
// levelsync kernel, because it depends entirely on the rowMap entries being
// used by these waves. As such, this loses performance when walking the row
// map outweights the spin-loop benefits. As of this writing, the levelsync
// LDS spin-loop is a net loser.
// Leaving this around for future studies.
// #define USE_LDS_SPINLOOP_LEVELSYNC


// Solves for 'y' in the equation 'A * y = alpha * x'
// In this kernel, we do not know what level each row is in. As such, we must
// dynamically figure this out. Each row has the potential to require data from
// a previous row. This happens when it has a non-zero in a column.
// i.e. having a non-zero value in column $foo means you must wait for row $foo
// to finish.
//
// The 'doneArray' has one entry per row. It starts out with each entry containing
// zeroes. When a row finishes and its output written, it knows its own level
// (which must be 1 more than the highest level of any row it relied on). As such,
// it puts that level into the doneArray. If you must wait on a previous row, you
// spinloop on that row's doneArray entry. Once it's non-zero, you know both that
// the data is ready, as well as what level that value came from (so you can
// calculate your own level).
//
// The doneArray can be used for future iterations, since the parllelism doesn't
// change between iterations. As such, we keep the doneArray around and call
// a different kernel that doesn't do the spin-loop waiting. To prep for that
// kernel, we also need to know how many rows are at each level. Thus, when a
// row finishes, it increments the numRowsAtLevel[] entry associated with its
// level. Also we set the maxDepth variable to the maximum of any level seen.
//__attribute__((reqd_work_group_size(WF_SIZE*WF_PER_WG, 1, 1)))
//__kernel void
__global__ void __launch_bounds__(WF_SIZE * WF_PER_WG, 1)
amd_spts_analyze_and_solve(
                            const size_t global_work_size,
#ifdef USE_ROCSHMEM
                            const int this_pe,
                            const int total_pes,
                            unsigned int * __restrict__ shadowDoneArray,
                            unsigned int * __restrict__ reqUpdateArray,
                            unsigned int * __restrict__ remoteInProgressArray,
                            unsigned int * __restrict__ oneBuf,
			    // 0: Naive puts
			    // 1: Naive gets
			    // 2: blocked puts
			    // 3: put/get hybrid
                            int rocshmem_algorithm,
							int rocshmem_put_block_size,
							int rocshmem_get_backoff_factor,
                            int spts_block_size,
#endif
                            const FPTYPE * __restrict__ vals,
                            const int * __restrict__ cols,
                            const int * __restrict__ rowPtrs,
                            const FPTYPE * __restrict__ vec_x,
                            FPTYPE * __restrict__ out_y,
                            const FPTYPE alpha,
                            unsigned int * __restrict__ doneArray,
                            unsigned int * __restrict__ numRowsAtLevel,
                            unsigned int * __restrict__ maxDepth,
                            unsigned long long * __restrict__ totalSpin)
{
    __shared__ FPTYPE *lds_ptr;
    lds_ptr = nullptr;
    __shared__ unsigned int *max_depth_ptr;
    max_depth_ptr = nullptr;
    __shared__ unsigned int *total_spins_ptr;
    total_spins_ptr = nullptr;
#ifdef LDS_REDUCTION
    __shared__ FPTYPE lds[WF_SIZE*WF_PER_WG];
    lds_ptr = lds;
#endif

    // If we want future kernel iterations to skip the "wait on previous rows"
    // work, we need to know what level set this row is in. This array is used
    // to calculate the depth of each dependency so we can calculate max+1.
#ifdef LDS_REDUCTION
    __shared__ unsigned int max_depth[WF_SIZE*WF_PER_WG];
    max_depth_ptr = max_depth;
    __shared__ unsigned int total_spins[WF_SIZE*WF_PER_WG];
    total_spins_ptr = total_spins;
#endif // LDS_REDUCTION
    unsigned int row_max_depth = 0;
    unsigned int spin_times = 0;
    const unsigned int wg_lid = hipThreadIdx_x;
    const unsigned int lid = wg_lid % WF_SIZE;

#ifdef USE_ROCSHMEM
    __shared__ rocshmem_ctx_t ctx;


    //if (wg_lid == OUTPUT_THREAD) {
    rocshmem_wg_init();
    rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);
    __syncthreads();
#endif

    // Which wavefront within this workgroup
    // also means which row within this workgroup's group of rows
    const unsigned int local_offset = wg_lid / WF_SIZE;
    // First row within this workgroup (within this group of rows)
    const unsigned int local_first_row = hipBlockIdx_x * WF_PER_WG;
    // Actual row this wavefront will work on.
    const unsigned int local_row = local_first_row + local_offset;

#ifdef USE_ROCSHMEM
    // Get the global row for this wavefront assuming a row-cyclic
    // decomposition.  Basically we need to account for other PEs here.
    int local_block_id = local_row / spts_block_size;
    const unsigned int block_offset = (local_block_id * spts_block_size * total_pes) +
        (this_pe * spts_block_size);
    const unsigned int row = block_offset + (local_row % spts_block_size);
    const unsigned int first_row = block_offset + (local_first_row % spts_block_size);
#else
    const unsigned int row = local_row;
    const unsigned int first_row = local_first_row;
#endif

    __shared__ FPTYPE diagonal[WF_PER_WG];

#ifdef USE_LDS_SPINLOOP
    // If we are trying to access an output that was produced by a wavefront
    // earlier in this workgroup, perform the transfer and spin-loop in LDS
    // to relieve global memory pressure.
    __shared__ unsigned int localDoneArray[WF_PER_WG];
    __shared__ FPTYPE localOutY[WF_PER_WG];
    __syncthreads();

    if (global_work_size > (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x)) {

    if (lid == 0)
    {
        localDoneArray[local_offset] = 0;
        localOutY[local_offset] = 0.;
    }
#else
    if (global_work_size > (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x)) {
#endif

    FPTYPE temp_sum = 0.;
    // Preload the first thread with alpha * x. We can bring this forward
    // because the 'x' vector in A*y=alpha*x is fixed and known already.
    // From this point on, we will subtract out values from rows of X from
    // alpha*x, and that will allow us to solve for entries of y.
    // Hauling this up to the top of the kernel increases performance because
    // it removes the memory load and multiply from the critical path of
    // "previous rows' inputs are ready, finish this and allow further rows
    // to start up as fast as possible."
    if (lid == OUTPUT_THREAD)
        temp_sum = alpha * vec_x[row];

    unsigned int start_of_this_row = rowPtrs[row];
    unsigned int end_of_this_row = rowPtrs[row+1];
    unsigned int start_point = start_of_this_row+lid;


    // This wavefront operates on a single row, from its beginning to end.
    for(unsigned int j = start_point; j < end_of_this_row; j+=WF_SIZE)
    {

        FPTYPE out_val;
        unsigned int local_done = 0;
        // Replace the two loads below with inline assembly that sets the
        // SLC bit. This forces the loads to essentially bypass the L2
        // to increase cache hit rate on other instructions. Vals and cols
        // are basically streamed in, so caching them doesn't help much.

        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        int local_col = __builtin_nontemporal_load(&cols[j]);
        // Haul loading from vals[] up near the load of cols[] so that we get
        // good coalsced loads.
        FPTYPE local_val = __builtin_nontemporal_load(&vals[j]);

        // diagonal. Skip this, we need to solve for it.
        if (local_col == row)
        {
            local_done = 1;
            diagonal[local_offset] = local_val;
	    local_val = 0.; // Make the out_val multiply below do nothing.
        }

        // While there are threads in this workgroup that have been unable to
        // get their input, loop and wait for the flag to exist.
        __asm__ volatile ("s_setprio 0");
#ifdef USE_ROCSHMEM
        int target_pe = (local_col / spts_block_size) % total_pes;
        int backoff_counter = 0;
        bool need_remote_notify = true;
		bool need_comm = true;
        bool first_time = true;

#endif

#ifdef USE_LDS_SPINLOOP
	if (local_col >= first_row)
	{
            while (!local_done)
            {
                // Check in the LDS if the value was produced by someone
                // within this workgroup.
                local_done = localDoneArray[local_col - first_row];
                out_val = localOutY[local_col - first_row];
        	asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
	    }
        }
#endif // USE_LDS_SPINLOOP
	while (!local_done)
        {
            // Replace this atomic with an assembly load with GLC bit set.
            // This forces the load to go to the coherence point, allowing
            // us to avoid deadlocks.
            // local_done = atomic_get_done(doneArray, local_col);
            __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc slc\n"
                "s_waitcnt vmcnt(0)"
                : "=v"(local_done)
                : "v"(&doneArray[local_col]));

            spin_times++;

#ifdef USE_ROCSHMEM
            if ((total_pes > 1) && (target_pe != this_pe) && (rocshmem_algorithm == 1)) {
				if (first_time) {
                    if (atomicCAS(&remoteInProgressArray[local_col], 0, 1) != 0)
                        need_comm = false;
                }
				first_time = false;
				if (need_comm)
					{
                    for (int i = 0; i < (backoff_counter * rocshmem_get_backoff_factor); i++)
                        __asm__ volatile("s_sleep 127");


                    rocshmem_ctx_getmem_nbi(ctx, &shadowDoneArray[local_col], &doneArray[local_col], sizeof(int), target_pe);
		        	//rocshmem_ctx_quiet(ctx);

                	__asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc slc\n"
                    	"s_waitcnt vmcnt(0)"
                    	: "=v"(local_done)
                    	: "v"(&shadowDoneArray[local_col]));


                	if (local_done)
                	{
                        rocshmem_ctx_getmem_nbi(ctx, &out_y[local_col], &out_y[local_col], sizeof(FPTYPE), target_pe);

                    	__asm__ volatile (MEM_PREFIX"_store_dword %0 %1 " OFF_MODIFIER " glc\n" WAKEUP
		        			:
                        	: "v"(&doneArray[local_col]),
                          	"v"(local_done));
                		} else {
                    		backoff_counter++;

             			}

            	}
			}

            if ((total_pes > 1) && (target_pe != this_pe) && (rocshmem_algorithm == 3)) {
                if (need_remote_notify) {
                    need_remote_notify = false;
                    //if (atomicCAS(&remoteInProgressArray[local_col], 0, 1) != 0)
                    //if (atomicCAS(&remoteInProgressArray[local_col], 0, 1) == 0)
		            {
                        rocshmem_ctx_putmem_nbi(ctx, &reqUpdateArray[local_col], oneBuf, sizeof(int), target_pe);
					   //printf("Put 111 blockIDx %d threadID %d target_pe  %d   local_col %d  oneBuf[0]= %d \n", hipBlockIdx_x, hipThreadIdx_x, target_pe, local_col, oneBuf[0]);

                        rocshmem_ctx_fence(ctx);
					   //printf("fence 222 blockIDx %d threadID %d target_pe  %d   local_col %d \n", hipBlockIdx_x, hipThreadIdx_x, target_pe, local_col);
                        rocshmem_ctx_getmem_nbi(ctx, &shadowDoneArray[local_col], &doneArray[local_col], sizeof(int), target_pe);
                        rocshmem_ctx_quiet(ctx);
					   //printf("Get 333  blockIDx %d threadID %d target_pe  %d   local_col %d shadowDone %d \n \n", hipBlockIdx_x, hipThreadIdx_x, target_pe, local_col, shadowDoneArray[local_col]);

                        __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc slc\n"
                            "s_waitcnt vmcnt(0)"
                            : "=v"(local_done)
                            : "v"(&shadowDoneArray[local_col]));

                        if (local_done)
                        {
                            rocshmem_ctx_getmem_nbi(ctx, &out_y[local_col], &out_y[local_col], sizeof(FPTYPE), target_pe);
			    			rocshmem_ctx_quiet(ctx);
                            __asm__ volatile (MEM_PREFIX"_store_dword %0 %1 " OFF_MODIFIER " glc\n" WAKEUP
                                    :
                                    : "v"(&doneArray[local_col]),
                                    "v"(local_done));
                        }
                    }
	            }
            }
#endif
        }

        __asm__ volatile ("s_setprio 1");
#ifdef USE_LDS_SPINLOOP
        if (local_col < first_row)
#endif
        {
            // The command below is manually replaced with GCN assembly with
            // the GLC bit set. This bypasses the L1, allowing us to do a
            // coherent load of the variable without needing atomics.
#ifdef USE_DOUBLE
            // out_val = as_double(atom_or((__global ulong *)&(out_y[local_col]), 0));
            __asm__ volatile (MEM_PREFIX"_load_dwordx2 %0 %1 " OFF_MODIFIER " glc\n"
                "s_waitcnt vmcnt(0)"
                : "=v"(out_val)
                : "v"(&out_y[local_col]));
#else
            // out_val = as_float(atomic_or((__global uint *)&(out_y[local_col]), 0));
            __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc\n"
                "s_waitcnt vmcnt(0)"
                : "=v"(out_val)
                : "v"(&out_y[local_col]));
#endif
        }
        temp_sum -= local_val * out_val;

        row_max_depth = max(local_done, row_max_depth);
    }
    __asm__ volatile ("s_setprio 1");

    // And if we care about the maximum depth, add it into OUTPUT_THREAD's
    // entry within the max_depth array.
    temp_sum = cross_lane_reduction_three(temp_sum, &row_max_depth, &spin_times,
            lds_ptr, max_depth_ptr, total_spins_ptr, start_of_this_row,
            end_of_this_row, wg_lid);
    row_max_depth++;

    // y = (x-sum_of_vals_from_A) / diag
    if (lid == OUTPUT_THREAD)
    {
#ifndef LDS_REDUCTION
        // Wait for local memory to quiesce for the diagonal
        // LDS_REDUCTION has such waits in it already.
        asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
#endif
        FPTYPE out_val = temp_sum / diagonal[local_offset];
        //out_y[row] = out_val;

#ifdef USE_DOUBLE
        __asm__ volatile (MEM_PREFIX"_store_dwordx2 %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : : "v" (&out_y[row]), "v"(out_val));
#else
        __asm__ volatile (MEM_PREFIX"_store_dword %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : : "v" (&out_y[row]), "v"(out_val));
#endif

        //out_y[row] = temp_sum / diagonal[local_offset]; // original divide
#ifdef USE_LDS_SPINLOOP
        localOutY[row - first_row] = out_val;
        localDoneArray[row - first_row] = row_max_depth;
#endif // USE_LDS_SPINLOOP
        //doneArray[row] = row_max_depth;
        __asm__ volatile (MEM_PREFIX"_store_dword %0 %1 " OFF_MODIFIER " glc\n" WAKEUP : : "v"(&doneArray[row]), "v"(row_max_depth));
        asm volatile ("s_waitcnt vmcnt(0)\n\t");

#ifdef USE_ROCSHMEM
    if (rocshmem_algorithm == 2 && total_pes > 1) {
        int CHUNK = rocshmem_put_block_size;
        bool sendTime = true;
        int row_base = (row / CHUNK) * CHUNK;
        int num_done = atomicAdd(&shadowDoneArray[row_base], 1);
        sendTime = (num_done == (CHUNK - 1));
        for(int p=0; p<total_pes; p++){
            if(p != this_pe && sendTime){
                rocshmem_ctx_putmem_nbi(ctx, &out_y[row_base], &out_y[row_base], sizeof(FPTYPE) * CHUNK, p);
                rocshmem_ctx_fence(ctx);
                rocshmem_ctx_putmem_nbi(ctx, &doneArray[row_base], &doneArray[row_base], sizeof(int) * CHUNK, p);
                rocshmem_ctx_quiet(ctx);
            }
        }
    }

	if (rocshmem_algorithm == 0) {
        for(int p=0; p<total_pes; p++){
            if(p != this_pe){
                rocshmem_ctx_putmem_nbi(ctx, &out_y[row], &out_y[row], sizeof(FPTYPE), p);
                rocshmem_ctx_fence(ctx);
                rocshmem_ctx_putmem_nbi(ctx, &doneArray[row], &doneArray[row], sizeof(int), p);
            }
        }
	}

	if (rocshmem_algorithm == 3) {
	    // Only broadcast update if another node explicitly registered for this row.  TODO:
	    // Make 2D array to scale
	    unsigned int need_broadcast;
        __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc slc\ns_waitcnt vmcnt(0)" : "=v"(need_broadcast) : "v"(&reqUpdateArray[row]));

	    if (need_broadcast == 1) {
            for(int p=0; p<total_pes; p++) {
                if (p != this_pe) {
                    rocshmem_ctx_putmem_nbi(ctx, &out_y[row], &out_y[row], sizeof(FPTYPE), p);
		    		rocshmem_ctx_fence(ctx);
                    rocshmem_ctx_putmem_nbi(ctx, &doneArray[row], &doneArray[row], sizeof(int), p);
                }
            }
	    }
	}
#endif

        // Must atomic these next two, since other WGs are doing the same thing
        // We're sending out "row_max_depth-1" because of 0-based indexing.
        // However, we needed to put a non-zero value into the doneArray up above
        // when we crammed row_max_depth in, so these two will be off by one.
        atomicAdd(&numRowsAtLevel[row_max_depth-1], 1);
        atomicMax(maxDepth, row_max_depth);
        atomicAdd(totalSpin, spin_times);
        // If you add this back in after doing a native_divide up above,
        // we can get *some* of the accuracy of a full Newton-Raphson
        // divide while maintaining the performance of the
        // native_divide() on the critical path.
        //out_y[row] = temp_sum / diagonal[local_offset];
    }
    }

    #ifdef USE_ROCSHMEM
    __syncthreads();
    //if (wg_lid == OUTPUT_THREAD)
    rocshmem_wg_ctx_destroy(ctx);
    rocshmem_wg_finalize();
    #endif
}

// Solves for 'y' in the equation 'A * y = alpha * x'
// In this kernel, we do not know what level each row is in. As such, we must
// dynamically figure this out. Each row has the potential to require data from
// a previous row. This happens when it has a non-zero in a column.
// i.e. having a non-zero value in column $foo means you must wait for row $foo
// to finish.
//
// The 'doneArray' has one entry per row. It starts out with each entry containing
// zeroes. When a row finishes and its output written, it knows its own level
// (which must be 1 more than the highest level of any row it relied on). As such,
// it puts that level into the doneArray. If you must wait on a previous row, you
// spinloop on that row's doneArray entry. Once it's non-zero, you know both that
// the data is ready, as well as what level that value came from (so you can
// calculate your own level).
//
// The doneArray can be used for future iterations, since the parllelism doesn't
// change between iterations. As such, we keep the doneArray around and call
// a different kernel that doesn't do the spin-loop waiting. To prep for that
// kernel, we also need to know how many rows are at each level. Thus, when a
// row finishes, it increments the numRowsAtLevel[] entry associated with its
// level. Also we set the maxDepth variable to the maximum of any level seen.
//__attribute__((reqd_work_group_size(WF_SIZE*WF_PER_WG, 1, 1)))
//__kernel void
__global__ void __launch_bounds__(WF_SIZE * WF_PER_WG, 1)
amd_spts_syncfree_solve(
                            size_t global_work_size,
                            const FPTYPE * __restrict__ vals,
                            const int * __restrict__ cols,
                            const int * __restrict__ rowPtrs,
                            const FPTYPE * __restrict__ vec_x,
                            FPTYPE * __restrict__ out_y,
                            const FPTYPE alpha,
                            unsigned int * __restrict__ doneArray,
                            unsigned int * __restrict__ numRowsAtLevel,
                            unsigned int * __restrict__ maxDepth,
                            unsigned long long * __restrict__ totalSpin)
{
    if (global_work_size <= hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) return;
    __shared__ FPTYPE *lds_ptr;
    lds_ptr = nullptr;
    __shared__ unsigned int *max_depth_ptr;
    max_depth_ptr = nullptr;
    __shared__ unsigned int *total_spins_ptr;
    total_spins_ptr = nullptr;
#ifdef LDS_REDUCTION
    __shared__ FPTYPE lds[WF_SIZE*WF_PER_WG];
    lds_ptr = lds;
#endif

    const unsigned int wg_lid = hipThreadIdx_x;
    const unsigned int lid = wg_lid % WF_SIZE;

    // Which wavefront within this workgroup
    // also means which row within this workgroup's group of rows
    const unsigned int local_offset = wg_lid / WF_SIZE;
    // First row within this workgroup (within this group of rows)
    const unsigned int first_row = hipBlockIdx_x * WF_PER_WG;
    // Actual row this wavefront will work on.
    const unsigned int row = first_row + local_offset;

    __shared__ FPTYPE diagonal[WF_PER_WG];

#ifdef USE_LDS_SPINLOOP
    // If we are trying to access an output that was produced by a wavefront
    // earlier in this workgroup, perform the transfer and spin-loop in LDS
    // to relieve global memory pressure.
    __shared__ unsigned int localDoneArray[WF_PER_WG];
    __shared__ FPTYPE localOutY[WF_PER_WG];
#endif

    FPTYPE temp_sum = 0.;
    // Preload the first thread with alpha * x. We can bring this forward
    // because the 'x' vector in A*y=alpha*x is fixed and known already.
    // From this point on, we will subtract out values from rows of X from
    // alpha*x, and that will allow us to solve for entries of y.
    // Hauling this up to the top of the kernel increases performance because
    // it removes the memory load and multiply from the critical path of
    // "previous rows' inputs are ready, finish this and allow further rows
    // to start up as fast as possible."
    if (lid == OUTPUT_THREAD)
        temp_sum = alpha * vec_x[row];

    unsigned int start_of_this_row = rowPtrs[row];
    unsigned int end_of_this_row = rowPtrs[row+1];
    unsigned int start_point = start_of_this_row+lid;
    // This wavefront operates on a single row, from its beginning to end.

    for(unsigned int j = start_point; j < end_of_this_row; j+=WF_SIZE)
    {
#ifdef USE_LDS_SPINLOOP
        if (lid == 0)
        {
            localDoneArray[local_offset] = 0;
            localOutY[local_offset] = 0.;
        }
#endif

        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        int local_col = -1;
        // Haul loading from vals[] up near the load of cols[] so that we get
        // good coalsced loads.
        FPTYPE local_val = 0.;
        unsigned int local_done = 0;

        // Replace the two loads below with inline assembly that sets the
        // SLC bit. This forces the loads to essentially bypass the L2
        // to increase cache hit rate on other instructions. Vals and cols
        // are basically streamed in, so caching them doesn't help much.
        // local_col = cols[j];
        // local_val = vals[j];
#ifdef USE_DOUBLE
	__asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dwordx2 %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#else
	__asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dword %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#endif

        // diagonal. Skip this, we need to solve for it.
        if (local_col == row)
        {
            local_done = 1;
            diagonal[local_offset] = local_val;
        }

        // While there are threads in this workgroup that have been unable to
        // get their input, loop and wait for the flag to exist.
        __asm__ volatile ("s_setprio 0");
        while (!local_done)
        {
#ifdef USE_LDS_SPINLOOP
            if (local_col >= first_row)
            {
                // Check in the LDS if the value was produced by someone
                // within this workgroup.
                local_done = localDoneArray[local_col - first_row];
                asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
            }
            else
#endif // USE_LDS_SPINLOOP
            {
                // Replace this atomic with an assembly load with GLC bit set.
                // This forces the load to go to the coherence point, allowing
                // us to avoid deadlocks.
                // local_done = atomic_get_done(doneArray, local_col);
                __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc slc\ns_waitcnt vmcnt(0)" : "=v"(local_done) : "v"(&doneArray[local_col]));
            }
            if (local_done)
            {
                FPTYPE out_val;
                __asm__ volatile ("s_setprio 1");
#ifdef USE_LDS_SPINLOOP
                if (local_col >= first_row)
                {
                    out_val = localOutY[local_col - first_row];
                    asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
                }
                else
#endif // USE_LDS_SPINLOOP
                {
                    // The command below is manually replaced with GCN assembly with
                    // the GLC bit set. This bypasses the L1, allowing us to do a
                    // coherent load of the variable without needing atomics.
#ifdef USE_DOUBLE
                    // out_val = as_double(atom_or((__global ulong *)&(out_y[local_col]), 0));
                    __asm__ volatile (MEM_PREFIX"_load_dwordx2 %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : "=v"(out_val) : "v"(&out_y[local_col]));
#else
                    // out_val = as_float(atomic_or((__global uint *)&(out_y[local_col]), 0));
                    __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : "=v"(out_val) : "v"(&out_y[local_col]));
#endif
                }
                temp_sum -= local_val * out_val;

            }
            else
            {
                (void)0;
            }
        }
    }
    __asm__ volatile ("s_setprio 1");

    // Take all of the temp_sum values and add them together into
    // OUTPUT_THREAD's temp_sum value.
    temp_sum = cross_lane_reduction(temp_sum, lds_ptr, start_of_this_row,
            end_of_this_row, wg_lid);

    // y = (x-sum_of_vals_from_A) / diag
    if (lid == OUTPUT_THREAD)
    {
#ifndef LDS_REDUCTION
        // Wait for local memory to quiesce for the diagonal
        // LDS_REDUCTION has such waits in it already.
        asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
#endif
        FPTYPE out_val = temp_sum / diagonal[local_offset];
        //out_y[row] = out_val;
#ifdef USE_DOUBLE
        __asm__ volatile (MEM_PREFIX"_store_dwordx2 %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : : "v" (&out_y[row]), "v"(out_val));
#else
        __asm__ volatile (MEM_PREFIX"_store_dword %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : : "v" (&out_y[row]), "v"(out_val));
#endif
        //out_y[row] = temp_sum / diagonal[local_offset]; // original divide
        int set_one = 1;
#ifdef USE_LDS_SPINLOOP
        localDoneArray[row - first_row] = 1;
        localOutY[row - first_row] = out_val;
#endif // USE_LDS_SPINLOOP
        //doneArray[row] = 1;
        __asm__ volatile (MEM_PREFIX"_store_byte %0 %1 " OFF_MODIFIER " glc\n" WAKEUP : : "v"(&doneArray[row]), "v"(set_one));
        // If you add this back in after doing a native_divide up above,
        // we can get *some* of the accuracy of a full Newton-Raphson
        // divide while maintaining the performance of the
        // native_divide() on the critical path.
        //out_y[row] = temp_sum / diagonal[local_offset];
    }
}

// Solves for 'y' in the equation 'A * y = alpha * x'
// In this kernel, every row is in the same level. As such, we can freely
// have every workgrup complete at its own pace.
// However, we must call this kernel multiple times, once per level.
//
// The rowMap tells us that, in this level, gid X works on row Y.
// We need this because each level of the solve can have different numbers
// of non-contiguous row. This version of our solver uses one kernel call
// per level.
//
// In addition, the 'total_rows_in_prev_levels' tells us how far in that array
// to look.
__global__ void __launch_bounds__(WF_SIZE * WF_PER_WG, 1)
amd_spts_levelset_solve(
                      size_t global_work_size,
                      const FPTYPE * __restrict__  vals,
                      const int * __restrict__  cols,
                      const int * __restrict__  rowPtrs,
                      const FPTYPE * __restrict__  vec_x,
                      FPTYPE * __restrict__  out_y,
                      const unsigned int * __restrict__  rowMap,
                      const unsigned int total_rows_in_prev_levels,
                      const FPTYPE alpha)
{
    if (global_work_size <= hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) return;
    __shared__ FPTYPE *lds_ptr;
    lds_ptr = nullptr;
#ifdef LDS_REDUCTION
    __shared__ FPTYPE lds[WF_SIZE*WF_PER_WG];
    lds_ptr = lds;
#endif

    // Which wavefront within this workgroup
    // also means which row within this workgroup's group of rows
    const unsigned int local_offset = hipThreadIdx_x / WF_SIZE;
    // First row within this workgroup (within this group of rows)
    const unsigned int first_row = hipBlockIdx_x * WF_PER_WG;

    const unsigned int wg_lid = hipThreadIdx_x;
    const unsigned int lid = wg_lid % WF_SIZE;

    const unsigned int row = rowMap[total_rows_in_prev_levels+first_row+local_offset];

    __shared__ FPTYPE diagonal[WF_PER_WG];
    FPTYPE temp_sum = 0.;

    // Preload the first thread with alpha * x. We can bring this forward
    // because the 'x' vector in A*y=alpha*x is fixed and known already.
    // From this point on, we will subtract out values from rows of X from
    // alpha*x, and that will allow us to solve for entries of y.
    if (lid == OUTPUT_THREAD)
        temp_sum = alpha * vec_x[row];

    unsigned int start_of_this_row = rowPtrs[row];
    unsigned int end_of_this_row = rowPtrs[row+1];
    unsigned int start_point = start_of_this_row+lid;

    // This workgroup operates on a single row, from its beginning to end.
    for(unsigned int j = start_point; j < end_of_this_row; j+=WF_SIZE)
    {
        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        int local_col = -1;
        // Haul loading from vals[] up near the load of cols[] so that we get
        // good coalsced loads.
        FPTYPE local_val = 0.;

        // Replace the two loads below with inline assembly that sets the
        // SLC bit. This forces the loads to essentially bypass the L2
        // to increase cache hit rate on other instructions. Vals and cols
        // are basically streamed in, so caching them doesn't help much.
        // local_col = cols[j];
        // local_val = vals[j];
#ifdef USE_DOUBLE
        __asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dwordx2 %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#else
        __asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dword %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#endif

        // diagonal. Skip this, we need to solve for it.
        if (local_col == row)
            diagonal[local_offset] = local_val;
        else
        {
            FPTYPE out_val = out_y[local_col];
            temp_sum -= local_val * out_val;
        }
    }
    // Take all of the temp_sum values and add them together into
    // OUTPUT_THREAD's temp_sum value.
    temp_sum = cross_lane_reduction(temp_sum, lds_ptr,
            start_of_this_row, end_of_this_row, wg_lid);

    // y = (x-sum_of_vals_from_A) / diag
    if (lid == OUTPUT_THREAD)
    {
#ifndef LDS_REDUCTION
        // Wait for local memory to quiesce for the diagonal
        // LDS_REDUCTION has such waits in it already.
        asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
#endif
        out_y[row] = temp_sum / diagonal[local_offset]; // original divide
        //out_y[row] = temp_sum / diagonal[local_offset]; // original divide
    }
}

// Solves for 'y' in the equation 'A * y = alpha * x'
// This kernel will only work if we launch a single workgroup that will
// solve multiple levels in a serial fashion. For each level, every thread
// within that level will try to solve for a different row.
// After solving for this level, the single workgroup hits a workgroup-wide
// barrier instruction waiting for all the other rows in this level to
// complete.
//
// We can only solve up to 1024 rows in a single level call right now,
// because each thread will solve a single row per level.
//
// This is a "CSR-Scalar" style analysis, where each thread is accessing
// a potentially very different area of both the CSR matrix and the vector.
// Performance may be bad, but this is very easy to write.
//
// The rowMap tells us that, within a level, thread X works on row Y.
// We need this because each level of the solve can have different numbers
// of non-contiguous row.
// In addition, the 'total_rows_in_prev_levels' tells us how far in that array
// to look.
//
// [start_level, end_level) tell us which entries in the rowMap we will go
// through in this kernel invocation.
__global__ void __launch_bounds__(WF_SIZE * WF_PER_WG, 1)
amd_spts_scalar_solve(
                      size_t global_work_size,
                      const FPTYPE * __restrict__  vals,
                      const int * __restrict__  cols,
                      const int * __restrict__  rowPtrs,
                      const FPTYPE * __restrict__  vec_x,
                      FPTYPE * __restrict__  out_y,
                      const FPTYPE alpha,
                      const unsigned int * __restrict__  rowMap,
                      const unsigned int * __restrict__  totalRowsInEachLevel,
                      const unsigned int total_rows_in_prev_levels,
                      const unsigned int start_level,
                      const unsigned int end_level)
{
    if (global_work_size <= hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) return;
    const unsigned int gid = hipBlockIdx_x;
    const unsigned int wg_lid = hipThreadIdx_x;
    const unsigned int lid = wg_lid % WF_SIZE;

    __shared__ unsigned int total_rows_seen_so_far;
    if (wg_lid == 0)
        total_rows_seen_so_far = 0;

    // We have a single workgroup, and it is going to walk through a
    // contiguous set of "levels" in the dependency graph.
    for (unsigned int current_level = start_level; current_level < end_level; current_level++)
    {
        // Every time we reach a new level, all of the threads within
        // this workgroup need to have completed their row's work.
        // This guarantees that we have synchronized.
        __syncthreads();
        if (wg_lid < totalRowsInEachLevel[current_level])
        {
            const unsigned int entry_in_row_map = total_rows_in_prev_levels + total_rows_seen_so_far + wg_lid;
            const unsigned int row = rowMap[entry_in_row_map];
            FPTYPE diagonal = 0.;
            FPTYPE temp_sum = alpha * vec_x[row];

            unsigned int start_of_this_row = rowPtrs[row];
            unsigned int end_of_this_row = rowPtrs[row+1];

            // This thread operates on a single row, from its beginning to end.
            for(unsigned int j = start_of_this_row; j < end_of_this_row; j++)
            {
                // local_col will tell us, for this iteration of the above for loop
                // (i.e. for this entry in this row), which columns contain the
                // non-zero values. We must then ensure that the output from the row
                // associated with the local_col is complete to ensure that we can
                // calculate the right answer.
                int local_col = cols[j];
                // Haul loading from vals[] up near the load of cols[] so that we get
                // good coalsced loads.
                FPTYPE local_val = vals[j];

                // diagonal. Skip this, we need to solve for it.
                if (local_col == row)
                    diagonal = local_val;
                else
                {
                    FPTYPE out_val;
#ifdef USE_DOUBLE
                    out_val = __ull2double_rd(atomicOr((unsigned long long *)&(out_y[local_col]), 0));
#else
                    out_val = as_float(atomicOr((uint *)&(out_y[local_col]), 0));
#endif
                    temp_sum -= local_val * out_val;
                }
            }

            FPTYPE out_val = temp_sum / diagonal;
            //FPTYPE out_val = temp_sum / diagonal; // original divide
            out_y[row] = out_val;
        }
        if (wg_lid == 0)
            total_rows_seen_so_far += totalRowsInEachLevel[current_level];
    }
}

// Solves for 'y' in the equation 'A * y = alpha * x'
// This kernel will only work if we launch a single workgroup that will
// solve multiple levels in a serial fashion. For each level, every wavefront
// within that level will try to solve for a different row.
// After solving for this level, the single workgroup hits a workgroup-wide
// barrier instruction waiting for all the other rows in this level to
// complete.
//
// Within a level, this algorithm will loop through the rows, so we should
// be able to handle levels of any size -- no synchronization is needed
// between the wavefronts working on a single level, since those rows are
// independent of one another.
//
// This is a "CSR-Vector" style execution, where each wavefront accesses
// coalesced values within its row, but where short rows waste thread
// resources.
//
// The rowMap tells us that, within a level, thread X works on row Y.
// We need this because each level of the solve can have different numbers
// of non-contiguous row.
// In addition, the 'total_rows_in_prev_levels' tells us how far in that array
// to look.
//
// [start_level, end_level) tell us which entries in the rowMap we will go
// through in this kernel invocation.
__global__ void __launch_bounds__(WF_SIZE * WF_PER_WG, 1)
amd_spts_vector_solve(
                      size_t global_work_size,
                      const FPTYPE * __restrict__  vals,
                      const int * __restrict__  cols,
                      const int * __restrict__  rowPtrs,
                      const FPTYPE * __restrict__  vec_x,
                      FPTYPE *  out_y,
                      const FPTYPE alpha,
                      const unsigned int * __restrict__  rowMap,
                      const unsigned int * __restrict__  totalRowsInEachLevel,
                      const unsigned int total_rows_in_prev_levels,
                      const unsigned int start_level,
                      const unsigned int end_level)
{
    if (global_work_size <= hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) return;
    __shared__ FPTYPE *lds_ptr;
    lds_ptr = nullptr;
#ifdef LDS_REDUCTION
    __shared__ FPTYPE lds[WF_SIZE*WF_PER_WG];
    lds_ptr = lds;
#endif
    __shared__ FPTYPE diagonal[WF_PER_WG];

    // First row within this workgroup (within this group of rows)
    const unsigned int first_row = hipBlockIdx_x * WF_PER_WG;

    const unsigned int gid = hipBlockIdx_x;
    const unsigned int wg_lid = hipThreadIdx_x;
    const unsigned int lid = wg_lid % WF_SIZE;
    const unsigned int wf_id = wg_lid / WF_SIZE;

    unsigned int cur_loc_row = wf_id;

    unsigned int total_rows_seen_so_far = 0;

    // We have a single workgroup, and it is going to walk through a
    // contiguous set of "levels" in the dependency graph.
    for (unsigned int current_level = start_level; current_level < end_level; current_level++)
    {
        // Every time we reach a new level, all of the wavefronts within
        // this workgroup need to have completed their row's work.
        // This guarantees that we have synchronized.
        __syncthreads();
        for (unsigned int cur_loc_row = wf_id; cur_loc_row < totalRowsInEachLevel[current_level]; cur_loc_row += WF_PER_WG)
        {
            const unsigned int entry_in_row_map = total_rows_in_prev_levels + total_rows_seen_so_far + cur_loc_row;
            const unsigned int row = rowMap[entry_in_row_map];
            FPTYPE temp_sum = 0.;

            if (lid == OUTPUT_THREAD)
                temp_sum = alpha * vec_x[row];

            unsigned int start_of_this_row = rowPtrs[row];
            unsigned int end_of_this_row = rowPtrs[row+1];

            // This thread operates on a single row, from its beginning to end.
            for(unsigned int j = start_of_this_row + lid; j < end_of_this_row; j += WF_SIZE)
            {
                // local_col will tell us, for this iteration of the above for loop
                // (i.e. for this entry in this row), which columns contain the
                // non-zero values. We must then ensure that the output from the row
                // associated with the local_col is complete to ensure that we can
                // calculate the right answer.
                int local_col = -1;
                // Haul loading from vals[] up near the load of cols[] so that we get
                // good coalsced loads.
                FPTYPE local_val = 0.;

                // Replace the two loads below with inline assembly that sets the
                // SLC bit. This forces the loads to essentially bypass the L2
                // to increase cache hit rate on other instructions. Vals and cols
                // are basically streamed in, so caching them doesn't help much.
                //local_col = cols[j];
                //local_val = vals[j];
#ifdef USE_DOUBLE
                __asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dwordx2 %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#else
                __asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dword %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#endif

                // diagonal. Skip this, we need to solve for it.
                if (local_col == row)
                    diagonal[wf_id] = local_val;
                else
                {
                    FPTYPE out_val;
                    out_val = out_y[local_col];
                    temp_sum -= local_val * out_val;
                }
            }

            // Take all of the temp_sum values and add them together into
            // OUTPUT_THREAD's temp_sum value.
            temp_sum = cross_lane_reduction(temp_sum, lds_ptr,
                    start_of_this_row, end_of_this_row, wg_lid);

            // y = (x-sum_of_vals_from_A) / diag
            if (lid == OUTPUT_THREAD)
            {
#ifndef LDS_REDUCTION
                // Wait for local memory to quiesce for the diagonal
                // LDS_REDUCTION has such waits in it already.
                asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
#endif
                FPTYPE out_val = temp_sum / diagonal[wf_id];
                //FPTYPE out_val = temp_sum / diagonal[wf_id]; // original divide
                out_y[row] = out_val;
            }
        }
        total_rows_seen_so_far += totalRowsInEachLevel[current_level];
    }
}

// Solves for 'y' in the equation 'A * y = alpha * x'
// This kernel is a simplified modification of the synchronization-free kernel.
// However, it is set up to work on rows that are in a contiguous series of
// levels. As such, this must be run after the initial analysis phase has
// produced a row map.
//
// Within a level, this kernel can use multiple workgroups to work on many
// rows simultaneously. In addition, multiple levels can be in flight at once,
// and this algorithm will use the synchronization-free spin-looping to produce
// the correct answer.
//
// However, we may not want to use *just* the synchronization-free spin-looping
// approach on all rows at the same time, as many rows deep in the dependency
// graph may just end up waiting, and spinning, for a long time. This spinning
// can slow down everyone else. As such, we partially break the dependency graph
// into multiple kernel invocations. This slightly reduces the theoretical
// parallelism, but it can make some invocations much faster due to less noise.
//
// The rowMap tells us that, within a level, thread X works on row Y.
// We need this because each level of the solve can have different numbers
// of non-contiguous row.
// In addition, the 'total_rows_in_prev_levels' tells us how far in that array
// to look, since previous kernel launches completed some previous rows.
__global__ void __launch_bounds__(WF_SIZE * WF_PER_WG, 1)
amd_spts_levelsync_solve(
                      size_t global_work_size,
                      const FPTYPE * __restrict__  vals,
                      const int * __restrict__  cols,
                      const int * __restrict__  rowPtrs,
                      const FPTYPE * __restrict__  vec_x,
                      FPTYPE * __restrict__  out_y,
                      const FPTYPE alpha,
                      unsigned int * __restrict__  doneArray,
                      const unsigned int * __restrict__  rowMap,
                      const unsigned int total_rows_in_prev_levels)
{
    if (global_work_size <= hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) return;
    __shared__ FPTYPE *lds_ptr;
    lds_ptr = nullptr;
#ifdef LDS_REDUCTION
    __shared__ FPTYPE lds[WF_SIZE*WF_PER_WG];
    lds_ptr = lds;
#endif
    __shared__ FPTYPE diagonal[WF_PER_WG];

    const unsigned int gid = hipBlockIdx_x;
    const unsigned int wg_lid = hipThreadIdx_x;
    const unsigned int lid = wg_lid % WF_SIZE;
    const unsigned int wf_id = wg_lid / WF_SIZE;

    const unsigned int row = rowMap[total_rows_in_prev_levels + (gid * WF_PER_WG) + wf_id];

    FPTYPE temp_sum = 0.;

    if (lid == OUTPUT_THREAD)
        temp_sum = alpha * vec_x[row];
    unsigned int start_of_this_row = rowPtrs[row];
    unsigned int end_of_this_row = rowPtrs[row+1];
    unsigned int start_point = start_of_this_row+lid;

    // This wavefront operates on a single row, from its beginning to end.
    for(unsigned int j = start_point; j < end_of_this_row; j+=WF_SIZE)
    {
        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        int local_col = -1;
        // Haul loading from vals[] up near the load of cols[] so that we get
        // good coalsced loads.
        FPTYPE local_val = 0.;
        unsigned int local_done = 0;

        // Replace the two loads below with inline assembly that sets the
        // SLC bit. This forces the loads to essentially bypass the L2
        // to increase cache hit rate on other instructions. Vals and cols
        // are basically streamed in, so caching them doesn't help much.
        // local_col = cols[j];
        // local_val = vals[j];
#ifdef USE_DOUBLE
        __asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dwordx2 %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#else
        __asm__ volatile (MEM_PREFIX"_load_dword %0 %2 " OFF_MODIFIER " slc\n" MEM_PREFIX"_load_dword %1 %3 " OFF_MODIFIER " slc\ns_waitcnt vmcnt(0)" : "=v"(local_col), "=v"(local_val) : "v"(&cols[j]), "v"(&vals[j]));
#endif

        // diagonal. Skip this, we need to solve for it.
        if (local_col == row)
        {
            local_done = 1;
            diagonal[wf_id] = local_val;
        }

        // While there are threads in this workgroup that have been unable to
        // get their input, loop and wait for the flag to exist.
        __asm__ volatile ("s_setprio 0");
        while (!local_done)
        {
            {
                // Replace this atomic with an assembly load with GLC bit set.
                // This forces the load to go to the coherence point, allowing
                // us to avoid deadlocks.
                // local_done = atomic_get_done(doneArray, local_col);
                __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc slc\ns_waitcnt vmcnt(0)" : "=v"(local_done) : "v"(&doneArray[local_col]));
            }
            if (local_done)
            {
                FPTYPE out_val;
                __asm__ volatile ("s_setprio 1");
                // The command below is manually replaced with GCN assembly with
                // the GLC bit set. This bypasses the L1, allowing us to do a
                // coherent load of the variable without needing atomics.
#ifdef USE_DOUBLE
                // out_val = as_double(atom_or((__global ulong *)&(out_y[local_col]), 0));
                __asm__ volatile (MEM_PREFIX"_load_dwordx2 %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : "=v"(out_val) : "v"(&out_y[local_col]));
#else
                // out_val = as_float(atomic_or((__global uint *)&(out_y[local_col]), 0));
                __asm__ volatile (MEM_PREFIX"_load_dword %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : "=v"(out_val) : "v"(&out_y[local_col]));
#endif
                temp_sum -= local_val * out_val;
            }
        }
    }
    __asm__ volatile ("s_setprio 1");
    // Take all of the temp_sum values and add them together into
    // OUTPUT_THREAD's temp_sum value.
    temp_sum = cross_lane_reduction(temp_sum, lds_ptr, start_of_this_row,
            end_of_this_row, wg_lid);
    // y = (x-sum_of_vals_from_A) / diag
    if (lid == OUTPUT_THREAD)
    {
#ifndef LDS_REDUCTION
        // Wait for local memory to quiesce for the diagonal
        // LDS_REDUCTION has such waits in it already.
        asm volatile ("s_waitcnt lgkmcnt(0)\n\t");
#endif
        FPTYPE out_val = temp_sum / diagonal[wf_id];
        //out_y[row] = out_val;
#ifdef USE_DOUBLE
        __asm__ volatile (MEM_PREFIX"_store_dwordx2 %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : : "v" (&out_y[row]), "v"(out_val));
#else
        __asm__ volatile (MEM_PREFIX"_store_dword %0 %1 " OFF_MODIFIER " glc\ns_waitcnt vmcnt(0)" : : "v" (&out_y[row]), "v"(out_val));
#endif
        //out_y[row] = temp_sum / diagonal[wf_id]; // original divide
        int set_one = 1;
        //doneArray[row] = 1;
        __asm__ volatile (MEM_PREFIX"_store_byte %0 %1 " OFF_MODIFIER " glc\n" WAKEUP : : "v"(&doneArray[row]), "v"(set_one));
        // If you add this back in after doing a native_divide up above,
        // we can get *some* of the accuracy of a full Newton-Raphson
        // divide while maintaining the performance of the
        // native_divide() on the critical path.
        //out_y[row] = temp_sum / diagonal[wf_id];
    }
}
