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
#ifndef GPUHelper_H
#define GPUHelper_H

#include "config.h"

#include <string>
#include <iostream>
#include <sstream>
#include "InputFlags.h"

#define ROW_BITS 32 // May be not the right place to define this macro
#define WG_BITS 24

static int SPTS_BLOCK_SIZE = 0;

#ifdef USE_ROCSHMEM
#define WF_PER_WG 1
#else
#define WF_PER_WG 16
#endif
#define WF_SIZE 64

#ifdef USE_HIP
    #include <hip/hip_runtime.h>
    typedef void * memPointer;
    typedef int memPointer_flags;
    typedef int gpuInt;
    typedef bool gpuBool;
    typedef hipEvent_t gpuEvent;
    typedef hipError_t gpuError;
    #define GPU_MEM_READ_ONLY 0
    #define GPU_MEM_READ_WRITE 0
    #define GPU_MEM_USE_HOST_PTR 0
    #define GPU_TRUE true
    #define GPU_FALSE false
#else
#include <CL/cl.h>
    typedef cl_mem memPointer;
    typedef cl_mem_flags memPointer_flags;
    typedef cl_int gpuInt;
    typedef cl_bool gpuBool;
    typedef cl_event gpuEvent;
    typedef cl_int gpuError;
    #define GPU_MEM_READ_ONLY CL_MEM_READ_ONLY
    #define GPU_MEM_READ_WRITE CL_MEM_READ_ONLY
    #define GPU_MEM_USE_HOST_PTR CL_MEM_USE_HOST_PTR
    #define GPU_TRUE CL_TRUE
    #define GPU_FALSE CL_FALSE
#endif

class GPUHelper
{
	public:
	GPUHelper() {}
	virtual int Init(const std::string &_filename, InputFlags &in_flags) = 0;
	virtual void checkStatus(gpuError status, const std::string errString) = 0;
	virtual void CopyToDevice(memPointer _d_buf, void *_h_buf, size_t _size, size_t _offset, gpuBool _blocking, gpuEvent *_ev) = 0;
	virtual void CopyToHost(memPointer _d_buf, void *_h_buf, size_t _size, size_t _offset, gpuBool _blocking, gpuEvent *_ev) = 0;
	virtual memPointer AllocateMem(const std::string name, size_t, memPointer_flags flags, void *) = 0;
	virtual void FreeMem(memPointer ptr) = 0;
    virtual void Flush() = 0;
};

#endif //GPUHelper_H
