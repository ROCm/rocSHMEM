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
#ifndef CLHelper_H
#define CLHelper_H

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <string>
#include <iostream>
#include <sstream>
#include "InputFlags.h"
#include "GPUHelper.h"
#include "hip/hip_runtime.h"

class HIPHelper : public GPUHelper
{
	public:
	HIPHelper() {}
	int Init(const std::string &_filename, InputFlags &in_flags);
	void checkStatus(gpuError status, const std::string errString);
	void CopyToDevice(memPointer _d_buf, void *_h_buf, size_t _size, size_t _offset, gpuBool _blocking, gpuEvent *_ev);
	void CopyToHost(memPointer _d_buf, void *_h_buf, size_t _size, size_t _offset, gpuBool _blocking, gpuEvent *_ev);
	memPointer AllocateMem(const std::string name, size_t, memPointer_flags flags, void *);
	void FreeMem(memPointer ptr) { hipFree(ptr); }
    void Flush() { hipDeviceSynchronize(); }
};

#endif //CLHelper_H

