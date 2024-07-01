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

#include "HIPHelper.h"
#include <cstring>
#include <string>
#include <iostream>

int HIPHelper::Init(const std::string &filename, InputFlags &in_flags)
{
    int device = 0;
    hipSetDevice(device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device /*deviceID*/);
    printf("info: running on device %s\n", props.name);
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);

    return 0;
}

void HIPHelper::checkStatus(gpuError status, const std::string errString)
{
    if (status != HIP_SUCCESS)
    {
        std::cerr << errString << " : " << hipGetErrorString(status) << std::endl;
        exit(-1);
    }
}

memPointer HIPHelper::AllocateMem(const std::string name,
                            size_t size,
                            memPointer_flags flags,
                            void *hostBuffer)
{
    void* buf;
    std::string errString = "HIP error allocating " + name + " !";
    checkStatus(hipMalloc(&buf, size), errString);
    printf("Allocating %s of size %zu at buf %p\n", name.c_str(), size, buf);
    return buf;
}

void HIPHelper::CopyToDevice(memPointer devBuffer,
                                void *hostBuffer,
                                size_t size,
                                size_t offset,
                                gpuBool blocking,
                                gpuEvent *ev)
{
    assert(offset == 0);
   memcpy(devBuffer, hostBuffer, size);
/*
    if (blocking == GPU_TRUE) {
        checkStatus(hipMemcpy(devBuffer, hostBuffer, size, hipMemcpyHostToDevice),
                    "HIP error copying data to device !");
    } else {
        checkStatus(hipMemcpyAsync(devBuffer, hostBuffer, size, hipMemcpyHostToDevice),
                    "HIP error copying data to device !");
    }
*/
}

void HIPHelper::CopyToHost(memPointer devBuffer,
                                void *hostBuffer,
                                size_t size,
                                size_t offset,
                                gpuBool blocking,
                                gpuEvent *ev)
{
    assert(offset == 0);
memcpy(hostBuffer, devBuffer, size);
/*
    if (blocking == GPU_TRUE) {
        checkStatus(hipMemcpy(hostBuffer, devBuffer, size, hipMemcpyDeviceToHost),
                    "HIP error copying data to device !");
    } else {
        checkStatus(hipMemcpyAsync(hostBuffer, devBuffer, size, hipMemcpyDeviceToHost),
                    "HIP error copying data to device !");
    }
*/
}
