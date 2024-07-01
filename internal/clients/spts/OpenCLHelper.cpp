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
#include "OpenCLHelper.h"
#include <cstring>
#include <string>
#include <iostream>

cl_context CLHelper::context = NULL;
cl_command_queue CLHelper::commandQueue = NULL;
cl_kernel CLHelper::SpTSKernel = NULL;
cl_kernel CLHelper::SpTSKernel_analyze = NULL;
cl_kernel CLHelper::SpTSKernel_levelset = NULL;
cl_kernel CLHelper::SpTSKernel_scalar = NULL;
cl_kernel CLHelper::SpTSKernel_vector = NULL;
cl_kernel CLHelper::SpTSKernel_levelsync = NULL;

const char * get_cl_err_string(cl_int err)
{
    switch (err)
    {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
        case CL_COMPILE_PROGRAM_FAILURE:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
#ifdef CL_VERSION_1_1
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
#ifdef cl_ext_device_fission
        case CL_DEVICE_PARTITION_FAILED_EXT:
            return "CL_DEVICE_PARTITION_FAILED_EXT";
        case CL_INVALID_PARTITION_COUNT_EXT:
            return "CL_INVALID_PARTITION_COUNT_EXT";
        case CL_INVALID_PARTITION_NAME_EXT:
            return "CL_INVALID_PARTITION_NAME_EXT";
#endif
#endif
#ifdef CL_VERSION_1_2
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
        case CL_INVALID_PIPE_SIZE:
            return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE:
            return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
        case CL_INVALID_SPEC_ID:
            return "CL_INVALID_SPEC_ID";
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
            return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
#ifdef cl_khr_icd
        case CL_PLATFORM_NOT_FOUND_KHR:
            return "CL_PLATFORM_NOT_FOUND_KHR";
#endif
        default:
            return "UNKNOWN CL ERROR CODE";
    }
}

void convertToStr(char **source, size_t* sourceSize, const std::string fname)
{
    FILE *fp = fopen(fname.c_str(), "r");
    fseek(fp, 0, SEEK_END);
    *sourceSize = ftell(fp);
    fseek(fp , 0, SEEK_SET);
    *source = (char *)malloc(*sourceSize * sizeof(char));
    fread(*source, 1, *sourceSize, fp);
    fclose(fp);

}

int CLHelper::Init(const std::string &filename, InputFlags &in_flags)
{
    cl_int status = 0;
    size_t deviceListSize;
    unsigned int i;

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_uint numPlatforms;
    platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"clGetPlatformIDs failed. %u",numPlatforms);
        return 1;
    }
    if (0 < numPlatforms) 
    {
        cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if(status != CL_SUCCESS)
        {
            fprintf(stderr, "clGetPlatformIDs failed: %s\n", get_cl_err_string(status) );
            return 1;
        }
        for (i = 0; i < numPlatforms; ++i) 
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);

            if(status != CL_SUCCESS)
            {
                fprintf(stderr,"clGetPlatformInfo failed: %s\n", get_cl_err_string(status));
                return 1;
            }

            platform = platforms[i];
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
            {
                break;
            }
        }
        free(platforms);
    }

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context_properties* cprops = (NULL == platform) ? NULL : cps;
    context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    if(status != CL_SUCCESS)
    {
        printf("status: %d",  status);
        fprintf(stderr,"Error: Creating Context. (clCreateContextFromType): %s\n", get_cl_err_string(status));
        return 1;
    }
    /* First, get the size of device list data */
    status = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(size_t), &deviceListSize, NULL);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Getting Context Info (device list size, clGetContextInfo): %s\n", get_cl_err_string(status));
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // Detect OpenCL devices
    /////////////////////////////////////////////////////////////////
    devices = (cl_device_id *)malloc(deviceListSize * sizeof(cl_device_id));
    if(devices == 0)
    {
        fprintf(stderr,"Error: No devices found: %s\n", get_cl_err_string(status));
        return 1;
    }

    /* Now, get the device list data */
    status = clGetContextInfo( context, CL_CONTEXT_DEVICES, deviceListSize*sizeof(cl_device_id), devices, NULL);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Getting Context Info (device list, clGetContextInfo): %s\n", get_cl_err_string(status));
        return 1;
    }

    char *deviceName;
    size_t dev_name_size = 0;

    int deviceNum = in_flags.GetValueInt("device");

    clGetDeviceInfo(devices[deviceNum], CL_DEVICE_NAME, sizeof(char*), NULL, &dev_name_size);
    deviceName = (char *)malloc(sizeof(char)*dev_name_size);

    clGetDeviceInfo(devices[deviceNum], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Device Name: %s\n", deviceName);

    bool use_gcn3 = false;
    bool use_gcn2 = false;
    char *found_gfx8 = strstr(deviceName, "gfx8");
    char *found_gfx7 = strstr(deviceName, "gfx7");
    if (found_gfx8 != NULL)
        use_gcn3 = true;
    if (found_gfx7 != NULL)
        use_gcn2 = true;

    free(deviceName);

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    commandQueue = clCreateCommandQueue(context, devices[deviceNum], CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Creating Command Queue. (clCreateCommandQueue): %s\n", get_cl_err_string(status));
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // Load CL file, build CL program object, create CL kernel object
    /////////////////////////////////////////////////////////////////
    char* source;
    size_t sourceSize;
    convertToStr(&source, &sourceSize, filename);
    
    syncfree_program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Loading Binary into cl_program (clCreateProgramWithBinary): %s\n", get_cl_err_string(status));
        return 1;
    }
    analyze_levelset_program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Loading Binary into cl_program (clCreateProgramWithBinary): %s\n", get_cl_err_string(status));
        return 1;
    }

    std::string buildFlags = "-x clc++ -Dcl_khr_int64_base_atomics=1 -cl-std=CL2.0";
    if (use_gcn3)
        buildFlags += " -DGCN3 ";
    if (use_gcn2)
        buildFlags += " -DGCN2 ";
    buildFlags += " -DROW_BITS=" + std::to_string(ROW_BITS);
    buildFlags += " -DWG_BITS=" + std::to_string(WG_BITS);
    buildFlags += " -DWF_SIZE=" + std::to_string(WF_SIZE);
    buildFlags += " -DWF_PER_WG=" + std::to_string(WF_PER_WG);
#ifdef USE_DOUBLE
    buildFlags += " -DDOUBLE";
#endif
    
    /* create a cl program executable for all the devices specified */
    status = clBuildProgram(analyze_levelset_program, 1, &devices[deviceNum], buildFlags.c_str(), NULL, NULL);
    if(status != CL_SUCCESS)
    {
        printf("Error: Building Analyze and Levelset Program (clBuildProgram): %d\n", status);
        char * errorbuf = (char*)calloc(sizeof(char),1024*1024);
        size_t size;
        clGetProgramBuildInfo(analyze_levelset_program, devices[deviceNum], CL_PROGRAM_BUILD_LOG, 1024*1024, errorbuf, &size);
        printf("%s ", errorbuf);
        return 1;
    }

    buildFlags += " -DSYNCFREE_KERNEL";
    status = clBuildProgram(syncfree_program, 1, &devices[deviceNum], buildFlags.c_str(), NULL, NULL);
    if(status != CL_SUCCESS)
    {
        printf("Error: Building Syncfree Program (clBuildProgram): %d\n", status);
        char * errorbuf = (char*)calloc(sizeof(char),1024*1024);
        size_t size;
        clGetProgramBuildInfo(syncfree_program, devices[deviceNum], CL_PROGRAM_BUILD_LOG, 1024*1024, errorbuf, &size);
        printf("%s ", errorbuf);
        return 1;
    }

    SpTSKernel = clCreateKernel(syncfree_program, "amd_spts_syncfree_solve", &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Creating Kernel from program. (SpTS): %s\n", get_cl_err_string(status));
        return 1;
    }

    SpTSKernel_analyze = clCreateKernel(analyze_levelset_program, "amd_spts_analyze_and_solve", &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Creating Kernel from program. (SpTS_analyze): %s\n", get_cl_err_string(status));
        return 1;
    }

    SpTSKernel_levelset = clCreateKernel(analyze_levelset_program, "amd_spts_levelset_solve", &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Creating Kernel from program. (SpTS_levelset): %s\n", get_cl_err_string(status));
        return 1;
    }

    SpTSKernel_scalar = clCreateKernel(analyze_levelset_program, "amd_spts_scalar_solve", &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Creating Kernel from program. (SpTS_scalar): %s\n", get_cl_err_string(status));
        return 1;
    }

    SpTSKernel_vector = clCreateKernel(analyze_levelset_program, "amd_spts_vector_solve", &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Creating Kernel from program. (SpTS_vector): %s\n", get_cl_err_string(status));
        return 1;
    }

    SpTSKernel_levelsync = clCreateKernel(analyze_levelset_program, "amd_spts_levelsync_solve", &status);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Creating Kernel from program. (SpTS_levelsync): %s\n", get_cl_err_string(status));
        return 1;
    }

    // All good
    return 0;
}

void CLHelper::checkStatus(cl_int status, const std::string errString)
{
    if (status != CL_SUCCESS)
    {
        std::cerr << errString << " : " << get_cl_err_string(status) << std::endl;
        exit(-1);
    }
}

memPointer CLHelper::AllocateMem(const std::string name,
                            size_t size, 
                            memPointer_flags flags, 
                            void *hostBuffer) 
{
    cl_mem buf;
    cl_int status;

    buf = clCreateBuffer(context, flags, size, hostBuffer, &status);
    std::string errString = "OpenCL error allocating " + name + " !";
    checkStatus(status, errString);

    return buf;
}

void CLHelper::CopyToDevice(memPointer devBuffer, 
                                void *hostBuffer,
                                size_t size,
                                size_t offset,
                                cl_bool blocking,
                                cl_event *ev)
{
    cl_int status;
    status = clEnqueueWriteBuffer(commandQueue, devBuffer, blocking, offset, size, hostBuffer, 0, NULL, ev);

    checkStatus(status, "OpenCL error copying data to device !");
}

void CLHelper::CopyToHost(memPointer devBuffer, 
                                void *hostBuffer,
                                size_t size,
                                size_t offset,
                                cl_bool blocking,
                                cl_event *ev)
{
    cl_int status;
    status = clEnqueueReadBuffer(commandQueue, devBuffer, blocking, offset, size, hostBuffer, 0, NULL, ev);

    checkStatus(status, "OpenCL error copying data to device !");
}

int64_t CLHelper::ComputeTime(cl_event event)
{
    int64_t start_time, end_time;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(int64_t), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(int64_t), &end_time, NULL);

    return end_time - start_time;
}
