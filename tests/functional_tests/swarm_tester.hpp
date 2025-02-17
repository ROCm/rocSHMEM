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

#ifndef _SWARM_TESTER_HPP_
#define _SWARM_TESTER_HPP_

#include "primitive_tester.hpp"

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void GetSwarmTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, char *s_buf,
                             char *r_buf, int size);

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class GetSwarmTester : public PrimitiveTester {
 public:
  explicit GetSwarmTester(TesterArguments args);
  virtual ~GetSwarmTester();

 protected:
  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) override;

  virtual void verifyResults(uint64_t size) override;
};

#endif
