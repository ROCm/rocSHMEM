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

#include "config.h"

#ifdef USE_HIP
#include "hip/hip_runtime.h"
#else
#include "OpenCLHelper.h"
#endif

#ifdef USE_RO_SHMEM
#include "mpi.h"
#endif

#include "MatrixMarketReader.h"
#include "SpTS.h"
#include <iostream>
#include <unistd.h>
#include <limits.h>

#ifdef USE_DOUBLE
typedef double FPTYPE;
#else
typedef float FPTYPE;
#endif

using namespace rocshmem;

int main(int argc, char *argv[])
{
    SparseTriangularSolve<FPTYPE> spts_obj;
    InputFlags &in_flags = spts_obj;
    in_flags.AddDerivedInputFlags();
    in_flags.Parse(argc, argv);
    FPTYPE alpha = in_flags.GetValueFloat("alpha");

    printf("Reading input file: %s...", in_flags.GetValueStr("filename").c_str());fflush(stdout);
    MatrixMarketReader<FPTYPE> mm_reader;
    if (mm_reader.MMReadFormat(in_flags.GetValueStr("filename"), in_flags) != 0)
    {
        fprintf(stderr, "ERROR reading input file !\n");
        exit(1);
    }
    printf("Done.\n");

    GPUHelper *GPU;
#ifdef USE_HIP
    printf("Initializing HIP runtime...\n\t");fflush(stdout);
    GPU = new HIPHelper();
    char buf[PATH_MAX + 1];
    readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string str(buf);
    printf("Going to try to open %s\n", (str.substr(0, str.rfind('/'))+"/spts_kernel.cl").c_str());
    if(GPU->Init((str.substr(0, str.rfind('/'))+ "/spts_kernel.cl").c_str(), in_flags) == 1)
    {
        fflush(stdout);
        fprintf(stderr,"\nError Initializing HIP Runtime !\n");
        exit(-1);
    }
#else
    printf("Initializing OpenCL runtime...\n\t");fflush(stdout);
    GPU = new CLHelper();
    char buf[PATH_MAX + 1];
    readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string str(buf);
    printf("Going to try to open %s\n", (str.substr(0, str.rfind('/'))+"/spts_kernel.cl").c_str());
    if(GPU->Init((str.substr(0, str.rfind('/'))+ "/spts_kernel.cl").c_str(), in_flags) == 1)
    {
        fflush(stdout);
        fprintf(stderr,"\nError Initializing OpenCL Runtime !\n");
        exit(-1);
    }
#endif
    printf("Done.\n");

    printf("Allocating sparse matrices...");fflush(stdout);
    spts_obj.AllocateSparseMatrix(mm_reader, in_flags, GPU);
    printf("Done.\n");

    printf("Converting COO to CSR...");fflush(stdout);
    spts_obj.ConvertFromCOOToCSR(mm_reader.GetCoordinates(), in_flags);
    printf("Done.\n");

    SPTS_BLOCK_SIZE = in_flags.GetValueInt("block_size");
    printf("Finding Stats For Parallel Decomposition...");fflush(stdout);
    spts_obj.FindStatsForParallelDecomposition();
    printf("Done.\n");

    printf("Allocating parallel sparse matrices...");fflush(stdout);
    spts_obj.AllocateParallelSparseMatrix(mm_reader, in_flags);
    printf("Done.\n");

    printf("Allocating vectors...");fflush(stdout);
    spts_obj.AllocateVectors(mm_reader);
    printf("Done.\n");

    float gflops = 0.f;
    int errors = 0;
    uint64_t ns_per_iter = 0;
    uint64_t ns_per_analysis_iter = 0;
    uint64_t ns_per_syncfree_iter = 0;
    uint64_t ns_per_levelset_iter = 0;
    uint64_t ns_per_levelsync_iter = 0;

    printf("Performing SpTS on the CPU with alpha=%f...", (float)alpha);fflush(stdout);
    spts_obj.CSRSpTSCPU(alpha);
    printf("Done.\n");

    printf("Checking results of CPU-side SpTS...");fflush(stdout);
    if (!spts_obj.CSRCheckCPU(alpha))
    {
        fflush(stdout);
        fprintf(stderr, "CPU-based results were 'wrong', likely due to FP rounding. Expect the CPU and GPU to differ wildly.\n");
        //exit(-1);
    }
    printf("Done.\n");

    printf("Performing %d iterations of SpTS on the GPU with alpha=%f...", in_flags.GetValueInt("iterations"), (float)alpha);fflush(stdout);
    gflops = spts_obj.CSRSpTSGPU(ns_per_iter, ns_per_analysis_iter, ns_per_syncfree_iter, ns_per_levelset_iter, ns_per_levelsync_iter, alpha);
    printf("Done.\n");

    if (in_flags.GetValueBool("verify")) {
        printf("Checking whether GPU SpTS caused non-deterministic errors...\n");fflush(stdout);
        int non_det_errors = spts_obj.NonDeterministicErrors();
        printf("Done.\n");
        if (non_det_errors)
            fprintf(stderr, "ERROR!! -- Saw %d GPU iterations that had non-deterministic differences.\n", non_det_errors);
        int max_errors = spts_obj.MaxErrors();
        if (max_errors)
        {
            if (max_errors > 1)
                printf(" -- %d rows differed between CPU and GPU results.\n", max_errors);
            else
                printf(" -- %d row differed between CPU and GPU results.\n", max_errors);
        }
        else
            printf("\n");
    }

    printf("File %s : SpTS Gflops: %f ms_per_iter: %lf ", in_flags.GetValueStr("filename").c_str(), gflops, ((double)ns_per_iter/1000000.));
    printf(" ( ms_per_analysis_iter: ");
    if (ns_per_analysis_iter == 0)
        printf("no_iter");
    else
        printf("%lf", ((double)ns_per_analysis_iter/1000000.));
    printf(" | ms_per_syncfree_iter: ");
    if (ns_per_syncfree_iter == 0)
        printf("no_iter");
    else
        printf("%lf", ((double)ns_per_syncfree_iter/1000000.));
    printf(" | ms_per_levelset_iter: ");
    if (ns_per_levelset_iter == 0)
        printf("no_iter");
    else
        printf("%lf", ((double)ns_per_levelset_iter/1000000.));
    printf(" | ms_per_levelsync_iter: ");
    if (ns_per_levelsync_iter == 0)
        printf("no_iter )");
    else
        printf("%lf )", ((double)ns_per_levelsync_iter/1000000.));

#ifdef USE_ROCSHMEM
    MPI_Allreduce(MPI_IN_PLACE, (void *) &ns_per_analysis_iter, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (spts_obj.Get_this_pe() == 0) {
       printf("\nRANK 0: analysis avg ms = %lf\n",
              ((double) ns_per_analysis_iter / 1000000.) / spts_obj.Get_total_pes());
    }
#endif

    return 0;
}
