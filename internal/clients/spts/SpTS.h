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
#ifndef SpTS_H
#define SpTS_H
#define TEST_NUM 9999999999ULL

#include "InputFlags.h"
#include "SparseMatrix.h"

#include "MatrixMarketReader.h"
#include <vector>
#include <float.h>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <thread>

#include <time.h>
#include <sys/time.h>

#include <unistd.h>


#ifdef USE_ROCSHMEM
#include "rocshmem.hpp"
#include "mpi.h"
#endif

#ifdef USE_HIP
#include "spts_kernel.h"
#endif

#ifdef DBL_DECIMAL_DIG
    #define OP_DBL_Digs (DBL_DECIMAL_DIG)
#else
    #ifdef DECIMAL_DIG
        #define OP_DBL_Digs (DECIMAL_DIG)
    #else
        #define OP_DBL_Digs (DBL_DIG + 3)
    #endif
#endif

#ifdef FLT_DECIMAL_DIG
  #define OP_FLT_Digs (FLT_DECIMAL_DIG)
#else
  #ifdef DECIMAL_DIG
    #define OP_FLT_Digs (DECIMAL_DIG)
  #else
    #define OP_FLT_Digs (FLT_DIG + 3)
  #endif
#endif

using namespace rocshmem;

template<typename FloatType>
class SparseTriangularSolve :
    public InputFlags, public SparseMatrix<FloatType>
{
    FloatType *x;
    FloatType *y;
    FloatType *y_zero;
    FloatType *yref;
    std::vector<uint64_t> rowBlocks;

    memPointer xDev;
    memPointer yDev;
    memPointer completedRowsDev;
    memPointer rowBlocksDev;
    memPointer doneArrayDev;
    memPointer shadowDoneArrayDev;
    memPointer remoteInProgressArrayDev;
    memPointer reqUpdateArrayDev;
    memPointer numRowsAtLevelDev;
    memPointer maxDepthDev;
    memPointer rowMapDev;
    memPointer totalSpinDev;
    memPointer oneBufDev;

    int nNZ;
    int nRows;
    int nCols;
    int numBlocks;
/*
    #ifdef USE_ROCSHMEM
    rocshmem_t* handle;
    #endif
*/
    std::unordered_map<int, FloatType> *observed_errors;
    int *errors_seen;

    public:

    SparseTriangularSolve() : nNZ(0), nRows(0), nCols(0), numBlocks(0)
    {
        x = NULL; y = NULL; y_zero = NULL, yref = NULL, observed_errors = NULL, errors_seen = NULL;
        xDev = yDev = completedRowsDev = remoteInProgressArrayDev = rowBlocksDev =  doneArrayDev = shadowDoneArrayDev = numRowsAtLevelDev = maxDepthDev = rowMapDev = totalSpinDev = oneBufDev = 0;

        #ifdef USE_ROCSHMEM
	int rocshmem_queues = (2560 / WF_PER_WG);
       	if (2560 % WF_PER_WG)
            rocshmem_queues++;
	printf("rocshmem_queues %d WF_PER_WG %d  \n",rocshmem_queues, WF_PER_WG);
        rocshmem_init(rocshmem_queues);

        this->Set_total_pes(rocshmem_n_pes());
        this->Set_this_pe(rocshmem_my_pe());
        #else
        this->Set_total_pes(1);
        this->Set_this_pe(0);
        #endif
    }

    void AddDerivedInputFlags();
    void AllocateVectors(MatrixMarketReader<FloatType> &mm_reader);
    void CSRSpTSCPU(FloatType alpha);
    bool CSRCheckCPU(FloatType alpha);

    float CSRSpTSGPU(uint64_t &ns_per_iter, uint64_t &ns_per_analysis_iter, uint64_t &ns_per_syncfree_iter, uint64_t &ns_per_levelset_iter, uint64_t &ns_per_levelsync_iter, FloatType alpha);

    int VerifyResults(int);
    int NonDeterministicErrors();
    int MaxErrors();
    int ComputeRowBlocks(std::vector<uint64_t> &, int *, int);

    ~SparseTriangularSolve()
    {
        if (x != NULL)
            delete[] x;
        if (y != NULL)
            delete[] y;
        if (y_zero != NULL)
            delete[] y_zero;
        if (yref != NULL)
            delete[] yref;
        if (errors_seen != NULL)
            delete[] errors_seen;

        if (xDev != 0)
            this->GPU->FreeMem(xDev);
        if (rowBlocksDev != 0)
            this->GPU->FreeMem(rowBlocksDev);
        if (completedRowsDev != 0)
            this->GPU->FreeMem(completedRowsDev);
        if (numRowsAtLevelDev != 0)
            this->GPU->FreeMem(numRowsAtLevelDev);
        if (maxDepthDev != 0)
            this->GPU->FreeMem(maxDepthDev);
        if (rowMapDev != 0)
            this->GPU->FreeMem(rowMapDev);
        if (totalSpinDev != 0)
            this->GPU->FreeMem(totalSpinDev);
        if (oneBufDev != 0)
            this->GPU->FreeMem(oneBufDev);
        if (remoteInProgressArrayDev != 0)
            this->GPU->FreeMem(remoteInProgressArrayDev);

        #ifndef USE_ROCSHMEM
        if (yDev != 0)
            this->GPU->FreeMem(yDev);
        if (doneArrayDev != 0)
            this->GPU->FreeMem(doneArrayDev);
        if (reqUpdateArrayDev != 0)
            this->GPU->FreeMem(reqUpdateArrayDev);
        if (shadowDoneArrayDev != 0)
            this->GPU->FreeMem(shadowDoneArrayDev);
        #else
        if (yDev != 0)
            rocshmem_free(yDev);
        if (doneArrayDev != 0)
            rocshmem_free(doneArrayDev);
        if (reqUpdateArrayDev != 0)
            rocshmem_free(reqUpdateArrayDev);
        if (shadowDoneArrayDev != 0)
            rocshmem_free(shadowDoneArrayDev);
        rocshmem_finalize();
        #endif
    }
};

    template<typename FloatType>
void SparseTriangularSolve<FloatType>::AddDerivedInputFlags()
{
    AddInputFlag("filename", 'f', "", "Matrix-Market File", "string");
    AddInputFlag("iterations", 'i', "10", "Number of SpTS Iterations (Default=10)", "int");
    AddInputFlag("exp_zeroes", 'z', "false", "Include Explicit Zeroes in Matrix-Market File (Default=false)", "bool");
    AddInputFlag("device", 'd', "0", "Choose the GPU to Execute SpTS (Default=0)", "int");
    AddInputFlag("alpha", 'A', "1.0", "A*y=alpha*x. Known vector 'x' is multiplied by scalar alpha befoer solving for vector 'y'. (Default=1.0)", "float");
    AddInputFlag("non_symmetric", 'n', "false", "Force the program to work on non-symmetric matrices. This will ignore the upper triangular entirely. (Default=false)", "bool");
    AddInputFlag("levelsync_size", 'l', "0", "Number of rows to launch in a level-sync kernel invocation (Default = auto-tune)", "int");
    AddInputFlag("verify", 'v', "false", "Verify results", "bool");
    AddInputFlag("rocshmem_algorithm", 'a', "0", "rocSHMEM algorithm type", "int");
    AddInputFlag("block_size", 'b', "32768", "Use get-based algorithm for rocSHMEM", "int");
	AddInputFlag("put_block_size", 'p', "1024", "Block size for puts", "int");
	AddInputFlag("get_backoff_factor", 'g', "128", "Backoff factor for gets", "int");
}

    template<typename FloatType>
void SparseTriangularSolve<FloatType>::AllocateVectors(
        MatrixMarketReader<FloatType> &mm_reader)
{
    nRows = mm_reader.GetNumRows();
    nCols = mm_reader.GetNumCols();
    nNZ = mm_reader.GetNumNonZeroes();

    x = new FloatType[nCols];
    y = new FloatType[nRows];
    y_zero = new FloatType[nRows];
    yref = new FloatType[nRows];
    observed_errors = new std::unordered_map<int, FloatType>[InputFlags::GetValueInt("iterations")];

    for(int i = 0; i < nRows; i++)
    {
        y[i] = (FloatType)0.0;
        y_zero[i] = (FloatType)0.0;
        yref[i] = (FloatType)0.0;
    }

    for(int i = 0; i < nCols; i++)
    {
        //x[i] = (FloatType)rand() / (FloatType)RAND_MAX;
        x[i] = 2.;
    }

    xDev = this->GPU->AllocateMem("xDev", nCols*sizeof(FloatType), GPU_MEM_READ_ONLY, NULL);
    #ifndef USE_ROCSHMEM
    yDev = this->GPU->AllocateMem("yDev", nRows*sizeof(FloatType), GPU_MEM_READ_WRITE, NULL);
    #else
    yDev = (memPointer) rocshmem_malloc(nRows*sizeof(FloatType));
    #endif
}

    template<typename FloatType>
void SparseTriangularSolve<FloatType>::CSRSpTSCPU(FloatType alpha)
{
    FloatType *NZvalues = SparseMatrix<FloatType>::GetVals();
    int *Cols = SparseMatrix<FloatType>::GetCols();
    int *rowptrs = SparseMatrix<FloatType>::GetRowPtrs();
    double internal_alpha = alpha;

    uint64_t local_nnz = 0;
    uint64_t remote_nnz = 0;
    uint64_t rows_with_nonlocal = 0;

    for(int i = 0; i < nRows; i++)
    {
        bool row_has_nonlocal = false;
        double diagonal = 0.;
        double temp = 0.;
        int diag_j = -1;
        for(int j = rowptrs[i]; j < rowptrs[i+1]; j++)
        {
            int ci = Cols[j];
            int row_pe = (i / SPTS_BLOCK_SIZE) % this->Get_total_pes();
            int col_pe = (ci / SPTS_BLOCK_SIZE) % this->Get_total_pes();

            int assigned_pe = (i / SPTS_BLOCK_SIZE) % this->Get_total_pes();
            if (assigned_pe == this->Get_this_pe()) {
                if (row_pe == col_pe) {
                    local_nnz++;
                } else {
                    row_has_nonlocal = true;
                    remote_nnz++;
                }
            }

            // Skip adding in the diagonal. We need to solve for that.
            if (ci != i)
            {
                if (i == TEST_NUM)
                    fprintf(stderr, "NZvalues[%d](%lf) * yref[%d](%lf)\n", j, NZvalues[j], ci, yref[ci]);
                temp += NZvalues[j] * yref[ci];
            }
            else
            {
                if (i==TEST_NUM)
                    fprintf(stderr, "\t\tDIAG = %lf\n", NZvalues[j]);
                diagonal = NZvalues[j];
                diag_j = j;
            }
        }
        if (row_has_nonlocal) rows_with_nonlocal++;
        if (diag_j == -1)
        {
            fflush(stdout);
            printf("\nERROR in SpTS CPU\n");
            printf("No diagonal found in row %d\n", i);
        }
        // y = (x-sum_of_vals_from_A) / diag
        double alpha_x = internal_alpha * (double)x[i];
        if (i == TEST_NUM)
        {
            char buf[128];
            char buf2[128];
            char buf3[128];
            snprintf(buf, sizeof(buf), "%.20f", alpha_x);
            snprintf(buf2, sizeof(buf2), "%.20f", internal_alpha);
            fprintf(stderr, "alpha_x: %s (%s * %lf)\n", buf, buf2, x[i]);
            snprintf(buf3, sizeof(buf3), "%.20f", temp);
            fprintf(stderr, "temp: %s\n", buf3);
        }
        yref[i] = (FloatType)((alpha_x - temp)/diagonal);
        if (i == TEST_NUM)
            fprintf(stderr, "\tsupposed answer [%d]: %lf\n", i, yref[i]);
    }
    double ratio = ((double) local_nnz) / ((double) remote_nnz + local_nnz);
    double rows_remote_ratio = ((double) rows_with_nonlocal) / ((double) this->nRows_p);
    if (this->Get_this_pe() == 0) {
        printf("\nRANK 0: global NNZ = %lu\n", remote_nnz + local_nnz);
        printf("RANK 0: global Rows = %d\n", nRows);
    }
    printf("\nLOCALITY  %d : Remote/Local cols %lu/%lu Fraction Columns Local %f Fraction Rows with Remote Columns %f\n", this->Get_this_pe(), remote_nnz, local_nnz, ratio, rows_remote_ratio);
}

    template<typename FloatType>
bool SparseTriangularSolve<FloatType>::CSRCheckCPU(FloatType alpha)
{
    FloatType *NZvalues = SparseMatrix<FloatType>::GetVals();
    int *Cols = SparseMatrix<FloatType>::GetCols();
    int *rowptrs = SparseMatrix<FloatType>::GetRowPtrs();
    double internal_alpha = alpha;
    bool all_worked = true;

#pragma omp parallel for
    for(int i = 0; i < nRows; i++)
    {
#pragma omp flush (all_worked)
        if (all_worked)
        {
            double temp = 0.;
            for(int j = rowptrs[i]; j < rowptrs[i+1]; j++)
            {
                int ci = Cols[j];
                // Skip anything that lies on the diagonal. We need to solve for that.
                temp += NZvalues[j] * yref[ci];
            }
            double compare_val = 0.;
            double alpha_x = internal_alpha * x[i];
            if(typeid(FloatType) == typeid(float))
            {
                compare_val = fabs(alpha_x*1e-3);
                if (compare_val < 10*FLT_EPSILON)
                    compare_val = 10*FLT_EPSILON;
                if ((FloatType)(alpha_x - compare_val) > (FloatType)temp || (FloatType)(alpha_x + compare_val) < (FloatType)temp)
                {
                    fflush(stdout);
                    fprintf(stderr, " CPU CALCULATION ERROR on row %d\n", i);
                    fprintf(stderr, "\tReal value for row %d: %.*e\n", i, OP_FLT_Digs-1, (float)alpha_x);
                    fprintf(stderr, "\tCalculated value for row %d: %.*e\n", i, OP_FLT_Digs-1, (float)temp);
                    all_worked = false;
#pragma omp flush (all_worked)
                }
            }
            else if(typeid(FloatType) == typeid(double))
            {
                compare_val = fabs(alpha_x*1e-4);
                if (compare_val < 10*DBL_EPSILON)
                    compare_val = 10*DBL_EPSILON;
                if ((FloatType)(alpha_x - compare_val) > (FloatType)temp || (FloatType)(alpha_x + compare_val) < (FloatType)temp)
                {
                    fflush(stdout);
                    fprintf(stderr, " CPU CALCULATION ERROR on row %d\n", i);
                    fprintf(stderr, "\tReal value for row %d: %.*le\n", i, OP_DBL_Digs-1, (double)alpha_x);
                    fprintf(stderr, "\tCalculated value for row %d: %.*le\n", i, OP_DBL_Digs-1, (double)temp);
                    all_worked = false;
#pragma omp flush (all_worked)
                }
            }
        }
    }
    return all_worked;
}

    template<>
int SparseTriangularSolve<float>::VerifyResults(int iteration)
{
    int errors = 0;

    #pragma omp parallel for
    for (int i = 0; i < nRows; i++)
    {
        int assigned_pe = (i / SPTS_BLOCK_SIZE) % this->Get_total_pes();
        if (this->Get_this_pe() == assigned_pe) {
            float compare_val = fabs(yref[i]*1e-3);
            if (compare_val < 10*FLT_EPSILON)
                compare_val = 10*FLT_EPSILON;
            if ((yref[i] - compare_val) > y[i] || (yref[i] + compare_val) < y[i])
            {
                #pragma omp critical
                {
                    if(errors == 0)
                    {
                        fflush(stdout);
                        fprintf(stderr, "\nDetected some differences between CPU and GPU results on iteration %d...", iteration);
                    }
                    fprintf(stderr, "%d GPU CALCULATION ERROR on row %d\n", this->Get_this_pe(), i);
                    fprintf(stderr, "\tCPU value for y[%d]: %.*e\n", i, OP_FLT_Digs-1, yref[i]);
                    fprintf(stderr, "\tGPU value for y[%d]: %.*e\n", i, OP_FLT_Digs-1, y[i]);
                    errors += 1;
                    observed_errors[iteration].insert(std::pair<int, float> (i, y[i]));
                }
            }
        }
    }
    return errors;
}

    template<>
int SparseTriangularSolve<double>::VerifyResults(int iteration)
{
    int errors = 0;
    #pragma omp parallel for
    for (int i = 0; i < nRows; i++)
    {
        double compare_val = fabs(yref[i]*1e-4);
        if (compare_val < 10*DBL_EPSILON)
            compare_val = 10*DBL_EPSILON;
        if ((yref[i] - compare_val) > y[i] || (yref[i] + compare_val) < y[i])
        {
            #pragma omp critical
            {
                if(errors == 0)
                {
                    fflush(stdout);
                    fprintf(stderr, "\nDetected differences between CPU and GPU results on iteration %d...", iteration);
                }
                fprintf(stderr, "GPU CALCULATION ERROR on row %d\n", i);
                fprintf(stderr, "\tCPU value for y[%d]: %.*e\n", i, OP_DBL_Digs-1, yref[i]);
                fprintf(stderr, "\tGPU value for y[%d]: %.*e\n", i, OP_DBL_Digs-1, y[i]);
                errors += 1;
                observed_errors[iteration].insert(std::pair<int, double> (i, y[i]));
            }
        }
    }
    return errors;
}

template<typename FloatType>
int SparseTriangularSolve<FloatType>::NonDeterministicErrors()
{
    int iter = InputFlags::GetValueInt("iterations");
    int non_det_errors = 0;
#ifdef ALL_SYNCFREE
    for (int i = 1; i < iter; i++)
    {
        if (errors_seen[i] != errors_seen[0])
        {
            non_det_errors++;
            if (non_det_errors == 1)
            {
                fprintf(stderr, "Different SpTS iterations saw different error counts -- non-deterministic bug possible.\n");
                fprintf(stderr, "\te.g. saw %d errors during iteration 0. Saw %d errors during iteration %i\n", errors_seen[0], errors_seen[i], i);
            }
        }
        else if (observed_errors[i] != observed_errors[0])
        {
            non_det_errors++;
            if (non_det_errors == 1)
            {
                fprintf(stderr, "ERRORS were seen. Different iterations saw errors on different rows -- non-deterministic bug possible.\n");
                fprintf(stderr, "\te.g. Iterations 0 and %d were different.\n", i);
            }
        }
    }
#else
    if (iter >= 1)
    {
        if (errors_seen[0] != errors_seen[1])
        {
            non_det_errors++;
            fprintf(stderr, "Different SpTS iterations saw different error counts -- non-deterministic bug possible.\n");
            fprintf(stderr, "\te.g. saw %d errors during iteration 0. Saw %d errors during iteration %i\n", errors_seen[0], errors_seen[1], 1);
        }
    }
    for (int i = 2; i < iter; i++)
    {
        if (errors_seen[i] != errors_seen[1])
        {
            non_det_errors++;
            if (non_det_errors == 1)
            {
                fprintf(stderr, "Different SpTS iterations saw different error counts -- non-deterministic bug possible.\n");
                fprintf(stderr, "\te.g. saw %d errors during iteration 1. Saw %d errors during iteration %i\n", errors_seen[1], errors_seen[i], i);
            }
        }
        else if (observed_errors[i] != observed_errors[1])
        {
            non_det_errors++;
            if (non_det_errors == 1)
            {
                fprintf(stderr, "ERRORS were seen. Different iterations saw errors on different rows -- non-deterministic bug possible.\n");
                fprintf(stderr, "\te.g. Iterations 1 and %d were different.\n", i);
            }
        }
    }
#endif
    return non_det_errors;
}

template<typename FloatType>
int SparseTriangularSolve<FloatType>::MaxErrors()
{
    int iter = InputFlags::GetValueInt("iterations");
    int max_errors = 0;
    for (int i = 0; i < iter; i++)
    {
        if (errors_seen[i] > max_errors)
            max_errors = errors_seen[i];
    }
    return max_errors;
}

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

static inline unsigned int flp2(unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}

// Short rows in CSR-Adaptive are batched together into a single row block.
// If there are a relatively small number of these, then we choose to do
// a horizontal reduction (groups of threads all reduce the same row).
// If there are many threads (e.g. more threads than the maximum size
// of our workgroup) then we choose to have each thread serially reduce
// the row.
// This function calculates the number of threads that could team up
// to reduce these groups of rows. For instance, if you have a
// workgroup size of 256 and 4 rows, you could have 64 threads
// working on each row. If you have 5 rows, only 32 threads could
// reliably work on each row because our reduction assumes power-of-2.
    template< typename rowBlockType >
static inline rowBlockType numThreadsForReduction(const rowBlockType num_rows)
{
#if defined(__INTEL_COMPILER)
    return 256 >> (_bit_scan_reverse(num_rows-1)+1);
#elif (defined(__clang__) && __has_builtin(__builtin_clz)) || \
    !defined(__clang) && \
    defined(__GNUG__) && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
    return (256 >> (8*sizeof(int)-__builtin_clz(num_rows-1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    unsigned long bit_returned;
    _BitScanReverse(&bit_returned, (num_rows-1));
    return 256 >> (bit_returned+1);
#else
    return flp2(256/num_rows);
#endif
}

    template<typename FloatType>
int SparseTriangularSolve<FloatType>::ComputeRowBlocks(std::vector<uint64_t> &rowBlocks,
        int *rowDelimiters,
        int nRows)
{
    rowBlocks.erase(rowBlocks.begin(), rowBlocks.end());
    rowBlocks.push_back(0);
    uint64_t sum = 0;
    uint64_t i, last_i = 0;

    // Check to ensure nRows can fit in 32 bits
    if ((uint64_t) nRows > (uint64_t)pow(2, ROW_BITS))
    {
        fflush(stdout);
        fprintf(stderr, "\nNumber of Rows in the Sparse Matrix is greater than what is supported at present (%d bits) !", ROW_BITS );
        exit(0);
    }

    int consecutive_long_rows = 0;
    for(i = 1; i <= nRows; i++)
    {
        int row_length = ( rowDelimiters[ i ] - rowDelimiters[ i - 1 ] );
        sum += row_length;

        // The following section of code calculates whether you're moving between
        // a series of "short" rows and a series of "long" rows.
        // This is because the reduction in CSR-Adaptive likes things to be
        // roughly the same length. Long rows can be reduced horizontally.
        // Short rows can be reduced one-thread-per-row. Try not to mix them.
        if ( row_length > 128 )
            consecutive_long_rows++;
        else if ( consecutive_long_rows > 0 )
        {
            // If it turns out we WERE in a long-row region, cut if off now.
            if (row_length < 32) // Now we're in a short-row region
                consecutive_long_rows = -1;
            else
                consecutive_long_rows++;
        }

        // If you just entered into a "long" row from a series of short rows,
        // then we need to make sure we cut off those short rows. Put them in
        // their own workgroup.
        if ( consecutive_long_rows == 1 )
        {
            // Assuming there *was* a previous workgroup. If not, nothing to do here.
            if( i - last_i > 1 )
            {
                rowBlocks.push_back( (i - 1) << (64 - ROW_BITS) );
                // If this row fits into CSR-Stream, calculate how many rows
                // can be used to do a parallel reduction.
                // Fill in the low-order bits with the numThreadsForRed
                if (((i-1) - last_i) > 2)
                    rowBlocks[rowBlocks.size() - 2] |= numThreadsForReduction((i - 1) - last_i);

                last_i = i-1;
                sum = row_length;
            }
        }
        else if (consecutive_long_rows == -1)
        {
            // We see the first short row after some long ones that
            // didn't previously fill up a row block.
            rowBlocks.push_back( (i - 1) << (64 - ROW_BITS) );
            if (((i-1) - last_i) > 2)
                rowBlocks[rowBlocks.size() - 2] |= numThreadsForReduction((i - 1) - last_i);

            last_i = i-1;
            sum = row_length;
            consecutive_long_rows = 0;
        }

        // Now, what's up with this row? What did it do?

        // exactly one row results in non-zero elements to be greater than blockSize
        // This is csr-vector case; bottom WG_BITS == workgroup ID
        if( ( i - last_i == 1 ) && sum > 1024 )
        {
            int numWGReq = ceil( (double)row_length / (1024) );

            // Check to ensure #workgroups can fit in WG_BITS bits, if not
            // then the last workgroup will do all the remaining work
            numWGReq = ( numWGReq < (int)pow( 2, WG_BITS ) ) ? numWGReq : (int)pow( 2, WG_BITS );

            for( int w = 1; w < numWGReq; w++ )
            {
                rowBlocks.push_back((i-1) << ROW_BITS);
                rowBlocks[rowBlocks.size() - 1] |= (uint64_t)w;
            }
            rowBlocks.push_back(i << ROW_BITS);

            last_i = i;
            sum = 0;
            consecutive_long_rows = 0;
        }
        // more than one row results in non-zero elements to be greater than blockSize
        // This is csr-stream case; bottom WG_BITS = number of parallel reduction threads
        else if( ( i - last_i > 1 ) && sum > 1024 )
        {
            i--; // This row won't fit, so back off one.
            rowBlocks.push_back( i << (64 - ROW_BITS) );
            if ((i - last_i) > 2)
                rowBlocks[rowBlocks.size() - 2] |= numThreadsForReduction(i - last_i);
            last_i = i;
            sum = 0;
            consecutive_long_rows = 0;
        }
        // This is csr-stream case; bottom WG_BITS = number of parallel reduction threads
        else if( sum == 1024 )
        {
            rowBlocks.push_back( i << (64 - ROW_BITS) );
            if ((i - last_i) > 2)
                rowBlocks[rowBlocks.size() - 2] |= numThreadsForReduction(i - last_i);
            last_i = i;
            sum = 0;
            consecutive_long_rows = 0;
        }
    }

    // If we didn't fill a row block with the last row, make sure we don't lose it.
    if ( (rowBlocks[rowBlocks.size() - 2] >> (64 - ROW_BITS)) != (uint64_t)(nRows) )
    {
        rowBlocks.push_back( (uint64_t)( nRows ) << (64 - ROW_BITS) );
        if ((nRows - last_i) > 2)
            rowBlocks[rowBlocks.size() - 2] |= numThreadsForReduction(i - last_i);
    }

    return rowBlocks.size();
}

    template<typename FloatType>
float SparseTriangularSolve<FloatType>::CSRSpTSGPU(uint64_t &ns_per_iter, uint64_t &ns_per_analysis_iter, uint64_t &ns_per_syncfree_iter, uint64_t &ns_per_levelset_iter, uint64_t &ns_per_levelsync_iter, FloatType alpha)
{
    gpuInt status;
    gpuEvent* event_array;
    #ifdef USE_HIP
    hipSetDevice(this->Get_this_pe());
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, this->Get_this_pe());
    printf("\nPE %d: PCIe BUS ID %d DEV ID %d\n", this->Get_this_pe(), props.pciBusID, props.pciDeviceID);
    event_array = (gpuEvent*)malloc(sizeof(gpuEvent) * 2);
    hipEventCreate(&event_array[0]);
    hipEventCreate(&event_array[1]);
    #else
    event_array = (gpuEvent*)malloc(sizeof(gpuEvent));
    #endif
    size_t global_work_size;
    size_t local_work_size = WF_PER_WG * WF_SIZE;


    /*************************** Setup and create buffers ********************/
    /****** Matrix Setup Code ******/
    /* Get the OpenCL buffers for the input matrix */
    memPointer bufNonZeroes = SparseMatrix<FloatType>::GetDevVals();
    memPointer bufColumnIndices = SparseMatrix<FloatType>::GetDevCols();
    memPointer bufRowPtrs = SparseMatrix<FloatType>::GetDevRowPtrs();
    /* Get the host buffers for the input matrix */
    FloatType *Avalues = SparseMatrix<FloatType>::GetVals();
    int *Acols = SparseMatrix<FloatType>::GetCols();
    int *rowptrs = SparseMatrix<FloatType>::GetRowPtrs();


    /****** Adaptive RowBlocks Setup Code ******/
    numBlocks = ComputeRowBlocks(rowBlocks, rowptrs, nRows);
    rowBlocksDev = this->GPU->AllocateMem("rowBlocks", numBlocks*sizeof(int64_t), GPU_MEM_READ_WRITE, NULL);
    uint64_t completedRows = 0;
    completedRowsDev = this->GPU->AllocateMem("completedRows", sizeof(uint64_t), GPU_MEM_READ_WRITE|GPU_MEM_USE_HOST_PTR, &completedRows);

    /****** SpTS Meta-Data Setup Code ******/
    /* Set up the OpenCL buffers for the SpTS meta-data */
    // TODO -- is this +1 in doneArray nRows+1 required? Why?
    #ifdef USE_ROCSHMEM
    doneArrayDev = rocshmem_malloc((nRows+1)*sizeof(uint32_t));
    reqUpdateArrayDev = rocshmem_malloc((nRows+1)*sizeof(uint32_t));
    shadowDoneArrayDev = rocshmem_malloc((nRows+1)*sizeof(uint32_t));
    #else
    doneArrayDev = this->GPU->AllocateMem("doneArray", (nRows+1)*sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    reqUpdateArrayDev = this->GPU->AllocateMem("reqUpdateArray", (nRows+1)*sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    shadowDoneArrayDev = this->GPU->AllocateMem("shadowDoneArray", (nRows+1)*sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    #endif
    remoteInProgressArrayDev = this->GPU->AllocateMem("remoteInProgressArray", (nRows+1)*sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    numRowsAtLevelDev = this->GPU->AllocateMem("numRowsAtLevel", (nRows)*sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    rowMapDev = this->GPU->AllocateMem("rowMap", (nRows+1)*sizeof(uint32_t), GPU_MEM_READ_ONLY, NULL);
    maxDepthDev = this->GPU->AllocateMem("maxDepth", sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    totalSpinDev = this->GPU->AllocateMem("totalSpin", sizeof(uint64_t), GPU_MEM_READ_WRITE, NULL);
    oneBufDev = this->GPU->AllocateMem("oneBuf", sizeof(uint32_t), GPU_MEM_READ_WRITE, NULL);
    /* Set up the host buffers for the SpTS meta-data */
    uint32_t *doneArray = (uint32_t*)calloc((nRows+1), sizeof(uint32_t));
    uint32_t *numRowsAtLevel = (uint32_t*)calloc(nRows, sizeof(uint32_t));
    uint32_t *rowMap = (uint32_t*)calloc((nRows+1), sizeof(uint32_t));
    uint32_t maxDepth = 0;
    uint64_t totalSpin = 0;

    uint32_t *nrows_plus1_zero = (uint32_t*)calloc((nRows+1), sizeof(uint32_t));
    uint64_t u64_zero = 0;
    uint32_t u32_zero = 0;

    //uint32_t uns_int_one = 0x42280000;
    uint32_t u32_one = 1;

    // TODO:  Gather and flatten out Avalues, Acols, and rowptrs based on
    // row cyclic decomposition.  For now, we just copy the hole vals, cols,
    // and row_ptrs matrix, even though we really only access 1/num_pes of the
    // whole thing.  We can do some more sophisticated stuff here if we run out
    // of space on the GPU or we don't like the copy overheads of the initial
    // buffers.

    /************************ Copy initial buffers to device *****************/
    /****** Copy matrix ******/
    this->GPU->CopyToDevice(bufNonZeroes, Avalues, this->nNZ*sizeof(FloatType), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(bufColumnIndices, Acols, this->nNZ*sizeof(int), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(bufRowPtrs, rowptrs, (this->nRows+1)*sizeof(int), 0, GPU_TRUE, NULL);

    /****** Copy vectors ******/
    this->GPU->CopyToDevice(xDev, x, nCols*sizeof(FloatType), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(yDev, y_zero, nRows*sizeof(FloatType), 0, GPU_TRUE, NULL);

    /****** Copy adaptive rowBlock information ******/
    this->GPU->CopyToDevice(rowBlocksDev, rowBlocks.data(), numBlocks*sizeof(int64_t), 0, GPU_TRUE, NULL);

    /****** Copy SpTS meta-data needed for analyze_and_solve run ******/
    this->GPU->CopyToDevice(doneArrayDev, nrows_plus1_zero, (nRows+1)*sizeof(uint32_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(shadowDoneArrayDev, nrows_plus1_zero, (nRows+1)*sizeof(uint32_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(reqUpdateArrayDev, nrows_plus1_zero, (nRows+1)*sizeof(uint32_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(remoteInProgressArrayDev, nrows_plus1_zero, (nRows+1)*sizeof(uint32_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(numRowsAtLevelDev, nrows_plus1_zero, nRows*sizeof(uint32_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(maxDepthDev, &u32_zero, sizeof(uint32_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(totalSpinDev, &u64_zero, sizeof(uint64_t), 0, GPU_TRUE, NULL);
    this->GPU->CopyToDevice(oneBufDev, &u32_one, sizeof(uint32_t), 0, GPU_TRUE, NULL);

    /************************** Set up iteration printing ********************/
    /* We want to print, ideally, every iteration that gets up 10% closer to
     * completion. This sets that up */
    int iter = InputFlags::GetValueInt("iterations");
    double print_iter = (float)iter / 10.;
    if (print_iter < 1.)
        print_iter = 1.;
    double next_to_print = 0.;


    /**************************** Set up perf analysis ************************/
    // For performance analysis, keep track of how much time we've spent doing
    // kernel work.
    // TODO -- Take in from the command line whether to get kernel or total time.
    // If doing total time, try launching all of the kernels at once and waiting
    // outside. This will apparently reduce the overheads.
    uint64_t total_kern_time = 0;
    uint64_t analyze_kern_time = 0;
    double analyze_kern_flops = 0.;
    uint64_t syncfree_kern_time = 0;
    uint64_t levelset_kern_time = 0;
    uint64_t levelsync_kern_time = 0;

    errors_seen = new int[iter];

    int analysis_iter = 0;
    int syncfree_iter = 0;
    int levelset_iter = 0;
    int levelsync_iter = 0;

    int level_sync_cutoff = InputFlags::GetValueInt("levelsync_size");
    bool syncfree_better = false;

    int total_workitems_per_workgroup = WF_SIZE * WF_PER_WG;
    //bool rocshmem_initialized = false;

    /*********************** Actual work of the benchmark *********************/
    for(int i = 0; i < iter; i++)
    {
        if (i == (int)next_to_print || i == (iter - 1))
        {
            printf("%d..", i+1);fflush(stdout);
            next_to_print += print_iter;
        }

#ifndef ALL_SYNCFREE
#ifdef ALL_ANALYZE
        // When we only want to run the analyze-and-solve mechanism, rather than
        // the more optimized syncfree algorithm, we always go into here.
        if (1)
#else
        // In any version of the program that has the possibility of running the
        // level-set algorithm, we need to start with the syncfree-and-analyze
        // version of the program, so that we can set up the potential to run the
        // level-set algorithm. This will take place on the first iteration.
        if (i == 0)
#endif
        {
            analysis_iter++;
            global_work_size = nRows * WF_SIZE;
            #ifndef USE_HIP
            CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
            CL->SetArgs(CLHelper::SpTSKernel_analyze, 0,
                    bufNonZeroes,
                    bufColumnIndices,
                    bufRowPtrs,
                    xDev,
                    yDev,
                    alpha,
                    doneArrayDev,
                    numRowsAtLevelDev,
                    maxDepthDev,
                    totalSpinDev);

            status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel_analyze, 1, NULL, &global_work_size, NULL, 0, NULL, &event_array[0]);
            CL->checkStatus(status,"clEnqueueNDRangeKernel failed");
            this->GPU->Flush();
            total_kern_time += CL->ComputeTime(event_array[0]);
            analyze_kern_time += CL->ComputeTime(event_array[0]);
            #else
            int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                    / total_workitems_per_workgroup;
            #ifdef USE_ROCSHMEM
            global_work_size = this->nRows_p * WF_SIZE;
            num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                 / total_workitems_per_workgroup;
	   /*
	    int rocshmem_queues = (2560 / WF_PER_WG);
	    if (2560 % WF_PER_WG)
		rocshmem_queues++;
            if (!rocshmem_initialized) {
            	int num_threads = InputFlags::GetValueInt("num_roshmem_threads");
                rocshmem_init(&handle, rocshmem_queues);
                rocshmem_initialized = true;
            }
		*/
            int rocshmem_algorithm = InputFlags::GetValueInt("rocshmem_algorithm");
			int rocshmem_put_block_size = InputFlags::GetValueInt("put_block_size");
			int rocshmem_get_backoff_factor = InputFlags::GetValueInt("get_backoff_factor");
	    switch (rocshmem_algorithm) {
		case 0:
                	printf("Using Put-based intra-kernel algorithm\n");
			break;
		case 1:
                	printf("Using Get-based intra-kernel algorithm (Backoff factor %d)\n", rocshmem_get_backoff_factor);
			break;
		case 2:
                	printf("Using blocked Put-based intra-kernel algorithm\n");
					printf("Using blocked Put-based intra-kernel algorithm (Block Size %d)\n", rocshmem_put_block_size);
			break;
		case 3:
                	printf("Using put/get hybrid intra-kernel algorithm\n");
			break;
		default:
			printf("Unknown rocSHMEM algorithm\n");
			exit(-1);
	   }
            size_t LDS_size;
            rocshmem_dynamic_shared(&LDS_size);
            printf("Work size %zu, wg size %d num workgroups %d  LDS %zu  thisPE %d  Global %d \n", global_work_size, total_workitems_per_workgroup, num_of_workgroups, LDS_size,  this->Get_this_pe(), this->Get_total_pes());
            MPI_Barrier(MPI_COMM_WORLD);
            hipEventRecord(event_array[0], NULL);
            hipLaunchKernelGGL(amd_spts_analyze_and_solve,
                    dim3(num_of_workgroups),
                    dim3(total_workitems_per_workgroup),
                    LDS_size, 0,
                    global_work_size,
                    this->Get_this_pe(),
                    this->Get_total_pes(),
                    static_cast<unsigned int *>(shadowDoneArrayDev),
                    static_cast<unsigned int *>(reqUpdateArrayDev),
                    static_cast<unsigned int *>(remoteInProgressArrayDev),
                    static_cast<unsigned int *>(oneBufDev),
                    rocshmem_algorithm,
					rocshmem_put_block_size,
					rocshmem_get_backoff_factor,
                    SPTS_BLOCK_SIZE,
                    static_cast<FPTYPE *>(bufNonZeroes),
                    static_cast<int *>(bufColumnIndices),
                    static_cast<int *>(bufRowPtrs),
                    static_cast<FPTYPE *>(xDev),
                    static_cast<FPTYPE *>(yDev),
                    alpha,
                    static_cast<unsigned int *>(doneArrayDev),
                    static_cast<unsigned int *>(numRowsAtLevelDev),
                    static_cast<unsigned int *>(maxDepthDev),
                    static_cast<unsigned long long *>(totalSpinDev));
            #else
            hipEventRecord(event_array[0], NULL);
            hipLaunchKernelGGL(amd_spts_analyze_and_solve,
                    dim3(num_of_workgroups),
                    dim3(total_workitems_per_workgroup),
                    0, 0,
                    global_work_size,
                    static_cast<FPTYPE *>(bufNonZeroes),
                    static_cast<int *>(bufColumnIndices),
                    static_cast<int *>(bufRowPtrs),
                    static_cast<FPTYPE *>(xDev),
                    static_cast<FPTYPE *>(yDev),
                    alpha,
                    static_cast<unsigned int *>(doneArrayDev),
                    static_cast<unsigned int *>(numRowsAtLevelDev),
                    static_cast<unsigned int *>(maxDepthDev),
                    static_cast<unsigned long long *>(totalSpinDev));
            #endif
            hipEventRecord(event_array[1], NULL);
            hipEventSynchronize(event_array[1]);

            #ifdef USE_ROCSHMEM
            // Wait for any outstanding network messages to finish up.  We
            // can have straggler updates to the doneArray that we don't
            // have any dependencies for but we still eed it to finish so
            // the below statistics can work correctly.
            //ro_shmem_dump_stats(handle);
            //ro_shmem_reset_stats(handle);
			//sleep(10);
			/* if( this->Get_this_pe() == 0 && (this->Get_total_pes() > 1)){
			 	PRINT_SQ(get_rtn_handle(handle), 0, 1, 0);
			 	PRINT_CQ(get_rtn_handle(handle), 0, 1, 0);
			 	PRINT_SQ(get_rtn_handle(handle), 0, 1, 1);
			 	PRINT_CQ(get_rtn_handle(handle), 0, 1, 1);
			 	PRINT_SQ(get_rtn_handle(handle), 0, 1, 2);
			 	PRINT_CQ(get_rtn_handle(handle), 0, 1, 2);
				}*/
            MPI_Barrier(MPI_COMM_WORLD);
            #endif
            float elapsed;
            hipEventElapsedTime(&elapsed, event_array[0], event_array[1]);
            total_kern_time += elapsed * 1000000;
            analyze_kern_time += elapsed * 1000000;
            #endif
            analyze_kern_flops = (2 * (double)nNZ * 1000000000.) / (double)analyze_kern_time;
            this->GPU->CopyToHost(yDev, y, nRows*sizeof(FloatType), 0, GPU_FALSE, NULL);
            this->GPU->CopyToHost(maxDepthDev, &maxDepth, sizeof(uint32_t), 0, GPU_FALSE, NULL);
            this->GPU->CopyToHost(doneArrayDev, doneArray, (nRows+1)*sizeof(uint32_t), 0, GPU_FALSE, NULL);
            this->GPU->CopyToHost(totalSpinDev, &totalSpin, sizeof(uint64_t), 0, GPU_FALSE, NULL);
            this->GPU->CopyToHost(numRowsAtLevelDev, numRowsAtLevel, nRows*sizeof(uint32_t), 0, GPU_TRUE, NULL);
            this->GPU->Flush();

            #ifdef USE_ROCSHMEM
            // Combine global statistics
            MPI_Allreduce(MPI_IN_PLACE, (void *) &maxDepth, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, (void *) &totalSpin, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

            // TODO: Broadcast out the doneArray and yDev values to all nodes.  This
            // is needed for the below calculations to work since in the 'pull'
            // distributed model we don't request data for rows that we don't
            // have a dependency on.
            #endif

            bool verify = InputFlags::GetValueBool("verify");
            if (verify) {
                printf("Performing results verification\n");
                errors_seen[i] = VerifyResults(i);
            }
            printf("\nTotalSpin: %lu\n", totalSpin);

            /* Prefix sum of the number of rows at each level, so that we can
             * calculate how much to offset each level into the rowMap */
            // TODO -- Do this prefix sum on the GPU while copying maxDepth and
            // doneArray back into the host. Set non-blocking on the previous ones.
            this->GPU->CopyToHost(numRowsAtLevelDev, numRowsAtLevel, nRows*sizeof(uint32_t), 0, GPU_TRUE, NULL);
            for (unsigned int joe = 1; joe < maxDepth; joe++)
                numRowsAtLevel[joe] = numRowsAtLevel[joe] + numRowsAtLevel[joe-1];

            /* Build up the rowMap so that each iteration of the no-wait solve
             * knows what it's global_id->row mapping is.
             * The general mechanism for this is as follows:
             * doneArray[row] holds the level that a particular row is in.
             *
             * We know the total number of levels needed (maxDepth), so rowMap
             * has maxDepth 'buckets'.
             *
             * The numRowsAtLevel array (after the above prefix-sum) tells us
             * how many values are in all of the previous buckets, so that
             * we can get an appropriate array offset for each bucket.
             *
             * The counters array keeps track of how many items are in each
             * bucket so far. Add this to the numRowsAtLevel[] offset.
             *
             * As we walk through all the rows, we check to see which level's
             * bucket we should put this row in. Add it at the end of the
             * current bucket, then increment the counter. */
            uint32_t *counters = (uint32_t *)calloc(maxDepth, sizeof(uint32_t));
/*            for (unsigned int this_row = 0; this_row < nRows; this_row++)
            {
                // We must subtract one here, because the first level is '1'
                // The GPU kernel does that because a value of '0' means
                // 'not done, keep waiting' in the analysis kernel.
                assert(doneArray[this_row] != 0);
                unsigned int this_rows_level = doneArray[this_row] - 1;
                unsigned int previous_level = this_rows_level - 1;
                unsigned int depth_offset;
                if (this_rows_level == 0) // can't check previous level
                    depth_offset = 0;
                else
                    depth_offset = numRowsAtLevel[previous_level];
                rowMap[depth_offset + counters[this_rows_level]] = this_row;
                counters[this_rows_level] += 1;
            } */
            free(counters);
            this->GPU->CopyToDevice(rowMapDev, rowMap, (nRows+1)*sizeof(uint32_t), 0, GPU_TRUE, NULL);
            free(event_array);
            #ifdef USE_HIP
            event_array = (gpuEvent*)malloc(maxDepth * sizeof(gpuEvent) * 2);
            for (int i = 0; i < maxDepth * 2; i++)
                hipEventCreate(&event_array[i]);
            #else
            event_array = (gpuEvent*)malloc(maxDepth * sizeof(gpuEvent));
            #endif
#ifdef ALL_ANALYZE
            // We will be coming back into this kernel. Time to reset its data.
            if (i != (iter - 1))
            {
                this->GPU->CopyToDevice(maxDepthDev, &u32_zero, sizeof(uint32_t), 0, GPU_FALSE, NULL);
                this->GPU->CopyToDevice(totalSpinDev, &u64_zero, sizeof(uint64_t), 0, GPU_FALSE, NULL);
                this->GPU->CopyToDevice(numRowsAtLevelDev, nrows_plus1_zero, nRows*sizeof(uint32_t), 0, GPU_FALSE, NULL);
            }
#endif
            this->GPU->CopyToDevice(yDev, y_zero, nRows*sizeof(FloatType), 0, GPU_FALSE, NULL);
            this->GPU->CopyToDevice(doneArrayDev, nrows_plus1_zero,(nRows+1)*sizeof(uint32_t), 0, GPU_FALSE, NULL);
            this->GPU->CopyToDevice(shadowDoneArrayDev, nrows_plus1_zero,(nRows+1)*sizeof(uint32_t), 0, GPU_FALSE, NULL);
            this->GPU->CopyToDevice(remoteInProgressArrayDev, nrows_plus1_zero,(nRows+1)*sizeof(uint32_t), 0, GPU_FALSE, NULL);
    	    this->GPU->CopyToDevice(reqUpdateArrayDev, nrows_plus1_zero, (nRows+1)*sizeof(uint32_t), 0, GPU_FALSE, NULL);
            this->GPU->Flush();
            // Either we always want to run just this function block
            // (ALL_ANALYZE), or the first iteration is the analyze-and-solve
            // kernel. Either way, don't continue to the code below this time.
            continue;
    }
#endif
    // If ALL_SYNCFREE is defined, we always run the amd_spts_syncfree_solve
    // kernel. We never try to speed it up by paying attention to the output
    // levels and running the levelset kernel.
    // If ALL_LEVELSET is set, we only run the analysis kernel up above to get the
    // level-set and do the first solve; after that we skip the
    // amd_spts_analyze_and_solve kernel and do the level-set based solve.
    // Otherwise, we dynamically choose between those kernels based on some
    // statistics that we gathered during the analyze-and-solve run.
#ifdef ALL_SYNCFREE
        if (1) // always run syncfree algorithm
#elif defined(ALL_LEVELSET) || defined(ALL_LEVELSYNC)
        if (0) // always *do not* run syncfree algorithm
#else
        if (totalSpin == 0 || analyze_kern_flops/totalSpin > 25000 || syncfree_better) // Try to run syncfree
#endif
        {
            syncfree_iter++;
            // TODO -- Eventually get this working with numer of RowBlocks - 1
            global_work_size = nRows * WF_SIZE;

            uint32_t current_iteration = 0;

            #ifdef USE_ROCSHMEM
            fprintf(stderr, "rocSHMEM not supported for selected algorithm\n");
            exit(-1);
            #endif

            #ifndef USE_HIP
            CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
            CL->SetArgs(CLHelper::SpTSKernel, 0,
                    bufNonZeroes,
                    bufColumnIndices,
                    bufRowPtrs,
                    xDev,
                    yDev,
                    alpha,
                    doneArrayDev,
                    numRowsAtLevelDev,
                    maxDepthDev,
                    totalSpinDev);

            status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event_array[0]);
            CL->checkStatus(status,"clEnqueueNDRangeKernel failed");
            current_iteration++;
            this->GPU->Flush();
            total_kern_time += CL->ComputeTime(event_array[0]);
            syncfree_kern_time += CL->ComputeTime(event_array[0]);
            #else
            int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                    / total_workitems_per_workgroup;
            hipEventRecord(event_array[0], NULL);
            hipLaunchKernelGGL(amd_spts_syncfree_solve,
                    dim3(num_of_workgroups),
                    dim3(total_workitems_per_workgroup),
                    0, 0,
                    global_work_size,
                    static_cast<FPTYPE *>(bufNonZeroes),
                    static_cast<int *>(bufColumnIndices),
                    static_cast<int *>(bufRowPtrs),
                    static_cast<FPTYPE *>(xDev),
                    static_cast<FPTYPE *>(yDev),
                    alpha,
                    static_cast<unsigned int *>(doneArrayDev),
                    static_cast<unsigned int *>(numRowsAtLevelDev),
                    static_cast<unsigned int *>(maxDepthDev),
                    static_cast<unsigned long long *>(totalSpinDev));
            hipEventRecord(event_array[1], NULL);
            hipEventSynchronize(event_array[1]);
            current_iteration++;
            float elapsed;
            hipEventElapsedTime(&elapsed, event_array[0], event_array[1]);
            total_kern_time += elapsed * 1000000;
            syncfree_kern_time += elapsed * 1000000;

            #endif

            this->GPU->CopyToHost(yDev, y, nRows*sizeof(FloatType), 0, GPU_TRUE, NULL);
            errors_seen[i] = VerifyResults(i);

            this->GPU->Flush();
            current_iteration = 0;
            completedRows = 0;
        }
#if defined(ALL_LEVELSYNC)
        else if (1) // always run levelset+syncfree combination
#elif defined (ALL_LEVELSET)
        else if (0) // Fall through to level-set
#else
        else if (1) // always run levelset+syncfree, never fall through to level-set only
#endif
        {
            // This is the "level-sync" algorithm, where we take the level-set
            // information and launch kernels that combine multiple levels
            // together. This allows us to find parallelism to run on the GPU,
            // even if technically there are some data dependencies between the
            // levels. Within the kernel, we use the synchronization-free algorithm
            // to ensure that we get the right answer.
            // This algorithm reduces the spin-loop overhead of the sync-free
            // algorithm if there are many levels, but it finds more parallelism
            // than the pure level-set algorithms which can only run on one CU.
            levelsync_iter++;

            // Keep track of total kernels we launch so we can watch for events.
            int total_enqueues = 0;

            /* The rowMap tells each workgroup within the kernel what
             * rows it is working on. However, each each kernel invocation
             * is working on a different level. Each level is in a separate
             * 'bucket' in the rowMap. We must tell each invocation how far
             * into the rowMap it much index. That's the depth_offset.
             * numRowsAtLevel (after the above prefix-sum) tells us how
             * many rows were in all previous levels combined. */
            unsigned int depth_offset = 0;
            unsigned int running_total = 0; // How many rows in this launch

            if (level_sync_cutoff == 0)
            {
                if (nRows/maxDepth < 32)
                    level_sync_cutoff = 2560;
                else
                    level_sync_cutoff = 81920;
            }

            #ifdef USE_ROCSHMEM
            fprintf(stderr, "rocSHMEM not supported for selected algorithm\n");
            exit(-1);
            #endif

            for (int this_level = 0; this_level < maxDepth; this_level++)
            {
                if (this_level != 0 && running_total == 0)
                    depth_offset = numRowsAtLevel[this_level-1];

                running_total = numRowsAtLevel[this_level] - depth_offset;

                if (running_total >= level_sync_cutoff)
                {
                    global_work_size = (running_total + (running_total % WF_PER_WG)) * WF_SIZE;
                    #ifndef USE_HIP
                    CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                    CL->SetArgs(CLHelper::SpTSKernel_levelsync, 0,
                            bufNonZeroes,
                            bufColumnIndices,
                            bufRowPtrs,
                            xDev,
                            yDev,
                            alpha,
                            doneArrayDev,
                            rowMapDev,
                            depth_offset);
                    status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel_levelsync, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event_array[total_enqueues]);
                    this->GPU->checkStatus(status,"clEnqueueNDRangeKernel failed");
                    #else
                    int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                            / total_workitems_per_workgroup;
                    hipEventRecord(event_array[total_enqueues * 2], NULL);
                    hipLaunchKernelGGL(amd_spts_levelsync_solve,
                            dim3(num_of_workgroups),
                            dim3(total_workitems_per_workgroup),
                            0, 0,
                            global_work_size,
                            static_cast<FPTYPE *>(bufNonZeroes),
                            static_cast<int *>(bufColumnIndices),
                            static_cast<int *>(bufRowPtrs),
                            static_cast<FPTYPE *>(xDev),
                            static_cast<FPTYPE *>(yDev),
                            alpha,
                            static_cast<unsigned int *>(doneArrayDev),
                            static_cast<unsigned int*>(rowMapDev),
                            depth_offset);
                    hipEventRecord(event_array[total_enqueues * 2 + 1], NULL);
                    #endif
                    total_enqueues++;
                    running_total = 0;
                }
            }
            if (running_total)
            {
                global_work_size = (running_total + (running_total % WF_PER_WG)) * WF_SIZE;
                #ifndef USE_HIP
                CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                CL->SetArgs(CLHelper::SpTSKernel_levelsync, 0,
                        bufNonZeroes,
                        bufColumnIndices,
                        bufRowPtrs,
                        xDev,
                        yDev,
                        alpha,
                        doneArrayDev,
                        rowMapDev,
                        depth_offset);
                status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel_levelsync, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event_array[total_enqueues]);
                this->GPU->checkStatus(status,"clEnqueueNDRangeKernel failed");
                #else
                int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                        / total_workitems_per_workgroup;
                hipEventRecord(event_array[total_enqueues * 2], NULL);
                hipLaunchKernelGGL(amd_spts_levelsync_solve,
                        dim3(num_of_workgroups),
                        dim3(total_workitems_per_workgroup),
                        0, 0,
                        global_work_size,
                        static_cast<FPTYPE *>(bufNonZeroes),
                        static_cast<int *>(bufColumnIndices),
                        static_cast<int *>(bufRowPtrs),
                        static_cast<FPTYPE *>(xDev),
                        static_cast<FPTYPE *>(yDev),
                        alpha,
                        static_cast<unsigned int *>(doneArrayDev),
                        static_cast<unsigned int*>(rowMapDev),
                        depth_offset);
                hipEventRecord(event_array[total_enqueues * 2 + 1], NULL);
                #endif
                total_enqueues++;
            }

            // After we cross this clFinish, all of the kernel invocations have
            // completed, and the final answer is in yDev. Now we should add up
            // all of the kernel runtimes from all levels to see how long this
            // levelset solve took.
            this->GPU->Flush();
            for (int this_enqueue = 0; this_enqueue < total_enqueues; this_enqueue++)
            {
                #ifndef USE_HIP
                CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                total_kern_time += CL->ComputeTime(event_array[this_enqueue]);
                levelsync_kern_time += CL->ComputeTime(event_array[this_enqueue]);
                #else
                float elapsed;
                hipEventElapsedTime(&elapsed, event_array[this_enqueue * 2], event_array[this_enqueue * 2 + 1]);
                total_kern_time += elapsed * 1000000;
                levelsync_kern_time += elapsed * 1000000;
                #endif
            }
            // The analyze kernel is about 15% slower than the syncfree kernel.
            // As such, if the level-sync verseion is < 15% faster, it's likely
            // that syncfree will win. Let's go back to doing that.
            if (i == 1 && (analyze_kern_time < (levelsync_kern_time * 1.15)))
                syncfree_better = true;
            this->GPU->CopyToHost(yDev, y, nRows*sizeof(FloatType), 0, GPU_TRUE, NULL);
            errors_seen[i] = VerifyResults(i);
        }
        else // Run level-set algorithm
        {
            // This is the level-set SpTS kernel, which can be done after the
            // first analyze-and-solve kernel. In this case, we know the levels
            // that each row is in, so we can launch one kernel per level with
            // exactly the right number of workgroups (one WG per row).
            // This means that we don't have any in-kernel atomics, spin-loops,
            // etc, so things run much faster. However, we much launch a
            // potentially large number of kernels.
            // Number of levels is maxDepth. */
            levelset_iter++;

            #ifdef USE_ROCSHMEM
            fprintf(stderr, "rocSHMEM not supported for selected algorithm\n");
            exit(-1);
            #endif

           // Keep track of total kernels we launch so we can watch for events.
            int total_enqueues = 0;

            unsigned int start_level = 0;
            unsigned int end_level = 0;
            unsigned int in_a_run = 0;
            unsigned int running_total = 0;

            // How far into the rowMap that lists which rows are in each level
            unsigned int depth_offset = 0;

            unsigned int total_vector = 0;
            unsigned int total_levelset = 0;
            for (int this_level = 0; this_level < maxDepth; this_level++)
            {
                unsigned int inner_depth_offset;
                if (this_level == 0)
                    inner_depth_offset = 0;
                else
                    inner_depth_offset = numRowsAtLevel[this_level-1];
                unsigned int total_in_this_depth = numRowsAtLevel[this_level] - inner_depth_offset;

                if (total_in_this_depth == 0)
                    continue;

                end_level = this_level;
                // Comment out this if(){} section to force us to always
                // launch the levelset kernel.
                if (total_in_this_depth <= 2*WF_PER_WG)
                {
                    running_total += total_in_this_depth;
                    if (in_a_run == 0)
                    {
                        start_level = this_level;
                        depth_offset = inner_depth_offset;
                        in_a_run = 1;
                    }
                }
                else
                {
                    if (in_a_run)
                    {
                        global_work_size = WF_SIZE * WF_PER_WG;
                        #ifndef USE_HIP
                        CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                        CL->SetArgs(CLHelper::SpTSKernel_vector, 0,
                                bufNonZeroes,
                                bufColumnIndices,
                                bufRowPtrs,
                                xDev,
                                yDev,
                                alpha,
                                rowMapDev,
                                numRowsAtLevelDev,
                                depth_offset,
                                start_level,
                                end_level);
                        status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel_vector, 1, NULL, &global_work_size, &global_work_size, 0, NULL, &event_array[total_enqueues]);
                        this->GPU->checkStatus(status,"clEnqueueNDRangeKernel failed");
                        #else
                        int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                                / total_workitems_per_workgroup;
                        hipEventRecord(event_array[total_enqueues * 2], NULL);
                        hipLaunchKernelGGL(amd_spts_vector_solve,
                                dim3(num_of_workgroups),
                                dim3(total_workitems_per_workgroup),
                                0, 0,
                                global_work_size,
                                static_cast<FPTYPE *>(bufNonZeroes),
                                static_cast<int *>(bufColumnIndices),
                                static_cast<int *>(bufRowPtrs),
                                static_cast<FPTYPE *>(xDev),
                                static_cast<FPTYPE *>(yDev),
                                alpha,
                                static_cast<unsigned int *>(rowMapDev),
                                static_cast<unsigned int *>(numRowsAtLevelDev),
                                depth_offset,
                                start_level,
                                end_level);
                        hipEventRecord(event_array[total_enqueues * 2 + 1], NULL);
                        #endif
                        total_enqueues++;
                        //printf("\n\tVector. offset %u Start %u End %u Rows in this enq %u\n", depth_offset, start_level, end_level, running_total);
                        in_a_run = start_level = end_level = running_total = 0;
                        depth_offset = numRowsAtLevel[this_level-1];
                        total_vector++;
                    }
                    global_work_size = WF_SIZE * total_in_this_depth;
                    #ifndef USE_HIP
                    CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                    CL->SetArgs(CLHelper::SpTSKernel_levelset, 0,
                            bufNonZeroes,
                            bufColumnIndices,
                            bufRowPtrs,
                            xDev,
                            yDev,
                            rowMapDev,
                            depth_offset,
                            alpha);
                    status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel_levelset, 1, NULL, &global_work_size, NULL, 0, NULL, &event_array[total_enqueues]);
                    this->GPU->checkStatus(status,"clEnqueueNDRangeKernel failed");
                    #else
                    int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                            / total_workitems_per_workgroup;
                    hipEventRecord(event_array[total_enqueues * 2], NULL);
                    hipLaunchKernelGGL(amd_spts_levelset_solve,
                            dim3(num_of_workgroups),
                            dim3(total_workitems_per_workgroup),
                            0, 0,
                            global_work_size,
                            static_cast<FPTYPE *>(bufNonZeroes),
                            static_cast<int *>(bufColumnIndices),
                            static_cast<int *>(bufRowPtrs),
                            static_cast<FPTYPE *>(xDev),
                            static_cast<FPTYPE *>(yDev),
                            static_cast<unsigned int *>(rowMapDev),
                            depth_offset,
                            alpha);
                    hipEventRecord(event_array[total_enqueues * 2 + 1], NULL);
                    #endif
                    total_enqueues++;
                    depth_offset = numRowsAtLevel[this_level];
                    total_levelset++;
                }
            }
            end_level++;
            if (in_a_run)
            {
                #ifndef USE_HIP
                CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                CL->SetArgs(CLHelper::SpTSKernel_vector, 0,
                        bufNonZeroes,
                        bufColumnIndices,
                        bufRowPtrs,
                        xDev,
                        yDev,
                        alpha,
                        rowMapDev,
                        numRowsAtLevelDev,
                        depth_offset,
                        start_level,
                        end_level);
                global_work_size = WF_SIZE * WF_PER_WG;
                status = clEnqueueNDRangeKernel(CLHelper::commandQueue, CLHelper::SpTSKernel_vector, 1, NULL, &global_work_size, &global_work_size, 0, NULL, &event_array[total_enqueues]);
                this->GPU->checkStatus(status,"clEnqueueNDRangeKernel failed");
                #else
                int num_of_workgroups = (global_work_size + total_workitems_per_workgroup - 1)
                                        / total_workitems_per_workgroup;
                hipEventRecord(event_array[total_enqueues * 2], NULL);
                hipLaunchKernelGGL(amd_spts_vector_solve,
                        dim3(num_of_workgroups),
                        dim3(total_workitems_per_workgroup),
                        0, 0,
                        global_work_size,
                        static_cast<FPTYPE *>(bufNonZeroes),
                        static_cast<int *>(bufColumnIndices),
                        static_cast<int *>(bufRowPtrs),
                        static_cast<FPTYPE *>(xDev),
                        static_cast<FPTYPE *>(yDev),
                        alpha,
                        static_cast<unsigned int *>(rowMapDev),
                        static_cast<unsigned int *>(numRowsAtLevelDev),
                        depth_offset,
                        start_level,
                        end_level);
                hipEventRecord(event_array[total_enqueues * 2 + 1], NULL);
                #endif
                total_enqueues++;
                //printf("\n\tVector. offset %u Start %u End %u Rows in this enq %u\n", depth_offset, start_level, end_level, running_total);
                in_a_run = start_level = end_level = running_total = 0;
                total_vector++;
            }

            if (i == 1)
                printf("\nTotal Vector: %u\nTotal levelset: %u\n", total_vector, total_levelset);
            // After we cross this clFinish, all of the kernel invocations have
            // completed, and the final answer is in yDev. Now we should add up
            // all of the kernel runtimes from all levels to see how long this
            // levelset solve took.
            this->GPU->Flush();
            for (int this_enqueue = 0; this_enqueue < total_enqueues; this_enqueue++)
            {
                #ifndef USE_HIP
                CLHelper *CL = dynamic_cast<CLHelper*>(this->GPU);
                total_kern_time += CL->ComputeTime(event_array[this_enqueue]);
                levelset_kern_time += CL->ComputeTime(event_array[this_enqueue]);
                #else
                float elapsed;
                hipEventElapsedTime(&elapsed, event_array[this_enqueue * 2], event_array[this_enqueue * 2 + 1]);
                total_kern_time += elapsed * 1000000;
                levelset_kern_time += elapsed * 1000000;
                #endif
            }
            this->GPU->CopyToHost(yDev, y, nRows*sizeof(FloatType), 0, GPU_TRUE, NULL);
            errors_seen[i] = VerifyResults(i);
        }

#ifndef ALL_SYNCFREE
        if (i == 1)
            printf("\nmaxDepth %d\n", maxDepth);
#endif
        if (i != (iter - 1))
        {
            this->GPU->CopyToDevice(yDev, y_zero, nRows*sizeof(FloatType), 0, GPU_FALSE, NULL);
            this->GPU->CopyToDevice(doneArrayDev, nrows_plus1_zero,(nRows+1)*sizeof(uint32_t), 0, GPU_FALSE, NULL);
            this->GPU->Flush();
        }
    }

    float gflops = 0.f;
    printf("\n\nnnz: %d\n", nNZ);
    gflops = (float)(2 * nNZ) / (float)(total_kern_time/iter);
    ns_per_iter = total_kern_time/iter;

    if (analysis_iter > 0)
        ns_per_analysis_iter = analyze_kern_time / analysis_iter;
    else
        ns_per_analysis_iter = 0;
    if (syncfree_iter > 0)
        ns_per_syncfree_iter = syncfree_kern_time / syncfree_iter;
    else
        ns_per_syncfree_iter = 0;
    if (levelset_iter > 0)
        ns_per_levelset_iter = levelset_kern_time / levelset_iter;
    else
        ns_per_levelset_iter = 0;
    if (levelsync_iter > 0)
        ns_per_levelsync_iter = levelsync_kern_time / levelsync_iter;
    else
        ns_per_levelsync_iter = 0;

    this->GPU->CopyToHost(yDev, y, nRows*sizeof(FloatType), 0, GPU_TRUE, NULL);

    if (doneArray)
        free(doneArray);
    if (numRowsAtLevel)
        free(numRowsAtLevel);
    if (rowMap)
        free(rowMap);
    if (event_array)
        free(event_array);

    return gflops;
}

#endif //SpTS_H
