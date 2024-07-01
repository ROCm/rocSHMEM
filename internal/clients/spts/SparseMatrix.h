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
#ifndef SparseMatrix_H
#define SparseMatrix_H

#include "GPUHelper.h"
#ifndef USE_HIP
#include "OpenCLHelper.h"
#include <CL/cl.h>
#else
#include "HIPHelper.h"
#endif

#include "InputFlags.h"
#include "MatrixMarketReader.h"
#include "OpenCLHelper.h"
#include <algorithm>
#include <cassert>

template<typename FloatType>
class SparseMatrix 
{
	
    public:
	int nRows;
	int nCols;
	int nNZ;
	
	int *cols;
	int *row_ptrs;
	
	FloatType *vals;
	
	memPointer d_cols;
	memPointer d_vals;
	memPointer d_row_ptrs;

    // info about parallel procs
    int this_pe;
    int total_pes;

    int nRows_p;
    int nCols_p;

    protected:

    GPUHelper *GPU;

	public:

	SparseMatrix() : nRows(0), nCols(0), nNZ(0), nRows_p(0), nCols_p(0)
	{
		cols = NULL;
		row_ptrs = NULL;
		vals = NULL;

        d_cols = NULL;
        d_vals = NULL;
        d_row_ptrs = NULL;

        this_pe = -1;//roc_shmem_my_pe(handle); // this pe
        total_pes = -1;//roc_shmem_n_pes(handle);  // total number of pes

    }
	void AllocateSparseMatrix(MatrixMarketReader<FloatType> &mm_reader,
						InputFlags &in_flags,
						GPUHelper *gpu);
    void AllocateParallelSparseMatrix(MatrixMarketReader<FloatType> &mm_reader,
            InputFlags &in_flags);
	void ConvertFromCOOToCSR(Coordinate<FloatType> *coords,
						InputFlags &in_flags);

    void PopulateParallelSparseMatrix(MatrixMarketReader<FloatType> &mm_reader,
            InputFlags &in_flags);

    void FindStatsForParallelDecomposition();

    void Set_total_pes(int val){
        this->total_pes = val;
    }
    void Set_this_pe(int val){
        this->this_pe = val;
    }

    int Get_total_pes(){
        return this->total_pes;
    }
    int Get_this_pe(){
        return this->this_pe;
    }

    int GetNumRows_p() {return nRows_p;}

	int *GetCols() { return cols; }
	FloatType *GetVals() { return vals; }
	int *GetRowPtrs() { return row_ptrs; }

	memPointer GetDevCols() {return d_cols; }
	memPointer GetDevVals() {return d_vals; }
	memPointer GetDevRowPtrs() {return d_row_ptrs; }

	~SparseMatrix()
	{
		delete[] cols;
		delete[] vals;
		delete[] row_ptrs;

		GPU->FreeMem(d_cols);
		GPU->FreeMem(d_vals);
		GPU->FreeMem(d_row_ptrs);
	}
};

template<typename FloatType>
void SparseMatrix<FloatType>::AllocateSparseMatrix(MatrixMarketReader<FloatType> &mm_reader,
					InputFlags &in_flags,
					GPUHelper *gpu)
{
    GPU = gpu;
	nRows = mm_reader.GetNumRows();
	nCols = mm_reader.GetNumCols();
	nNZ = mm_reader.GetNumNonZeroes();
    printf("Allocating a sparse matrix with-- nRows: %d nCols: %d nNZ: %d\n", nRows, nCols, nNZ);

    assert(total_pes != -1);
    assert(this_pe != -1);

    #ifdef USE_RO_SHMEM
    if (nRows != nCols){
        fprintf(stderr, "RO_SHMEM port requires the global matrix to be "
                "square!\n");
        exit(-1);
    }
    #endif

	cols = new int[nNZ];
    if (cols == NULL)
    {
        fprintf(stderr, "Failed to allocate host-side cols array !\n");
        exit(-1);
    }
	vals = new FloatType[nNZ];
    if (vals == NULL)
    {
        fprintf(stderr, "Failed to allocate host-side vals array !\n");
        exit(-1);
    }
	row_ptrs = new int[nRows + 1];
    if (row_ptrs == NULL)
    {
        fprintf(stderr, "Failed to allocate host-side row_ptrs array !\n");
        exit(-1);
    }
}

template<typename FloatType>
bool CoordinateCompare(const Coordinate<FloatType> &c1, const Coordinate<FloatType> &c2)
{
	if(c1.x != c2.x)
		return (c1.x < c2.x);
	else
		return (c1.y < c2.y);
}

template<typename FloatType>
void SparseMatrix<FloatType>::ConvertFromCOOToCSR(Coordinate<FloatType> *coords,
					InputFlags &in_flags)
{
	std::sort(coords, coords + nNZ, CoordinateCompare<FloatType>);

	int current_row = 1;
    bool has_seen_diagonal = false;
	row_ptrs[0] = 0;
	for (int i = 0; i < nNZ; i++)
	{
		cols[i] = coords[i].y;
		vals[i] = coords[i].val;
        //fprintf(stderr,"Row %d Col %d Val %lf (cur_row: %d)\n", coords[i].x, coords[i].y, coords[i].val, current_row-1);

		while(coords[i].x >= current_row)
        {
            // We've reached the end of a row. Did we see a diagonal?
            // If not, the triangular solve will be underconstrained.
            if (!has_seen_diagonal)
            {
                fprintf(stderr, "ERROR Converting the COO to CSR.\n");
                fprintf(stderr, "\tMissing diagonal on row %d\n", current_row-1);
                exit(-1);
            }
            has_seen_diagonal = false;
			row_ptrs[current_row] = i;
            current_row++;
        }
        if (coords[i].x == coords[i].y)
            has_seen_diagonal = true;

	}
	row_ptrs[current_row++] = nNZ;
    while (current_row <= nRows)
    {
        if (!has_seen_diagonal)
        {
            fprintf(stderr, "ERROR Converting the COO to CSR.\n");
            fprintf(stderr, "\tNo values on row %d, so no diagonal.\n", current_row-1);
            exit(-1);
        }
        has_seen_diagonal = false;
        row_ptrs[current_row++] = nNZ;
    }
}

template<typename FloatType>
void SparseMatrix<FloatType>::AllocateParallelSparseMatrix(MatrixMarketReader<FloatType> &mm_reader,
        InputFlags &in_flags)
{
    d_cols = GPU->AllocateMem("cols", nNZ*sizeof(int), 0, NULL);
    d_vals = GPU->AllocateMem("vals", nNZ*sizeof(FloatType), 0, NULL);
    d_row_ptrs = GPU->AllocateMem("row_ptrs", (nRows+1)*sizeof(int), 0, NULL);
}

template<typename FloatType>
void SparseMatrix<FloatType>::FindStatsForParallelDecomposition()
{

    assert(SPTS_BLOCK_SIZE % 64 == 0);

    // Rows left over in the potentially partial final block
    int left_over_last_block = nRows % SPTS_BLOCK_SIZE;
    printf("%d: lolb %d\n", this_pe, left_over_last_block);
    // Number of complete blocks, not including any partial block at the end
    int total_blocks = nRows / SPTS_BLOCK_SIZE;
    printf("%d: totb %d\n", this_pe, total_blocks);

    // Everyone has at least this many rows
    nRows_p = (total_blocks / total_pes) * SPTS_BLOCK_SIZE;
    printf("%d: initial nRows_p %d\n", this_pe, nRows_p);

    // Last cycle might not assign to all PEs
    int straggler_blocks = total_blocks % total_pes;
    if (this_pe < straggler_blocks)
        nRows_p += SPTS_BLOCK_SIZE;
    printf("%d: straggler nRows_p %d\n", this_pe, nRows_p);
    
    // Last block of last cycle might have less than SPTS_BLOCK_SIZE rows
    if (left_over_last_block) {
        int final_pe = ((total_blocks + 1) % total_pes) - 1;
        if (final_pe == -1)
            final_pe = total_pes - 1;
        if (this_pe == final_pe)
            nRows_p += left_over_last_block;
    }
    printf("%d: final nRows_p %d\n", this_pe, nRows_p);

    if (nRows_p <= 0) {
        fprintf(stderr, "Block Size %d too small for input row size %d with "
                "%d number of nodes.  Please decrease the block size or "
                "decrease the number of nodes\n", SPTS_BLOCK_SIZE, nRows,
                total_pes);
        exit(-1);
    }

    // print to check!
    printf("\nPE: %d total_rows: %d my_rows: %d\n", this_pe, nRows, nRows_p);

    nCols_p = nCols; // 1D decomposition
}

#endif
