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
#ifndef MatrixMarketReader_H
#define MatrixMarketReader_H
/*
Portions of this file include code provided by The National Institute of
Standards and Technology (NIST).  The code includes
macro definitions from mmio.h and is subject to the following disclaimer.

Software Disclaimer

NIST-developed software is provided by NIST as a public service. You may use,
copy and distribute copies of the software in any medium, provided that you
keep intact this entire notice. You may improve, modify and create derivative
works of the software or any portion of the software, and you may copy and
distribute such modifications or works. Modified works should carry a notice
stating that you changed the software and should note the date and nature of
any such change. Please explicitly acknowledge the National Institute of
Standards and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS" NIST MAKES NO WARRANTY
OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW,
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST
NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR
THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and
distributing the software and you assume all risks associated with its use,
including but not limited to the risks and costs of program errors, compliance
with applicable laws, damage to or loss of data, programs or equipment, and
the unavailability or interruption of operation. This software is not intended
to be used in any situation where a failure could cause risk of injury or
damage to property. The software developed by NIST employees is not subject
to copyright protection within the United States.
*/

#include <string>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <iostream>
#include "InputFlags.h"
#include <typeinfo>
#include "mmio.h"

// Class declaration

template<typename FloatType>
struct Coordinate {
	int x;
	int y;
	FloatType val;
};

template <typename FloatType>
class MatrixMarketReader
{
	char Typecode[4];
	int nNZ;
	int nRows;
	int nCols;
	int isSymmetric;
	int isDoubleMem;
	Coordinate<FloatType> *coords;
	bool *has_seen_diag;

	public:
	MatrixMarketReader() : nNZ(0), nRows(0), nCols(0), isSymmetric(0), isDoubleMem(0)
	{
        for (int i = 0; i < sizeof(Typecode); i++)
            Typecode[i] = '\0';
		coords = NULL;
	}
	bool MMReadFormat(const std::string &_filename, InputFlags &_in_flags);
	bool MMReadBanner(FILE *_infile);
	bool MMReadMtxCrdSize(FILE *_infile);
	void MMGenerateCOOFromFile(FILE *_infile, InputFlags &_in_flags);

	int GetNumRows() { return nRows; }
	int GetNumCols() { return nCols; }
	int GetNumNonZeroes() { return nNZ; }
	int GetSymmetric() { return isSymmetric; }

	char *GetTypecode() { return Typecode; }
	Coordinate<FloatType> *GetCoordinates() { return coords; }

	~MatrixMarketReader() 
	{
		delete[] coords;
	}
};

// Class definition

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadFormat(const std::string &filename, InputFlags &in_flags)
{
	FILE *mm_file = fopen(filename.c_str(), "r");
	if( mm_file == NULL)
	{
		printf("Cannot Open Matrix-Market File !\n");
		return 1;
	}

	int status = MMReadBanner(mm_file);
	if(status != 0)
	{
		printf("Error Reading Banner in Matrix-Market File !\n");
		return 1;
	}
    
	if(! mm_is_coordinate(Typecode)) 
	{printf(" only handling coordinate format\n"); return(1);}

	if(mm_is_complex(Typecode)) {
		printf("Error: cannot handle complex format\n");
		return (1);
	}

	if(mm_is_symmetric(Typecode))
		isSymmetric = 1;

	status = MMReadMtxCrdSize(mm_file);
	if(status != 0) { 
		printf("Error reading Matrix Market crd_size %d\n",status); 
		return(1);
    }

    if(mm_is_symmetric(Typecode))
        coords = new Coordinate<FloatType>[nNZ+nRows];
    else if (in_flags.GetValueBool("non_symmetric"))
        coords = new Coordinate<FloatType>[nNZ+nRows]; // This is too large, but oh well.
    else
    {
        fprintf(stderr, "Error: Input matrix is NOT symmetric. This will not work for SpTS.\n");
        return (1);
    }

    has_seen_diag = new bool[nRows];
    for (int i = 0; i < nRows; i++)
        has_seen_diag[i] = false;

    MMGenerateCOOFromFile(mm_file, in_flags);
    return 0;
}

template<typename FloatType>
void FillCoordData(char Typecode[],
				Coordinate<FloatType> *coords, 
				bool *has_seen_diag,
				int &actual_nnz, 
				int ir,
				int ic,
				FloatType val)
{
    int new_x = ir - 1;
    int new_y = ic - 1;
    if (new_y > new_x)
    {
        // Skip stuff in the upper diagonal
        // Just keep our lower diag.
        return;
    }
    if (new_y == new_x)
        has_seen_diag[new_x] = true;
    coords[actual_nnz].x = new_x;
    coords[actual_nnz].y = new_y;	
    coords[actual_nnz ++].val = val;
}

template<typename FloatType>
void FixupMissingDiags(char Typecode[],
                Coordinate<FloatType> *coords,
                int &actual_nnz,
                int nRows,
				bool *has_seen_diag,
                InputFlags &in_flags)
{
    for(int i = 0; i < nRows; i++)
    {
        if (has_seen_diag[i] == false)
        {
            coords[actual_nnz].x = i;
            coords[actual_nnz].y = i;
            coords[actual_nnz ++].val = 1.;
        }
    }
}

template<typename FloatType>
void MatrixMarketReader<FloatType>::MMGenerateCOOFromFile(FILE *infile,
										InputFlags &in_flags)
{
	int actual_nnz = 0;
	FloatType val;
	int ir, ic;

	int exp_zeroes = in_flags.GetValueBool("exp_zeroes");

	for(int i = 0; i < nNZ; i++)
	{
		if(mm_is_real(Typecode))
		{
			if(typeid(FloatType) == typeid(float))
				fscanf(infile, "%d %d %f\n", &ir, &ic, (float*)(&val));
			else if(typeid(FloatType) == typeid(double))
				fscanf(infile, "%d %d %lf\n", &ir, &ic, (double*)(&val));

			if(exp_zeroes == 0 && val == 0) 
				continue;
			else
				FillCoordData(Typecode, coords, has_seen_diag, actual_nnz, ir, ic, val);
		}
		else if (mm_is_integer(Typecode))
		{
            if(typeid(FloatType) == typeid(float))
                fscanf(infile, "%d %d %f\n", &ir, &ic, (float*)(&val));
            else if(typeid(FloatType) == typeid(double))
                fscanf(infile, "%d %d %lf\n", &ir, &ic, (double*)(&val));

			if(exp_zeroes == 0 && val == 0) 
				continue;
			else
				FillCoordData(Typecode, coords, has_seen_diag, actual_nnz, ir, ic, val);

		}
		else if(mm_is_pattern(Typecode))
		{
			fscanf(infile, "%d %d", &ir, &ic);
			//val = ((FloatType) MAX_RAND_VAL * (rand() / (RAND_MAX + 1.0)));
            val = 3.;
			
			if(exp_zeroes == 0 && val == 0) 
				continue;
			else
				FillCoordData(Typecode, coords, has_seen_diag, actual_nnz, ir, ic, val);
		}
	}
	FixupMissingDiags(Typecode, coords, actual_nnz, nRows, has_seen_diag, in_flags);
	nNZ = actual_nnz;
    printf("\n\tNNZ in the lower triangular and fixedup diagonal: %d\n", nNZ);
}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadBanner(FILE *infile)
{
	char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH]; 
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;

    mm_clear_typecode(Typecode);  

    if (fgets(line, MM_MAX_LINE_LENGTH, infile) == NULL) 
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);  
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(Typecode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(Typecode);
    else if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(Typecode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(Typecode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(Typecode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(Typecode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(Typecode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(Typecode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(Typecode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(Typecode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(Typecode);
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;

}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadMtxCrdSize(FILE *infile)
{
	char line[MM_MAX_LINE_LENGTH];
	int num_items_read;

	/* now continue scanning until you reach the end-of-comments */
	do 
	{
        if (fgets(line,MM_MAX_LINE_LENGTH, infile) == NULL) 
            return MM_PREMATURE_EOF;
	}while (line[0] == '%');

	/* line[] is either blank or has M,N, nz */
	if (sscanf(line, "%d %d %d", &nRows, &nCols, &nNZ) == 3)
		return 0;
	else
		do
		{ 
			num_items_read = fscanf(infile, "%d %d %d", &nRows, &nCols, &nNZ); 
			if (num_items_read == EOF) return MM_PREMATURE_EOF;
		}
		while (num_items_read != 3);

	return 0;
}
#endif // MatrixMarketReader_H
