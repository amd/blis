/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/*===================================================================
 * File Name :  test_dtl.c
 * 
 * Description : Unit test cases for dtl.
 * 
 *==================================================================*/

#if 0 // Disable this for normal build.

#include "aocltpdef.h"
#include "aocldtl.h"

int aocl_allocate(double**A, double** B, double** C, int N)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO);

	*A = (double*)malloc(sizeof(double) * N);
	if (*A == NULL)
	{
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_MAJOR, "Error allocating memory to A");
		return 1;
	}

	*B = (double*)malloc(sizeof(double) * N);
	if (*B == NULL)
	{
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_MAJOR, "Error allocating memory to B");
		return 1;
	}

	*C = (double*)malloc(sizeof(double) * N);
	if (*C == NULL)
	{
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_MAJOR, "Error allocating memory to C");
		return 1;
	}

	for (int i = 0; i < N; i++)
	{
		(*A)[i] = (double)((i + 1) * 1.0);
		(*B)[i] = (double)((i - 1) * 1.0);
		(*C)[i] = (double)((i) * 1.0);
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, " aocl_allocate()");
	return 0;
}

void sumV(double* A, double* B, double* C, int N)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO);
	if ((A == NULL) || (B == NULL) || (C == NULL))
	{
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_MAJOR, "Invalid Pointers");
		return;
	}
	for (int i = 0; i < N; i++)
	{
		C[i] += A[i] + B[i];
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
}

int main(void)
{
	int status = 0;
	double* A = NULL;
	double* B = NULL;
	double* C = NULL;

	printf("Initializing\n");
	AOCL_DTL_INITIALIZE(AOCL_DTL_LEVEL_ALL);

	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO);

	status = aocl_allocate(&A, &B, &C, 120);
	if (status != 0)
	{
		printf("Error allocating memory\n");

		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_CRITICAL, "Error in function aocl_allocate()");
		AOCL_DTL_UNINITIALIZE();
		exit(1);
	}

	sumV(A, B, C, 120);
	
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	AOCL_DTL_UNINITIALIZE();

	return 0;
}
#endif
