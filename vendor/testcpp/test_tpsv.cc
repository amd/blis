/*

   BLISPP
   C++ test driver for BLIS CPP gemm routine and reference blis gemm routine.

   Copyright (C) 2019 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include <complex>
#include <iostream>
#include "blis.hh"
#include "test.hh"

using namespace blis;
using namespace std;
#define N 2
/*
 * Test application assumes matrices to be column major, non-transposed
 */
inline int logical_index(int i, int n, int increment)
{
  return increment < 0 ? (n - 1 - i) * (-increment) : i * increment;
}

inline int lower_packed_index(int n, int row, int column)
{
  return column * n - column * (column - 1) / 2 + row - column;
}

template< typename T >
void ref_tpsv(int n, const T* A, T* X, int incX)
{
  for (int i = 0; i < n; ++i)
  {
    const int x_index = logical_index(i, n, incX);
    T value = X[x_index];
    for (int j = 0; j < i; ++j)
      value -= A[lower_packed_index(n, i, j)] *
           X[logical_index(j, n, incX)];
    X[x_index] = value / A[lower_packed_index(n, i, i)];
  }
}

template< typename T >
void test_tpsv(  ) 
{
 int n;
 n = N;

 T A[] = { -0.381f, 0.53f, 0.451f };
 T X[] = { 0.144f, 0.032f };
 int incX = -1;
 T X_ref[] = { 0.144f, 0.032f };

 ref_tpsv(n, A, X_ref, incX);
   
   

	blis::tpsv(
	    CblasColMajor,
	    CblasLower,
	    CblasNoTrans,
	    CblasNonUnit,
            n,
            A,
            X,
            incX
            );


  const double threshold = test_threshold<T>(2.0 * n,
                                              18.0, 70.0, 1.5, 3.1,
                                              9.0, 1.0, 1.0, 1.0);
  if(computeErrorV(n, X, X_ref, threshold)==1)
	    printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
    else
	    printf("%s TEST PASS\n" , __PRETTY_FUNCTION__); 


}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_tpsv<double>( );
    test_tpsv<float>( );
    test_tpsv<complex<float>>( );
    test_tpsv<complex<double>>( );
    return 0;

}	
