/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"
#include "level2/ref_spmv.h"

/*
 * ==========================================================================
 * spmv performs the matrix-vector operation
 *    y := alpha*A*x + beta*y
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n packed symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_spmv( char storage, char uploa, gtint_t n,
    T *alpha, T *ap, T *xp, gtint_t incx, T *beta,
    T *yp, gtint_t incy )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uploa;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_uplo( uploa, &cblas_uploa );

    typedef void (*Fptr_ref_cblas_spmv)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                         const T, const T*, const T*, f77_int, const T, T*, f77_int);

    Fptr_ref_cblas_spmv ref_cblas_spmv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_spmv = (Fptr_ref_cblas_spmv)refCBLASModule.loadSymbol("cblas_sspmv");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_spmv = (Fptr_ref_cblas_spmv)refCBLASModule.loadSymbol("cblas_dspmv");
    }
    else
    {
      throw std::runtime_error("Error in ref_spmv.cpp: Invalid typename is passed to function template.");
    }
    if (!ref_cblas_spmv) {
        throw std::runtime_error("Error in ref_spmv.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_spmv( cblas_order, cblas_uploa, n, *alpha, ap, xp, incx, *beta, yp, incy );
}

// Explicit template instantiations
template void ref_spmv<float>( char, char, gtint_t, float *,
              float *, float *, gtint_t, float *, float *, gtint_t );
template void ref_spmv<double>( char, char, gtint_t, double *,
              double *, double *, gtint_t, double *, double *, gtint_t );

} //end of namespace testinghelpers
