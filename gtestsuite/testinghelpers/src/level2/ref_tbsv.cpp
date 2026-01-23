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
#include "level2/ref_tbsv.h"

/*
 * ==========================================================================
 * tbsv Solves a triangular system of equations with a single value for the
 *      right side
 *    b := inv(transa(A)) * x_orig
 * where b and x are n element vectors and A is an n by n unit, or non-unit,
 * upper or lower band triangular matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_tbsv( char storage, char uploa, char transa, char diaga,
    gtint_t n, gtint_t k, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uploa;
    enum CBLAS_TRANSPOSE cblas_trans;
    enum CBLAS_DIAG cblas_diaga;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_uplo( uploa, &cblas_uploa );
    char_to_cblas_trans( transa, &cblas_trans );
    char_to_cblas_diag( diaga, &cblas_diaga );

    typedef void (*Fptr_ref_cblas_tbsv)( const CBLAS_ORDER, const CBLAS_UPLO,
                                         const CBLAS_TRANSPOSE, CBLAS_DIAG ,
                                         f77_int, f77_int, const T*, f77_int, T*, f77_int );
    Fptr_ref_cblas_tbsv ref_cblas_tbsv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_tbsv = (Fptr_ref_cblas_tbsv)refCBLASModule.loadSymbol("cblas_stbsv");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_tbsv = (Fptr_ref_cblas_tbsv)refCBLASModule.loadSymbol("cblas_dtbsv");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_tbsv = (Fptr_ref_cblas_tbsv)refCBLASModule.loadSymbol("cblas_ctbsv");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_tbsv = (Fptr_ref_cblas_tbsv)refCBLASModule.loadSymbol("cblas_ztbsv");
    }
    else
    {
      throw std::runtime_error("Error in ref_tbsv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_tbsv) {
        throw std::runtime_error("Error in ref_tbsv.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_tbsv( cblas_order, cblas_uploa, cblas_trans, cblas_diaga, n, k, ap, lda, xp, incx );
}

// Explicit template instantiations
template void ref_tbsv<float>( char , char , char , char , gtint_t , gtint_t ,
                              float *, gtint_t , float *, gtint_t );
template void ref_tbsv<double>( char , char , char , char , gtint_t , gtint_t ,
                              double *, gtint_t , double *, gtint_t );
template void ref_tbsv<scomplex>( char , char , char , char , gtint_t , gtint_t ,
                              scomplex *, gtint_t , scomplex *, gtint_t );
template void ref_tbsv<dcomplex>( char , char , char , char , gtint_t , gtint_t ,
                              dcomplex *, gtint_t , dcomplex *, gtint_t );

} //end of namespace testinghelpers
