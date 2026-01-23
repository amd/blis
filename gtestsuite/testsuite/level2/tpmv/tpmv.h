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

#pragma once

#include "blis.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

/**
 * @brief Performs the operation
  *    x := transa(A) * x
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     transa specifies the form of op( A ) to be used in matrix multiplication
 * @param[in]     diaga  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in,out] xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.

 */

template<typename T>
static void tpmv_( char uploa, char transa, char diaga, gtint_t n,
                         T *ap, T *xp, gtint_t incx )
{
    if constexpr (std::is_same<T, float>::value)
        stpmv_( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dtpmv_( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctpmv_( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztpmv_( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else
        throw std::runtime_error("Error in testsuite/level2/tpmv.h: Invalid typename in tpmv_().");
}

template<typename T>
static void tpmv_blis_impl( char uploa, char transa, char diaga, gtint_t n,
                         T *ap, T *xp, gtint_t incx )
{
    if constexpr (std::is_same<T, float>::value)
        stpmv_blis_impl( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dtpmv_blis_impl( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctpmv_blis_impl( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztpmv_blis_impl( &uploa, &transa, &diaga, &n, ap, xp, &incx );
    else
        throw std::runtime_error("Error in testsuite/level2/tpmv.h: Invalid typename in tpmv_blis_impl().");
}

template<typename T>
static void cblas_tpmv( char storage, char uploa, char transa, char diaga,
                      gtint_t n, T *ap, T *xp, gtint_t incx )
{

    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uploa;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_DIAG cblas_diaga;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_uplo( uploa, &cblas_uploa );
    testinghelpers::char_to_cblas_trans( transa, &cblas_transa );
    testinghelpers::char_to_cblas_diag( diaga, &cblas_diaga );

    if constexpr (std::is_same<T, float>::value)
        cblas_stpmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dtpmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_ctpmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_ztpmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, xp, incx );
    else
        throw std::runtime_error("Error in testsuite/level2/tpmv.h: Invalid typename in cblas_tpmv().");
}

template<typename T>
static void tpmv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *ap, T *xp, gtint_t incx )
{
#if (defined TEST_BLAS_LIKE || defined TEST_CBLAS)
    T one;
    testinghelpers::initone(one);
#endif

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    uploa = static_cast<char>(std::toupper(static_cast<unsigned char>(uploa)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    diaga = static_cast<char>(std::toupper(static_cast<unsigned char>(diaga)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char uploa_cpy = uploa;
    char transa_cpy = transa;
    char diaga_cpy = diaga;
    gtint_t n_cpy = n;
    gtint_t incx_cpy = incx;

    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = ( n * ( n + 1 ) ) / 2;
    if (ap && size_ap > 0)
    {
        ap_cpy = new T[size_ap];
        memcpy( ap_cpy, ap, size_ap * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if(( storage == 'c' || storage == 'C' ))
        tpmv_<T>( uploa, transa, diaga, n, ap, xp, incx );
    else
        throw std::runtime_error("Error in testsuite/level2/tpmv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if(( storage == 'c' || storage == 'C' ))
        tpmv_blis_impl<T>( uploa, transa, diaga, n, ap, xp, incx );
    else
        throw std::runtime_error("Error in testsuite/level2/tpmv.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_tpmv<T>( storage, uploa, transa, diaga, n, ap, xp, incx );
#else
    throw std::runtime_error("Error in testsuite/level2/tpmv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "uploa", uploa, uploa_cpy );
    computediff<char>( "transa", transa, transa_cpy );
    computediff<char>( "diaga", diaga, diaga_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        computediff<T>( "A", size_ap, ap, ap_cpy, 1, true );
        delete[] ap_cpy;
    }
#endif
}
