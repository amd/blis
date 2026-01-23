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
 * @brief Performs the operation:
 *        y := alpha*A*x + beta*y
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number of rows of the matrix A
 * @param[in]     k      specifies the number of super-diagonals of the matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 * @param[in]     xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] yp     specifies pointer which points to the first element of yp
 * @param[in]     incy   specifies storage spacing between elements of yp.
 */

template<typename T>
static void sbmv_( char uploa, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
                    T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    if constexpr (std::is_same<T, float>::value)
        ssbmv_( &uploa, &n, &k, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dsbmv_( &uploa, &n, &k, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else
        throw std::runtime_error("Error in testsuite/level2/sbmv.h: Invalid typename in sbmv_().");
}

template<typename T>
static void sbmv_blis_impl( char uploa, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
                    T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    if constexpr (std::is_same<T, float>::value)
        ssbmv_blis_impl( &uploa, &n, &k, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dsbmv_blis_impl( &uploa, &n, &k, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else
        throw std::runtime_error("Error in testsuite/level2/sbmv.h: Invalid typename in sbmv_blis_impl().");
}

template<typename T>
static void cblas_sbmv( char storage, char uploa, gtint_t n, gtint_t k, T* alpha,
    T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uplo;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_uplo( uploa, &cblas_uplo );

    if constexpr (std::is_same<T, float>::value)
        cblas_ssbmv( cblas_order, cblas_uplo, n, k, *alpha, ap, lda, xp, incx, *beta, yp, incy );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dsbmv( cblas_order, cblas_uplo, n, k, *alpha, ap, lda, xp, incx, *beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/sbmv.h: Invalid typename in cblas_sbmv().");
}

template<typename T>
static void sbmv( char storage, char uploa, gtint_t n, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp,
    gtint_t incy )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    uploa = static_cast<char>(std::toupper(static_cast<unsigned char>(uploa)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char uploa_cpy = uploa;
    gtint_t n_cpy = n;
    gtint_t k_cpy = k;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t incx_cpy = incx;
    T* beta_cpy = beta;
    gtint_t incy_cpy = incy;

    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = testinghelpers::matsize( 'c', 'n', k+1, n, lda );
    if (ap && size_ap > 0)
    {
        ap_cpy = new T[size_ap];
        memcpy( ap_cpy, ap, size_ap * sizeof( T ) );
    }
    T* xp_cpy = nullptr;
    gtint_t size_xp = testinghelpers::buff_dim( n, incx );
    if (xp && size_xp > 0)
    {
        xp_cpy = new T[size_xp];
        memcpy( xp_cpy, xp, size_xp * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        sbmv_<T>( uploa, n, k, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/sbmv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        sbmv_blis_impl<T>( uploa, n, k, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/sbmv.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_sbmv<T>( storage, uploa, n, k, alpha, ap, lda, xp, incx, beta, yp, incy );
#else
    throw std::runtime_error("Error in testsuite/level2/sbmv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "uploa", uploa, uploa_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "k", k, k_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy, true );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    if (beta) computediff<T>( "beta", *beta, *beta_cpy, true );
    computediff<gtint_t>( "incy", incy, incy_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        computediff<T>( "A", 'c', k+1, n, ap, ap_cpy, lda, true );
        delete[] ap_cpy;
    }

    if (xp && size_xp > 0)
    {
        computediff<T>( "x", n, xp, xp_cpy, incx, true );
        delete[] xp_cpy;
    }
#endif
}
