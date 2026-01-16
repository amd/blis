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
 *   y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
 *   y := alpha*A**H*x + beta*y,
 *
 * or y := beta * y + alpha * transa(A) * x
 *
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies the number of rows of the matrix A
 * @param[in]     n      specifies the number of columns of the matrix A
 * @param[in]     kl     specifies the number of sub-diagonals of the matrix A
 * @param[in]     ku     specifies the number of super-diagonals of the matrix A
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
static void gbmv_( char transa, gtint_t m, gtint_t n, gtint_t kl, gtint_t ku, T* alpha, T* ap, gtint_t lda,
  T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    if constexpr (std::is_same<T, float>::value)
        sgbmv_( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dgbmv_( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgbmv_( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgbmv_( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else
        throw std::runtime_error("Error in testsuite/level2/gbmv.h: Invalid typename in gbmv_().");
}

template<typename T>
static void gbmv_blis_impl( char transa, gtint_t m, gtint_t n, gtint_t kl, gtint_t ku, T* alpha, T* ap, gtint_t lda,
  T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    if constexpr (std::is_same<T, float>::value)
        sgbmv_blis_impl( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dgbmv_blis_impl( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgbmv_blis_impl( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgbmv_blis_impl( &transa, &m, &n, &kl, &ku, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else
        throw std::runtime_error("Error in testsuite/level2/gbmv.h: Invalid typename in gbmv_blis_impl().");
}

template<typename T>
static void cblas_gbmv( char storage, char trans, gtint_t m, gtint_t n, gtint_t kl, gtint_t ku, T* alpha,
    T* ap, gtint_t lda,  T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans;

    testinghelpers::char_to_cblas_order( storage, &cblas_order );
    testinghelpers::char_to_cblas_trans( trans, &cblas_trans );

    if constexpr (std::is_same<T, float>::value)
        cblas_sgbmv( cblas_order, cblas_trans, m, n, kl, ku, *alpha, ap, lda, xp, incx, *beta, yp, incy );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dgbmv( cblas_order, cblas_trans, m, n, kl, ku, *alpha, ap, lda, xp, incx, *beta, yp, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cgbmv( cblas_order, cblas_trans, m, n, kl, ku, alpha, ap, lda, xp, incx, beta, yp, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zgbmv( cblas_order, cblas_trans, m, n, kl, ku, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/gbmv.h: Invalid typename in cblas_gbmv().");
}

template<typename T>
static void gbmv( char storage, char trans, gtint_t m, gtint_t n,
    gtint_t kl, gtint_t ku, T* alpha, T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{

#ifdef TEST_UPPERCASE_ARGS
    storage = static_cast<char>(std::toupper(static_cast<unsigned char>(storage)));
    trans = static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char storage_cpy = storage;
    char trans_cpy = trans;
    gtint_t m_cpy = m;
    gtint_t n_cpy = n;
    gtint_t kl_cpy = kl;
    gtint_t ku_cpy = ku;
    T* alpha_cpy = alpha;
    gtint_t lda_cpy = lda;
    gtint_t incx_cpy = incx;
    T* beta_cpy = beta;
    gtint_t incy_cpy = incy;

    // Create copy of input arrays so we can check that they are not altered.
    T* ap_cpy = nullptr;
    gtint_t size_ap = testinghelpers::matsize( storage, trans, m, n, lda );
    if (ap && size_ap > 0)
    {
        ap_cpy = new T[size_ap];
        memcpy( ap_cpy, ap, size_ap * sizeof( T ) );
    }
    T* xp_cpy = nullptr;
    gtint_t size_xp;
    if(( trans == 'n' ) || ( trans == 'N' ))
        size_xp = testinghelpers::buff_dim( n, incx );
    else
        size_xp = testinghelpers::buff_dim( m, incx );
    if (xp && size_xp > 0)
    {
        xp_cpy = new T[size_xp];
        memcpy( xp_cpy, xp, size_xp * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        gbmv_<T>( trans, m, n, kl, ku, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/gbmv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_BLAS_BLIS_IMPL
    if( storage == 'c' || storage == 'C' )
        gbmv_blis_impl<T>( trans, m, n, kl, ku, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/gbmv.h: BLAS_BLIS_IMPL interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_gbmv<T>( storage, trans, m, n, kl, ku, alpha, ap, lda, xp, incx, beta, yp, incy );
#else
    throw std::runtime_error("Error in testsuite/level2/gbmv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "storage", storage, storage_cpy );
    computediff<char>( "trans", trans, trans_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "kl", kl, kl_cpy );
    computediff<gtint_t>( "ku", ku, ku_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    if (beta) computediff<T>( "beta", *beta, *beta_cpy );
    computediff<gtint_t>( "incy", incy, incy_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (ap && size_ap > 0)
    {
        if(( trans == 'n' ) || ( trans == 'N' ))
            computediff<T>( "A", storage, m, n, ap, ap_cpy, lda, true );
        else
            computediff<T>( "A", storage, n, m, ap, ap_cpy, lda, true );
        delete[] ap_cpy;
    }

    if (xp && size_xp > 0)
    {
        if(( trans == 'n' ) || ( trans == 'N' ))
            computediff<T>( "x", n, xp, xp_cpy, incx, true );
        else
            computediff<T>( "x", m, xp, xp_cpy, incx, true );
        delete[] xp_cpy;
    }
#endif
}
