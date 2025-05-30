/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
 * @brief Finds the index of the first element that has the maximum absolute value.
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 *
 * If n < 1 or incx <= 0, return 0.
 * If n == 1, return 1(BLAS) or 0(CBLAS).
 */

template<typename T>
static gtint_t amaxv_(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
        idx = isamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        idx = idamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        idx = icamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        idx = izamax_( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in amaxv_().");

    return idx;
}

template<typename T>
static gtint_t amaxv_blis_impl(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
        idx = isamax_blis_impl( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        idx = idamax_blis_impl( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        idx = icamax_blis_impl( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        idx = izamax_blis_impl( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in amaxv_blis_impl().");

    return idx;
}

template<typename T>
static gtint_t cblas_amaxv(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
      idx = cblas_isamax( n, x, incx );
    else if constexpr (std::is_same<T, double>::value)
      idx = cblas_idamax( n, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
      idx = cblas_icamax( n, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
      idx = cblas_izamax( n, x, incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in cblas_amaxv().");

    return idx;
}

template<typename T>
static gtint_t typed_amaxv(gtint_t n, T* x, gtint_t incx)
{
    gtint_t idx = 0;
    if constexpr (std::is_same<T, float>::value)
        bli_samaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, double>::value)
        bli_damaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_camaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zamaxv( n, x, incx, &idx );
    else
        throw std::runtime_error("Error in testsuite/level1/amaxddv.h: Invalid typename in typed_amaxv().");

    return idx;
}

template<typename T>
static gtint_t amaxv(gtint_t n, T* x, gtint_t incx)
{

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    gtint_t n_cpy = n;
    gtint_t incx_cpy = incx;

    // Create copy of input arrays so we can check that they are not altered.
    T* x_cpy = nullptr;
    gtint_t size_x = testinghelpers::buff_dim( n, incx );
    if (x && size_x > 0)
    {
        x_cpy = new T[size_x];
        memcpy( x_cpy, x, size_x * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS_LIKE
    // Since we would be comparing against CBLAS which is 0-based and BLAS
    // which is 1-based, we need decrement the result of BLAS call by 1.
    // Exception is IIT tests which return 0 in both BLAS and CBLAS.

  #ifdef TEST_BLAS
    gtint_t idx = amaxv_<T>(n, x, incx);
  #elif TEST_BLAS_BLIS_IMPL
    gtint_t idx = amaxv_blis_impl<T>(n, x, incx);
  #endif
    if ( n < 1 || incx <= 0 )
        return idx;
    else
        return idx - 1;

#elif TEST_CBLAS
    return cblas_amaxv<T>(n, x, incx);
#elif TEST_BLIS_TYPED
    return typed_amaxv(n, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/amaxv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (x && size_x > 0)
    {
        computediff<T>( "x", n, x, x_cpy, incx, true );
        delete[] x_cpy;
    }
#endif
}
