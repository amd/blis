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

#include "gbmv.h"
#include "level2/ref_gbmv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include <stdexcept>
#include <algorithm>
#include "inc/data_pool.h"

template<typename T>
void test_gbmv( char storage, char transa, gtint_t m, gtint_t n, gtint_t kl, gtint_t ku,
                T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy,
                double thresh,
                bool is_evt_test = false, T a_exval = T{0}, T x_exval = T{0},
                T y_exval = T{0} )
{
    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', kl+ku+1, n, lda_inc );

    //----------------------------------------------------------
    //        Initialize matrices with random numbers.
    //----------------------------------------------------------

    // Set index based on n and incx to get varied data
    get_pool<T, 1, 5>().set_index(n, incx);
    get_pool<T, 1, 3>().set_index(n, incx);

    std::vector<T> a = get_pool<T, 1, 5>().get_random_matrix( storage, 'n', kl+ku+1, n, lda );

    // Get correct vector lengths.
    gtint_t lenx = ( testinghelpers::chknotrans( transa ) ) ? n : m ;
    gtint_t leny = ( testinghelpers::chknotrans( transa ) ) ? m : n ;

    std::vector<T> x = get_pool<T, 1, 3>().get_random_vector(lenx, incx);

    std::vector<T> y( testinghelpers::buff_dim(leny, incy) );
    if (beta != testinghelpers::ZERO<T>())
        get_pool<T, -3, 3>().randomgenerators( leny, incy, y.data() );
    else
    {
        // Vector Y should not be read, only set.
        testinghelpers::set_vector( leny, incy, y.data(), testinghelpers::aocl_extreme<T>() );
    }

    if ( is_evt_test )
    {
        // Add extreme value to A matrix
        dim_t n_idx = rand() % n;
        dim_t m_idx = rand() % (kl+ku+1);
        a[ m_idx + (n_idx * lda) ] = a_exval;

        // Add extreme value to x vector
        x[ (rand() % lenx) * std::abs(incx) ] = x_exval;

        // Add extreme value to y vector
        y[ (rand() % leny) * std::abs(incy) ] = y_exval;
    }

    // Copying the contents of y to y_ref
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gbmv<T>( storage, transa, m, n, kl, ku, &alpha, a.data(), lda,
             x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gbmv<T>( storage, transa, m, n, kl, ku, alpha, a.data(),
                                 lda, x.data(), incx, beta, y_ref.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", leny, y.data(), y_ref.data(), incy, thresh, is_evt_test );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class gbmvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,gtint_t,gtint_t,T,T,gtint_t,gtint_t,gtint_t>> str) const {
        char storage        = std::get<0>(str.param);
        char transa         = std::get<1>(str.param);
        gtint_t m           = std::get<2>(str.param);
        gtint_t n           = std::get<3>(str.param);
        gtint_t kl          = std::get<4>(str.param);
        gtint_t ku          = std::get<5>(str.param);
        T alpha             = std::get<6>(str.param);
        T beta              = std::get<7>(str.param);
        gtint_t incx        = std::get<8>(str.param);
        gtint_t incy        = std::get<9>(str.param);
        gtint_t lda_inc     = std::get<10>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_kl_" + std::to_string(kl);
        str_name += "_ku_" + std::to_string(ku);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        return str_name;
    }
};

template <typename T>
class gbmvEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,gtint_t,gtint_t,T,T,gtint_t,gtint_t,T,T,T,gtint_t>> str) const {
        char storage        = std::get<0>(str.param);
        char transa         = std::get<1>(str.param);
        gtint_t m           = std::get<2>(str.param);
        gtint_t n           = std::get<3>(str.param);
        gtint_t kl          = std::get<4>(str.param);
        gtint_t ku          = std::get<5>(str.param);
        T alpha             = std::get<6>(str.param);
        T beta              = std::get<7>(str.param);
        gtint_t incx        = std::get<8>(str.param);
        gtint_t incy        = std::get<9>(str.param);
        T a_exval           = std::get<10>(str.param);
        T x_exval           = std::get<11>(str.param);
        T y_exval           = std::get<12>(str.param);
        gtint_t lda_inc     = std::get<13>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_kl_" + std::to_string(kl);
        str_name += "_ku_" + std::to_string(ku);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name = str_name + "_a_exval_" + testinghelpers::get_value_string(a_exval);
        str_name = str_name + "_x_exval_" + testinghelpers::get_value_string(x_exval);
        str_name = str_name + "_y_exval_" + testinghelpers::get_value_string(y_exval);

        return str_name;
    }
};
