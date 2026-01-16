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

#include "tpsv.h"
#include "level2/ref_tpsv.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include "common/testing_helpers.h"
#include "inc/data_pool.h"

template<typename T>
void test_tpsv(
                char storage,
                char uploa,
                char transa,
                char diaga,
                gtint_t n,
                gtint_t incx,
                double thresh,
                bool is_evt_test = false,
                T evt_x = T{0},
                T evt_a = T{0}
             )
{
    using RT = typename testinghelpers::type_info<T>::real_type;

    dim_t len_a = ( n * ( n + 1 ) ) / 2;

    //----------------------------------------------------------
    //        Initialize matrices  with random numbers.
    //----------------------------------------------------------

    // Set index based on n and incx to get varied data
    get_pool<T, 0, 1>().set_index(n, incx);
    get_pool<T, 1, 3>().set_index(n, incx);
    std::vector<T> a = get_pool<T, 0, 1>().get_random_vector(len_a, 1);
    // This will use the index from where it was
    std::vector<T> x = get_pool<T, 1, 3>().get_random_vector(n, incx);

    // Make A matrix diagonal dominant to make sure that algorithm doesn't diverge
    // This makes sure that the tpsv problem is solvable

    // Packed storage (1-based) accessed as AP(i + j*(j-1)/2) for upper ('U') or AP(i + (2*n - j)*(j-1)/2)

    if ( uploa == 'l' || uploa == 'L' )
    {
        for ( dim_t a_dim = 1; a_dim <= n; ++a_dim )
        {
            a[ a_dim + (2*n - a_dim)*(a_dim-1)/2 - 1 ] = a[ a_dim + (2*n - a_dim)*(a_dim-1)/2 - 1 ] + T{RT(n)};
        }
    }
    else
    {
        for ( dim_t a_dim = 1; a_dim <= n; ++a_dim )
        {
            a[ a_dim + a_dim*(a_dim-1)/2 - 1 ] = a[ a_dim + a_dim*(a_dim-1)/2 - 1 ] + T{RT(n)};
        }
    }

    // add extreme values to the X vector
    if ( is_evt_test )
    {
        x[ (rand() % n) * std::abs(incx) ] = evt_x;
    }

    // add extreme values to the A matrix
    if ( is_evt_test )
    {
        dim_t n_idx = rand() % n;
        dim_t m_idx = (std::max)((dim_t)0, n_idx - 1);

        // Change to 1-based indexing
        n_idx = n_idx + 1;
        m_idx = m_idx + 1;

        if ( uploa == 'l' || uploa == 'L' )
        {
            a[ m_idx + n_idx*(n_idx-1)/2 -1 ]  = evt_a;
            a[ n_idx + m_idx*(m_idx-1)/2 -1 ]  = evt_a;
        }
        else
        {
            a[ m_idx + (2*n - n_idx)*(n_idx-1)/2 -1 ]  = evt_a;
            a[ n_idx + (2*n - m_idx)*(m_idx-1)/2 -1 ]  = evt_a;
        }
    }

    // skipped making A triangular
    // A matrix being a non triangular matrix could be a better test
    // because we are expected to read only from the upper or lower triangular
    // part of the data, contents of the rest of the matrix should not change the
    // result.
    // testinghelpers::make_triangular<T>( storage, uploa, n, a, lda );

    // Create a copy of x so that we can check reference results.
    std::vector<T> x_ref(x);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    tpsv<T>( storage, uploa, transa, diaga, n, a.data(), x.data(), incx );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_tpsv<T>( storage, uploa, transa, diaga, n, a.data(), x_ref.data(), incx );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "x", n, x.data(), x_ref.data(), incx, thresh, is_evt_test );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class tpsvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t>> str) const {
        char storage     = std::get<0>(str.param);
        char uploa       = std::get<1>(str.param);
        char transa      = std::get<2>(str.param);
        char diaga       = std::get<3>(str.param);
        gtint_t n        = std::get<4>(str.param);
        gtint_t incx     = std::get<5>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploa_" + std::string(&uploa, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_diaga_" + std::string(&diaga, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        return str_name;
    }
};

template <typename T>
class tpsvEVTPrint
{
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,T,T>> str) const {
        char storage    = std::get<0>(str.param);
        char uploa      = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char diaga      = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t incx    = std::get<5>(str.param);
        T xexval        = std::get<6>(str.param);
        T aexval        = std::get<7>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploa_" + std::string(&uploa, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_diaga_" + std::string(&diaga, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name    = str_name + "_ex_x_" + testinghelpers::get_value_string(xexval);
        str_name    = str_name + "_ex_a_" + testinghelpers::get_value_string(aexval);
        return str_name;
    }
};
