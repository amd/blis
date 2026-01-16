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

#include "spr.h"
#include "level2/ref_spr.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include "inc/data_pool.h"

template<typename T>
void test_spr( char storage, char uploa, gtint_t n, T alpha,
               gtint_t incx, double thresh )
{
    dim_t len_a = ( n * ( n + 1 ) ) / 2;

    //----------------------------------------------------------
    //        Initialize matrices with random integer numbers.
    //----------------------------------------------------------

    // Set index based on n and incx to get varied data
    get_pool<T, -2, 5>().set_index(n, incx);
    get_pool<T, -3, 3>().set_index(n, incx);
    std::vector<T> a = get_pool<T, -2, 5>().get_random_vector(len_a, 1);
    // This will use the index from where it was
    std::vector<T> x = get_pool<T, -3, 3>().get_random_vector(n, incx);

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    spr<T>( storage, uploa, n, &alpha, x.data(), incx, a.data() );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_spr<T>( storage, uploa, n, alpha,
                                      x.data(), incx, a_ref.data() );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, thresh );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class sprGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,T,gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char uploa      = std::get<1>(str.param);
        gtint_t n       = std::get<2>(str.param);
        T alpha         = std::get<3>(str.param);
        gtint_t incx    = std::get<4>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploa_" + std::string(&uploa, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        return str_name;
    }
};
