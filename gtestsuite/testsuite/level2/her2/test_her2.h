/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "her2.h"
#include "level2/ref_her2.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_her2( char storage, char uploa, char conjx, char conjy, gtint_t n,
    T alpha, gtint_t incx, gtint_t incy, gtint_t lda_inc, double thresh )
{
    // Compute the leading dimensions of a.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', n, n, lda_inc );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, storage, 'n', n, n, lda );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, n, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -2, 5, n, incy );

    testinghelpers::make_herm<T>( storage, uploa, n, a.data(), lda );
    testinghelpers::make_triangular<T>( storage, uploa, n, a.data(), lda );

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    her2<T>( storage, uploa, conjx, conjy, n, &alpha, x.data(), incx,
                                              y.data(), incy, a.data(), lda );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_her2<T>( storage, uploa, conjx, conjy, n, &alpha,
                           x.data(), incx, y.data(), incy, a_ref.data(), lda );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "A", storage, n, n, a.data(), a_ref.data(), lda, thresh );
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class her2GenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,T,gtint_t,gtint_t,gtint_t>> str) const {
        char sfm       = std::get<0>(str.param);
        char uploa     = std::get<1>(str.param);
        char conjx     = std::get<2>(str.param);
        char conjy     = std::get<3>(str.param);
        gtint_t n      = std::get<4>(str.param);
        T alpha = std::get<5>(str.param);
        gtint_t incx   = std::get<6>(str.param);
        gtint_t incy   = std::get<7>(str.param);
        gtint_t ld_inc = std::get<8>(str.param);

        std::string str_name = API_PRINT;
        str_name    = str_name + "_" + sfm;
        str_name    = str_name + "_" + uploa+conjx+conjy;
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name    = str_name + "_" + std::to_string(ld_inc);
        return str_name;
    }
};
