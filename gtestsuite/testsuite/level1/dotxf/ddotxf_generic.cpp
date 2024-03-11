/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include <gtest/gtest.h>
#include "test_dotxf.h"

class ddotxffGenericTest :
        public ::testing::TestWithParam<std::tuple<char,    // conj_x
                                                   char,    // conj_a
                                                   gtint_t, // m
                                                   gtint_t, // b
                                                   double,  // alpha
                                                   gtint_t, // inca
                                                   gtint_t, // lda
                                                   gtint_t, // incx
                                                   double,  // beta
                                                   gtint_t  // incy
                                                   >> {};
// Tests using random integers as vector elements.
TEST_P( ddotxffGenericTest, FunctionalTest )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    char conj_x = std::get<0>(GetParam());
    conj_t conjx;
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    char conj_a = std::get<1>(GetParam());
    conj_t conja;
    testinghelpers::char_to_blis_conj( conj_a, &conja );
    gint_t m = std::get<2>(GetParam());
    gint_t b = std::get<3>(GetParam());
    T alpha = std::get<4>(GetParam());

    // stride size for x:
    gtint_t inca = std::get<5>(GetParam());
    // stride size for y:
    gtint_t lda = std::get<6>(GetParam());
    gtint_t incx = std::get<7>(GetParam());
    T beta = std::get<8>(GetParam());
    gtint_t incy = std::get<9>(GetParam());

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_dotxf<T>( conjx, conja, m, b, &alpha, inca, lda, incx, &beta, incy );
}

// Test-case logger : Used to print the test-case details
class ddotxfGenericTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   double,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   double,
                                                   gtint_t>> str) const {
        char conja    = std::get<0>(str.param);
        char conjx    = std::get<1>(str.param);
        gtint_t m     = std::get<2>(str.param);
        gtint_t b  = std::get<3>(str.param);
        double alpha  = std::get<4>(str.param);
        gtint_t incx     = std::get<7>(str.param);
        double beta  = std::get<8>(str.param);
        gtint_t incy  = std::get<9>(str.param);

        std::string str_name = "bli_";

        str_name += ( conja == 'n' )? "_conja_n" : "_conja_t";
        str_name += ( conjx == 'n' )? "_conjx_n" : "_conjx_t";
        str_name += "_m" + std::to_string(m);
        str_name += "_b" + std::to_string(b);
        std::string alpha_str = ( alpha >= 0) ? std::to_string(int(alpha)) : "m" + std::to_string(int(std::abs(alpha)));
        str_name = str_name + "_alpha" + alpha_str;
        std::string beta_str = ( beta >= 0) ? std::to_string(int(beta)) : "m" + std::to_string(int(std::abs(beta)));
        str_name = str_name + "_beta" + beta_str;
        std::string incx_str = ( incx >= 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx" + incx_str;
        std::string incy_str = ( incy >= 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_incy" + incy_str;
        return str_name;
    }
};

// Black box testing for generic and main use of ddotxf.
INSTANTIATE_TEST_SUITE_P(
        FunctionalTest,
        ddotxffGenericTest,
        ::testing::Combine(
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Values('n'),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of matrix
            ::testing::Range(gtint_t(6), gtint_t(10), 1),                    // b size of matrix
            ::testing::Values(double(0.0), double(1.0), double(2.3)),        // alpha
            ::testing::Values(gtint_t(0)),                                // lda increament
            ::testing::Values(gtint_t(1)),                                   // stride size for a
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(double(0.0), double(1.0)),                     // beta
            ::testing::Values(gtint_t(1))                                    // stride size for y
        ),
        ::ddotxfGenericTestPrint()
    );

