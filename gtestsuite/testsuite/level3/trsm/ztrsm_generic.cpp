/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_trsm.h"

class ztrsmAPI :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   gtint_t,
                                                   gtint_t>> {};

TEST_P(ztrsmAPI, FunctionalTest)
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies matrix A appears left or right in
    // the matrix multiplication
    char side = std::get<1>(GetParam());
    // specifies upper or lower triangular part of A is used
    char uploa = std::get<2>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<3>(GetParam());
    // denotes whether matrix a in unit or non-unit diagonal
    char diaga = std::get<4>(GetParam());
    // matrix size m
    gtint_t m  = std::get<5>(GetParam());
    // matrix size n
    gtint_t n  = std::get<6>(GetParam());
    // specifies alpha value
    T alpha = std::get<7>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    gtint_t ldb_inc = std::get<9>(GetParam());

    // Set the threshold for the errors:
    double thresh = 1.5*(std::max)(m, n)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
}

class ztrsmPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, char, char, gtint_t, gtint_t, dcomplex, gtint_t, gtint_t>> str) const {
        char sfm        = std::get<0>(str.param);
        char side       = std::get<1>(str.param);
        char uploa      = std::get<2>(str.param);
        char transa     = std::get<3>(str.param);
        char diaga      = std::get<4>(str.param);
        gtint_t m       = std::get<5>(str.param);
        gtint_t n       = std::get<6>(str.param);
        dcomplex alpha  = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "blas_";
#elif TEST_CBLAS
        std::string str_name = "cblas_";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_";
#endif
        str_name = str_name + "_stor_" + sfm;
        str_name = str_name + "_side_" + side;
        str_name = str_name + "_uploa_" + uploa;
        str_name = str_name + "_transa_" + transa;
        str_name = str_name + "_diag_" + diaga;
        str_name = str_name + "_m_" + std::to_string(m);
        str_name = str_name + "_n_" + std::to_string(n);
        std::string alpha_str = testinghelpers::get_value_string(alpha);
        str_name = str_name + "_alpha_" + alpha_str;
        gtint_t mn;
        testinghelpers::set_dim_with_side( side, m, n, &mn );
        str_name = str_name + "_lda_" +
                   std::to_string(testinghelpers::get_leading_dimension( sfm, transa, mn, mn, lda_inc ));
        str_name = str_name + "_ldb_" +
                   std::to_string(testinghelpers::get_leading_dimension( sfm, 'n', m, n, ldb_inc ));
        return str_name;
    }
};

/**
 * @brief Test ZTRSM native path, which starts from size 501 for BLAS api
 *        and starts from size 0 for BLIS api.
 */
INSTANTIATE_TEST_SUITE_P(
        Native,
        ztrsmAPI,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','c','t'),                                  // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 53, 520),                                   // m
            ::testing::Values(1, 38, 511),                                   // n
            ::testing::Values(dcomplex{2.0,-1.0}),                           // alpha
            ::testing::Values(gtint_t(20)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(33))                                   // increment to the leading dim of b
        ),
        ::ztrsmPrint()
    );

/**
 * @brief Test ZTRSM small avx2 path all fringe cases
 *        Kernel size for avx2 small path is 4x3, testing in range of
 *        1 to 4 ensures all finge cases are being tested.
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2_fringe,
        ztrsmAPI,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(1), gtint_t(5), 1),                     // m
            ::testing::Range(gtint_t(1), gtint_t(5), 1),                     // n
            ::testing::Values(dcomplex{2.0,-3.4}),                           // alpha
            ::testing::Values(gtint_t(56)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(33))                                   // increment to the leading dim of b
        ),
        ::ztrsmPrint()
    );

/**
 * @brief Test ZTRSM small avx2 path, this code path is used in range 0 to 500
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2,
        ztrsmAPI,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(17, 500),                                      // m
            ::testing::Values(48, 500),                                      // n
            ::testing::Values(dcomplex{2.0,-3.4}),                           // alpha
            ::testing::Values(gtint_t(54)),                                  // increment to the leading dim of a
            ::testing::Values(gtint_t(37))                                   // increment to the leading dim of b
        ),
        ::ztrsmPrint()
    );

/**
 * @brief Test ZTRSM with differnt values of alpha
 *      code paths covered:
 *          TRSV              -> 1
 *          TRSM_AVX2_small   -> 3
 *          TRSM_NATIVE       -> 501
 */
INSTANTIATE_TEST_SUITE_P(
        Alpha,
        ztrsmAPI,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 3, 501),                                    // n
            ::testing::Values(1, 3, 501),                                    // m
            ::testing::Values(dcomplex{2.0, 0.0}, dcomplex{0.0, -10.0},
                              dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0}),      // alpha
            ::testing::Values(gtint_t(0), gtint_t(65)),                      // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(23))                       // increment to the leading dim of b
        ),
        ::ztrsmPrint()
    );