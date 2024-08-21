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
#include "test_omatcopy.h"

class domatcopyGeneric :
        public ::testing::TestWithParam<std::tuple<char,        // storage
                                                   char,        // trans
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   double,      // alpha
                                                   gtint_t,     // lda_inc
                                                   gtint_t,     // ldb_inc
                                                   bool>> {};   // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(domatcopyGeneric);

// Tests using random numbers as vector elements.
TEST_P( domatcopyGeneric, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the storage format of the input matrices
    char storage = std::get<0>(GetParam());
    // denotes the trans value for the operation
    char trans = std::get<1>(GetParam());
    // m dimension
    gtint_t m = std::get<2>(GetParam());
    // n dimension
    gtint_t n = std::get<3>(GetParam());
    // alpha
    T alpha = std::get<4>(GetParam());
    // lda_inc for A
    gtint_t lda_inc = std::get<5>(GetParam());
    // ldb_inc for B
    gtint_t ldb_inc = std::get<6>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<7>(GetParam());

    double thresh = 0.0;
    // Set the threshold for the errors
    if( ( alpha != testinghelpers::ZERO<T>() || alpha != testinghelpers::ONE<T>() ) )
      thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_omatcopy<T>( storage, trans, m, n, alpha, lda_inc, ldb_inc, thresh, is_memory_test );
}

#if defined(TEST_BLAS_LIKE) && (defined(REF_IS_MKL) || defined(REF_IS_OPENBLAS))
// Black box testing for generic and main use of domatcopy.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        domatcopyGeneric,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format(currently only for BLAS testing)
            ::testing::Values('n', 't', 'r', 'c'),                           // trans(and/or conj) value
                                                                             // 'n' - no-transpose, 't' - transpose
                                                                             // 'r' - conjugate,    'c' - conjugate-transpose
            ::testing::Values(gtint_t(10), gtint_t(55), gtint_t(243)),       // m
            ::testing::Values(gtint_t(10), gtint_t(55), gtint_t(243)),       // n
            ::testing::Values(2.0, -3.0, 1.0, 0.0),                          // alpha
            ::testing::Values(gtint_t(0), gtint_t(25)),                      // increment of lda
            ::testing::Values(gtint_t(0), gtint_t(17)),                      // increment of ldb
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::omatcopyGenericPrint<double>()
    );
#endif
