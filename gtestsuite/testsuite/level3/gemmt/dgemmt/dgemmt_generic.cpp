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

#include <gtest/gtest.h>
#include "level3/gemmt/test_gemmt.h"

class dgemmtGeneric :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   double,       // alpha
                                                   double,       // beta
                                                   gtint_t,      // lda_inc
                                                   gtint_t,      // ldb_inc
                                                   gtint_t,      // ldc_inc
                                                   bool>> {};    // is memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmtGeneric);

TEST_P( dgemmtGeneric, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies if the upper or lower triangular part of C is used
    char uplo = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // matrix size k
    gtint_t k  = std::get<5>(GetParam());
    // specifies alpha value
    T alpha = std::get<6>(GetParam());
    // specifies beta value
    T beta = std::get<7>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<8>(GetParam());
    gtint_t ldb_inc = std::get<9>(GetParam());
    gtint_t ldc_inc = std::get<10>(GetParam());
    bool is_mem_test = std::get<11>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite gemmt.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) &&
             (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 4.1;
#else
        double adj = 4.0;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemmt<T>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
#endif
}

#ifndef TEST_BLIS_TYPED
INSTANTIATE_TEST_SUITE_P(
        skinny_fringe_cases,
        dgemmtGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uplo u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','t'),                                      // transb
            ::testing::Range(gtint_t(1), gtint_t(30), 5),                    // n
            ::testing::Range(gtint_t(1), gtint_t(30), 5),                    // k
            ::testing::Values(1.0, 0.0, -2.4, 3.1),                          // alpha
            ::testing::Values(1.0, 0.0, -2.4, 3.1),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(153)),                     // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(122)),                     // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(195)),                     // increment to the leading dim of c
            ::testing::Values(true, false)                                   // is memory test
        ),
        ::gemmtMemGenericPrint<double>()
    );

INSTANTIATE_TEST_SUITE_P(
        skinny,
        dgemmtGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uplo u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','t'),                                      // transb
            ::testing::Values(35, 537, 799),                                 // n
            ::testing::Values(35, 537, 799),                                 // k
            ::testing::Values(1.0, 0.0, -2.4, 3.1),                          // alpha
            ::testing::Values(1.0, 0.0, -2.4, 3.1),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(153)),                     // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(122)),                     // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(195)),                     // increment to the leading dim of c
            ::testing::Values(true, false)                                   // is memory test
        ),
        ::gemmtMemGenericPrint<double>()
    );

INSTANTIATE_TEST_SUITE_P(
        Large,
        dgemmtGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uplo u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','t'),                                      // transb
            ::testing::Values(800, 1500),                                    // n
            ::testing::Values(800, 1500),                                    // k
            ::testing::Values(1.0, 0.0, -2.4, 3.1),                          // alpha
            ::testing::Values(1.0, 0.0, -2.4, 3.1),                          // beta
            ::testing::Values(gtint_t(0), gtint_t(153)),                     // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(122)),                     // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(195)),                     // increment to the leading dim of c
            ::testing::Values(true, false)                                   // is memory test
        ),
        ::gemmtMemGenericPrint<double>()
    );
#endif
