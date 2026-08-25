/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level3/trsm/test_trsm.h"

class ztrsmGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // side
                                                   char,          // uplo
                                                   char,          // transa
                                                   char,          // diaga
                                                   gtint_t,       // m
                                                   gtint_t,       // n
                                                   dcomplex,      // alpha
                                                   gtint_t>> {};  // inc to the lda and ldb

class ztrsmGeneric2 :
        public ::testing::TestWithParam<std::tuple<char,          // storage format
                                                   char,          // side
                                                   char,          // uplo
                                                   char,          // transa
                                                   char,          // diaga
                                                   gtint_t,       // m
                                                   gtint_t,       // n
                                                   dcomplex,      // alpha
                                                   gtint_t,       // lda_inc
                                                   gtint_t>> {};  // ldb_inc

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ztrsmGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ztrsmGeneric2);

using T = dcomplex;
void ztrsmGeneric( char storage, char side, char uploa, char transa, char diaga, gtint_t m, gtint_t n,
                   T alpha, gtint_t lda_inc, gtint_t ldb_inc )
{
    // Set the threshold for the errors:
    // Check gtestsuite trsm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multiplier for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (m == 0 || n == 0 || alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else
        if ( side == 'l' || side == 'L' )
            thresh = 3*m*testinghelpers::getEpsilon<T>();
        else
            thresh = 3*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_trsm<T>( storage, side, uploa, transa, diaga, m, n, alpha, lda_inc, ldb_inc, thresh );
#endif
}

TEST_P( ztrsmGeneric1, API )
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

    gtint_t ldb_inc = lda_inc;

    ztrsmGeneric( storage, side, uploa, transa, diaga, m, n,
                  alpha, lda_inc, ldb_inc );
}

TEST_P( ztrsmGeneric2, API )
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

    ztrsmGeneric( storage, side, uploa, transa, diaga, m, n,
                  alpha, lda_inc, ldb_inc );
}

/**
 * @brief Test ZTRSM native path, which starts from size 501 for BLAS api
 *        and starts from size 0 for BLIS api.
 */
INSTANTIATE_TEST_SUITE_P(
        Native,
        ztrsmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n','c','t'),                                  // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(1, 53, 520),                                   // m
            ::testing::Values(1, 38, 511),                                   // n
            ::testing::Values(dcomplex{0.0,0.0},dcomplex{1.0,0.0},
                              dcomplex{-1.0,0.0},dcomplex{2.0,-3.4}),        // alpha
            ::testing::Values(gtint_t(0),gtint_t(3))                         // increment to the leading dim of a and b
        ),
        ::trsmGeneric1Print<dcomplex>()
    );

/**
 * @brief Test ZTRSM small avx2 path all fringe cases
 *        Kernel size for avx2 small path is 4x3, testing in range of
 *        1 to 4 ensures all fringe cases are being tested.
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2_fringe,
        ztrsmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(1), gtint_t(6), 1),                     // m
            ::testing::Range(gtint_t(1), gtint_t(6), 1),                     // n
            ::testing::Values(dcomplex{0.0,0.0},dcomplex{1.0,0.0},
                              dcomplex{-1.0,0.0},dcomplex{2.0,-3.4}),        // alpha
            ::testing::Values(gtint_t(0),gtint_t(3))                         // increment to the leading dim of a and b
        ),
        ::trsmGeneric1Print<dcomplex>()
    );

/**
 * @brief Test ZTRSM small avx2 path, this code path is used in range 0 to 500
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX2,
        ztrsmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(17, 500),                                      // m
            ::testing::Values(48, 500),                                      // n
            ::testing::Values(dcomplex{0.0,0.0},dcomplex{1.0,0.0},
                              dcomplex{-1.0,0.0},dcomplex{2.0,-3.4}),        // alpha
            ::testing::Values(gtint_t(0),gtint_t(3))                         // increment to the leading dim of a and b
        ),
        ::trsmGeneric1Print<dcomplex>()
    );


/**
 * @brief Test ZTRSM small avx512 path all fringe cases
 *        Kernel size for avx512 small path is 4x4 and 12x4, testing in range of
 *        1 to 24 ensures all fringe cases are being tested (1 to 12 for trsm and 12 to 24 for gemm subproblem).
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX512_fringe,
        ztrsmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 't'),                                     // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Range(gtint_t(1), gtint_t(25), 1),                    // m
            ::testing::Range(gtint_t(1), gtint_t(25), 1),                    // n
            ::testing::Values(dcomplex{0.0,0.0},dcomplex{1.0,0.0},
                              dcomplex{-1.0,0.0},dcomplex{2.0,-3.4}),        // alpha
            ::testing::Values(gtint_t(0),gtint_t(3))                         // increment to the leading dim of a and b
        ),
        ::trsmGeneric1Print<dcomplex>()
    );

/**
 * @brief Test ZTRSM small avx512 path, Same test also covers small_zen5 kernels when run with
 *        BLIS_ARCH_TYPE=zen5
 */
INSTANTIATE_TEST_SUITE_P(
        Small_AVX512,
        ztrsmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format
            ::testing::Values('l','r'),                                      // side  l:left, r:right
            ::testing::Values('u','l'),                                      // uplo  u:upper, l:lower
            ::testing::Values('n', 't'),                                     // transa
            ::testing::Values('n','u'),                                      // diaga , n=nonunit u=unit
            ::testing::Values(17, 500),                                      // m
            ::testing::Values(48, 500),                                      // n
            ::testing::Values(dcomplex{0.0,0.0},dcomplex{1.0,0.0},
                              dcomplex{-1.0,0.0},dcomplex{2.0,-3.4}),        // alpha
            ::testing::Values(gtint_t(0),gtint_t(3))                         // increment to the leading dim of a and b
        ),
        ::trsmGeneric1Print<dcomplex>()
    );

