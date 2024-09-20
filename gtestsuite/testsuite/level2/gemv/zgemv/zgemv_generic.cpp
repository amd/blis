/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level2/gemv/test_gemv.h"

using T = dcomplex;

class zgemvGeneric :
        public ::testing::TestWithParam<std::tuple<char,        // storage format
                                                   char,        // transa
                                                   char,        // conjx
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   T,           // alpha
                                                   T,           // beta
                                                   gtint_t,     // incx
                                                   gtint_t,     // incy
                                                   gtint_t,     // lda_inc
                                                   bool>> {};   // is_memory_test

TEST_P( zgemvGeneric, API )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<1>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<2>(GetParam());
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // specifies beta value
    T beta = std::get<6>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<7>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<8>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<9>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<10>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite gemv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() && (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 2.1;
#endif
        if(( transa == 'n' ) || ( transa == 'N' ))
            thresh = adj*(3*n+1)*testinghelpers::getEpsilon<T>();
        else
            thresh = adj*(3*m+1)*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemv<T>( storage, transa, conjx, m, n, alpha, lda_inc, incx, beta, incy, thresh, is_memory_test );
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        BlackboxSmall,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(1), gtint_t(12), gtint_t(20)),                    // m
            ::testing::Values(gtint_t(1), gtint_t(17), gtint_t(20)),                    // n
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // alpha
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // beta
            ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(-1)),          // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(5), gtint_t(-2)),          // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(7)),                       // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        BlackboxMedium,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(25),
                              gtint_t(211)
                            ),                                               // m
            ::testing::Values(gtint_t(25),
                              gtint_t(173)
                            ),                                               // n
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // alpha
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // beta
            ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(-1)),          // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(5), gtint_t(-1)),          // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(7)),                       // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

#if 1
INSTANTIATE_TEST_SUITE_P(
        Blackbox_Large,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(2127)),                                // m
            ::testing::Values(gtint_t(2127)),                                // n
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // alpha
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // beta
            ::testing::Values(gtint_t(1), gtint_t(211)),                     // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(11)),                      // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(57)),                      // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_LargeM,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(5099)),                                // m
            ::testing::Values(gtint_t(1), gtint_t(17),
                              gtint_t(173)),                                 // n
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // alpha
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // beta
            ::testing::Values(gtint_t(1), gtint_t(211)),                     // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(11)),                      // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(57)),                      // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_LargeN,
        zgemvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('n', 'c', 't'),                                // transa
            ::testing::Values('n'),                                          // conjx
            ::testing::Values(gtint_t(1), 
                              gtint_t(173)),                                 // m
            ::testing::Values(gtint_t(5099)),                                // n
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // alpha
            ::testing::Values(T{0.0, 0.0}, T{1.0, 0.0}, T{-1.0, 0.0},
                              T{1.1, -2.0} ),                                // beta
            ::testing::Values(gtint_t(1), gtint_t(211)),                     // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(11)),                      // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(57)),                      // increment to the leading dim of a
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::gemvGenericPrint<T>()
    );
#endif
