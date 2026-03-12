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

#include <gtest/gtest.h>
#include "level2/gbmv/test_gbmv.h"

using T = double;

class dgbmvGeneric :
        public ::testing::TestWithParam<std::tuple<char,        // storage format
                                                   char,        // transa
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   gtint_t,     // kl
                                                   gtint_t,     // ku
                                                   T,           // alpha
                                                   T,           // beta
                                                   gtint_t,     // incx
                                                   gtint_t,     // incy
                                                   gtint_t      // lda_inc
                                        >> {};
                                        
TEST_P( dgbmvGeneric, API )
{
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<1>(GetParam());
    // matrix size m
    gtint_t m  = std::get<2>(GetParam());
    // matrix size n
    gtint_t n  = std::get<3>(GetParam());
    // sub-diagonal size kl
    gtint_t kl  = std::get<4>(GetParam());
    // super-diagonal size ku
    gtint_t ku  = std::get<5>(GetParam());
    // specifies alpha value
    T alpha = std::get<6>(GetParam());
    // specifies beta value
    T beta = std::get<7>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<8>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<9>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<10>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite gbmv.h or netlib source code for reminder of the
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
        if(( transa == 'n' ) || ( transa == 'N' ))
            thresh = (3*n+1)*testinghelpers::getEpsilon<T>();
        else
            thresh = (3*m+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gbmv<T>( storage, transa, m, n, kl, ku, alpha, lda_inc, incx, beta, incy, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gbmv<T>( storage, transa, m, n, kl, ku, alpha, lda_inc, incx, beta, incy, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gbmv<T>( storage, transa, m, n, kl, ku, alpha, lda_inc, incx, beta, incy, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gbmv<T>( storage, transa, m, n, kl, ku, alpha, lda_inc, incx, beta, incy, thresh );
#endif
}

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        BlackboxSmall,
        dgbmvGeneric,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
            ,'r'
#endif
            ),                                                              // storage format
            ::testing::Values('n', 'c', 't'),                               // transa
            ::testing::Values(gtint_t(1), gtint_t(2), gtint_t(13)),         // m
            ::testing::Values(gtint_t(1), gtint_t(2), gtint_t(16)),         // n
            ::testing::Values(gtint_t(0), gtint_t(1), gtint_t(7)),          // kl
            ::testing::Values(gtint_t(0), gtint_t(1), gtint_t(3)),          // ku
            ::testing::Values( 0.0, 1.0, -1.0, -1.2 ),                      // alpha
            ::testing::Values( 0.0, 1.0, -1.0, 2.1 ),                       // beta
            ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(-1)),         // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(5), gtint_t(-2)),         // stride size for y
            ::testing::Values(gtint_t(0), gtint_t(7))                       // increment to the leading dim of a
        ),
        ::gbmvGenericPrint<T>()
    );
