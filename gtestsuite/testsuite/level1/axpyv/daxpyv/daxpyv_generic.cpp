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
#include "level1/axpyv/test_axpyv.h"

class daxpyvGeneric :
        public ::testing::TestWithParam<std::tuple<char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   double>> {};
// Tests using random integers as vector elements.
TEST_P( daxpyvGeneric, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());
    // alpha
    T alpha = std::get<4>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpyv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else if (alpha == testinghelpers::ONE<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
        thresh = 2*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_axpyv<T>( conj_x, n, incx, incy, alpha, thresh );
#endif
}

// Black box testing for generic and main use of daxpy.
INSTANTIATE_TEST_SUITE_P(
        unitStride,
        daxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'
            #ifndef TEST_BLIS_TYPED
                            , 'c'
#endif
            ),                                          // n: use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(35), gtint_t(112)), // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(double(0.0), double(1.0),
                              double(-1.0), double(4.1))                     // alpha
        ),
        ::axpyvGenericPrint<double>()
    );

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        nonUnitIncrements,
        daxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'
            #ifdef TEST_BLIS_TYPED
                            , 'c'
            #endif
            ),                                           // n: use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(2)
            #ifdef TEST_BLIS_TYPED
                            , gtint_t(-2)
            #endif
            ),                                   // stride size for x
            ::testing::Values(gtint_t(3)
            #ifdef TEST_BLIS_TYPED
                            , gtint_t(-4)
            #endif
            ),                                                              // stride size for y
            ::testing::Values(double(0.0), double(1.0),
                              double(-1.0), double(4.1))                     // alpha
        ),
        ::axpyvGenericPrint<double>()
    );

// The following instantiator is enabled only if BLIS has been configured for openmp
// with aocl-dynamic enabled.
#if defined(BLIS_ENABLE_OPENMP) && defined(AOCL_DYNAMIC)
// Checking for the thresholds with unit strides
INSTANTIATE_TEST_SUITE_P(
        AOCLDynamicThresholds_unitStrides,
        daxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                         // n: use x, c: use conj(x)
            ::testing::Values(// Sizes are based on the thresholds
                              gtint_t(4000),                // nt_ideal = 1
                              gtint_t(11000),               // nt_ideal = 4
                              gtint_t(300000),              // nt_ideal = 8
                              gtint_t(750000),              // nt_ideal = 16
                              gtint_t(2600000),             // nt_ideal = 32
                              gtint_t(4000000)),            // nt_ideal = 64

            ::testing::Values(gtint_t(1)),                  // stride size for x
            ::testing::Values(gtint_t(1)),                  // stride size for y
            ::testing::Values(double(0.0), double(1.0),
                              double(-1.0), double(4.1))    // alpha
        ),
        ::axpyvGenericPrint<double>()
    );

// Checking for the thresholds with non-unit strides
INSTANTIATE_TEST_SUITE_P(
        AOCLDynamicThresholds_nonUnitStrides,
        daxpyvGeneric,
        ::testing::Combine(
            ::testing::Values('n'),                         // n: use x, c: use conj(x)
            ::testing::Values(// Sizes are based on the thresholds
                              gtint_t(4000),                // nt_ideal = 1
                              gtint_t(11000),               // nt_ideal = 4
                              gtint_t(300000),              // nt_ideal = 8
                              gtint_t(750000),              // nt_ideal = 16
                              gtint_t(2600000),             // nt_ideal = 32
                              gtint_t(4000000)),            // nt_ideal = 64

            ::testing::Values(gtint_t(3)),                  // stride size for x
            ::testing::Values(gtint_t(3)),                  // stride size for y
            ::testing::Values(double(0.0), double(1.0),
                              double(-1.0), double(4.1))    // alpha
        ),
        ::axpyvGenericPrint<double>()
    );
#endif
