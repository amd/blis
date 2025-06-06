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
#include "test_scalv_ukr.h"
#include "common/blis_version_defs.h"

class zscalvGeneric :
        public ::testing::TestWithParam<std::tuple<zscalv_ker_ft,   // Function pointer for zscalv kernels
                                                   char,            // conj_alpha
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   dcomplex,        // alpha
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zscalvGeneric);

// Tests using random integers as vector elements.
TEST_P( zscalvGeneric, UKR )
{
    using T = dcomplex;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // denotes the kernel to be tested:
    zscalv_ker_ft ukr = std::get<0>(GetParam());
    // denotes whether alpha or conj(alpha) will be used:
    char conj_alpha = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // alpha:
    T alpha = std::get<4>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<5>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite scalv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_scalv_ukr<T, T, zscalv_ker_ft>( ukr, conj_alpha, n, incx, alpha, thresh, is_memory_test );
}

// ----------------------------------------------
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Tests for bli_zscalv_zen_int (AVX2) kernel.
/**
 * Loops:
 * L8      - Main loop, handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
*/
#ifdef K_bli_zscalv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_zscalv_zen_int_unitPositiveStride,
        zscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zscalv_zen_int),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(16),       // L8 (executed twice)
                                gtint_t(15),       // L8 upto LScalar
                                gtint_t( 8),       // L8
                                gtint_t( 4),       // L4
                                gtint_t( 2),       // L2
                                gtint_t( 1)        // LScalar
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)      // unit stride
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                dcomplex{-5.1, -7.3},
                                dcomplex{ 0.0,  0.0},
                                dcomplex{ 7.3,  5.1}
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<dcomplex,zscalv_ker_ft>())
    );
#endif

#ifdef K_bli_zscalv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_zscalv_zen_int_nonUnitPositiveStrides,
        zscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zscalv_zen_int),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                dcomplex{-5.1, -7.3},
                                dcomplex{ 0.0,  0.0},
                                dcomplex{ 7.3,  5.1}
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<dcomplex,zscalv_ker_ft>())
    );
#endif
#endif
// ----------------------------------------------
// -----  End ZEN1/2/3 (AVX2) Kernel Tests  -----
// ----------------------------------------------


// ----------------------------------------------
// -----  Begin ZEN4 (AVX512) Kernel Tests  -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
// Tests for bli_zscalv_zen_int_avx512 (AVX512) kernel.
/**
 * Loops:
 * L48     - Main loop, handles 48 elements
 * L32     - Main loop, handles 32 elements
 * L16     - Main loop, handles 16 elements
 * L8      - Main loop, handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
*/
#ifdef K_bli_zscalv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_zscalv_zen_int_avx512_unitPositiveStride,
        zscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zscalv_zen_int_avx512),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(143),       // L48 x2 + L32 + L8 + L4 + L2 + LScalar
                                gtint_t(127),       // L48 x2 + L16 + L8 + L4 + L2 + LScalar
                                gtint_t(48),        // L48
                                gtint_t(47),        // L32 + L16 + L8 + L4 + L2 + LScalar
                                gtint_t(32),        // L32
                                gtint_t(16),        // L16
                                gtint_t( 8),        // L8
                                gtint_t( 4),        // L4
                                gtint_t( 2),        // L2
                                gtint_t( 1)         // LScalar
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)      // unit stride
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                dcomplex{-5.1, -7.3},
                                dcomplex{ 0.0,  0.0},
                                dcomplex{ 7.3,  5.1}
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<dcomplex,zscalv_ker_ft>())
    );
#endif

#ifdef K_bli_zscalv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_zscalv_zen_int_avx512_nonUnitPositiveStrides,
        zscalvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zscalv_zen_int_avx512),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                dcomplex{-5.1, -7.3},
                                dcomplex{ 0.0,  0.0},
                                dcomplex{ 7.3,  5.1}
            ),
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::scalvUKRPrint<dcomplex,zscalv_ker_ft>())
    );
#endif
#endif
