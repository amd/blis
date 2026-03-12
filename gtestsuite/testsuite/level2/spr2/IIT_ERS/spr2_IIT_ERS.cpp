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
#include "common/testing_helpers.h"
#include "level2/spr2/test_spr2.h"
#include "inc/check_error.h"

template <typename T>
class spr2_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double> TypeParam; // The supported datatypes from BLAS calls for spr2
TYPED_TEST_SUITE(spr2_IIT_ERS, TypeParam); // Defining individual testsuites based on the datatype support.

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

// When info == 1
TYPED_TEST(spr2_IIT_ERS, invalid_storage)
{
    using T = TypeParam;
    static const char UPLO = 'u';
    static const gtint_t N = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t incx = 1;
    gtint_t incy = 1;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    spr2<T>( 'x', UPLO, N, &alpha, nullptr, incx, nullptr, incy, nullptr );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    // Call BLIS spr2 with a invalid value for storage.
    spr2<T>( 'x', UPLO, N, &alpha, x.data(), incx, y.data(), incy, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for spr2):
    1. When UPLO != 'U' || UPLO != 'L' (info = 1)
    2. When n < 0 (info = 2)
    3. When incx = 0 (info = 5)
    4. When incy = 0 (info = 7)

*/

// When info == 1
TYPED_TEST(spr2_IIT_ERS, invalid_uplo)
{
    using T = TypeParam;
    static const char STORAGE = 'c';
    static const gtint_t N = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t incx = 1;
    gtint_t incy = 1;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    spr2<T>( STORAGE, 'p', N, nullptr, nullptr, incx, nullptr, incy, nullptr );
#else
    spr2<T>( STORAGE, 'p', N, &alpha, nullptr, incx, nullptr, incy, nullptr );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    // Call BLIS spr2 with a invalid value for uplo.
    spr2<T>( STORAGE, 'p', N, &alpha, x.data(), incx, y.data(), incy, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

// When info == 2
TYPED_TEST(spr2_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    gtint_t incx = 1;
    gtint_t incy = 1;
    static const gtint_t N = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    spr2<T>( STORAGE, UPLO, -1, nullptr, nullptr, incx, nullptr, incy, nullptr );
#else
    spr2<T>( STORAGE, UPLO, -1, &alpha, nullptr, incx, nullptr, incy, nullptr );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    // Call BLIS spr2 with a invalid value for n.
    spr2<T>( STORAGE, UPLO, -1, &alpha, x.data(), incx, y.data(), incy, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

// When info == 5
TYPED_TEST(spr2_IIT_ERS, incx_eq_zero)
{
    using T = TypeParam;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const gtint_t N = 7;
    gtint_t incx = 1;
    gtint_t incy = 1;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    spr2<T>( STORAGE, UPLO, N, nullptr, nullptr, 0, nullptr, incy, nullptr );
#else
    spr2<T>( STORAGE, UPLO, N, &alpha, nullptr, 0, nullptr, incy, nullptr );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    // Call BLIS spr2 with a invalid value for incx.
    spr2<T>( STORAGE, UPLO, N, &alpha, x.data(), 0, y.data(), incy, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}

// When info == 7
TYPED_TEST(spr2_IIT_ERS, incy_eq_zero)
{
    using T = TypeParam;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const gtint_t N = 7;
    gtint_t incx = 1;
    gtint_t incy = 1;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    spr2<T>( STORAGE, UPLO, N, nullptr, nullptr, incx, nullptr, 0, nullptr );
#else
    spr2<T>( STORAGE, UPLO, N, &alpha, nullptr, incx, nullptr, 0, nullptr );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    // Call BLIS spr2 with a invalid value for incy.
    spr2<T>( STORAGE, UPLO, N, &alpha, x.data(), incx, y.data(), 0, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

/*
    Early Return Scenarios(ERS) :

    The spr2 API is expected to return early in the following cases:

    1. When n == 0.
    2. When alpha == 0.

*/

// When n is 0
TYPED_TEST(spr2_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const gtint_t N = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t incx = 1;
    gtint_t incy = 1;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    spr2<T>( STORAGE, UPLO, 0, nullptr, nullptr, incx, nullptr, incy, nullptr );
#else
    spr2<T>( STORAGE, UPLO, 0, &alpha, nullptr, incx, nullptr, incy, nullptr );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    spr2<T>( STORAGE, UPLO, 0, &alpha, x.data(), incx, y.data(), incy, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When alpha is 0
TYPED_TEST(spr2_IIT_ERS, alpha_zero)
{
    using T = TypeParam;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const gtint_t N = 4;
    gtint_t incx = 1;
    gtint_t incy = 1;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    T alpha;
    testinghelpers::initzero<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    spr2<T>( STORAGE, UPLO, N, &alpha, nullptr, incx, nullptr, incy, nullptr );
#else
    spr2<T>( STORAGE, UPLO, N, &alpha, nullptr, incx, nullptr, incy, nullptr );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    dim_t len_a = ( N * ( N + 1 ) ) / 2;
    std::vector<T> a = testinghelpers::get_random_vector<T>( -2, 5, len_a, 1 );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> a_ref(a);

    spr2<T>( STORAGE, UPLO, N, &alpha, x.data(), incx, y.data(), incy, a.data() );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", len_a, a.data(), a_ref.data(), 1, true );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif

