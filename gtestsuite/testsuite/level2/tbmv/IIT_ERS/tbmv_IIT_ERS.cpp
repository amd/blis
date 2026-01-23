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

#include "level2/tbmv/test_tbmv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include "common/wrong_inputs_helpers.h"
#include <stdexcept>
#include <algorithm>
#include <gtest/gtest.h>

template <typename T>
class tbmv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(tbmv_IIT_ERS, TypeParam);

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

/**
 * @brief Test tbmv when STORAGE argument is incorrect
 *        when info == 1
 *
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_storage)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( 'x', UPLO, TRANS, DIAG, N, KB, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( 'x', UPLO, TRANS, DIAG, N, KB, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for tbmv):
    1. When UPLO  != 'L' || UPLO   != 'U'                  (info = 1)
    2. When TRANS != 'N' || TRANS  != 'T' || TRANS != 'C'  (info = 2)
    3. When DIAG  != 'U' || DIAG   != 'N'                  (info = 3)
    4. When n < 0                                          (info = 4)
    5. When k < 0                                          (info = 5)
    6. When lda < k+1                                      (info = 7)
    7. When incx == 0                                      (info = 9)
*/


/**
 * @brief Test tbmv when UPLO argument is incorrect
 *        when info == 1
 *
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_UPLO)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, 'A', TRANS, DIAG, N, KB, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, KB, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, 'A', TRANS, DIAG, N, KB, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

/**
 * @brief Test tbmv when TRANS argument is incorrect
 *        when info == 2
 *
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_TRANS)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, 'A', DIAG, N, KB, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, 'A', DIAG, N, KB, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif
}

/**
 * @brief Test tbmv when DIAG argument is incorrect
 *        when info == 3
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_DIAG)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, TRANS, 'A', N, KB, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+3 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, TRANS, 'A', N, KB, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+3 );
#endif
}

/**
 * @brief Test tbmv when N is negative
 *        when info == 4
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_n)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, -1, KB, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, -1, KB, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}

/**
 * @brief Test tbmv when K is negative
 *        when info == 5
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_k)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, N, -1, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, N, -1, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}


/**
 * @brief Test tbmv when lda < max(1, N)
 *        when info == 7
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_lda)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, N, KB, nullptr, LDA_B - 1, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, N, KB, a.data(), LDA_B - 1, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

/**
 * @brief Test tbmv when INCX == 0
 *        when info == 9
 */
TYPED_TEST(tbmv_IIT_ERS, invalid_incx)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, N, KB, nullptr, LDA_B, nullptr, 0);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, N, KB, a.data(), LDA_B, x.data(), 0);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}


/*
    Early Return Scenarios(ERS) :

    The tbmv API is expected to return early in the following cases:

    1. When n == 0.

*/

/**
 * @brief Test tbmv when N is zero
 */
TYPED_TEST(tbmv_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, 0, KB, nullptr, LDA_B, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, 'n', KB+1, N, LDA_B);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    tbmv<T>( STORAGE, UPLO, TRANS, DIAG, 0, KB, a.data(), LDA_B, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif
