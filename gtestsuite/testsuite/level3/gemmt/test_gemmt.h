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

#pragma once

#include "gemmt.h"
#include "level3/ref_gemmt.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include "common/testing_helpers.h"
#include "inc/data_pool.h"

template<typename T>
void test_gemmt( char storage, char uploc, char transa, char transb, gtint_t n,
    gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc, T alpha,
    T beta, double thresh, bool is_mem_test=false, bool is_evt_test=false,
    T evt_a=T{0.0}, T evt_b=T{0.0}, T evt_c=T{0.0} )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );

    //----------------------------------------------------------
    //         Initialize matrices with random numbers
    //----------------------------------------------------------
    T *a_ptr, *b_ptr, *c_ptr;
    dim_t size_a = testinghelpers::matsize(storage, transa, n, k, lda) * sizeof(T);
    testinghelpers::ProtectedBuffer a(size_a, false, is_mem_test );
    a_ptr = (T*)a.greenzone_1;
    // Set index to a starting position for this test
    get_pool<T>().set_index(n, lda_inc, k);
    get_pool<T>().randomgenerators( storage, n, k, a_ptr, transa, lda);

    if ( is_evt_test )
    {
        srand(SRAND_SEED);

        dim_t n_rand = rand() % (std::min)(n, k);
        dim_t k_rand = rand() % (std::min)(n, k);
        a_ptr[n_rand + k_rand * lda] = evt_a;
    }

    dim_t size_b = testinghelpers::matsize(storage, transb, k, n, ldb) * sizeof(T);
    testinghelpers::ProtectedBuffer b(size_b, false, is_mem_test );
    b_ptr = (T*)b.greenzone_1;
    get_pool<T>().randomgenerators( storage, k, n, b_ptr, transb, ldb);

    if ( is_evt_test )
    {
        dim_t n_rand = rand() % (std::min)(k, n);
        dim_t k_rand = rand() % (std::min)(k, n);
        b_ptr[n_rand + k_rand * ldb] = evt_b;
    }

    dim_t size_c = testinghelpers::matsize(storage, 'n', n, n, ldc) * sizeof(T);
    testinghelpers::ProtectedBuffer c(size_c, false, is_mem_test );
    c_ptr = (T*)c.greenzone_1;
    //if (beta != testinghelpers::ZERO<T>())
    if (1)
    {
        get_pool<T>().randomgenerators( storage, n, n, c_ptr, 'n', ldc);
        if ( is_evt_test )
        {
            dim_t n_rand = rand() % n;
            dim_t k_rand = rand() % n;
            c_ptr[n_rand + k_rand * ldc] = evt_c;
        }
    }
    else
    {
        // Matrix C should not be read, only set.
        testinghelpers::set_matrix( storage, n, n, c_ptr, 'n', ldc, testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(testinghelpers::matsize(storage, 'n', n, n, ldc));
    memcpy(c_ref.data(), c_ptr, size_c);

    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        //----------------------------------------------------------
        //                  Call BLIS function
        //----------------------------------------------------------
        gemmt<T>( storage, uploc, transa, transb, n, k, &alpha, a_ptr, lda,
                  b_ptr, ldb, &beta, c_ptr, ldc );
        if (is_mem_test)
        {
            memcpy(a.greenzone_2, a.greenzone_1, size_a);
            memcpy(b.greenzone_2, b.greenzone_1, size_b);
            memcpy(c.greenzone_2, c_ref.data(), size_c);

            gemmt<T>( storage, uploc, transa, transb, n, k, &alpha, (T*)a.greenzone_2, lda,
                      (T*)b.greenzone_2, ldb, &beta, (T*)c.greenzone_2, ldc );
        }

    }
    catch(const std::exception& e)
    {
        // reset to default signal handler
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // show failure in case seg fault was detected
        FAIL() << "Memory Test Failed";
    }
    // reset to default signal handler
    testinghelpers::ProtectedBuffer::stop_signal_handler();

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gemmt<T>( storage, uploc, transa, transb, n, k, alpha,
               a_ptr, lda, b_ptr, ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "C", storage, n, n, c_ptr, c_ref.data(), ldc, thresh, is_evt_test );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class gemmtGeneric1Print {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,T,T,gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char uploc      = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char transb     = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha  = std::get<6>(str.param);
        T beta   = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);

        gtint_t ldb_inc = lda_inc;
        gtint_t ldc_inc = lda_inc;

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploc_" + std::string(&uploc, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class gemmtGeneric3Print {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,T,T,gtint_t,gtint_t,gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char uploc      = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char transb     = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha  = std::get<6>(str.param);
        T beta   = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploc_" + std::string(&uploc, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};

template <typename T>
class gemmtMemGeneric1Print {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,T,T,gtint_t,bool>> str) const {
        char storage    = std::get<0>(str.param);
        char uploc      = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char transb     = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha    = std::get<6>(str.param);
        T beta     = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        bool is_mem_test = std::get<9>(str.param);

        gtint_t ldb_inc = lda_inc;
        gtint_t ldc_inc = lda_inc;

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploc_" + std::string(&uploc, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        str_name = str_name + (is_mem_test ? "_mem_test_enabled" : "_mem_test_disabled");
        return str_name;
    }
};

template <typename T>
class gemmtMemGeneric3Print {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,T,T,gtint_t,gtint_t,gtint_t,bool>> str) const {
        char storage    = std::get<0>(str.param);
        char uploc      = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char transb     = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha    = std::get<6>(str.param);
        T beta     = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);
        bool is_mem_test = std::get<11>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploc_" + std::string(&uploc, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        str_name = str_name + (is_mem_test ? "_mem_test_enabled" : "_mem_test_disabled");
        return str_name;
    }
};

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class gemmtEVTPrint
{
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,gtint_t,T,T,gtint_t,gtint_t,gtint_t,T,T,T>> str) const {
        char storage    = std::get<0>(str.param);
        char uploc      = std::get<1>(str.param);
        char transa     = std::get<2>(str.param);
        char transb     = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        gtint_t k       = std::get<5>(str.param);
        T alpha    = std::get<6>(str.param);
        T beta     = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ldb_inc = std::get<9>(str.param);
        gtint_t ldc_inc = std::get<10>(str.param);
        T aexval   = std::get<11>(str.param);
        T bexval   = std::get<12>(str.param);
        T cexval   = std::get<13>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uploc_" + std::string(&uploc, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_transb_" + std::string(&transb, 1);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_ex_a_" + testinghelpers::get_value_string(aexval);
        str_name = str_name + "_ex_b_" + testinghelpers::get_value_string(bexval);
        str_name = str_name + "_ex_c_" + testinghelpers::get_value_string(cexval);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, k, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, k, n, ldb_inc );
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        return str_name;
    }
};

// Two options for instantiate block, one to set lda_inc = ldb_inc = ldc_inc to
// reduce Cartesian product, and one to allow all to be varied independently.
class zgemmtGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   dcomplex,     // alpha
                                                   dcomplex,     // beta
                                                   gtint_t,      // inc to the lda, ldb and ldc
                                                   bool>> {};    // is memory test

class zgemmtGeneric3 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   dcomplex,     // alpha
                                                   dcomplex,     // beta
                                                   gtint_t,      // lda_inc
                                                   gtint_t,      // ldb_inc
                                                   gtint_t,      // ldc_inc
                                                   bool>> {};    // is memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zgemmtGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zgemmtGeneric3);

using TZ = dcomplex;
void zgemmtGeneric( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
                    TZ alpha, TZ beta, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
                    bool is_mem_test )
{
    // Set the threshold for the errors:
    // Check gtestsuite gemmt.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multiplier for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<TZ>() || k == 0) &&
             (beta == testinghelpers::ZERO<TZ>() || beta == testinghelpers::ONE<TZ>()))
        thresh = 0.0;
    else
    {
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 1.05;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<TZ>();
    }

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemmt<TZ>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemmt<TZ>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemmt<TZ>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemmt<TZ>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
#endif
}

TEST_P( zgemmtGeneric1, API )
{
    using T = dcomplex;
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
    bool is_mem_test = std::get<9>(GetParam());

    gtint_t ldb_inc = lda_inc;
    gtint_t ldc_inc = lda_inc;

    zgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

TEST_P( zgemmtGeneric3, API )
{
    using T = dcomplex;
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

    zgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

// Two options for instantiate block, one to set lda_inc = ldb_inc = ldc_inc to
// reduce Cartesian product, and one to allow all to be varied independently.
class dgemmtGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   double,       // alpha
                                                   double,       // beta
                                                   gtint_t,      // inc to the lda, ldb and ldc
                                                   bool>> {};    // is memory test

class dgemmtGeneric3 :
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

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmtGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmtGeneric3);

using TD = double;
void dgemmtGeneric( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
                    TD alpha, TD beta, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
                    bool is_mem_test )
{
    // Set the threshold for the errors:
    // Check gtestsuite gemmt.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multiplier for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<TD>() || k == 0) &&
             (beta == testinghelpers::ZERO<TD>() || beta == testinghelpers::ONE<TD>()))
        thresh = 0.0;
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 4.1;
#else
        double adj = 4.0;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<TD>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemmt<TD>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemmt<TD>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemmt<TD>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemmt<TD>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
#endif
}

TEST_P( dgemmtGeneric1, API )
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
    bool is_mem_test = std::get<9>(GetParam());

    gtint_t ldb_inc = lda_inc;
    gtint_t ldc_inc = lda_inc;

    dgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

TEST_P( dgemmtGeneric3, API )
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

    dgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

// Two options for instantiate block, one to set lda_inc = ldb_inc = ldc_inc to
// reduce Cartesian product, and one to allow all to be varied independently.
class cgemmtGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   scomplex,     // alpha
                                                   scomplex,     // beta
                                                   gtint_t,      // inc to the lda, ldb and ldc
                                                   bool>> {};    // is memory test

class cgemmtGeneric3 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   scomplex,     // alpha
                                                   scomplex,     // beta
                                                   gtint_t,      // lda_inc
                                                   gtint_t,      // ldb_inc
                                                   gtint_t,      // ldc_inc
                                                   bool>> {};    // is memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cgemmtGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cgemmtGeneric3);

using TC = scomplex;
void cgemmtGeneric( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
                    TC alpha, TC beta, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
                    bool is_mem_test )
{
    // Set the threshold for the errors:
    // Check gtestsuite gemmt.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multiplier for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<TC>() || k == 0) &&
             (beta == testinghelpers::ZERO<TC>() || beta == testinghelpers::ONE<TC>()))
        thresh = 0.0;
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 1.2;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<TC>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemmt<TC>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemmt<TC>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemmt<TC>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemmt<TC>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
#endif
}

TEST_P( cgemmtGeneric1, API )
{
    using T = scomplex;
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
    bool is_mem_test = std::get<9>(GetParam());

    gtint_t ldb_inc = lda_inc;
    gtint_t ldc_inc = lda_inc;

    cgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

TEST_P( cgemmtGeneric3, API )
{
    using T = scomplex;
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

    cgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

// Two options for instantiate block, one to set lda_inc = ldb_inc = ldc_inc to
// reduce Cartesian product, and one to allow all to be varied independently.
class sgemmtGeneric1 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   float,        // alpha
                                                   float,        // beta
                                                   gtint_t,      // inc to the lda, ldb and ldc
                                                   bool>> {};    // is memory test

class sgemmtGeneric3 :
        public ::testing::TestWithParam<std::tuple<char,         // storage
                                                   char,         // uplo
                                                   char,         // transa
                                                   char,         // transb
                                                   gtint_t,      // n
                                                   gtint_t,      // k
                                                   float,        // alpha
                                                   float,        // beta
                                                   gtint_t,      // lda_inc
                                                   gtint_t,      // ldb_inc
                                                   gtint_t,      // ldc_inc
                                                   bool>> {};    // is memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmtGeneric1);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(sgemmtGeneric3);

using TS = float;
void sgemmtGeneric( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
                    TS alpha, TS beta, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
                    bool is_mem_test )
{
    // Set the threshold for the errors:
    // Check gtestsuite gemmt.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multiplier for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<TS>() || k == 0) &&
             (beta == testinghelpers::ZERO<TS>() || beta == testinghelpers::ONE<TS>()))
        thresh = 0.0;
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 1.5;
#endif
        thresh = adj*(3*k+1)*testinghelpers::getEpsilon<TS>();
    }
    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_gemmt<TS>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_gemmt<TS>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_gemmt<TS>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_gemmt<TS>( storage, uplo, transa, transb, n, k, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, is_mem_test );
#endif
}

TEST_P( sgemmtGeneric1, API )
{
    using T = float;
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
    bool is_mem_test = std::get<9>(GetParam());

    gtint_t ldb_inc = lda_inc;
    gtint_t ldc_inc = lda_inc;

    sgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}

TEST_P( sgemmtGeneric3, API )
{
    using T = float;
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

    sgemmtGeneric( storage, uplo, transa, transb, n, k,
                   alpha, beta, lda_inc, ldb_inc, ldc_inc, is_mem_test );
}
