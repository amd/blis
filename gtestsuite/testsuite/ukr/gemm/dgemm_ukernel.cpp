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
#include "blis.h"
#include "common/testing_helpers.h"
#include "test_gemm_ukr.h"

class dgemmUkrSUP :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char, dgemmsup_ker_ft, gtint_t, char, char, bool, bool>> {};
// m, n, k, alpha, beta,  storage of c, dgemm sup kernel, micro-kernel MR block, transa, transb, memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmUkrSUP);

TEST_P(dgemmUkrSUP, sup_kernel)
{
    using T = double;
    gtint_t m      = std::get<0>(GetParam());  // dimension m
    gtint_t n      = std::get<1>(GetParam());  // dimension n
    gtint_t k      = std::get<2>(GetParam());  // dimension k
    T alpha        = std::get<3>(GetParam());  // alpha
    T beta         = std::get<4>(GetParam());  // beta
    char storageC   = std::get<5>(GetParam()); // storage scheme for C matrix
    dgemmsup_ker_ft kern_ptr = std::get<6>(GetParam()); //pointer to the gemm kernel
    gtint_t MR  = std::get<7>(GetParam());
    char transa = std::get<8>(GetParam());
    char transb = std::get<9>(GetParam());
    bool row_pref = std::get<10>(GetParam());
    bool memory_test = std::get<11>(GetParam());

    test_gemmsup_ukr(kern_ptr, transa, transb, m, n, k, alpha, beta, storageC, MR, row_pref, memory_test);

}// end of function


class dgemmUkrSUPPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char, dgemmsup_ker_ft, gtint_t, char, char, bool, bool>>  str) const {

        gtint_t m       = std::get<0>(str.param);
        gtint_t n       = std::get<1>(str.param);
        gtint_t k       = std::get<2>(str.param);
        double alpha    = std::get<3>(str.param);
        double beta     = std::get<4>(str.param);
        char storageC    = std::get<5>(str.param);
        char trnsa       = std::get<8>(str.param);
        char trnsb       = std::get<9>(str.param);
        bool memory_test       = std::get<11>(str.param);

        std::string str_name;
        str_name = str_name + "_" + trnsa;
        str_name = str_name + "_" + trnsb;
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + storageC;
        str_name += ( memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_HASWELL) && defined(GTEST_AVX2FMA3)

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8m_row_stored_c,
        dgemmUkrSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8m),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
        ),
        ::dgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8m_col_stored_c,
        dgemmUkrSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8m),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('t'),                                 // transb
            ::testing::Values(true),                                // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
        ),
        ::dgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rd_haswell_asm_6x8m_col_stored_c,
        dgemmUkrSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rd_haswell_asm_6x8m),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
        ),
        ::dgemmUkrSUPPrint()
    );


INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8n_col_stored_c,
        dgemmUkrSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8n),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
        ),
        ::dgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rv_haswell_asm_6x8n_row_stored_c,
        dgemmUkrSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_haswell_asm_6x8n),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
        ),
        ::dgemmUkrSUPPrint()
    );

INSTANTIATE_TEST_SUITE_P (
        bli_dgemmsup_rd_haswell_asm_6x8n_col_stored_c,
        dgemmUkrSUP,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(7), 1),            // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(17), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rd_haswell_asm_6x8n),    // dgemm_sup kernel
            ::testing::Values(gtint_t(6)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(true),                                // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
        ),
        ::dgemmUkrSUPPrint()
    );
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)

 INSTANTIATE_TEST_SUITE_P (
         bli_dgemmsup_rv_zen4_asm_24x8m_col_stored_c,
         dgemmUkrSUP,
         ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(25), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(25), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('c'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_zen4_asm_24x8m),      // dgemm_sup kernel
            ::testing::Values(gtint_t(8)),                          // Micro kernel block MR
            ::testing::Values('n'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(false),                               // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
         ),
         ::dgemmUkrSUPPrint()
     );

 INSTANTIATE_TEST_SUITE_P (
         bli_dgemmsup_rv_zen4_asm_24x8m_row_stored_c,
         dgemmUkrSUP,
         ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(25), 1),           // values of m
            ::testing::Range(gtint_t(1), gtint_t(9), 1),            // values of n
            ::testing::Range(gtint_t(0), gtint_t(25), 1),           // values of k
            ::testing::Values(2.0, 1.0, -1.0),                      // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),                 // beta value
            ::testing::Values('r'),                                 // storage of c
            ::testing::Values(bli_dgemmsup_rv_zen4_asm_24x8m),      // dgemm_sup kernel
            ::testing::Values(gtint_t(8)),                          // Micro kernel block MR
            ::testing::Values('t'),                                 // transa
            ::testing::Values('n'),                                 // transb
            ::testing::Values(false),                               // row preferred kernel?
            ::testing::Values(true, false)                          // memory test
         ),
         ::dgemmUkrSUPPrint()
     );
#endif

class dgemmUkrNat :
        public ::testing::TestWithParam<std::tuple<gtint_t, double, double, char, gtint_t, gtint_t, dgemm_ukr_ft, bool>> {};
// k, alpha, beta, storage of c, m, n, dgemm native kernel, memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmUkrNat);

TEST_P(dgemmUkrNat, native_kernel_testing)
{
    using T = double;
    gtint_t k      = std::get<0>(GetParam());  // dimension k
    T alpha        = std::get<1>(GetParam());  // alpha
    T beta         = std::get<2>(GetParam());  // beta
    char storage   = std::get<3>(GetParam());  // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m = std::get<4>(GetParam());
    gtint_t n = std::get<5>(GetParam());
    dgemm_ukr_ft kern_ptr = std::get<6>(GetParam());
    bool memory_test = std::get<7>(GetParam());
    test_gemmnat_ukr(storage, m, n, k, alpha, beta, kern_ptr, memory_test);
}// end of function



class dgemmUkrNatPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, double, double, char, gtint_t, gtint_t, dgemm_ukr_ft, bool>>  str) const {
        gtint_t k       = std::get<0>(str.param);
        double alpha    = std::get<1>(str.param);
        double beta     = std::get<2>(str.param);
        char storage    = std::get<3>(str.param);
        bool memory_test = std::get<7>(str.param);

        std::string str_name;
        str_name += "_k_" + std::to_string(k);
               str_name += "_alpha_" + testinghelpers::get_value_string(alpha);;
        str_name += "_beta_" + testinghelpers::get_value_string(beta);;
        str_name = str_name + "_storage_" + storage;
        str_name += ( memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_zen4_asm_32x6,
    dgemmUkrNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1),   // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(32),                          // values of m
        ::testing::Values(6),                           // values of n
        ::testing::Values(bli_dgemm_zen4_asm_32x6),
        ::testing::Values(true, false)                          // memory test
    ),
    ::dgemmUkrNatPrint()
);

INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_zen4_asm_8x24,
    dgemmUkrNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1), // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(8),                           // values of m
        ::testing::Values(24),                          // values of n
        ::testing::Values(bli_dgemm_zen4_asm_8x24),
        ::testing::Values(true, false)                          // memory test
    ),
    ::dgemmUkrNatPrint()
);
#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_haswell_asm_6x8,
    dgemmUkrNat,
    ::testing::Combine(
        ::testing::Range(gtint_t(0), gtint_t(17), 1), // values of k
        ::testing::Values(2.0, 1.0, -1.0),              // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),         // beta value
        ::testing::Values('r', 'c'),                    // storage
        ::testing::Values(6),                           // values of m
        ::testing::Values(8),                           // values of n
        ::testing::Values(bli_dgemm_haswell_asm_6x8),
        ::testing::Values(true, false)                          // memory test
    ),
    ::dgemmUkrNatPrint()
);
#endif

//Function pointer specific to dgemm kernel that handles
//special case where k=1.
typedef err_t (*gemm_k1_kernel)
     (
        dim_t  m,
        dim_t  n,
        dim_t  k,
        double*    alpha,
        double*    a, const inc_t lda,
        double*    b, const inc_t ldb,
        double*    beta,
        double*    c, const inc_t ldc
    );

//Since AOCL BLAS is having separate kernel optimized to handle k=1 cases
//dgemm computation, a micro-kernel testing added that validates dgemm kernel
//for k=1 case.

class dgemmUkrk1 :
        public ::testing::TestWithParam<std::tuple<double, double, char, gtint_t, gtint_t, gemm_k1_kernel, bool>> {};
// k, alpha, beta, storage of c, m, n, dgemm k1 kernel, memory test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmUkrk1);

TEST_P(dgemmUkrk1, k1_kernel_testing)
{
    using T = double;
    gtint_t k      = 1;
    T alpha        = std::get<0>(GetParam());  // alpha
    T beta         = std::get<1>(GetParam());  // beta
    char storage   = std::get<2>(GetParam());  // indicates storage of all matrix operands
    // Fix m and n to MR and NR respectively.
    gtint_t m = std::get<3>(GetParam());
    gtint_t n = std::get<4>(GetParam());
    gemm_k1_kernel kern_ptr = std::get<5>(GetParam());
    bool memory_test = std::get<6>(GetParam());
    test_gemmk1_ukr(kern_ptr, m, n, k, storage, alpha, beta, memory_test);
}// end of function



class dgemmUkrk1Print {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<double, double, char, gtint_t, gtint_t, gemm_k1_kernel, bool>>  str) const {
        gtint_t k       = 1;
        double alpha    = std::get<0>(str.param);
        double beta     = std::get<1>(str.param);
        char storage    = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        bool memory_test = std::get<6>(str.param);

        std::string str_name;
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name = str_name + "_" + storage;
        str_name += ( memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};


#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_24x8_avx512_k1_nn,
    dgemmUkrk1,
    ::testing::Combine(

        ::testing::Values(2.0, 1.0, -1.0),             // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),        // beta value
        ::testing::Values('c'),                        // storage
        ::testing::Range(gtint_t(1), gtint_t(25), 1),  // values of m
        ::testing::Range(gtint_t(1), gtint_t(9), 1),   // values of n
        ::testing::Values(bli_dgemm_24x8_avx512_k1_nn),
        ::testing::Values(true, false)                 // memory test
    ),
    ::dgemmUkrk1Print()
);

#endif

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
INSTANTIATE_TEST_SUITE_P (
    bli_dgemm_8x6_avx2_k1_nn,
    dgemmUkrk1,
    ::testing::Combine(
        ::testing::Values(2.0, 1.0, -1.0),           // alpha value
        ::testing::Values(1.0, 0.0, -1.0, 2.3),      // beta value
        ::testing::Values('c'),                      // storage
        ::testing::Range(gtint_t(1), gtint_t(9), 1), // values of m
        ::testing::Range(gtint_t(1), gtint_t(7), 1), // values of n
        ::testing::Values(bli_dgemm_8x6_avx2_k1_nn),
        ::testing::Values(true, false)               // memory test
    ),
    ::dgemmUkrk1Print()
);
#endif

#ifdef BLIS_ENABLE_SMALL_MATRIX

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
class dgemmSmallUkernel :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char, bool>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(dgemmSmallUkernel);

//m, n, k, alpha, beta, storage scheme, memory test
TEST_P(dgemmSmallUkernel, gemm_small)
{
    using T = double;
    gtint_t m      = std::get<0>(GetParam());      // dimension m
    gtint_t n      = std::get<1>(GetParam());      // dimension n
    gtint_t k      = std::get<2>(GetParam());      // dimension k
    T alpha        = std::get<3>(GetParam());      // alpha
    T beta         = std::get<4>(GetParam());      // beta
    char storage   = std::get<5>(GetParam());      // indicates storage of all matrix operands
    bool memory_test   = std::get<6>(GetParam());  // memory test enable or disable


    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, 'n', k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, 0 );

    const num_t dt = BLIS_DOUBLE;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao = BLIS_OBJECT_INITIALIZER;
    obj_t       bo = BLIS_OBJECT_INITIALIZER;
    obj_t       betao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       co = BLIS_OBJECT_INITIALIZER;

    dim_t       m0_a, n0_a;
    dim_t       m0_b, n0_b;

    bli_set_dims_with_trans(BLIS_NO_TRANSPOSE, m, k, &m0_a, &n0_a);
    bli_set_dims_with_trans(BLIS_NO_TRANSPOSE, k, n, &m0_b, &n0_b);

    bli_obj_init_finish_1x1(dt, (double*)&alpha, &alphao);
    bli_obj_init_finish_1x1(dt, (double*)&beta, &betao);

    if(memory_test == true)
    {
        srand(time(NULL));
        double *a, *b, *c, *cref = NULL;
        // Allocate memory for A
        testinghelpers::ProtectedBuffer a_buf( m * k * lda * sizeof(double), false, memory_test );
        // Allocate memory for B
        testinghelpers::ProtectedBuffer b_buf( k * n * ldb * sizeof(double), false, memory_test );
        testinghelpers::ProtectedBuffer c_buf( m * n * ldc * sizeof(double), false, memory_test );

        a = (double*)a_buf.greenzone_1;
        b = (double*)b_buf.greenzone_1;
        c = (double*)c_buf.greenzone_1;

        cref = (double*)malloc(m * n * ldc * sizeof(double));

        testinghelpers::datagenerators::randomgenerators<double>( -2, 8, 'c', m, k, (a), 'n', lda);
        memset(b, rand() % 5, n*k*ldb*sizeof(double));
        memset(cref, rand() % 3, m*n*ldc*sizeof(double));
        memcpy(c, cref, m*n*ldc*sizeof(double));

        bli_obj_init_finish(dt, m, k, (double*)a, 1, lda, &ao);
        bli_obj_init_finish(dt, k, n, (double*)b, 1, ldb, &bo);
        bli_obj_init_finish(dt, m, n, (double*)c, 1, ldc, &co);

        bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &ao);
        bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &bo);

        // add signal handler for segmentation fault
        testinghelpers::ProtectedBuffer::start_signal_handler();
        try
        {
            bli_dgemm_small ( &alphao,
                              &ao,
                              &bo,
                              &betao,
                              &co,
                              NULL,
                              NULL
                            );

            if(memory_test == true)
            {
                a = (double*)a_buf.greenzone_2;
                b = (double*)b_buf.greenzone_2;
                c = (double*)c_buf.greenzone_2;

                memcpy(a, a_buf.greenzone_1, m * k * lda * sizeof(double));
                memcpy(b, b_buf.greenzone_1, n * k * ldb * sizeof(double));
                memcpy(c, cref, m * n * ldc * sizeof(double));

                bli_dgemm_small ( &alphao,
                                &ao,
                                &bo,
                                &betao,
                                &co,
                                NULL,
                                NULL
                                );
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

        // Set the threshold for the errors:
        // Check gtestsuite gemm.h or netlib source code for reminder of the
        // functionality from which we estimate operation count per element
        // of output, and hence the multipler for epsilon.
        double thresh;
        if (m == 0 || n == 0)
            thresh = 0.0;
        else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
                  beta == testinghelpers::ONE<T>()))
            thresh = 0.0;
        else
            thresh = (3*k+1)*testinghelpers::getEpsilon<T>();
            //thresh = (4*k+1)*testinghelpers::getEpsilon<T>();

        // call reference implementation
        testinghelpers::ref_gemm<T>( storage, 'n', 'n', m, n, k, alpha,
                                    a, lda, b, ldb, beta, cref, ldc);
        // Check component-wise error
        computediff<T>( "C", storage, m, n, c, cref, ldc, thresh );

        free(cref);
    }
    else 
    {
        //----------------------------------------------------------
        //         Initialize matrics with random numbers
        //----------------------------------------------------------
        std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, 'n', m, k, lda );
        std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storage, 'n', k, n, ldb );
        std::vector<T> c = testinghelpers::get_random_matrix<T>( -3, 5, storage, 'n', m, n, ldc );

        std::vector<T> c_ref(c);

        bli_obj_init_finish(dt, m0_a, n0_a, (double*)a.data(), 1, lda, &ao);
        bli_obj_init_finish(dt, m0_b, n0_b, (double*)b.data(), 1, ldb, &bo);
        bli_obj_init_finish(dt, m, n, (double*)c.data(), 1, ldc, &co);

        bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &ao);
        bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &bo);

        bli_dgemm_small ( &alphao,
                          &ao,
                          &bo,
                          &betao,
                          &co,
                          NULL,
                          NULL
                        );

        // Set the threshold for the errors:
        // Check gtestsuite gemm.h or netlib source code for reminder of the
        // functionality from which we estimate operation count per element
        // of output, and hence the multipler for epsilon.
        double thresh;
        if (m == 0 || n == 0)
            thresh = 0.0;
        else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
                  beta == testinghelpers::ONE<T>()))
            thresh = 0.0;
        else
            thresh = (3*k+1)*testinghelpers::getEpsilon<T>();
            //thresh = (4*k+1)*testinghelpers::getEpsilon<T>();

        // call reference implementation
        testinghelpers::ref_gemm<T>( storage, 'n', 'n', m, n, k, alpha,
                                    a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc);
        // Check component-wise error
        computediff<T>( "C", storage, m, n, c.data(), c_ref.data(), ldc, thresh );
    }

}// end of function



class dgemmSmallUkernelPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, double, double, char, bool>>  str) const {
        gtint_t m       = std::get<0>(str.param);
        gtint_t n       = std::get<1>(str.param);
        gtint_t k       = std::get<2>(str.param);
        double alpha    = std::get<3>(str.param);
        double beta     = std::get<4>(str.param);
        char storage    = std::get<5>(str.param);
        bool memory_test    = std::get<6>(str.param);

        std::string str_name;
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name = str_name + "_" + storage;
        str_name += ( memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

INSTANTIATE_TEST_SUITE_P (
        bli_dgemm_small,
        dgemmSmallUkernel,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(21), 1), // values of m
            ::testing::Range(gtint_t(1), gtint_t(11), 1), // values of n
            ::testing::Range(gtint_t(1), gtint_t(20), 1), // values of k
            ::testing::Values(2.0, 1.0, -1.0),            // alpha value
            ::testing::Values(1.0, 0.0, -1.0, 2.3),       // beta value
            ::testing::Values('c'),                       // storage
            ::testing::Values(true, false)                // memory test
        ),
        ::dgemmSmallUkernelPrint()
    );
#endif

#endif
