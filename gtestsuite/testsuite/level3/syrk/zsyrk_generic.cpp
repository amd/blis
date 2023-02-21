#include <gtest/gtest.h>
#include "test_syrk.h"

class zsyrkTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   dcomplex,
                                                   gtint_t,
                                                   gtint_t,
                                                   char>> {};

TEST_P(zsyrkTest, RandomData) {
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
    // matrix size m
    gtint_t m  = std::get<3>(GetParam());
    // matrix size k
    gtint_t k  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // specifies beta value
    T beta = std::get<6>(GetParam());
    // lda, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<7>(GetParam());
    gtint_t ldc_inc = std::get<8>(GetParam());
    // specifies the datatype for randomgenerators
    char datatype   = std::get<9>(GetParam());

    // Set the threshold for the errors:
    double thresh = m*k*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_syrk<T>(storage, uplo, transa, m, k, lda_inc, ldc_inc, alpha, beta, thresh, datatype);
}

class zsyrkTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, gtint_t, gtint_t, dcomplex, dcomplex, gtint_t, gtint_t, char>> str) const {
        char sfm        = std::get<0>(str.param);
        char uplo       = std::get<1>(str.param);
        char tsa        = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t k       = std::get<4>(str.param);
        dcomplex alpha  = std::get<5>(str.param);
        dcomplex beta   = std::get<6>(str.param);
        gtint_t lda_inc = std::get<7>(str.param);
        gtint_t ldc_inc = std::get<8>(str.param);
        char datatype   = std::get<9>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "zsyrk_";
#elif TEST_CBLAS
        std::string str_name = "cblas_zsyrk";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_zsyrk";
#endif
        str_name = str_name + "_" + sfm+sfm+sfm;
        str_name = str_name + "_" + uplo;
        str_name = str_name + "_" + tsa;
        str_name = str_name + "_" + std::to_string(m);
        str_name = str_name + "_" + std::to_string(k);
        std::string alpha_str = ( alpha.real > 0) ? std::to_string(int(alpha.real)) : ("m" + std::to_string(int(std::abs(alpha.real))));
                    alpha_str = alpha_str + "pi" + (( alpha.imag > 0) ? std::to_string(int(alpha.imag)) : ("m" + std::to_string(int(std::abs(alpha.imag)))));
        str_name = str_name + "_a" + alpha_str;
        std::string beta_str = ( beta.real > 0) ? std::to_string(int(beta.real)) : ("m" + std::to_string(int(std::abs(beta.real))));
                    beta_str = beta_str + "pi" + (( beta.imag > 0) ? std::to_string(int(beta.imag)) : ("m" + std::to_string(int(std::abs(beta.imag)))));
        str_name = str_name + "_a" + beta_str;
        str_name = str_name + "_" + std::to_string(lda_inc);
        str_name = str_name + "_" + std::to_string(ldc_inc);
        str_name = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        zsyrkTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // u:upper, l:lower
            ::testing::Values('n','t'),                                      // transa
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // m
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // k
            ::testing::Values(dcomplex{2.0, -1.0}, dcomplex{-2.0, 3.0}),     // alpha
            ::testing::Values(dcomplex{-3.0, 2.0}, dcomplex{4.0, -1.0}),     // beta
            ::testing::Values(gtint_t(0), gtint_t(2)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(5)),                       // increment to the leading dim of c
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : dcomplex  datatype type tested
        ),
        ::zsyrkTestPrint()
    );