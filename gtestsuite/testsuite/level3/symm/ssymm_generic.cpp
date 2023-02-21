#include <gtest/gtest.h>
#include "test_symm.h"

class ssymmTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   gtint_t,
                                                   float,
                                                   float,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   char>> {};

TEST_P(ssymmTest, RandomData) {
    using T = float;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // specifies if the symmetric matrix A appears left or right in
    // the matrix multiplication
    char side = std::get<1>(GetParam());
    // specifies if the upper or lower triangular part of A is used
    char uplo = std::get<2>(GetParam());
    // denotes whether matrix a is n,c
    char conja = std::get<3>(GetParam());
    // denotes whether matrix b is n,c,t,h
    char transb = std::get<4>(GetParam());
    // matrix size m
    gtint_t m  = std::get<5>(GetParam());
    // matrix size n
    gtint_t n  = std::get<6>(GetParam());
    // specifies alpha value
    T alpha = std::get<7>(GetParam());
    // specifies beta value
    T beta = std::get<8>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<9>(GetParam());
    gtint_t ldb_inc = std::get<10>(GetParam());
    gtint_t ldc_inc = std::get<11>(GetParam());
    // specifies the datatype for randomgenerators
    char datatype   = std::get<12>(GetParam());

    // Set the threshold for the errors:
    double thresh = 8*m*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_symm<T>(storage, side, uplo, conja, transb, m, n, lda_inc, ldb_inc, ldc_inc, alpha, beta, thresh, datatype);
}

class ssymmTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, char, char, char, char, gtint_t, gtint_t, float, float, gtint_t, gtint_t, gtint_t,char>> str) const {
        char sfm        = std::get<0>(str.param);
        char side       = std::get<1>(str.param);
        char uplo       = std::get<2>(str.param);
        char conja      = std::get<3>(str.param);
        char tsb        = std::get<4>(str.param);
        gtint_t m       = std::get<5>(str.param);
        gtint_t n       = std::get<6>(str.param);
        float alpha     = std::get<7>(str.param);
        float beta      = std::get<8>(str.param);
        gtint_t lda_inc = std::get<9>(str.param);
        gtint_t ldb_inc = std::get<10>(str.param);
        gtint_t ldc_inc = std::get<11>(str.param);
        char datatype   = std::get<12>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "ssymm_";
#elif TEST_CBLAS
        std::string str_name = "cblas_ssymm";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_ssymm";
#endif
        str_name = str_name + "_" + sfm+sfm+sfm;
        str_name = str_name + "_" + side + uplo;
        str_name = str_name + "_" + conja + tsb;
        str_name = str_name + "_" + std::to_string(m);
        str_name = str_name + "_" + std::to_string(n);
        std::string alpha_str = ( alpha > 0) ? std::to_string(int(alpha)) : "m" + std::to_string(int(std::abs(alpha)));
        str_name = str_name + "_a" + alpha_str;
        std::string beta_str = ( beta > 0) ? std::to_string(int(beta)) : "m" + std::to_string(int(std::abs(beta)));
        str_name = str_name + "_b" + beta_str;
        str_name = str_name + "_" + std::to_string(lda_inc);
        str_name = str_name + "_" + std::to_string(ldb_inc);
        str_name = str_name + "_" + std::to_string(ldc_inc);
        str_name = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        ssymmTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('l','r'),                                      // l:left, r:right
            ::testing::Values('u','l'),                                      // u:upper, l:lower
            ::testing::Values('n'),                                          // conja
            ::testing::Values('n'),                                          // transb
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // m
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // n
            ::testing::Values( 1.0, -2.0),                                   // alpha
            ::testing::Values(-1.0,  1.0),                                   // beta
            ::testing::Values(gtint_t(0), gtint_t(3)),                       // increment to the leading dim of a
            ::testing::Values(gtint_t(0), gtint_t(1)),                       // increment to the leading dim of b
            ::testing::Values(gtint_t(0), gtint_t(9)),                       // increment to the leading dim of c
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : dcomplex  datatype type tested
        ),
        ::ssymmTestPrint()
    );