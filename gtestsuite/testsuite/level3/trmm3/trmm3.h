#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *        C := beta * C_orig + alpha * transa(A) * transb(B)
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     side   specifies if the symmetric matrix A appears left or right in
                         the matrix multiplication
 * @param[in]     uplo   specifies if the upper or lower triangular part of A is used
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     diaga  specifies whether upper or lower triangular part of the matrix A
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies the number of rows and cols of the  matrix
                         op( A ) and rows of the matrix C and B
 * @param[in]     n      specifies the number of columns of the matrix
                         op( B ) and the number of columns of the matrix C
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in]     bp     specifies pointer which points to the first element of bp
 * @param[in]     rsb    specifies row increment of bp.
 * @param[in]     csb    specifies column increment of bp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp
 * @param[in]     rsc    specifies row increment of cp.
 * @param[in]     csc    specifies column increment of cp.
 */

template<typename T>
static void typed_trmm3( char storage, char side, char uplo, char trnsa,
    char diag, char trnsb, gtint_t m, gtint_t n, T *alpha, T *a, gtint_t lda,
    T *b, gtint_t ldb, T *beta, T *c, gtint_t ldc )
{
    side_t  sidea;
    uplo_t  uploa;
    trans_t transa;
    trans_t transb;
    diag_t  diaga;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_side( side, &sidea );
    testinghelpers::char_to_blis_uplo( uplo, &uploa );
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );
    testinghelpers::char_to_blis_diag( diag, &diaga );

    dim_t rsa,csa;
    dim_t rsb,csb;
    dim_t rsc,csc;

    rsa=rsb=rsc=1;
    csa=csb=csc=1;
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csb = ldb ;
        csc = ldc ;
    } else {
        rsa = lda ;
        rsb = ldb ;
        rsc = ldc ;
    }

    if constexpr (std::is_same<T, float>::value) {
        bli_strmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else if constexpr (std::is_same<T, double>::value) {
        bli_dtrmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else if constexpr (std::is_same<T, scomplex>::value)  {
        bli_ctrmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else if constexpr (std::is_same<T, dcomplex>::value)  {
        bli_ztrmm3( sidea, uploa, transa, diaga, transb, m, n, alpha,
                      a, rsa, csa, b, rsb, csb, beta, c, rsc, csc );
    }
    else
        throw std::runtime_error("Error in testsuite/level3/trmm3.h: Invalid typename in typed_trmm3().");
}

template<typename T>
static void trmm3( char storage, char side, char uploa, char transa, char diaga,
                  char transb, gtint_t m, gtint_t n, T *alpha, T *ap, gtint_t lda,
                  T *bp, gtint_t ldb, T *beta, T *c, gtint_t ldc )
{
#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: BLAS interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: BLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, alpha,
                                            ap, lda, bp, ldb, beta, c, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/trmm3.h: No interfaces are set to be tested.");
#endif
}
