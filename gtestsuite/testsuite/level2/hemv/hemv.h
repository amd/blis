#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *        y := alpha*A*x + beta*y
 *     or y := beta * y + alpha * conja(A) * conjx(x)
 * @param[in]     conja  specifies the form of A to be used in
                         the matrix-vector multiplication (BLIS_TYPED only)
 * @param[in]     conjx  specifies the form of xp to be used in
                         the matrix-vector multiplication (BLIS_TYPED only)
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 * @param[in]     xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] yp     specifies pointer which points to the first element of yp
 * @param[in]     incy   specifies storage spacing between elements of yp.
 */

template<typename T>
static void hemv_( char uploa, gtint_t n, T* alpha, T* ap, gtint_t lda,
                    T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    if constexpr (std::is_same<T, scomplex>::value)
        chemv_( &uploa, &n, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zhemv_( &uploa, &n, alpha, ap, &lda, xp, &incx, beta, yp, &incy );
    else
        throw std::runtime_error("Error in testsuite/level2/hemv.h: Invalid typename in hemv_().");
}

template<typename T>
static void cblas_hemv( char storage, char uploa, gtint_t n, T* alpha,
    T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp, gtint_t incy )
{
    enum CBLAS_ORDER cblas_order;
    if( storage == 'c' || storage == 'C' )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uplo;
    if( (uploa == 'u') || (uploa == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    if constexpr (std::is_same<T, scomplex>::value)
        cblas_chemv( cblas_order, cblas_uplo, n, alpha, ap, lda, xp, incx, beta, yp, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zhemv( cblas_order, cblas_uplo, n, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/hemv.h: Invalid typename in cblas_hemv().");
}

template<typename T>
static void typed_hemv( char storage, char uplo, char conj_a, char conj_x,
    gtint_t n, T* alpha, T* a, gtint_t lda, T* x, gtint_t incx, T* beta,
    T* y, gtint_t incy )
{
    uplo_t uploa;
    conj_t conja;
    conj_t conjx;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );
    testinghelpers::char_to_blis_conj ( conj_a, &conja );
    testinghelpers::char_to_blis_conj ( conj_x, &conjx );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, scomplex>::value)
        bli_chemv( uploa, conja, conjx, n, alpha, a, rsa, csa, x, incx, beta, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zhemv( uploa, conja, conjx, n, alpha, a, rsa, csa, x, incx, beta, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/hemv.h: Invalid typename in typed_hemv().");
}

template<typename T>
static void hemv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T* alpha, T* ap, gtint_t lda, T* xp, gtint_t incx, T* beta, T* yp,
    gtint_t incy )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        hemv_<T>( uploa, n, alpha, ap, lda, xp, incx, beta, yp, incy );
    else
        throw std::runtime_error("Error in testsuite/level2/hemv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_hemv<T>( storage, uploa, n, alpha, ap, lda, xp, incx, beta, yp, incy );
#elif TEST_BLIS_TYPED
    typed_hemv<T>( storage, uploa, conja, conjx, n, alpha, ap, lda, xp, incx, beta, yp, incy );
#else
    throw std::runtime_error("Error in testsuite/level2/hemv.h: No interfaces are set to be tested.");
#endif
}