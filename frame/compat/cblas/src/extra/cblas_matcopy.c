/*
 * cblas_matcopy.c
 *
 * CBLAS interface to the BLAS-like extension routines ?imatcopy (in-place
 * scale/transpose) and ?omatcopy (out-of-place scale/transpose).
 *
 * BLIS provides only Fortran-bound, column-major workers
 * (?imatcopy_blis_impl / ?omatcopy_blis_impl) that take a trans character and
 * (rows, cols, lda, ldb). These wrappers add the standard MKL/OpenBLAS-style
 * cblas_?imatcopy / cblas_?omatcopy entry points on top of them: for
 * CblasRowMajor the (rows, cols) are swapped (a row-major rows x cols matrix
 * is the same memory as a column-major cols x rows matrix) while trans, lda
 * and ldb are passed through unchanged.
 *
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 */

#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
#include "cblas.h"
#include "cblas_f77.h"

// Map a CBLAS_TRANSPOSE enum (or a raw 'N'/'T'/'C'/'R' char) to a BLAS trans
// character; returns '\0' for an unrecognized value.
static f77_char bli_cblas_matcopy_trans( enum CBLAS_TRANSPOSE trans )
{
    switch ( ( int )trans )
    {
        case CblasNoTrans:   return 'N';
        case CblasTrans:     return 'T';
        case CblasConjTrans: return 'C';
        case 'N': case 'n':  return 'N';
        case 'T': case 't':  return 'T';
        case 'C': case 'c':  return 'C';
        case 'R': case 'r':  return 'R';
        default:             return '\0';
    }
}

// Validate Order and Trans the way the other CBLAS entry points do (e.g.
// cblas_sgemm): on an illegal value, report it through cblas_xerbla and return
// '\0' so the caller returns early with a predictable error path.
static f77_char bli_cblas_matcopy_check( enum CBLAS_ORDER Order,
                                         enum CBLAS_TRANSPOSE Trans,
                                         const char* rout )
{
    if ( Order != CblasColMajor && Order != CblasRowMajor )
    {
        cblas_xerbla( 1, rout, "Illegal Order setting, %d\n", ( int )Order );
        return '\0';
    }
    f77_char trans = bli_cblas_matcopy_trans( Trans );
    if ( trans == '\0' )
        cblas_xerbla( 2, rout, "Illegal Trans setting, %d\n", ( int )Trans );
    return trans;
}

void cblas_simatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, float alpha,
                      float* a, f77_int lda, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_simatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    simatcopy_blis_impl( &trans, &m, &n, &alpha, a, &lda, &ldb );
}

void cblas_dimatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, double alpha,
                      double* a, f77_int lda, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_dimatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    dimatcopy_blis_impl( &trans, &m, &n, &alpha, a, &lda, &ldb );
}

void cblas_cimatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, const void* alpha,
                      void* a, f77_int lda, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_cimatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    cimatcopy_blis_impl( &trans, &m, &n, ( const scomplex* )alpha,
                         ( scomplex* )a, &lda, &ldb );
}

void cblas_zimatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, const void* alpha,
                      void* a, f77_int lda, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_zimatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    zimatcopy_blis_impl( &trans, &m, &n, ( const dcomplex* )alpha,
                         ( dcomplex* )a, &lda, &ldb );
}

void cblas_somatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, float alpha,
                      const float* a, f77_int lda, float* b, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_somatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    somatcopy_blis_impl( &trans, &m, &n, &alpha, a, &lda, b, &ldb );
}

void cblas_domatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, double alpha,
                      const double* a, f77_int lda, double* b, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_domatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    domatcopy_blis_impl( &trans, &m, &n, &alpha, a, &lda, b, &ldb );
}

void cblas_comatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, const void* alpha,
                      const void* a, f77_int lda, void* b, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_comatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    comatcopy_blis_impl( &trans, &m, &n, ( const scomplex* )alpha,
                         ( const scomplex* )a, &lda, ( scomplex* )b, &ldb );
}

void cblas_zomatcopy( enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE Trans,
                      f77_int rows, f77_int cols, const void* alpha,
                      const void* a, f77_int lda, void* b, f77_int ldb )
{
    f77_char trans = bli_cblas_matcopy_check( Order, Trans, "cblas_zomatcopy" );
    if ( trans == '\0' ) return;
    f77_int  m = ( Order == CblasRowMajor ) ? cols : rows;
    f77_int  n = ( Order == CblasRowMajor ) ? rows : cols;
    zomatcopy_blis_impl( &trans, &m, &n, ( const dcomplex* )alpha,
                         ( const dcomplex* )a, &lda, ( dcomplex* )b, &ldb );
}

#endif
