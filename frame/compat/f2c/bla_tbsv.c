/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

// Make thread settings local to each thread calling BLIS routines.
// (The definition resides in bli_rntm.c.)
extern BLIS_THREAD_LOCAL rntm_t tl_rntm;

/* ctbsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ 
int PASTEF77S(c,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_scomplex *a, const bla_integer *lda, bla_scomplex *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    bla_scomplex q__1, q__2, q__3;

    /* Builtin functions */
    //void bla_c_div(bla_scomplex *, bla_scomplex *, bla_scomplex *), bla_r_cnjg(bla_scomplex *, bla_scomplex *);

    /* Local variables */
    bla_integer info;
    bla_scomplex temp;
    bla_integer i__, j, l;
    //extern bla_logical PASTE_LSAME(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kplus1, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTE_XERBLA(bla_character *, bla_integer *, ftnlen);
    bla_logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTBSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular band matrix, with ( k + 1 ) */
/*  diagonals. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   conjg( A' )*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - COMPLEX          array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */

    // Initialize info_value to 0
    gint_t info_value = 0;
    bli_rntm_set_info_value_only( info_value, &tl_rntm );

    info = 0;
    if (! PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTE_LSAME(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	PASTE_XERBLA("CTBSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = PASTE_LSAME(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = PASTE_LSAME(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed by sequentially with one pass through A. */

    if (PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_creal(x[i__1]) != 0.f || bli_cimag(x[i__1]) != 0.f) 
			{
			l = kplus1 - j;
			if (nounit) {
			    i__1 = j;
			    bla_c_div(&q__1, &x[j], &a[kplus1 + j * a_dim1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__1] );
			}
			i__1 = j;
			bli_csets( (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__4 = l + i__ + j * a_dim1;
			    bli_csets( (bli_creal(temp) * bli_creal(a[i__4]) - bli_cimag(temp) * bli_cimag(a[i__4])), (bli_creal(temp) * bli_cimag(a[i__4]) + bli_cimag(temp) * bli_creal(a[i__4])), q__2 );
			    bli_csets( (bli_creal(x[i__3]) - bli_creal(q__2)), (bli_cimag(x[i__3]) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
/* L10: */
			}
		    }
/* L20: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    kx -= *incx;
		    i__1 = jx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_creal(x[i__1]) != 0.f || bli_cimag(x[i__1]) != 0.f) 
			{
			ix = kx;
			l = kplus1 - j;
			if (nounit) {
			    i__1 = jx;
			    bla_c_div(&q__1, &x[jx], &a[kplus1 + j * a_dim1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__1] );
			}
			i__1 = jx;
			bli_csets( (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = l + i__ + j * a_dim1;
			    bli_csets( (bli_creal(temp) * bli_creal(a[i__4]) - bli_cimag(temp) * bli_cimag(a[i__4])), (bli_creal(temp) * bli_cimag(a[i__4]) + bli_cimag(temp) * bli_creal(a[i__4])), q__2 );
			    bli_csets( (bli_creal(x[i__3]) - bli_creal(q__2)), (bli_cimag(x[i__3]) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			    ix -= *incx;
/* L30: */
			}
		    }
		    jx -= *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f) 
			{
			l = 1 - j;
			if (nounit) {
			    i__2 = j;
			    bla_c_div(&q__1, &x[j], &a[j * a_dim1 + 1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			}
			i__2 = j;
			bli_csets( (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = l + i__ + j * a_dim1;
			    bli_csets( (bli_creal(temp) * bli_creal(a[i__5]) - bli_cimag(temp) * bli_cimag(a[i__5])), (bli_creal(temp) * bli_cimag(a[i__5]) + bli_cimag(temp) * bli_creal(a[i__5])), q__2 );
			    bli_csets( (bli_creal(x[i__4]) - bli_creal(q__2)), (bli_cimag(x[i__4]) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__3] );
/* L50: */
			}
		    }
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    kx += *incx;
		    i__2 = jx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f) 
			{
			ix = kx;
			l = 1 - j;
			if (nounit) {
			    i__2 = jx;
			    bla_c_div(&q__1, &x[jx], &a[j * a_dim1 + 1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__2] );
			}
			i__2 = jx;
			bli_csets( (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = l + i__ + j * a_dim1;
			    bli_csets( (bli_creal(temp) * bli_creal(a[i__5]) - bli_cimag(temp) * bli_cimag(a[i__5])), (bli_creal(temp) * bli_cimag(a[i__5]) + bli_cimag(temp) * bli_creal(a[i__5])), q__2 );
			    bli_csets( (bli_creal(x[i__4]) - bli_creal(q__2)), (bli_cimag(x[i__4]) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), x[i__3] );
			    ix += *incx;
/* L70: */
			}
		    }
		    jx += *incx;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x  or  x := inv( conjg( A') )*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    bli_csets( (bli_creal(x[i__2])), (bli_cimag(x[i__2])), temp );
		    l = kplus1 - j;
		    if (noconj) {
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			    i__2 = l + i__ + j * a_dim1;
			    i__3 = i__;
			    bli_csets( (bli_creal(a[i__2]) * bli_creal(x[i__3]) - bli_cimag(a[i__2]) * bli_cimag(x[i__3])), (bli_creal(a[i__2]) * bli_cimag(x[i__3]) + bli_cimag(a[i__2]) * bli_creal(x[i__3])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L90: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &a[kplus1 + j * a_dim1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    } else {
/* Computing MAX */
			i__4 = 1, i__2 = j - *k;
			i__3 = j - 1;
			for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__4 = i__;
			    bli_csets( (bli_creal(q__3) * bli_creal(x[i__4]) - bli_cimag(q__3) * bli_cimag(x[i__4])), (bli_creal(q__3) * bli_cimag(x[i__4]) + bli_cimag(q__3) * bli_creal(x[i__4])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L100: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &a[kplus1 + j * a_dim1]);
			    bla_c_div(&q__1, &temp, &q__2);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    }
		    i__3 = j;
		    bli_csets( (bli_creal(temp)), (bli_cimag(temp)), x[i__3] );
/* L110: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__3 = jx;
		    bli_csets( (bli_creal(x[i__3])), (bli_cimag(x[i__3])), temp );
		    ix = kx;
		    l = kplus1 - j;
		    if (noconj) {
/* Computing MAX */
			i__3 = 1, i__4 = j - *k;
			i__2 = j - 1;
			for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			    i__3 = l + i__ + j * a_dim1;
			    i__4 = ix;
			    bli_csets( (bli_creal(a[i__3]) * bli_creal(x[i__4]) - bli_cimag(a[i__3]) * bli_cimag(x[i__4])), (bli_creal(a[i__3]) * bli_cimag(x[i__4]) + bli_cimag(a[i__3]) * bli_creal(x[i__4])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    ix += *incx;
/* L120: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &a[kplus1 + j * a_dim1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    } else {
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__2 = ix;
			    bli_csets( (bli_creal(q__3) * bli_creal(x[i__2]) - bli_cimag(q__3) * bli_cimag(x[i__2])), (bli_creal(q__3) * bli_cimag(x[i__2]) + bli_cimag(q__3) * bli_creal(x[i__2])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    ix += *incx;
/* L130: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &a[kplus1 + j * a_dim1]);
			    bla_c_div(&q__1, &temp, &q__2);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    }
		    i__4 = jx;
		    bli_csets( (bli_creal(temp)), (bli_cimag(temp)), x[i__4] );
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L140: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    bli_csets( (bli_creal(x[i__1])), (bli_cimag(x[i__1])), temp );
		    l = 1 - j;
		    if (noconj) {
/* Computing MIN */
			i__1 = *n, i__4 = j + *k;
			i__2 = j + 1;
			for (i__ = f2c_min(i__1,i__4); i__ >= i__2; --i__) {
			    i__1 = l + i__ + j * a_dim1;
			    i__4 = i__;
			    bli_csets( (bli_creal(a[i__1]) * bli_creal(x[i__4]) - bli_cimag(a[i__1]) * bli_cimag(x[i__4])), (bli_creal(a[i__1]) * bli_cimag(x[i__4]) + bli_cimag(a[i__1]) * bli_creal(x[i__4])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L150: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &a[j * a_dim1 + 1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    } else {
/* Computing MIN */
			i__2 = *n, i__1 = j + *k;
			i__4 = j + 1;
			for (i__ = f2c_min(i__2,i__1); i__ >= i__4; --i__) {
			    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__2 = i__;
			    bli_csets( (bli_creal(q__3) * bli_creal(x[i__2]) - bli_cimag(q__3) * bli_cimag(x[i__2])), (bli_creal(q__3) * bli_cimag(x[i__2]) + bli_cimag(q__3) * bli_creal(x[i__2])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
/* L160: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &a[j * a_dim1 + 1]);
			    bla_c_div(&q__1, &temp, &q__2);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    }
		    i__4 = j;
		    bli_csets( (bli_creal(temp)), (bli_cimag(temp)), x[i__4] );
/* L170: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__4 = jx;
		    bli_csets( (bli_creal(x[i__4])), (bli_cimag(x[i__4])), temp );
		    ix = kx;
		    l = 1 - j;
		    if (noconj) {
/* Computing MIN */
			i__4 = *n, i__2 = j + *k;
			i__1 = j + 1;
			for (i__ = f2c_min(i__4,i__2); i__ >= i__1; --i__) {
			    i__4 = l + i__ + j * a_dim1;
			    i__2 = ix;
			    bli_csets( (bli_creal(a[i__4]) * bli_creal(x[i__2]) - bli_cimag(a[i__4]) * bli_cimag(x[i__2])), (bli_creal(a[i__4]) * bli_cimag(x[i__2]) + bli_cimag(a[i__4]) * bli_creal(x[i__2])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    ix -= *incx;
/* L180: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &a[j * a_dim1 + 1]);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    } else {
/* Computing MIN */
			i__1 = *n, i__4 = j + *k;
			i__2 = j + 1;
			for (i__ = f2c_min(i__1,i__4); i__ >= i__2; --i__) {
			    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__1 = ix;
			    bli_csets( (bli_creal(q__3) * bli_creal(x[i__1]) - bli_cimag(q__3) * bli_cimag(x[i__1])), (bli_creal(q__3) * bli_cimag(x[i__1]) + bli_cimag(q__3) * bli_creal(x[i__1])), q__2 );
			    bli_csets( (bli_creal(temp) - bli_creal(q__2)), (bli_cimag(temp) - bli_cimag(q__2)), q__1 );
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			    ix -= *incx;
/* L190: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &a[j * a_dim1 + 1]);
			    bla_c_div(&q__1, &temp, &q__2);
			    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp );
			}
		    }
		    i__2 = jx;
		    bli_csets( (bli_creal(temp)), (bli_cimag(temp)), x[i__2] );
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of CTBSV . */

} /* ctbsv_ */

/* dtbsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ 
int PASTEF77S(d,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_double *a, const bla_integer *lda, bla_double *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    bla_integer info;
    bla_double temp;
    bla_integer i__, j, l;
    //extern bla_logical PASTE_LSAME(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kplus1, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTE_XERBLA(bla_character *, bla_integer *, ftnlen);
    bla_logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTBSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular band matrix, with ( k + 1 ) */
/*  diagonals. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   A'*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */

    // Initialize info_value to 0
    gint_t info_value = 0;
    bli_rntm_set_info_value_only( info_value, &tl_rntm );

    info = 0;
    if (! PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTE_LSAME(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	PASTE_XERBLA("DTBSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = PASTE_LSAME(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed by sequentially with one pass through A. */

    if (PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[j] != 0.) 
			{
			l = kplus1 - j;
			if (nounit) {
			    x[j] /= a[kplus1 + j * a_dim1];
			}
			temp = x[j];
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    x[i__] -= temp * a[l + i__ + j * a_dim1];
/* L10: */
			}
		    }
/* L20: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    kx -= *incx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[jx] != 0.) 
			{
			ix = kx;
			l = kplus1 - j;
			if (nounit) {
			    x[jx] /= a[kplus1 + j * a_dim1];
			}
			temp = x[jx];
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    x[ix] -= temp * a[l + i__ + j * a_dim1];
			    ix -= *incx;
/* L30: */
			}
		    }
		    jx -= *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[j] != 0.) 
			{
			l = 1 - j;
			if (nounit) {
			    x[j] /= a[j * a_dim1 + 1];
			}
			temp = x[j];
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[i__] -= temp * a[l + i__ + j * a_dim1];
/* L50: */
			}
		    }
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    kx += *incx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[jx] != 0.) 
			{
			ix = kx;
			l = 1 - j;
			if (nounit) {
			    x[jx] /= a[j * a_dim1 + 1];
			}
			temp = x[jx];
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[ix] -= temp * a[l + i__ + j * a_dim1];
			    ix += *incx;
/* L70: */
			}
		    }
		    jx += *incx;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A')*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    l = kplus1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *k;
		    i__4 = j - 1;
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			temp -= a[l + i__ + j * a_dim1] * x[i__];
/* L90: */
		    }
		    if (nounit) {
			temp /= a[kplus1 + j * a_dim1];
		    }
		    x[j] = temp;
/* L100: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = kx;
		    l = kplus1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *k;
		    i__3 = j - 1;
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			temp -= a[l + i__ + j * a_dim1] * x[ix];
			ix += *incx;
/* L110: */
		    }
		    if (nounit) {
			temp /= a[kplus1 + j * a_dim1];
		    }
		    x[jx] = temp;
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L120: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    l = 1 - j;
/* Computing MIN */
		    i__1 = *n, i__3 = j + *k;
		    i__4 = j + 1;
		    for (i__ = f2c_min(i__1,i__3); i__ >= i__4; --i__) {
			temp -= a[l + i__ + j * a_dim1] * x[i__];
/* L130: */
		    }
		    if (nounit) {
			temp /= a[j * a_dim1 + 1];
		    }
		    x[j] = temp;
/* L140: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = kx;
		    l = 1 - j;
/* Computing MIN */
		    i__4 = *n, i__1 = j + *k;
		    i__3 = j + 1;
		    for (i__ = f2c_min(i__4,i__1); i__ >= i__3; --i__) {
			temp -= a[l + i__ + j * a_dim1] * x[ix];
			ix -= *incx;
/* L150: */
		    }
		    if (nounit) {
			temp /= a[j * a_dim1 + 1];
		    }
		    x[jx] = temp;
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of DTBSV . */

} /* dtbsv_ */

/* stbsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ 
int PASTEF77S(s,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_real *a, const bla_integer *lda, bla_real *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    bla_integer info;
    bla_real temp;
    bla_integer i__, j, l;
    //extern bla_logical PASTE_LSAME(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kplus1, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTE_XERBLA(bla_character *, bla_integer *, ftnlen);
    bla_logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  STBSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular band matrix, with ( k + 1 ) */
/*  diagonals. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   A'*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - REAL             array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - REAL             array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */

    // Initialize info_value to 0
    gint_t info_value = 0;
    bli_rntm_set_info_value_only( info_value, &tl_rntm );

    info = 0;
    if (! PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTE_LSAME(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	PASTE_XERBLA("STBSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = PASTE_LSAME(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed by sequentially with one pass through A. */

    if (PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[j] != 0.f) 
			{
			l = kplus1 - j;
			if (nounit) {
			    x[j] /= a[kplus1 + j * a_dim1];
			}
			temp = x[j];
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    x[i__] -= temp * a[l + i__ + j * a_dim1];
/* L10: */
			}
		    }
/* L20: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    kx -= *incx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[jx] != 0.f) 
			{
			ix = kx;
			l = kplus1 - j;
			if (nounit) {
			    x[jx] /= a[kplus1 + j * a_dim1];
			}
			temp = x[jx];
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    x[ix] -= temp * a[l + i__ + j * a_dim1];
			    ix -= *incx;
/* L30: */
			}
		    }
		    jx -= *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[j] != 0.f) 
			{
			l = 1 - j;
			if (nounit) {
			    x[j] /= a[j * a_dim1 + 1];
			}
			temp = x[j];
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[i__] -= temp * a[l + i__ + j * a_dim1];
/* L50: */
			}
		    }
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    kx += *incx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (x[jx] != 0.f) 
			{
			ix = kx;
			l = 1 - j;
			if (nounit) {
			    x[jx] /= a[j * a_dim1 + 1];
			}
			temp = x[jx];
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[ix] -= temp * a[l + i__ + j * a_dim1];
			    ix += *incx;
/* L70: */
			}
		    }
		    jx += *incx;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A')*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    l = kplus1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *k;
		    i__4 = j - 1;
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			temp -= a[l + i__ + j * a_dim1] * x[i__];
/* L90: */
		    }
		    if (nounit) {
			temp /= a[kplus1 + j * a_dim1];
		    }
		    x[j] = temp;
/* L100: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = kx;
		    l = kplus1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *k;
		    i__3 = j - 1;
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			temp -= a[l + i__ + j * a_dim1] * x[ix];
			ix += *incx;
/* L110: */
		    }
		    if (nounit) {
			temp /= a[kplus1 + j * a_dim1];
		    }
		    x[jx] = temp;
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L120: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    l = 1 - j;
/* Computing MIN */
		    i__1 = *n, i__3 = j + *k;
		    i__4 = j + 1;
		    for (i__ = f2c_min(i__1,i__3); i__ >= i__4; --i__) {
			temp -= a[l + i__ + j * a_dim1] * x[i__];
/* L130: */
		    }
		    if (nounit) {
			temp /= a[j * a_dim1 + 1];
		    }
		    x[j] = temp;
/* L140: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = kx;
		    l = 1 - j;
/* Computing MIN */
		    i__4 = *n, i__1 = j + *k;
		    i__3 = j + 1;
		    for (i__ = f2c_min(i__4,i__1); i__ >= i__3; --i__) {
			temp -= a[l + i__ + j * a_dim1] * x[ix];
			ix -= *incx;
/* L150: */
		    }
		    if (nounit) {
			temp /= a[j * a_dim1 + 1];
		    }
		    x[jx] = temp;
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of STBSV . */

} /* stbsv_ */

/* ztbsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ 
int PASTEF77S(z,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_dcomplex *a, const bla_integer *lda, bla_dcomplex *x, const bla_integer *incx)
{
    /* System generated locals */
    bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    bla_dcomplex z__1, z__2, z__3;

    /* Builtin functions */
    //void bla_z_div(bla_dcomplex *, bla_dcomplex *, bla_dcomplex *), bla_d_cnjg(
	//    bla_dcomplex *, bla_dcomplex *);

    /* Local variables */
    bla_integer info;
    bla_dcomplex temp;
    bla_integer i__, j, l;
    //extern bla_logical PASTE_LSAME(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kplus1, ix, jx, kx = 0;
    //extern /* Subroutine */ int PASTE_XERBLA(bla_character *, bla_integer *, ftnlen);
    bla_logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTBSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular band matrix, with ( k + 1 ) */
/*  diagonals. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   conjg( A' )*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */

    // Initialize info_value to 0
    gint_t info_value = 0;
    bli_rntm_set_info_value_only( info_value, &tl_rntm );

    info = 0;
    if (! PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTE_LSAME(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	PASTE_XERBLA("ZTBSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = PASTE_LSAME(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = PASTE_LSAME(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed by sequentially with one pass through A. */

    if (PASTE_LSAME(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_zreal(x[i__1]) != 0. || bli_zimag(x[i__1]) != 0.) 
			{
			l = kplus1 - j;
			if (nounit) {
			    i__1 = j;
			    bla_z_div(&z__1, &x[j], &a[kplus1 + j * a_dim1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__1] );
			}
			i__1 = j;
			bli_zsets( (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__4 = l + i__ + j * a_dim1;
			    bli_zsets( (bli_zreal(temp) * bli_zreal(a[i__4]) - bli_zimag(temp) * bli_zimag(a[i__4])), (bli_zreal(temp) * bli_zimag(a[i__4]) + bli_zimag(temp) * bli_zreal(a[i__4])), z__2 );
			    bli_zsets( (bli_zreal(x[i__3]) - bli_zreal(z__2)), (bli_zimag(x[i__3]) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
/* L10: */
			}
		    }
/* L20: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    kx -= *incx;
		    i__1 = jx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_zreal(x[i__1]) != 0. || bli_zimag(x[i__1]) != 0.) 
			{
			ix = kx;
			l = kplus1 - j;
			if (nounit) {
			    i__1 = jx;
			    bla_z_div(&z__1, &x[jx], &a[kplus1 + j * a_dim1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__1] );
			}
			i__1 = jx;
			bli_zsets( (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__1 = f2c_max(i__2,i__3);
			for (i__ = j - 1; i__ >= i__1; --i__) {
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = l + i__ + j * a_dim1;
			    bli_zsets( (bli_zreal(temp) * bli_zreal(a[i__4]) - bli_zimag(temp) * bli_zimag(a[i__4])), (bli_zreal(temp) * bli_zimag(a[i__4]) + bli_zimag(temp) * bli_zreal(a[i__4])), z__2 );
			    bli_zsets( (bli_zreal(x[i__3]) - bli_zreal(z__2)), (bli_zimag(x[i__3]) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			    ix -= *incx;
/* L30: */
			}
		    }
		    jx -= *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0.) 
			{
			l = 1 - j;
			if (nounit) {
			    i__2 = j;
			    bla_z_div(&z__1, &x[j], &a[j * a_dim1 + 1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			}
			i__2 = j;
			bli_zsets( (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = l + i__ + j * a_dim1;
			    bli_zsets( (bli_zreal(temp) * bli_zreal(a[i__5]) - bli_zimag(temp) * bli_zimag(a[i__5])), (bli_zreal(temp) * bli_zimag(a[i__5]) + bli_zimag(temp) * bli_zreal(a[i__5])), z__2 );
			    bli_zsets( (bli_zreal(x[i__4]) - bli_zreal(z__2)), (bli_zimag(x[i__4]) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__3] );
/* L50: */
			}
		    }
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    kx += *incx;
		    i__2 = jx;
			//When matrix A is singular or near singular, the solution to Ax = b is non-trivial
			//Therefore inverse of A doesn't exist. Here by commenting out the below lines, 
			//we end up generating NAN or inf
			//if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0.) 
			{
			ix = kx;
			l = 1 - j;
			if (nounit) {
			    i__2 = jx;
			    bla_z_div(&z__1, &x[jx], &a[j * a_dim1 + 1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__2] );
			}
			i__2 = jx;
			bli_zsets( (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
/* Computing MIN */
			i__3 = *n, i__4 = j + *k;
			i__2 = f2c_min(i__3,i__4);
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = l + i__ + j * a_dim1;
			    bli_zsets( (bli_zreal(temp) * bli_zreal(a[i__5]) - bli_zimag(temp) * bli_zimag(a[i__5])), (bli_zreal(temp) * bli_zimag(a[i__5]) + bli_zimag(temp) * bli_zreal(a[i__5])), z__2 );
			    bli_zsets( (bli_zreal(x[i__4]) - bli_zreal(z__2)), (bli_zimag(x[i__4]) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), x[i__3] );
			    ix += *incx;
/* L70: */
			}
		    }
		    jx += *incx;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x  or  x := inv( conjg( A') )*x. */

	if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    bli_zsets( (bli_zreal(x[i__2])), (bli_zimag(x[i__2])), temp );
		    l = kplus1 - j;
		    if (noconj) {
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			    i__2 = l + i__ + j * a_dim1;
			    i__3 = i__;
			    bli_zsets( (bli_zreal(a[i__2]) * bli_zreal(x[i__3]) - bli_zimag(a[i__2]) * bli_zimag(x[i__3])), (bli_zreal(a[i__2]) * bli_zimag(x[i__3]) + bli_zimag(a[i__2]) * bli_zreal(x[i__3])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L90: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &a[kplus1 + j * a_dim1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    } else {
/* Computing MAX */
			i__4 = 1, i__2 = j - *k;
			i__3 = j - 1;
			for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__4 = i__;
			    bli_zsets( (bli_zreal(z__3) * bli_zreal(x[i__4]) - bli_zimag(z__3) * bli_zimag(x[i__4])), (bli_zreal(z__3) * bli_zimag(x[i__4]) + bli_zimag(z__3) * bli_zreal(x[i__4])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L100: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &a[kplus1 + j * a_dim1]);
			    bla_z_div(&z__1, &temp, &z__2);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    }
		    i__3 = j;
		    bli_zsets( (bli_zreal(temp)), (bli_zimag(temp)), x[i__3] );
/* L110: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__3 = jx;
		    bli_zsets( (bli_zreal(x[i__3])), (bli_zimag(x[i__3])), temp );
		    ix = kx;
		    l = kplus1 - j;
		    if (noconj) {
/* Computing MAX */
			i__3 = 1, i__4 = j - *k;
			i__2 = j - 1;
			for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			    i__3 = l + i__ + j * a_dim1;
			    i__4 = ix;
			    bli_zsets( (bli_zreal(a[i__3]) * bli_zreal(x[i__4]) - bli_zimag(a[i__3]) * bli_zimag(x[i__4])), (bli_zreal(a[i__3]) * bli_zimag(x[i__4]) + bli_zimag(a[i__3]) * bli_zreal(x[i__4])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    ix += *incx;
/* L120: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &a[kplus1 + j * a_dim1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    } else {
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__2 = ix;
			    bli_zsets( (bli_zreal(z__3) * bli_zreal(x[i__2]) - bli_zimag(z__3) * bli_zimag(x[i__2])), (bli_zreal(z__3) * bli_zimag(x[i__2]) + bli_zimag(z__3) * bli_zreal(x[i__2])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    ix += *incx;
/* L130: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &a[kplus1 + j * a_dim1]);
			    bla_z_div(&z__1, &temp, &z__2);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    }
		    i__4 = jx;
		    bli_zsets( (bli_zreal(temp)), (bli_zimag(temp)), x[i__4] );
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L140: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    bli_zsets( (bli_zreal(x[i__1])), (bli_zimag(x[i__1])), temp );
		    l = 1 - j;
		    if (noconj) {
/* Computing MIN */
			i__1 = *n, i__4 = j + *k;
			i__2 = j + 1;
			for (i__ = f2c_min(i__1,i__4); i__ >= i__2; --i__) {
			    i__1 = l + i__ + j * a_dim1;
			    i__4 = i__;
			    bli_zsets( (bli_zreal(a[i__1]) * bli_zreal(x[i__4]) - bli_zimag(a[i__1]) * bli_zimag(x[i__4])), (bli_zreal(a[i__1]) * bli_zimag(x[i__4]) + bli_zimag(a[i__1]) * bli_zreal(x[i__4])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L150: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &a[j * a_dim1 + 1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    } else {
/* Computing MIN */
			i__2 = *n, i__1 = j + *k;
			i__4 = j + 1;
			for (i__ = f2c_min(i__2,i__1); i__ >= i__4; --i__) {
			    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__2 = i__;
			    bli_zsets( (bli_zreal(z__3) * bli_zreal(x[i__2]) - bli_zimag(z__3) * bli_zimag(x[i__2])), (bli_zreal(z__3) * bli_zimag(x[i__2]) + bli_zimag(z__3) * bli_zreal(x[i__2])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
/* L160: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &a[j * a_dim1 + 1]);
			    bla_z_div(&z__1, &temp, &z__2);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    }
		    i__4 = j;
		    bli_zsets( (bli_zreal(temp)), (bli_zimag(temp)), x[i__4] );
/* L170: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__4 = jx;
		    bli_zsets( (bli_zreal(x[i__4])), (bli_zimag(x[i__4])), temp );
		    ix = kx;
		    l = 1 - j;
		    if (noconj) {
/* Computing MIN */
			i__4 = *n, i__2 = j + *k;
			i__1 = j + 1;
			for (i__ = f2c_min(i__4,i__2); i__ >= i__1; --i__) {
			    i__4 = l + i__ + j * a_dim1;
			    i__2 = ix;
			    bli_zsets( (bli_zreal(a[i__4]) * bli_zreal(x[i__2]) - bli_zimag(a[i__4]) * bli_zimag(x[i__2])), (bli_zreal(a[i__4]) * bli_zimag(x[i__2]) + bli_zimag(a[i__4]) * bli_zreal(x[i__2])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    ix -= *incx;
/* L180: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &a[j * a_dim1 + 1]);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    } else {
/* Computing MIN */
			i__1 = *n, i__4 = j + *k;
			i__2 = j + 1;
			for (i__ = f2c_min(i__1,i__4); i__ >= i__2; --i__) {
			    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__1 = ix;
			    bli_zsets( (bli_zreal(z__3) * bli_zreal(x[i__1]) - bli_zimag(z__3) * bli_zimag(x[i__1])), (bli_zreal(z__3) * bli_zimag(x[i__1]) + bli_zimag(z__3) * bli_zreal(x[i__1])), z__2 );
			    bli_zsets( (bli_zreal(temp) - bli_zreal(z__2)), (bli_zimag(temp) - bli_zimag(z__2)), z__1 );
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			    ix -= *incx;
/* L190: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &a[j * a_dim1 + 1]);
			    bla_z_div(&z__1, &temp, &z__2);
			    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp );
			}
		    }
		    i__2 = jx;
		    bli_zsets( (bli_zreal(temp)), (bli_zimag(temp)), x[i__2] );
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of ZTBSV . */

} /* ztbsv_ */

#ifdef BLIS_ENABLE_BLAS

int PASTEF77(s,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_real *a, const bla_integer *lda, bla_real *x, const bla_integer *incx)
{
  return PASTEF77S(s,tbsv)( uplo, trans, diag, n, k, a, lda, x, incx );
}

int PASTEF77(d,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_double *a, const bla_integer *lda, bla_double *x, const bla_integer *incx)
{
  return PASTEF77S(d,tbsv)( uplo, trans, diag, n, k, a, lda, x, incx );
}

int PASTEF77(c,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_scomplex *a, const bla_integer *lda, bla_scomplex *x, const bla_integer *incx)
{
  return PASTEF77S(c,tbsv)( uplo, trans, diag, n, k, a, lda, x, incx );
}

int PASTEF77(z,tbsv)(const bla_character *uplo, const bla_character *trans, const bla_character *diag, const bla_integer *n, const bla_integer *k, const bla_dcomplex *a, const bla_integer *lda, bla_dcomplex *x, const bla_integer *incx)
{
  return PASTEF77S(z,tbsv)( uplo, trans, diag, n, k, a, lda, x, incx );
}

#endif

