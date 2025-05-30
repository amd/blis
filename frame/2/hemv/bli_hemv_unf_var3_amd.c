/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uplo, \
       conj_t  conja, \
       conj_t  conjx, \
       conj_t  conjh, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  one        = PASTEMAC(ch,1); \
	ctype*  zero       = PASTEMAC(ch,0); \
	ctype*  A11; \
	ctype*  A21; \
	ctype*  a10t; \
	ctype*  alpha11; \
	ctype*  a21; \
	ctype*  x1; \
	ctype*  x2; \
	ctype*  chi11; \
	ctype*  y1; \
	ctype*  y2; \
	ctype*  y01; \
	ctype*  psi11; \
	ctype*  y21; \
	ctype   conjx_chi11; \
	ctype   alpha_chi11; \
	ctype   alpha11_temp; \
	siz_t   i, k, j; \
	dim_t   b_fuse, f; \
	dim_t   n_ahead; \
	dim_t   f_ahead, f_behind; \
	inc_t   rs_at, cs_at; \
	conj_t  conj0, conj1; \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_at = rs_a; \
		cs_at = cs_a; \
\
		conj0 = bli_apply_conj( conjh, conja ); \
		conj1 = conja; \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_at = cs_a; \
		cs_at = rs_a; \
\
		conj0 = conja; \
		conj1 = bli_apply_conj( conjh, conja ); \
	} \
\
	/* If beta is zero, use setv. Otherwise, scale by beta. */ \
	if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		/* y = 0; */ \
		PASTEMAC2(ch,setv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  zero, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
	else \
	{ \
		/* y = beta * y; */ \
		PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  beta, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
\
	PASTECH(ch,dotxaxpyf_ker_ft) kfp_xf; \
\
	/* Query the context for the kernel function pointer and fusing factor. */ \
	kfp_xf = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXAXPYF_KER, cntx ); \
	b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_XF, cntx ); \
\
	for ( i = 0; i < m; i += f ) \
	{ \
		f        = bli_determine_blocksize_dim_f( i, m, b_fuse ); \
		n_ahead  = m - i - f; \
		A11      = a + (i  )*rs_at + (i  )*cs_at; \
		A21      = a + (i+f)*rs_at + (i  )*cs_at; \
		x1       = x + (i  )*incx; \
		x2       = x + (i+f)*incx; \
		y1       = y + (i  )*incy; \
		y2       = y + (i+f)*incy; \
\
		/* y1 = y1 + alpha * A11 * x1;  (variant 4) */ \
		for ( k = 0; k < f; ++k ) \
		{ \
			f_behind = k; \
			f_ahead  = f - k - 1; \
			a10t     = A11 + (k  )*rs_at + (0  )*cs_at; \
			alpha11  = A11 + (k  )*rs_at + (k  )*cs_at; \
			a21      = A11 + (k+1)*rs_at + (k  )*cs_at; \
			chi11    = x1  + (k  )*incx; \
			y01      = y1  + (0  )*incy; \
			psi11    = y1  + (k  )*incy; \
			y21      = y1  + (k+1)*incy; \
\
			/* y01 = y01 + alpha * a10t' * chi11; */ \
			PASTEMAC(ch,copycjs)( conjx, *chi11, conjx_chi11 ); \
			PASTEMAC(ch,scal2s)( *alpha, conjx_chi11, alpha_chi11 ); \
			if ( bli_is_conj( conj0 ) ) \
			{ \
				for ( j = 0; j < f_behind; ++j ) \
					PASTEMAC(ch,axpyjs)( alpha_chi11, *(a10t + j*cs_at), *(y01 + j*incy) ); \
			} \
			else \
			{ \
				for ( j = 0; j < f_behind; ++j ) \
					PASTEMAC(ch,axpys)( alpha_chi11, *(a10t + j*cs_at), *(y01 + j*incy) ); \
			} \
\
			/* For hemv, explicitly set the imaginary component of alpha11 to
			   zero. */ \
			PASTEMAC(ch,copycjs)( conja, *alpha11, alpha11_temp ); \
			if ( bli_is_conj( conjh ) ) \
				PASTEMAC(ch,seti0s)( alpha11_temp ); \
\
			/* psi11 = psi11 + alpha * alpha11 * chi11; */ \
			PASTEMAC(ch,axpys)( alpha_chi11, alpha11_temp, *psi11 ); \
\
			/* y21 = y21 + alpha * a21 * chi11; */ \
			if ( bli_is_conj( conj1 ) ) \
			{ \
				for ( j = 0; j < f_ahead; ++j ) \
					PASTEMAC(ch,axpyjs)( alpha_chi11, *(a21 + j*rs_at), *(y21 + j*incy) ); \
			} \
			else \
			{ \
				for ( j = 0; j < f_ahead; ++j ) \
					PASTEMAC(ch,axpys)( alpha_chi11, *(a21 + j*rs_at), *(y21 + j*incy) ); \
			} \
		} \
\
		/* y1 = y1 + alpha * A21' * x2;  (dotxf) */ \
		/* y2 = y2 + alpha * A21  * x1;  (axpyf) */ \
		kfp_xf \
		( \
		  conj0, \
		  conj1, \
		  conjx, \
		  conjx, \
		  n_ahead, \
		  f, \
		  alpha, \
		  A21, rs_at, cs_at, \
		  x2,  incx, \
		  x1,  incx, \
		  one, \
		  y1,  incy, \
		  y2,  incy, \
		  cntx  \
		); \
	} \
}

void bli_pre_hemv_8x8
     (
       double *a,
       double *x,
       double *y,
       double *alpha,
       dim_t cs_a,
       dim_t rs_a
      );

void bli_dhemv_unf_var3
     (
       uplo_t  uplo,
       conj_t  conja,
       conj_t  conjx,
       conj_t  conjh,
       dim_t   m,
       double*  alpha,
       double*  a, inc_t rs_a, inc_t cs_a,
       double*  x, inc_t incx,
       double*  beta,
       double*  y, inc_t incy,
       cntx_t* cntx
     )
{
	const num_t dt = PASTEMAC(d,type);

	double*  one        = PASTEMAC(d,1);
	double*  zero       = PASTEMAC(d,0);
	double*  A11;
	double*  A21;
	double*  a10t;
	double*  alpha11;
	double*  a21;
	double*  x1;
	double*  x2;
	double*  chi11;
	double*  y1;
	double*  y2;
	double*  y01;
	double*  psi11;
	double*  y21;
	double   conjx_chi11;
	double   alpha_chi11;
	double   alpha11_temp;
	dim_t   i, k, j;
	dim_t   b_fuse, f;
	dim_t   n_ahead;
	dim_t   f_ahead, f_behind;
	inc_t   rs_at, cs_at;
	conj_t  conj0 = 0, conj1 = 0;

	/* The algorithm will be expressed in terms of the lower triangular
	 * case; the upper triangular case is supported by swapping the row
	 * and column strides of A and toggling some conj parameters. */
	if ( bli_is_lower( uplo ) )
	{
		rs_at = rs_a;
		cs_at = cs_a;
	}
	else /* if ( bli_is_upper( uplo ) ) */
	{
		rs_at = cs_a;
		cs_at = rs_a;
	}

	/* If beta is zero, use setv. Otherwise, scale by beta. */
	if ( PASTEMAC(d,eq0)( *beta ) )
	{
		/* y = 0; */
		PASTEMAC2(d,setv,BLIS_TAPI_EX_SUF)
		(
		  BLIS_NO_CONJUGATE,
		  m,
		  zero,
		  y, incy,
		  cntx,
		  NULL
		);
	}
	else
	{
		/* y = beta * y; */
		PASTEMAC2(d,scalv,BLIS_TAPI_EX_SUF)
		(
		  BLIS_NO_CONJUGATE,
		  m,
		  beta,
		  y, incy,
		  cntx,
		  NULL
		);
	}

	PASTECH(d,dotxaxpyf_ker_ft) kfp_dotxaxpyf_ker;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE)
	{
		kfp_dotxaxpyf_ker = bli_ddotxaxpyf_zen_int_8;
		b_fuse = 8;
	}
	else
	{
		if ( cntx == NULL ) cntx = bli_gks_query_cntx();
		kfp_dotxaxpyf_ker =
			bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXAXPYF_KER, cntx);
		b_fuse =
			bli_cntx_get_blksz_def_dt( dt, BLIS_XF, cntx );
	}

	for ( i = 0; i < m; i += f )
	{
		f        = bli_determine_blocksize_dim_f( i, m, b_fuse );
		n_ahead  = m - i - f;
		A11      = a + (i  )*rs_at + (i  )*cs_at;
		A21      = a + (i+f)*rs_at + (i  )*cs_at;
		x1       = x + (i  )*incx;
		x2       = x + (i+f)*incx;
		y1       = y + (i  )*incy;
		y2       = y + (i+f)*incy;

		/* y1 = y1 + alpha * A11 * x1;  (variant 4) */
		if((f == 8) && (incx == 1) && (incy == 1) && (rs_at == 1))
		{
			/*this helper function handles unit stride only*/
			bli_pre_hemv_8x8(A11, x1, y1, alpha, cs_at, rs_at);
		}
		else
		{
			for ( k = 0; k < f; ++k )
			{
				f_behind = k;
				f_ahead  = f - k - 1;
				a10t     = A11 + (k  )*rs_at + (0  )*cs_at;
				alpha11  = A11 + (k  )*rs_at + (k  )*cs_at;
				a21      = A11 + (k+1)*rs_at + (k  )*cs_at;
				chi11    = x1  + (k  )*incx;
				y01      = y1  + (0  )*incy;
				psi11    = y1  + (k  )*incy;
				y21      = y1  + (k+1)*incy;

				/* y01 = y01 + alpha * a10t' * chi11; */
				PASTEMAC(d,copycjs)( conjx,
						*chi11, conjx_chi11 );
				PASTEMAC(d,scal2s)( *alpha, conjx_chi11,
						alpha_chi11 );
				{
					for ( j = 0; j < f_behind; ++j )
					{
						PASTEMAC(d,axpys)
							( alpha_chi11,
							  *(a10t + j*cs_at),
							  *(y01 + j*incy) );
					}
				}

				PASTEMAC(d,copycjs)( conja, *alpha11,
						alpha11_temp );

				/* psi11 = psi11 + alpha * alpha11 * chi11; */
				PASTEMAC(d,axpys)( alpha_chi11, alpha11_temp,
						*psi11 );

				/* y21 = y21 + alpha * a21 * chi11; */
				for ( j = 0; j < f_ahead; ++j )
				{
					PASTEMAC(d,axpys)( alpha_chi11,
							*(a21 + j*rs_at),
							*(y21 + j*incy) );
				}
			}
		}

		/* y1 = y1 + alpha * A21' * x2;  (dotxf) */
		/* y2 = y2 + alpha * A21  * x1;  (axpyf) */
		kfp_dotxaxpyf_ker
		(
		  conj0,
		  conj1,
		  conjx,
		  conjx,
		  n_ahead,
		  f,
		  alpha,
		  A21, rs_at, cs_at,
		  x2,  incx,
		  x1,  incx,
		  one,
		  y1,  incy,
		  y2,  incy,
		  cntx
		);
	}
}

GENTFUNC(float, s, hemv_unf_var3)
GENTFUNC(scomplex, c, hemv_unf_var3)
GENTFUNC(dcomplex, z, hemv_unf_var3)


