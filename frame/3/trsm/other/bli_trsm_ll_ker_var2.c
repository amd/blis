/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)
     (
       doff_t  diagoffa,
       pack_t  schema_a,
       pack_t  schema_b,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       void*   alpha1,
       void*   a, inc_t cs_a, dim_t pd_a, inc_t ps_a,
       void*   b, inc_t rs_b, dim_t pd_b, inc_t ps_b,
       void*   alpha2,
       void*   c, inc_t rs_c, inc_t cs_c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     );

static FUNCPTR_T GENARRAY(ftypes,trsm_ll_ker_var2);


void bli_trsm_ll_ker_var2
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	num_t     dt_exec   = bli_obj_exec_dt( c );

	doff_t    diagoffa  = bli_obj_diag_offset( a );

	pack_t    schema_a  = bli_obj_pack_schema( a );
	pack_t    schema_b  = bli_obj_pack_schema( b );

	dim_t     m         = bli_obj_length( c );
	dim_t     n         = bli_obj_width( c );
	dim_t     k         = bli_obj_width( a );

	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     cs_a      = bli_obj_col_stride( a );
	dim_t     pd_a      = bli_obj_panel_dim( a );
	inc_t     ps_a      = bli_obj_panel_stride( a );

	void*     buf_b     = bli_obj_buffer_at_off( b );
	inc_t     rs_b      = bli_obj_row_stride( b );
	dim_t     pd_b      = bli_obj_panel_dim( b );
	inc_t     ps_b      = bli_obj_panel_stride( b );

	void*     buf_c     = bli_obj_buffer_at_off( c );
	inc_t     rs_c      = bli_obj_row_stride( c );
	inc_t     cs_c      = bli_obj_col_stride( c );

	void*     buf_alpha1;
	void*     buf_alpha2;

	FUNCPTR_T f;

	// Grab the address of the internal scalar buffer for the scalar
	// attached to B (the non-triangular matrix). This will be the alpha
	// scalar used in the gemmtrsm subproblems (ie: the scalar that would
	// be applied to the packed copy of B prior to it being updated by
	// the trsm subproblem). This scalar may be unit, if for example it
	// was applied during packing.
	buf_alpha1 = bli_obj_internal_scalar_buffer( b );

	// Grab the address of the internal scalar buffer for the scalar
	// attached to C. This will be the "beta" scalar used in the gemm-only
	// subproblems that correspond to micro-panels that do not intersect
	// the diagonal. We need this separate scalar because it's possible
	// that the alpha attached to B was reset, if it was applied during
	// packing.
	buf_alpha2 = bli_obj_internal_scalar_buffer( c );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( diagoffa,
	   schema_a,
	   schema_b,
	   m,
	   n,
	   k,
	   buf_alpha1,
	   buf_a, cs_a, pd_a, ps_a,
	   buf_b, rs_b, pd_b, ps_b,
	   buf_alpha2,
	   buf_c, rs_c, cs_c,
	   cntx,
	   rntm,
	   thread );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffa, \
       pack_t  schema_a, \
       pack_t  schema_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       void*   alpha1, \
       void*   a, inc_t cs_a, dim_t pd_a, inc_t ps_a, \
       void*   b, inc_t rs_b, dim_t pd_b, inc_t ps_b, \
       void*   alpha2, \
       void*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     ) \
{ \
	const num_t     dt          = PASTEMAC(ch,type); \
\
	/* Alias some constants to simpler names. */ \
	const dim_t     MR          = pd_a; \
	const dim_t     NR          = pd_b; \
	const dim_t     PACKMR      = cs_a; \
	const dim_t     PACKNR      = rs_b; \
\
	/* Cast the micro-kernel address to its function pointer type. */ \
	PASTECH(ch,gemmtrsm_ukr_ft) \
	               gemmtrsm_ukr = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMMTRSM_L_UKR, cntx ); \
	PASTECH(ch,gemm_ukr_ft) \
	                   gemm_ukr = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
\
	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */ \
	ctype           ct[ BLIS_STACK_BUF_MAX_SIZE \
	                    / sizeof( ctype ) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const bool      col_pref    = bli_cntx_l3_vir_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx ); \
	const inc_t     rs_ct       = ( col_pref ? 1 : NR ); \
	const inc_t     cs_ct       = ( col_pref ? MR : 1 ); \
\
	ctype* restrict zero        = PASTEMAC(ch,0); \
	ctype* restrict minus_one   = PASTEMAC(ch,m1); \
	ctype* restrict a_cast      = a; \
	ctype* restrict b_cast      = b; \
	ctype* restrict c_cast      = c; \
	ctype* restrict alpha1_cast = alpha1; \
	ctype* restrict alpha2_cast = alpha2; \
	ctype* restrict b1; \
	ctype* restrict c1; \
\
	doff_t          diagoffa_i; \
	dim_t           k_full; \
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	dim_t           k_a1011; \
	dim_t           k_a10; \
	dim_t           off_a10; \
	dim_t           off_a11; \
	dim_t           i, j; \
	inc_t           rstep_a; \
	inc_t           cstep_b; \
	inc_t           rstep_c, cstep_c; \
	inc_t           istep_a; \
	inc_t           istep_b; \
	inc_t           off_scl; \
	inc_t           ss_a_num; \
	inc_t           ss_a_den; \
	inc_t           ps_a_cur; \
	inc_t           is_a_cur; \
	auxinfo_t       aux; \
\
	/*
	   Assumptions/assertions:
	     rs_a == 1
	     cs_a == PACKMR
	     pd_a == MR
	     ps_a == stride to next micro-panel of A
	     rs_b == PACKNR
	     cs_b == 1
	     pd_b == NR
	     ps_b == stride to next micro-panel of B
	     rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/ \
\
	/* Safety trap: Certain indexing within this macro-kernel does not
	   work as intended if both MR and NR are odd. */ \
	if ( ( bli_is_odd( PACKMR ) && bli_is_odd( NR ) ) || \
	     ( bli_is_odd( PACKNR ) && bli_is_odd( MR ) ) ) bli_abort(); \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Safeguard: If matrix A is above the diagonal, it is implicitly zero.
	   So we do nothing. */ \
	if ( bli_is_strictly_above_diag_n( diagoffa, m, k ) ) return; \
\
	/* Compute k_full as k inflated up to a multiple of MR. This is
	   needed because some parameter combinations of trsm reduce k
	   to advance past zero regions in the triangular matrix, and
	   when computing the imaginary stride of B (the non-triangular
	   matrix), which is used by 4m1/3m1 implementations, we need
	   this unreduced value of k. */ \
	k_full = ( k % MR != 0 ? k + MR - ( k % MR ) : k ); \
\
	/* Compute indexing scaling factor for for 4m or 3m. This is
	   needed because one of the packing register blocksizes (PACKMR
	   or PACKNR) is used to index into the micro-panels of the non-
	   triangular matrix when computing with a diagonal-intersecting
	   micro-panel of the triangular matrix. In the case of 4m or 3m,
	   real values are stored in both sub-panels, and so the indexing
	   needs to occur in units of real values. The value computed
	   here is divided into the complex pointer offset to cause the
	   pointer to be advanced by the correct value. */ \
	if ( bli_is_4mi_packed( schema_a ) || \
	     bli_is_3mi_packed( schema_a ) || \
	     bli_is_rih_packed( schema_a ) ) off_scl = 2; \
	else                                 off_scl = 1; \
\
	/* Compute the storage stride scaling. Usually this is just 1.
	   However, in the case of interleaved 3m, we need to scale the
	   offset by 3/2. Note that real-only, imag-only, and summed-only
	   packing formats are not applicable here since trsm is a two-
	   operand operation only (unlike trmm, which is capable of three-
	   operand). */ \
	if ( bli_is_3mi_packed( schema_a ) ) { ss_a_num = 3; ss_a_den = 2; } \
	else                                 { ss_a_num = 1; ss_a_den = 1; } \
\
	/* If there is a zero region above where the diagonal of A intersects the
	   left edge of the block, adjust the pointer to C and treat this case as
	   if the diagonal offset were zero. This skips over the region that was
	   not packed. (Note we assume the diagonal offset is a multiple of MR;
	   this assumption will hold as long as the cache blocksizes are each a
	   multiple of MR and NR.) */ \
	if ( diagoffa < 0 ) \
	{ \
		i        = -diagoffa; \
		m        = m - i; \
		diagoffa = 0; \
		c_cast   = c_cast + (i  )*rs_c; \
	} \
\
	/* Check the k dimension, which needs to be a multiple of MR. If k
	   isn't a multiple of MR, we adjust it higher to satisfy the micro-
	   kernel, which is expecting to perform an MR x MR triangular solve.
	   This adjustment of k is consistent with what happened when A was
	   packed: all of its bottom/right edges were zero-padded, and
	   furthermore, the panel that stores the bottom-right corner of the
	   matrix has its diagonal extended into the zero-padded region (as
	   identity). This allows the trsm of that bottom-right panel to
	   proceed without producing any infs or NaNs that would infect the
	   "good" values of the corresponding block of B. */ \
	if ( k % MR != 0 ) k += MR - ( k % MR ); \
\
	/* NOTE: We don't need to check that m is a multiple of PACKMR since we
	   know that the underlying buffer was already allocated to have an m
	   dimension that is a multiple of PACKMR, with the region between the
	   last row and the next multiple of MR zero-padded accordingly. */ \
\
	/* Clear the temporary C buffer in case it has any infs or NaNs. */ \
	PASTEMAC(ch,set0s_mxn)( MR, NR, \
	                        ct, rs_ct, cs_ct ); \
\
	/* Compute number of primary and leftover components of the m and n
       dimensions. */ \
	n_iter = n / NR; \
	n_left = n % NR; \
\
	m_iter = m / MR; \
	m_left = m % MR; \
\
	if ( n_left ) ++n_iter; \
	if ( m_left ) ++m_iter; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	rstep_a = ps_a; \
\
	cstep_b = ps_b; \
\
	rstep_c = rs_c * MR; \
	cstep_c = cs_c * NR; \
\
	istep_a = PACKMR * k; \
	istep_b = PACKNR * k_full; \
\
	if ( bli_is_odd( istep_a ) ) istep_a += 1; \
	if ( bli_is_odd( istep_b ) ) istep_b += 1; \
\
	/* Save the pack schemas of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_schema_a( schema_a, &aux ); \
	bli_auxinfo_set_schema_b( schema_b, &aux ); \
\
	/* Save the imaginary stride of B to the auxinfo_t object. */ \
	bli_auxinfo_set_is_b( istep_b, &aux ); \
\
	b1 = b_cast; \
	c1 = c_cast; \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = 0; j < n_iter; ++j ) \
	{ \
		if( bli_trsm_my_iter( j, thread ) ) { \
\
		ctype* restrict a1; \
		ctype* restrict c11; \
		ctype* restrict b2; \
\
		a1  = a_cast; \
		c11 = c1 + (0  )*rstep_c; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b1; \
\
		/* Loop over the m dimension (MR rows at a time). */ \
		for ( i = 0; i < m_iter; ++i ) \
		{ \
			diagoffa_i = diagoffa + ( doff_t )i*MR; \
\
			m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
			/* If the current panel of A intersects the diagonal, use a
			   special micro-kernel that performs a fused gemm and trsm.
			   If the current panel of A resides below the diagonal, use a
			   a regular gemm micro-kernel. Otherwise, if it is above the
			   diagonal, it was not packed (because it is implicitly zero)
			   and so we do nothing. */ \
			if ( bli_intersects_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				ctype* restrict a10; \
				ctype* restrict a11; \
				ctype* restrict b01; \
				ctype* restrict b11; \
				ctype* restrict a2; \
\
				/* Compute various offsets into and lengths of parts of A. */ \
				off_a10 = 0; \
				k_a1011 = diagoffa_i + MR; \
				k_a10   = k_a1011 - MR; \
				off_a11 = k_a10; \
\
				/* Compute the panel stride for the current diagonal-
				   intersecting micro-panel. */ \
				is_a_cur  = k_a1011 * PACKMR; \
				is_a_cur += ( bli_is_odd( is_a_cur ) ? 1 : 0 ); \
				ps_a_cur  = ( is_a_cur * ss_a_num ) / ss_a_den; \
\
				/* Compute the addresses of the panel A10 and the triangular
				   block A11. */ \
				a10 = a1; \
				/* a11 = a1 + ( k_a10 * PACKMR ) / off_scl; */ \
				a11 = bli_ptr_inc_by_frac( a1, sizeof( ctype ), k_a10 * PACKMR, off_scl ); \
\
				/* Compute the addresses of the panel B01 and the block
				   B11. */ \
				b01 = b1 + ( off_a10 * PACKNR ) / off_scl; \
				b11 = b1 + ( off_a11 * PACKNR ) / off_scl; \
\
				/* Compute the addresses of the next panels of A and B. */ \
				a2 = a1 + ps_a_cur; \
				if ( bli_is_last_iter( i, m_iter, 0, 1 ) ) \
				{ \
					a2 = a_cast; \
					b2 = b1; \
					/*if ( bli_is_last_iter( j, n_iter, 0, 1 ) ) */\
					if ( j + bli_thread_num_threads(thread) >= n_iter ) \
						b2 = b_cast; \
				} \
\
				/* Save addresses of next panels of A and B to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_next_a( a2, &aux ); \
				bli_auxinfo_set_next_b( b2, &aux ); \
\
				/* Save the 4m1/3m1 imaginary stride of A to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_is_a( is_a_cur, &aux ); \
\
				/* Handle interior and edge cases separately. */ \
				if ( m_cur == MR && n_cur == NR ) \
				{ \
					/* Invoke the fused gemm/trsm micro-kernel. */ \
					gemmtrsm_ukr \
					( \
					  k_a10, \
					  alpha1_cast, \
					  a10, \
					  a11, \
					  b01, \
					  b11, \
					  c11, rs_c, cs_c, \
					  &aux, \
					  cntx  \
					); \
				} \
				else \
				{ \
					/* Invoke the fused gemm/trsm micro-kernel. */ \
					gemmtrsm_ukr \
					( \
					  k_a10, \
					  alpha1_cast, \
					  a10, \
					  a11, \
					  b01, \
					  b11, \
					  ct, rs_ct, cs_ct, \
					  &aux, \
					  cntx  \
					); \
\
					/* Copy the result to the bottom edge of C. */ \
					PASTEMAC(ch,copys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        c11, rs_c,  cs_c ); \
				} \
\
				a1 += ps_a_cur; \
			} \
			else if ( bli_is_strictly_below_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				ctype* restrict a2; \
\
				/* Compute the addresses of the next panels of A and B. */ \
				a2 = a1 + rstep_a; \
				if ( bli_is_last_iter( i, m_iter, 0, 1 ) ) \
				{ \
					a2 = a_cast; \
					b2 = b1; \
					/*if ( bli_is_last_iter( j, n_iter, 0, 1 ) ) */\
					if ( j + bli_thread_num_threads(thread) >= n_iter ) \
						b2 = b_cast; \
				} \
\
				/* Save addresses of next panels of A and B to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_next_a( a2, &aux ); \
				bli_auxinfo_set_next_b( b2, &aux ); \
\
				/* Save the 4m1/3m1 imaginary stride of A to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_is_a( istep_a, &aux ); \
\
				/* Handle interior and edge cases separately. */ \
				if ( m_cur == MR && n_cur == NR ) \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					gemm_ukr \
					( \
					  k, \
					  minus_one, \
					  a1, \
					  b1, \
					  alpha2_cast, \
					  c11, rs_c, cs_c, \
					  &aux, \
					  cntx  \
					); \
				} \
				else \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					gemm_ukr \
					( \
					  k, \
					  minus_one, \
					  a1, \
					  b1, \
					  zero, \
					  ct, rs_ct, cs_ct, \
					  &aux, \
					  cntx  \
					); \
\
					/* Add the result to the edge of C. */ \
					PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        alpha2_cast, \
					                        c11, rs_c,  cs_c ); \
				} \
\
				a1 += rstep_a; \
			} \
\
			c11 += rstep_c; \
		} \
		} \
\
		b1 += cstep_b; \
		c1 += cstep_c; \
	} \
\
/*
if ( bli_is_4mi_packed( schema_a ) ){ \
PASTEMAC(d,fprintm)( stdout, "trsm4m1_ll_ker_var2: b_r before", k, n, \
                     ( double* )b,    rs_b, 1, "%4.1f", "" ); \
PASTEMAC(d,fprintm)( stdout, "trsm4m1_ll_ker_var2: b_i before", k, n, \
                     ( double* )b+72, rs_b, 1, "%4.1f", "" ); \
}else{ \
PASTEMAC(d,fprintm)( stdout, "trsmnat_ll_ker_var2: b_r before", k, n, \
                     ( double* )b,   2*rs_b, 2, "%4.1f", "" ); \
PASTEMAC(d,fprintm)( stdout, "trsmnat_ll_ker_var2: b_i before", k, n, \
                     ( double* )b+1, 2*rs_b, 2, "%4.1f", "" ); \
} \
*/ \
\
/*
PASTEMAC(d,fprintm)( stdout, "trsm_ll_ker_var2: a11p_r computed", MR, MR, \
                     ( double* )a11, 1, PACKMR, "%4.1f", "" ); \
*/ \
\
/*
if ( bli_is_4mi_packed( schema_a ) ){ \
PASTEMAC(d,fprintm)( stdout, "trsm4m1_ll_ker_var2: b_r after", k, n, \
                     ( double* )b,    rs_b, 1, "%4.1f", "" ); \
PASTEMAC(d,fprintm)( stdout, "trsm4m1_ll_ker_var2: b_i after", k, n, \
                     ( double* )b+72, rs_b, 1, "%4.1f", "" ); \
}else{ \
PASTEMAC(d,fprintm)( stdout, "trsmnat_ll_ker_var2: b_r after", k, n, \
                     ( double* )b,   2*rs_b, 2, "%4.1f", "" ); \
PASTEMAC(d,fprintm)( stdout, "trsmnat_ll_ker_var2: b_i after", k, n, \
                     ( double* )b+1, 2*rs_b, 2, "%4.1f", "" ); \
} \

PASTEMAC(d,fprintm)( stdout, "trsm_ll_ker_var2: b_r", m, n, \
                     ( double* )c,    1, cs_c, "%4.1f", "" ); \
PASTEMAC(d,fprintm)( stdout, "trsm_ll_ker_var2: b_i", m, n, \
                     ( double* )c + 8*9, 1, cs_c, "%4.1f", "" ); \
*/ \
\
/*
PASTEMAC(ch,fprintm)( stdout, "trsm_ll_ker_var2: a1 (diag)", MR, k_a1011, a1, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_ll_ker_var2: a11 (diag)", MR, MR, a11, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_ll_ker_var2: b1 (diag)", k_a1011, NR, bp_i, NR, 1, "%5.2f", "" );  \
PASTEMAC(ch,fprintm)( stdout, "trsm_ll_ker_var2: bp11 (diag)", MR, NR, bp11, NR, 1, "%5.2f", "" );  \
*/ \
\
/*
PASTEMAC(ch,fprintm)( stdout, "trsm_ll_ker_var2: a1 (ndiag)", MR, k, a1, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_ll_ker_var2: b1 (ndiag)", k, NR, bp, NR, 1, "%5.2f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( trsm_ll_ker_var2 )

