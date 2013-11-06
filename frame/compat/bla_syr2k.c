/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77(ch,blasname)( \
                            f77_char* uploc, \
                            f77_char* trans, \
                            f77_int*  m, \
                            f77_int*  k, \
                            ftype*    alpha, \
                            ftype*    a, f77_int* lda, \
                            ftype*    b, f77_int* ldb, \
                            ftype*    beta, \
                            ftype*    c, f77_int* ldc  \
                          ) \
{ \
	uplo_t  blis_uploc; \
	trans_t blis_trans; \
	dim_t   m0, k0; \
	inc_t   rs_a, cs_a; \
	inc_t   rs_b, cs_b; \
	inc_t   rs_c, cs_c; \
	err_t   init_result; \
\
	/* Initialize BLIS (if it is not already initialized). */ \
	bli_init_safe( &init_result ); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname)( MKSTR(ch), \
	                       MKSTR(blasname), \
	                       uploc, \
	                       trans, \
	                       m, \
	                       k, \
	                       lda, \
	                       ldb, \
	                       ldc ); \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_uplo( *uploc, &blis_uploc ); \
	bli_param_map_netlib_to_blis_trans( *trans, &blis_trans ); \
\
	/* The real domain ssyr2k and dsyr2k in netlib BLAS treat a trans value
	   of 'C' (conjugate-transpose) as 'T' (transpose only). So, we have
	   to go out of our way a little to support this behavior. */ \
	if ( bli_is_real( PASTEMAC(ch,type) ) && \
	     bli_is_conjtrans( blis_trans ) ) \
	{ \
		blis_trans = BLIS_TRANSPOSE; \
	} \
\
	/* Convert/typecast negative values of m and k to zero. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *k, k0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	rs_a = 1; \
	cs_a = *lda; \
	rs_b = 1; \
	cs_b = *ldb; \
	rs_c = 1; \
	cs_c = *ldc; \
\
	/* Call BLIS interface. */ \
	PASTEMAC(ch,blisname)( blis_uploc, \
	                       blis_trans, \
	                       blis_trans, \
	                       m0, \
	                       k0, \
	                       alpha, \
	                       a, rs_a, cs_a, \
	                       b, rs_b, cs_b, \
	                       beta, \
	                       c, rs_c, cs_c ); \
\
	/* Finalize BLIS (if it was initialized above). */ \
	bli_finalize_safe( init_result ); \
}

#ifdef BLIS_ENABLE_BLAS2BLIS
INSERT_GENTFUNC_BLAS( syr2k, syr2k )
#endif

