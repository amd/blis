/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC0(opname) \
     ( \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       cntl_t* cntl, \
       thrinfo_t* thread  \
     );

GENPROT( gemmt_ker_var2 )



//
// Prototype BLAS-like interfaces with void pointer operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname, uplo, varname ) \
\
void PASTEMACT(ch,opname,uplo,varname) \
     ( \
       pack_t  schema_a, \
       pack_t  schema_b, \
       dim_t   m_off, \
       dim_t   n_off, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       void*   alpha, \
       void*   a, inc_t cs_a, inc_t is_a, \
                  dim_t pd_a, inc_t ps_a, \
       void*   b, inc_t rs_b, inc_t is_b, \
                  dim_t pd_b, inc_t ps_b, \
       void*   beta, \
       void*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     );

INSERT_GENTPROT_GEMMT( gemmt, ker_var2 )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC0(opname) \
	( \
	  trans_t trans, \
	  obj_t*  alpha, \
	  obj_t*  a, \
	  obj_t*  b, \
	  obj_t*  beta, \
	  obj_t*  c, \
	  stor3_t eff_id, \
	  cntx_t* cntx, \
	  rntm_t* rntm, \
	  thrinfo_t* thread  \
	 );

GENPROT( gemmtsup_ref_var1n )
GENPROT( gemmtsup_ref_var2m )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname, uplo, varname ) \
\
void PASTEMACT(ch,opname,uplo,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          eff_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     );

INSERT_GENTPROT_GEMMT( gemmtsup, ref_var1n )
INSERT_GENTPROT_GEMMT( gemmtsup, ref_var2m )	


