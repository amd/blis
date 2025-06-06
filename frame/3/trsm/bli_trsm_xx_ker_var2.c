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

static trsm_var_oft vars[2][2] =
{
	{ bli_trsm_ll_ker_var2, bli_trsm_lu_ker_var2 },
	{ bli_trsm_rl_ker_var2, bli_trsm_ru_ker_var2 }
};

void bli_trsm_xx_ker_var2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_6);
	dim_t        side;
	dim_t        uplo;
	trsm_var_oft f;
	AOCL_progress_callback AOCL_progress_local_ptr = AOCL_progress_ptr;

	// Set two bools: one based on the implied side parameter (the structure
	// of the root object) and one based on the uplo field of the triangular
	// matrix's root object (whether that is matrix A or matrix B).
	if ( bli_obj_root_is_triangular( a ) )
	{
		side = 0;
		if ( bli_obj_root_is_lower( a ) ) uplo = 0;
		else                              uplo = 1;
	}
	else // if ( bli_obj_root_is_triangular( b ) )
	{
		side = 1;
		if ( bli_obj_root_is_lower( b ) ) uplo = 0;
		else                              uplo = 1;
	}

	// Index into the variant array to extract the correct function pointer.
	f = vars[side][uplo];

	// Call the macrokernel.
	f
	(
	  a,
	  b,
	  c,
	  cntx,
	  rntm,
	  cntl,
	  thread
	);

	/* Send progress update if the user has enabled it */
	if (AOCL_progress_local_ptr)
	{
		// Get the size of block processed in
		// this iteration, add it to the accumulated
		// total and send the update.
		dim_t m = bli_obj_length(c);
		dim_t n = bli_obj_width(c);
		dim_t k = bli_obj_width(a);

		num_t dt = bli_obj_dt(c);
		char *dt_api = NULL;
		dim_t dt_api_len = 5;

		// Running total for current thread.
		tls_aoclprogress_counter += m * n * k;

		// Send the update only if number of elements processes so far
		// has exceeded the freqency of reporting.
		if ((tls_aoclprogress_counter - tls_aoclprogress_last_update) >=
			 AOCL_PROGRESS_FREQUENCY)
		{

			// reset the last update counter for next iteration.
			tls_aoclprogress_last_update = tls_aoclprogress_counter;

			switch (dt)
			{
			case BLIS_FLOAT:
				dt_api = "strsm";
				break;
			case BLIS_DOUBLE:
				dt_api = "dtrsm";
				break;
			case BLIS_SCOMPLEX:
				dt_api = "ctrsm";
				break;
			case BLIS_DCOMPLEX:
				dt_api = "ztrsm";
				break;
			default:
				dt_api = " trsm";
			}

			(*AOCL_progress_local_ptr)(dt_api,
			                           dt_api_len,
			                           tls_aoclprogress_counter,
			                           AOCL_gettid(),
			                           bli_rntm_num_threads(rntm));
		}
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_6);
}

