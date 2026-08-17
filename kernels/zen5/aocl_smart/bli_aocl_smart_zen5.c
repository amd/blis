/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

/* This function determines if we need to take SUP or native path
   for given matrix sizes for zen5 configuration.
   * Returns TRUE if the dimensions fall under SUP range
   * Returns FALSE if the dimensions fall under Native range
*/
bool bli_cntx_gemmsup_thresh_is_met_zen5( obj_t* a, obj_t* b, obj_t* c, cntx_t* cntx )
{
	num_t       dt          =   bli_obj_dt( c );

	if( dt == BLIS_DOUBLE )
	{
		dim_t k           =   bli_obj_width_after_trans( a );
		dim_t m, n;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		if ( bli_cntx_l3_sup_ker_dislikes_storage_of( c, stor_id, cntx ) )
		{
			m = bli_obj_width(c);
			n = bli_obj_length(c);
		}
		else
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
		}
		// For skinny sizes where one/two dimensions are small
		if((m < 1000) || (n < 1000) || (k < 116)) return TRUE;
		// // For all combinations in small sizes
		if((m < 2200) && (n < 2200) && (k < 2200)) return TRUE;
		return FALSE;
	}
	else if( dt == BLIS_DCOMPLEX )
	{
		dim_t k           =   bli_obj_width_after_trans( a );
		dim_t m, n;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		if ( bli_cntx_l3_sup_ker_dislikes_storage_of( c, stor_id, cntx ) )
		{
			m = bli_obj_width(c);
			n = bli_obj_length(c);
		}
		else
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
		}
		// Tuning for conjugate ZGEMM inputs: take the SUP path only within a
		// small-ZGEMM size envelope; larger conjugate shapes use the native path.
		// The guards compare per-operand element counts (m*k, k*n, m*n) and the
		// total m*n*k product against tuned size limits. NOTE: Zen6 also uses this
		// selector (config/zen6 wires BLIS_GEMM here), so this routing applies to
		// Zen4, Zen5 and Zen6.
		if ( ( bli_obj_has_conj( a ) == TRUE ) || ( bli_obj_has_conj( b ) == TRUE ) )
		{
			// Stay on SUP only when an operand is tiny AND the total volume is
			// small; otherwise the native packed path wins.
			const double conj_sup_operand_elems_max = 500.0;
			const double conj_sup_mnk_max           = 7500.0;
			const double a_elems = ( double )m * ( double )k; // elements in A
			const double b_elems = ( double )k * ( double )n; // elements in B
			const double c_elems = ( double )m * ( double )n; // elements in C
			const double mnk     = ( double )m * ( double )n * ( double )k;
			if ( ( ( a_elems < conj_sup_operand_elems_max ) ||
			       ( b_elems < conj_sup_operand_elems_max ) ||
			       ( c_elems < conj_sup_operand_elems_max ) ) &&
			     ( mnk < conj_sup_mnk_max ) )
			{
				return TRUE;
			}
			return FALSE;
		}
		// For skinny sizes where m and/or n is small
		// The threshold for m is a single value, but for n, it is
		// also based on the packing size of A, since the kernels are
		// column preferential
        if ( ( ( m <= 1380 ) || ( ( n <= 1520 ) && ( k <= 128 ) ) ) && ( m + n + k < 6400 ) ) return TRUE;

		return FALSE;
	}
	else if( dt == BLIS_SCOMPLEX )
	{
		dim_t k           =   bli_obj_width_after_trans( a );
		dim_t m, n;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		if ( bli_cntx_l3_sup_ker_dislikes_storage_of( c, stor_id, cntx ) )
		{
			m = bli_obj_width(c);
			n = bli_obj_length(c);
		}
		else
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
		}

		// The threshold conditionals are as follows:
		if( n <= 540 )
		{
			if( n <= 420 ) return TRUE;
			else if( m <= 1260 ) return TRUE;
		}
		else
		{
			if( m <= 420 )
			{
				if( m <= 180 ) return TRUE;
				else if( n <= 2100 ) return TRUE;
			}
			else
			{
				if( k <= 540 )
				{
					if( n <= 1260 ) return TRUE;
					else if( m <= 900 ) return TRUE;
				}
			}
		}
		return FALSE;
	}
	else // dt == BLIS_FLOAT
	{
		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		const dim_t m = bli_obj_length( c );
		const dim_t n = bli_obj_width( c ); 
		const dim_t k = bli_obj_width_after_trans( a );
		const dim_t n_threads = bli_thread_get_num_threads();

		int64_t min_m_n = bli_min(m, n);
		int64_t m_n_k = m * n * k;

		if ( stor_id == BLIS_CRC || stor_id == BLIS_RRC)
		{
			// These inputs go to the SGEMM RD kernel
			if ( min_m_n <= 16.28 *  n_threads || (m_n_k / (double)n_threads) < 103802408.0)
			{
				// go to SUP
				return TRUE;
			}
		}
		else
		{
			// These inputs go to the SGEMM RV kernel
			if ( min_m_n <= 4.875 *  n_threads || (m_n_k / (double)n_threads) < 1211034577.0)
			{
				// go to SUP
				return TRUE;
			}
		}

		// in all other cases, goto the native code path
		return FALSE;
	}
}

/* This function determines the ideal blocksizes for given datatype
   and num_threads.
*/
void bli_dynamic_blkszs_zen5( dim_t m, dim_t n, dim_t n_threads, cntx_t* cntx, num_t dt )
{
	// dynamic blocksizes enabled only for double and single datatype.
	if (dt != BLIS_DOUBLE && dt != BLIS_FLOAT) return;

	if (dt == BLIS_FLOAT)
	{
		// For floats, KC has to be changed based on the size of C for the following reasons. 
        // When C is large, a larger KC is better because KC determines how many accesses over
        // C needs to be performed. A larger KC means that the number of write access over C
        // is smaller. Moreover a larger KC ensures more reuse of the packed A and B panels. 
        // But when C is small, we have a different effect, here, since the packed
        // sizes of A and B are larger, these don’t fit in the caches anymore and we have more 
        // cache misses because of this. Moreover, the packing of A becomes inefficient since 
        // we need to pack 8xk (half a cache line) we essentially stream through A and P while
        // only loading and storing half a cache line (we waste half of this memory bandwidth)
        // and having KC smaller ensures that A and P both stay in the L1/L2 cache so that the 
        // next iteration is faster and at least the fetched memory is not wasted by excessive
        // flushing into higher layers of memory
		if ( !( n_threads >= 128 && m >= 18000 && n >= 18000 ) )
		{
			bli_cntx_set_blksz_def_dt( BLIS_FLOAT, BLIS_KC, 192, cntx );
			bli_cntx_set_blksz_max_dt( BLIS_FLOAT, BLIS_KC, 192, cntx );
		}
		return;
	}

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	dim_t mc, kc, nc;
	model_t model = bli_init_model_query_id();

	// determine ideal blocksize
	if ( model == BLIS_MODEL_TURIN_DENSE )
	{
		if (n_threads == 1 )
		{
			mc = 88, kc = 384, nc = 4032;
		}
		else
		{
			// these blocksizes are tuned for M >> K, N >> K and K < 500
			mc = 120, kc = 576, nc = 4008;
		}
	}
	else // BLIS_MODEL_TURIN
	{
		if (n_threads == 1)
		{
			mc = 80, kc = 384, nc = 4032;
		}
		else
		{
			// these blocksizes are tuned for M >> K, N >> K and K < 500
			mc = 120, kc = 512, nc = 2016;
		}
	}

	// set blocksizes
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   192,  mc,    72,    48 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,  kc,   128,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8064,  nc,  2040,  1020 );

	bli_cntx_set_blkszs
	(
		BLIS_NAT, 3,
		BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
		BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
		BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
		cntx
	);
}
