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
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "trsm_ukr";
static char*     o_types                   = "m";  // c
static char*     p_types                   = "u";  // uploa
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_trsm_ukr_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_trsm_ukr_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     );

void libblis_test_trsm_ukr_impl
     (
       iface_t   iface,
       side_t    side,
       obj_t*    a,
       obj_t*    b,
       obj_t*    c,
       cntx_t*   cntx
     );

void libblis_test_trsm_ukr_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         a,
       obj_t*         b,
       obj_t*         b_orig,
       double*        resid
     );



void libblis_test_trsm_ukr_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
	libblis_test_randm( tdata, params, &(op->ops->randm) );
	libblis_test_setv( tdata, params, &(op->ops->setv) );
	libblis_test_normfv( tdata, params, &(op->ops->normfv) );
	libblis_test_subv( tdata, params, &(op->ops->subv) );
	libblis_test_scalv( tdata, params, &(op->ops->scalv) );
	libblis_test_copym( tdata, params, &(op->ops->copym) );
	libblis_test_scalm( tdata, params, &(op->ops->scalm) );
	libblis_test_gemv( tdata, params, &(op->ops->gemv) );
	libblis_test_trsv( tdata, params, &(op->ops->trsv) );
}



void libblis_test_trsm_ukr
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{

	// Return early if this test has already been done.
	if ( libblis_test_op_is_done( op ) ) return;

	// Return early if operation is disabled.
	if ( libblis_test_op_is_disabled( op ) ||
	     libblis_test_l3ukr_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_trsm_ukr_deps( tdata, params, op );

	// Execute the test driver for each implementation requested.
	//if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( tdata,
		                        params,
		                        op,
		                        BLIS_TEST_SEQ_UKERNEL,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_trsm_ukr_experiment );
	}
}


// Import the register blocksizes used by the micro-kernel(s).
extern blksz_t* gemm_mr;
extern blksz_t* gemm_nr;
extern blksz_t* gemm_kr;

void libblis_test_trsm_ukr_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     )
{
	unsigned int n_repeats = params->n_repeats;
	unsigned int i;

	double       time_min  = DBL_MAX;
	double       time;

	num_t        datatype;

	dim_t        m, n;
	inc_t        ldap, ldbp;

	char         sc_a = 'c';
	char         sc_b = 'r';

	side_t       side = BLIS_LEFT;
	uplo_t       uploa;

	obj_t        a, b, c;
	obj_t        ap, bp;
	obj_t        c_save;

	cntx_t*      cntx;


	// Query a context.
	cntx = bli_gks_query_cntx();

	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Fix m and n to MR and NR, respectively.
	m = bli_cntx_get_blksz_def_dt( datatype, BLIS_MR, cntx );
	n = bli_cntx_get_blksz_def_dt( datatype, BLIS_NR, cntx );

	// Also query PACKMR and PACKNR as the leading dimensions to ap and bp,
	// respectively.
	ldap = bli_cntx_get_blksz_max_dt( datatype, BLIS_MR, cntx );
	ldbp = bli_cntx_get_blksz_max_dt( datatype, BLIS_NR, cntx );

	// Store the register blocksizes so that the driver can retrieve the
	// values later when printing results.
	op->dim_aux[0] = m;
	op->dim_aux[1] = n;

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );

	// Create test scalars.

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_a,      m, m, &a );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_b,      m, n, &b );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &c );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &c_save );

	// Set the structure, uplo, and diagonal offset properties of A.
	bli_obj_set_struc( BLIS_TRIANGULAR, &a );
	bli_obj_set_uplo( uploa, &a );
	bli_obj_set_diag_offset( 0, &a );

	// Randomize A, make it densely triangular.
	libblis_test_mobj_randomize( params, TRUE, &a );
	libblis_test_mobj_load_diag( params, &a );
	bli_mktrim( &a );

	// Randomize B.
	libblis_test_mobj_randomize( params, TRUE, &b );

	// Randomize C and save C.
	libblis_test_mobj_randomize( params, TRUE, &c );
	bli_copym( &c, &c_save );

#if 0
	// Create pack objects for a and b, and pack them to ap and bp,
	// respectively.
	cntl_t* cntl_a = libblis_test_pobj_create
	(
	  BLIS_MR,
	  BLIS_MR,
	  BLIS_INVERT_DIAG,
	  BLIS_PACKED_ROW_PANELS,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &a, &ap,
	  cntx
	);
	cntl_t* cntl_b = libblis_test_pobj_create
	(
	  BLIS_MR,
	  BLIS_NR,
	  BLIS_NO_INVERT_DIAG,
	  BLIS_PACKED_COL_PANELS,
	  BLIS_BUFFER_FOR_B_PANEL,
	  &b, &bp,
	  cntx
	);
#endif

	// Create the packed objects. Use packmr and packnr as the leading
	// dimensions of ap and bp, respectively. Note that we use the ldims
	// instead of the matrix dimensions for allocation purposes here.
	// This is a little hacky and was prompted when trying to support
	// configurations such as power9 that employ duplication/broadcasting
	// of elements in one of the packed matrix objects. Thankfully, packm
	// doesn't care about those dimensions and instead relies on
	// information taken from the source object. Thus, this is merely
	// about coaxing bli_obj_create() in allocating enough space for our
	// purposes.
	bli_obj_create( datatype, ldap, m, 1, ldap, &ap );
	bli_obj_create( datatype, m, ldbp, ldbp, 1, &bp );

	// Set up the objects for packing. Calling packm_init_pack() does everything
	// except checkout a memory pool block and save its address to the obj_t's.
	// However, it does overwrite the buffer field of packed object with that of
	// the source object (as a side-effect of bli_obj_alias_to(); that buffer
	// field would normally be overwritten yet again by the address from the
	// memory pool block). So, we have to save the buffer address that was
	// allocated so we can re-store it to the object afterward.
	void* buf_ap = bli_obj_buffer( &ap );
	void* buf_bp = bli_obj_buffer( &bp );
	// trsm_ukr are derived from gemm kernels therefore packing is done with
	// gemm blocksizes
	bli_packm_init_pack( BLIS_INVERT_DIAG, BLIS_GEMM, BLIS_PACKED_ROW_PANELS,
	                     BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
	                     BLIS_MR, BLIS_KR, &a, &ap, cntx );
	bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_GEMM, BLIS_PACKED_COL_PANELS,
	                     BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
	                     BLIS_KR, BLIS_NR, &b, &bp, cntx );
	bli_obj_set_buffer( buf_ap, &ap );
	bli_obj_set_buffer( buf_bp, &bp );

	// Set the diagonal offset of ap.
	bli_obj_set_diag_offset( 0, &ap );

	// Set the uplo field of ap since the default for packed objects is
	// BLIS_DENSE, and the _ukernel() wrapper needs this information to
	// know which set of micro-kernels (lower or upper) to choose from.
	bli_obj_set_uplo( uploa, &ap );

	// Pack the data from the source objects.
	bli_packm_blk_var1( &a, &ap, cntx, NULL, &BLIS_PACKM_SINGLE_THREADED );
	bli_packm_blk_var1( &b, &bp, cntx, NULL, &BLIS_PACKM_SINGLE_THREADED );

#if 0
bli_printm( "a", &a, "%5.2f", "" );
bli_printm( "ap", &ap, "%5.2f", "" );
#endif

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		// Re-pack the contents of b to bp.
		//bli_packm_blk_var1( &b, &bp, cntx, cntl_b, &BLIS_PACKM_SINGLE_THREADED );
		bli_packm_blk_var1( &b, &bp, cntx, NULL, &BLIS_PACKM_SINGLE_THREADED );

		bli_copym( &c_save, &c );

		time = bli_clock();

		libblis_test_trsm_ukr_impl( iface, side,
		                            &ap, &bp, &c,
		                            cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 1.0 * m * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &b ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_trsm_ukr_check( params, side, &ap, &c, &b, resid );

	// Zero out performance and residual if output matrix is empty.
	//libblis_test_check_empty_problem( &c, perf, resid );

#if 0
	// Free the control tree nodes and release their cached mem_t entries
	// back to the memory broker.
	bli_cntl_free( NULL, cntl_a, &BLIS_PACKM_SINGLE_THREADED );
	bli_cntl_free( NULL, cntl_b, &BLIS_PACKM_SINGLE_THREADED );
#endif

	// Free the packed objects.
	bli_obj_free( &ap );
	bli_obj_free( &bp );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &c_save );
}



void libblis_test_trsm_ukr_impl
     (
       iface_t   iface,
       side_t    side,
       obj_t*    a,
       obj_t*    b,
       obj_t*    c,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_UKERNEL:
		bli_trsm_ukernel( a, b, c, cntx );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_trsm_ukr_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         a,
       obj_t*         b,
       obj_t*         b_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( b );
	num_t  dt_real = bli_obj_dt_proj_to_real( b );

	dim_t  m       = bli_obj_length( b );
	dim_t  n       = bli_obj_width( b );

	obj_t  norm;
	obj_t  t, v, w, z;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized and triangular.
	// - b_orig is randomized.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   B := inv(transa(A)) * B_orig    (side = left)
	//   B := B_orig * inv(transa(A))    (side = right)
	//
	// is functioning correctly if
	//
	//   normfv( v - z )
	//
	// is negligible, where
	//
	//   v = B * t
	//
	//   z = ( inv(transa(A)) * B ) * t     (side = left)
	//     = inv(transa(A)) * B * t
	//     = inv(transa(A)) * w
	//
	//   z = ( B * inv(transa(A)) ) * t     (side = right)
	//     = B * tinv(ransa(A)) * t
	//     = B * w

	bli_obj_scalar_init_detached( dt_real, &norm );

	if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, m, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}
	else // else if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, n, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}

	libblis_test_vobj_randomize( params, TRUE, &t );

	bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &v );

#if 0
bli_printm( "a11", a, "%5.2f", "" );
#endif

	// Restore the diagonal of a11 to its original, un-inverted state
	// (needed for trsv).
	bli_invertd( a );

	if ( bli_is_left( side ) )
	{
		bli_gemv( &BLIS_ONE, b_orig, &t, &BLIS_ZERO, &w );
		bli_trsv( &BLIS_ONE, a, &w );
		bli_copyv( &w, &z );
	}
	else
	{
		bli_copyv( &t, &w );
		bli_trsv( &BLIS_ONE, a, &w );
		bli_gemv( &BLIS_ONE, b_orig, &w, &BLIS_ZERO, &z );
	}

	bli_subv( &z, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w );
	bli_obj_free( &z );
}

