/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include <immintrin.h>
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_f32_kern_macros.h"
#include "../int4_utils_avx512.h"

#ifndef LPGEMM_BF16_JIT

// 6xlt16 bf16s4f32of32 fringe kernel
LPGEMM_N_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6xlt16m)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_6xLT16_DISABLE,
						  &&POST_OPS_BIAS_6xLT16,
						  &&POST_OPS_RELU_6xLT16,
						  &&POST_OPS_RELU_SCALE_6xLT16,
						  &&POST_OPS_GELU_TANH_6xLT16,
						  &&POST_OPS_GELU_ERF_6xLT16,
						  &&POST_OPS_CLIP_6xLT16,
						  &&POST_OPS_DOWNSCALE_6xLT16,
						  &&POST_OPS_MATRIX_ADD_6xLT16,
						  &&POST_OPS_SWISH_6xLT16,
						  &&POST_OPS_MATRIX_MUL_6xLT16,
						  &&POST_OPS_TANH_6xLT16,
						  &&POST_OPS_SIGMOID_6xLT16
						};

	dim_t MR = 6;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t group_size = pre_ops_attr.group_size;

    // B matrix storage bfloat type
	__m512bh b0;
	__m256i b0_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	 __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp = _mm512_set1_epi8(0x08);

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8;

	/* Regs to store zero-point values */
	__m512i zero_point, zero_point0;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

	__mmask16 lmask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

    __m512i mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		// Registers to use for accumulating C.
		__m512 c_float_0p0 = _mm512_setzero_ps();

		__m512 c_float_1p0 = _mm512_setzero_ps();

		__m512 c_float_2p0 = _mm512_setzero_ps();

		__m512 c_float_3p0 = _mm512_setzero_ps();

		__m512 c_float_4p0 = _mm512_setzero_ps();

		__m512 c_float_5p0 = _mm512_setzero_ps();

		dim_t group_start = pre_ops_attr.pre_op_b_i / group_size;
		dim_t group_end   = ( pre_ops_attr.pre_op_b_i + k0 - 1 ) / group_size;

		bfloat16* a_group = (bfloat16*) a;
		int8_t* b_group = (int8_t*)b;

		if( pre_ops_attr.zero_point_len > 0 )
		{
			dim_t pre_op_sf_off = 0;
			dim_t pre_op_zp_off = 0;

			for( dim_t group = group_start; group <= group_end; group++ )
			{

				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{

					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 =_mm512_maskz_loadu_ps( load_mask,
						                               (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT(
							                    (__m256i)_mm256_maskz_loadu_epi16( load_mask,
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );

				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *( ( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					// float temp[16];
					// _mm512_storeu_ps( temp, scale0 );
					// printf("group:%ld, scale_factors:", group);
					// for( dim_t i = 0; i < 16; i++ )
					// {
					// 	printf( " %f ", temp[i] );
					// } printf("\n");
					scale1 = scale0;
				}

				if( pre_ops_attr.zero_point_len > 1 )
				{
					pre_op_zp_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					zero_point = _mm512_maskz_loadu_epi8( lmask, ( pre_ops_attr.zero_point +
                                              pre_op_zp_off ) );
				}
				else
				{
					pre_op_zp_off = group;
					zero_point = _mm512_set1_epi8( *( ( int8_t* )( pre_ops_attr.zero_point
					                                   + pre_op_zp_off) ) );
				}

				zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 = _mm256_maskz_loadu_epi8( lmask,(__m256i const *)
					                     ( b_group + ( ( rs_b * kr ) / 2 ) ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					__m512i zero_reg = _mm512_setzero_si512();

					/* Interleave zero_point values with zeroes */
					zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

					b0_s4 = _mm256_maskz_loadu_epi8( lmask,(__m256i const *)
					                     ( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				}
			} // group loop

		} // zero-point condition
		else
		{
			for( dim_t group = group_start; group <= group_end; group++ )
			{
				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				// calculate offsets
				dim_t pre_op_sf_off = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{

					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_maskz_loadu_ps( load_mask,
						                      (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT(
						                        (__m256i)_mm256_maskz_loadu_epi16( load_mask,
						                         (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );

				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *( ( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
				}

				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 = _mm256_maskz_loadu_epi8( lmask, (__m256i const *)
					                      ( b_group + ( ( rs_b * kr ) / 2 ) ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					b0_s4 = _mm256_maskz_loadu_epi8( lmask,(__m256i const *)
					                     ( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				}
			} // group loop
		} // zero-point condition

		// Load alpha and beta
		__m512 selector1 = _mm512_set1_ps( alpha );
		__m512 selector2 = _mm512_set1_ps( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

			c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

			c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

			c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );

			c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );

			c_float_5p0 = _mm512_mul_ps( selector1, c_float_5p0 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

				// c[0,0-15]
				BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_0p0, 0, 0, \
								selector1, selector2 );

				// c[1,0-15]
				BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_1p0, 1, 0, \
								selector1, selector2 );

				// c[2,0-15]
				BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_2p0, 2, 0, \
								selector1, selector2 );

				// c[3,0-15]
				BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_3p0, 3, 0, \
								selector1, selector2 );

				// c[4,0-15]
				BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_4p0, 4, 0, \
								selector1, selector2 );

				// c[5,0-15]
				BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_5p0, 5, 0, \
								selector1, selector2 );
			}
			else
			{
				__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

				// c[0,0-15]
				F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_0p0, ir, 0, 0, \
								selector1, selector2);

				// c[1,0-15]
				F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_1p0, ir, 1, 0, \
								selector1, selector2);

				// c[2,0-15]
				F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_2p0, ir, 2, 0, \
								selector1, selector2);

				// c[3,0-15]
				F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_3p0, ir, 3, 0, \
								selector1, selector2);

				// c[4,0-15]
				F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_4p0, ir, 4, 0, \
								selector1, selector2);

				// c[5,0-15]
				F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_5p0, ir, 5, 0, \
								selector1, selector2);
			}
		}
		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps
						(
						  bias_mask,
						  ( float* )post_ops_list_temp->op_args1 +
						  post_ops_attr.post_op_c_j
						);
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector1, c_float_5p0 );
			}
			else
			{
				__m512 selector3;
				__m512 selector4;
				__m512 selector5;
				__m512 selector6;
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
					BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
					BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
					BF16_F32_BIAS_BCAST(selector6, bias_mask, 5);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 3 ) );
					selector5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 4 ) );
					selector6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 5 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector6, c_float_5p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			// c[5,0-15]
			c_float_5p0 = _mm512_max_ps( selector1, c_float_5p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_5p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_6xLT16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
		    __m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512(c_float_5p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_6xLT16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512(c_float_5p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_6xLT16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(c_float_5p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_6xLT16:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();
		__m512 selector5 = _mm512_setzero_ps();
		__m512 selector6 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();
		__m512 zero_point4 = _mm512_setzero_ps();
		__m512 zero_point5 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector6 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point4 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point5 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
				selector6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 5 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
				zero_point4 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
				zero_point5 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 5 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector5,zero_point4);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector6,zero_point5);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_6xLT16:
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( load_mask,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,1);

					// c[2:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,2);

					// c[3:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,3);

					// c[4:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,4);

					// c[5:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr2,1);

					// c[2:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr3,2);

					// c[3:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr4,3);

					// c[4:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr5,4);

					// c[5:0-15]
					BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,1);

					// c[2:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,2);

					// c[3:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,3);

					// c[4:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,4);

					// c[5:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr2,1);

					// c[2:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr3,2);

					// c[3:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr4,3);

					// c[4:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr5,4);

					// c[5:0-15]
					F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_6xLT16:
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( load_mask,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr2,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr3,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr4,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr5,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,1);

					// c[2:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,2);

					// c[3:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,3);

					// c[4:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,4);

					// c[5:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr2,1);

					// c[2:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr3,2);

					// c[3:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr4,3);

					// c[4:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr5,4);

					// c[5:0-15]
					F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_6xLT16:
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(c_float_5p0, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_6xLT16:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(c_float_3p0, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(c_float_4p0, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(c_float_5p0, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_6xLT16:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_3p0, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_4p0, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_5p0, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_6xLT16_DISABLE:
		;
		// Store the results.
		if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

			// c[1,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

			// c[2,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

			// c[3,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

			// c[4,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);

			// c[5,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_5p0,5,0);
		}

		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_ps( c + ( rs_c * ( ir + 0 ) ), load_mask, c_float_0p0 );

			// c[1,0-15]
			_mm512_mask_storeu_ps( c + ( rs_c * ( ir + 1 ) ), load_mask, c_float_1p0 );

			// c[2,0-15]
			_mm512_mask_storeu_ps( c + ( rs_c * ( ir + 2 ) ), load_mask, c_float_2p0 );

			// c[3,0-15]
			_mm512_mask_storeu_ps( c + ( rs_c * ( ir + 3 ) ), load_mask, c_float_3p0 );

			// c[4,0-15]
			_mm512_mask_storeu_ps( c + ( rs_c * ( ir + 4 ) ), load_mask, c_float_4p0 );

			// c[5,0-15]
			_mm512_mask_storeu_ps( c + ( rs_c * ( ir + 5 ) ), load_mask, c_float_5p0 );

		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 5 );
			lpgemm_rowvar_bf16s4f32of32_5xlt16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta, n0_rem,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 4 );
			lpgemm_rowvar_bf16s4f32of32_4xlt16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta, n0_rem,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 3 );
			lpgemm_rowvar_bf16s4f32of32_3xlt16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta, n0_rem,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 2 );
			lpgemm_rowvar_bf16s4f32of32_2xlt16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta, n0_rem,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 1 );
			lpgemm_rowvar_bf16s4f32of32_1xlt16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta, n0_rem,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
	}

}

// 6x16 bf16 fringe kernel
LPGEMM_N_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x16m)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_6x16_DISABLE,
						  &&POST_OPS_BIAS_6x16,
						  &&POST_OPS_RELU_6x16,
						  &&POST_OPS_RELU_SCALE_6x16,
						  &&POST_OPS_GELU_TANH_6x16,
						  &&POST_OPS_GELU_ERF_6x16,
						  &&POST_OPS_CLIP_6x16,
						  &&POST_OPS_DOWNSCALE_6x16,
						  &&POST_OPS_MATRIX_ADD_6x16,
						  &&POST_OPS_SWISH_6x16,
						  &&POST_OPS_MATRIX_MUL_6x16,
						  &&POST_OPS_TANH_6x16,
						  &&POST_OPS_SIGMOID_6x16
						};
	dim_t MR = 6;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t group_size = pre_ops_attr.group_size;

	// B matrix storage bfloat type
	__m512bh b0;
	__m256i b0_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	__m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp = _mm512_set1_epi8(0x08);

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	 /* Regs to store zero-point values */
	__m512i zero_point, zero_point0;

	/* Regs tp sore masks to interleave zero-point values */
	__m512i mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
	                             0x5717561655155414, 0x5313521251115010,
	                             0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
	                             0x4707460645054404, 0x4303420241014000 );

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		// Registers to use for accumulating C.
		__m512 c_float_0p0 = _mm512_setzero_ps();

		__m512 c_float_1p0 = _mm512_setzero_ps();

		__m512 c_float_2p0 = _mm512_setzero_ps();

		__m512 c_float_3p0 = _mm512_setzero_ps();

		__m512 c_float_4p0 = _mm512_setzero_ps();

		__m512 c_float_5p0 = _mm512_setzero_ps();

		dim_t group_start = pre_ops_attr.pre_op_b_i / group_size;
		dim_t group_end   = ( pre_ops_attr.pre_op_b_i + k0 - 1 ) / group_size;

		bfloat16* a_group = (bfloat16*) a;
		int8_t* b_group = (int8_t*)b;

		if( pre_ops_attr.zero_point_len > 0 )
		{
			dim_t pre_op_sf_off = 0;
			dim_t pre_op_zp_off = 0;

			for( dim_t group = group_start; group <= group_end; group++ )
			{

				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{
					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );

				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *(( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
				}

				if( pre_ops_attr.zero_point_len > 1 )
				{
					pre_op_zp_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;
					zero_point = _mm512_maskz_loadu_epi8( 0xFFFF, ( pre_ops_attr.zero_point +
                                              pre_op_zp_off ) );
				}
				else
				{
					pre_op_zp_off = group;
					zero_point = _mm512_set1_epi8( *( ( int8_t* ) pre_ops_attr.zero_point
					                                   + pre_op_zp_off) );
				}

				zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );


				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const *)
					                      ( b_group + ( ( rs_b * kr ) / 2 ) ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
													sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					__m512i zero_reg = _mm512_setzero_si512();

					/* Interleave zero_point values with zeroes */
					zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

					b0_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const *)( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
													sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				} // k_partial_pieces
			} // group loop

		} // zero-point condition
		else
		{
			dim_t pre_op_sf_off = 0;

			for( dim_t group = group_start; group <= group_end; group++ )
			{

				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{
					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );

				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *(( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
				}

				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 =_mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )
					                     ( b_group + ( ( rs_b * kr ) / 2 ) ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
													sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					b0_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const *)( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
													sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-15]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
				} // k_partial_pieces
			} // group loop
		} // zero-point condition

		// Load alpha and beta
		__m512 selector1 = _mm512_set1_ps( alpha );
		__m512 selector2 = _mm512_set1_ps( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

			c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

			c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

			c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );

			c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );

			c_float_5p0 = _mm512_mul_ps( selector1, c_float_5p0 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{

				// c[0,0-15]
				BF16_F32_BETA_OP( c_float_0p0, ir, 0, 0, \
								selector1, selector2 );

				// c[1,0-15]
				BF16_F32_BETA_OP( c_float_1p0, ir, 1, 0, \
								selector1, selector2 );

				// c[2,0-15]
				BF16_F32_BETA_OP( c_float_2p0, ir, 2, 0, \
								selector1, selector2 );

				// c[3,0-15]
				BF16_F32_BETA_OP( c_float_3p0, ir, 3, 0, \
								selector1, selector2 );

				// c[4,0-15]
				BF16_F32_BETA_OP( c_float_4p0, ir, 4, 0, \
								selector1, selector2 );

				// c[5,0-15]
				BF16_F32_BETA_OP( c_float_5p0, ir, 5, 0, \
								selector1, selector2 );
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0, ir, 0, 0, \
								selector1, selector2);

				// c[1,0-15]
				F32_F32_BETA_OP(c_float_1p0, ir, 1, 0, \
								selector1, selector2);

				// c[2,0-15]
				F32_F32_BETA_OP(c_float_2p0, ir, 2, 0, \
								selector1, selector2);

				// c[3,0-15]
				F32_F32_BETA_OP(c_float_3p0, ir, 3, 0, \
								selector1, selector2);

				// c[4,0-15]
				F32_F32_BETA_OP(c_float_4p0, ir, 4, 0, \
								selector1, selector2);

				// c[5,0-15]
				F32_F32_BETA_OP(c_float_5p0, ir, 5, 0, \
								selector1, selector2);
			}
		}
		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6x16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector1, c_float_5p0 );
			}
			else
			{
				__m512 selector3;
				__m512 selector4;
				__m512 selector5;
				__m512 selector6;
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
					BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
					BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
					BF16_F32_BIAS_BCAST(selector6, bias_mask, 5);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 3 ) );
					selector5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 4 ) );
					selector6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 5 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector6, c_float_5p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6x16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			// c[5,0-15]
			c_float_5p0 = _mm512_max_ps( selector1, c_float_5p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6x16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_5p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_6x16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512(c_float_5p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_6x16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512(c_float_5p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_6x16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(c_float_5p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_6x16:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();
		__m512 selector5 = _mm512_setzero_ps();
		__m512 selector6 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();
		__m512 zero_point4 = _mm512_setzero_ps();
		__m512 zero_point5 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector6 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point4 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point5 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
				selector6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 5 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
				zero_point4 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
				zero_point5 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 5 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector5,zero_point4);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector6,zero_point5);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_6x16:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,1);

					// c[2:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,2);

					// c[3:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,3);

					// c[4:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,4);

					// c[5:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr2,1);

					// c[2:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr3,2);

					// c[3:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr4,3);

					// c[4:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr5,4);

					// c[5:0-15]
					BF16_F32_MATRIX_ADD_1COL(selector1,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,1);

					// c[2:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,2);

					// c[3:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,3);

					// c[4:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,4);

					// c[5:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr2,1);

					// c[2:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr3,2);

					// c[3:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr4,3);

					// c[4:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr5,4);

					// c[5:0-15]
					F32_F32_MATRIX_ADD_1COL(selector1,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_6x16:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr2,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr3,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr4,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr5,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL(selector1,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,1);

					// c[2:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,2);

					// c[3:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,3);

					// c[4:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,4);

					// c[5:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,5);
				}
				else
				{
					// c[0:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr1,0);

					// c[1:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr2,1);

					// c[2:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr3,2);

					// c[3:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr4,3);

					// c[4:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr5,4);

					// c[5:0-15]
					F32_F32_MATRIX_MUL_1COL(selector1,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_6x16:
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(c_float_5p0, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_6x16:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(c_float_3p0, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(c_float_4p0, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(c_float_5p0, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_6x16:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_3p0, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_4p0, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_5p0, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_6x16_DISABLE:
		;
		if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Generate a mask16 of all 1's.
			__m512i selector_a = _mm512_setzero_epi32();
			__m512i selector_b = _mm512_set1_epi32( 10 );
			__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

			// c[1,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

			// c[2,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

			// c[3,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

			// c[4,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);

			// c[5,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_5p0,5,0);
		}

		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_float_0p0 );

			// c[1,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_float_1p0 );

			// c[2,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_float_2p0 );

			// c[3,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_float_3p0 );

			// c[4,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_float_4p0 );

			// c[5,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_float_5p0 );
		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}


	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 5 );
			lpgemm_rowvar_bf16s4f32of32_5x16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 4 );
			lpgemm_rowvar_bf16s4f32of32_4x16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			int cs_a_use = ( cs_a == 2) ? 2 : ( ( cs_a / 6 ) * 3 );
			lpgemm_rowvar_bf16s4f32of32_3x16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 2 );
			lpgemm_rowvar_bf16s4f32of32_2x16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 1 );
			lpgemm_rowvar_bf16s4f32of32_1x16
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
	}

}

// 6x32 bf16 fringe kernel
LPGEMM_N_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x32m)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_6x32_DISABLE,
						  &&POST_OPS_BIAS_6x32,
						  &&POST_OPS_RELU_6x32,
						  &&POST_OPS_RELU_SCALE_6x32,
						  &&POST_OPS_GELU_TANH_6x32,
						  &&POST_OPS_GELU_ERF_6x32,
						  &&POST_OPS_CLIP_6x32,
						  &&POST_OPS_DOWNSCALE_6x32,
						  &&POST_OPS_MATRIX_ADD_6x32,
						  &&POST_OPS_SWISH_6x32,
						  &&POST_OPS_MATRIX_MUL_6x32,
						  &&POST_OPS_TANH_6x32,
						  &&POST_OPS_SIGMOID_6x32
						};

	dim_t MR = 6;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t group_size = pre_ops_attr.group_size;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	__m256i b0_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;


	__m512i shift_idx_64;
	MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
	__m512i sign_comp = _mm512_set1_epi8(0x08);

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1, scale2, scale3;

	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	/* Regs to store zero-point values */
    __m512i zero_point, zero_point0;

	/* Regs to store masks to inteleave zero-point values */
	__m512i mask_zp1;

	mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);



	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		// Registers to use for accumulating C.
		__m512 c_float_0p0 = _mm512_setzero_ps();
		__m512 c_float_0p1 = _mm512_setzero_ps();

		__m512 c_float_1p0 = _mm512_setzero_ps();
		__m512 c_float_1p1 = _mm512_setzero_ps();

		__m512 c_float_2p0 = _mm512_setzero_ps();
		__m512 c_float_2p1 = _mm512_setzero_ps();

		__m512 c_float_3p0 = _mm512_setzero_ps();
		__m512 c_float_3p1 = _mm512_setzero_ps();

		__m512 c_float_4p0 = _mm512_setzero_ps();
		__m512 c_float_4p1 = _mm512_setzero_ps();

		__m512 c_float_5p0 = _mm512_setzero_ps();
		__m512 c_float_5p1 = _mm512_setzero_ps();


		dim_t group_start = pre_ops_attr.pre_op_b_i / group_size;
		dim_t group_end   = ( pre_ops_attr.pre_op_b_i + k0 - 1 ) / group_size;

		bfloat16* a_group = (bfloat16*) a;
		int8_t* b_group = (int8_t*)b;

		if( pre_ops_attr.zero_point_len > 0 )
		{
			dim_t pre_op_sf_off = 0;
			dim_t pre_op_zp_off = 0;

			for( dim_t group = group_start; group <= group_end; group++ )
			{

				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{
					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
						scale2 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 );
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
						scale2 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
					scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
					scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *( ( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
					scale2 = scale0;
					scale3 = scale0;
				}

				if( pre_ops_attr.zero_point_len > 1 )
				{
					pre_op_zp_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;
					zero_point = _mm512_maskz_loadu_epi8( 0xFFFFFFFF,
					                      ( pre_ops_attr.zero_point +
                                              pre_op_zp_off ) );
				}
				else
				{
					pre_op_zp_off = group;
					zero_point = _mm512_set1_epi8( *( ( int8_t* )( pre_ops_attr.zero_point
					                                   + pre_op_zp_off) ) );
				}

				zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group + ( rs_b * kr ) / 2 ) );


					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
											CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
											CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-31] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-31] = a[5,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					__m512i zero_reg = _mm512_setzero_si512();

					/* Interleave zero_point values with zeroes */
					zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
					                                    sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-47] = a[5,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
				} // k_partial_pieces

			} // group loop
		} // zero-point condition
		else
		{
			for( dim_t group = group_start; group <= group_end; group++ )
			{
				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				// calculate offsets
				dim_t pre_op_sf_off = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{
					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
						scale2 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 );
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
						scale2 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
					scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
					scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *( ( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
					scale2 = scale0;
					scale3 = scale0;
				}
				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group + ( rs_b * kr ) / 2 ) );


					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
											CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
											CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-31] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-31] = a[5,kr:kr+2]*b[kr:kr+2,0-31]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
					                                    sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-47] = a[5,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
				} // k_partial_pieces
			} // group loop
		} // zero-point condition

		// Load alpha and beta
		__m512 selector1 = _mm512_set1_ps( alpha );
		__m512 selector2 = _mm512_set1_ps( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
			c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

			c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
			c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );

			c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
			c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );

			c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
			c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );

			c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
			c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );

			c_float_5p0 = _mm512_mul_ps( selector1, c_float_5p0 );
			c_float_5p1 = _mm512_mul_ps( selector1, c_float_5p1 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{

				// c[0,0-15]
				BF16_F32_BETA_OP( c_float_0p0, ir, 0, 0, selector1, selector2 );

				// c[0, 16-31]
				BF16_F32_BETA_OP( c_float_0p1, ir, 0, 1, selector1, selector2 );

				// c[1,0-15]
				BF16_F32_BETA_OP( c_float_1p0, ir, 1, 0, selector1, selector2 );

				// c[1, 16-31]
				BF16_F32_BETA_OP( c_float_1p1, ir, 1, 1, selector1, selector2 );

				// c[2,0-15]
				BF16_F32_BETA_OP( c_float_2p0, ir, 2, 0, selector1, selector2 );

				// c[2, 16-31]
				BF16_F32_BETA_OP( c_float_2p1, ir, 2, 1, selector1, selector2 );

				// c[3,0-15]
				BF16_F32_BETA_OP( c_float_3p0, ir, 3, 0, selector1, selector2 );

				// c[3, 16-31]
				BF16_F32_BETA_OP( c_float_3p1, ir, 3, 1, selector1, selector2 );

				// c[4,0-15]
				BF16_F32_BETA_OP( c_float_4p0, ir, 4, 0, selector1, selector2 );

				// c[4, 16-31]
				BF16_F32_BETA_OP( c_float_4p1, ir, 4, 1, selector1, selector2 );

				// c[5,0-15]
				BF16_F32_BETA_OP( c_float_5p0, ir, 5, 0, selector1, selector2 );

				// c[5, 16-31]
				BF16_F32_BETA_OP( c_float_5p1, ir, 5, 1, selector1, selector2 );
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP( c_float_0p0, ir, 0, 0, selector1, selector2 );

				// c[0, 16-31]
				F32_F32_BETA_OP( c_float_0p1, ir, 0, 1, selector1, selector2 );

				// c[1,0-15]
				F32_F32_BETA_OP( c_float_1p0, ir, 1, 0, selector1, selector2 );

				// c[1, 16-31]
				F32_F32_BETA_OP( c_float_1p1, ir, 1, 1, selector1, selector2 );

				// c[2,0-15]
				F32_F32_BETA_OP( c_float_2p0, ir, 2, 0, selector1, selector2 );

				// c[2, 16-31]
				F32_F32_BETA_OP( c_float_2p1, ir, 2, 1, selector1, selector2 );

				// c[3,0-15]
				F32_F32_BETA_OP( c_float_3p0, ir, 3, 0, selector1, selector2 );

				// c[3, 16-31]
				F32_F32_BETA_OP( c_float_3p1, ir, 3, 1, selector1, selector2 );

				// c[4,0-15]
				F32_F32_BETA_OP( c_float_4p0, ir, 4, 0, selector1, selector2 );

				// c[4, 16-31]
				F32_F32_BETA_OP( c_float_4p1, ir, 4, 1, selector1, selector2 );

				// c[5,0-15]
				F32_F32_BETA_OP( c_float_5p0, ir, 5, 0, selector1, selector2 );

				// c[5, 16-31]
				F32_F32_BETA_OP( c_float_5p1, ir, 5, 1, selector1, selector2 );
			}

		}
		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6x32:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
					BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				}
				else
				{
					selector1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

				// c[4, 16-31]
				c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector1, c_float_5p0 );

				// c[5, 16-31]
				c_float_5p1 = _mm512_add_ps( selector2, c_float_5p1 );
			}
			else
			{
				__m512 selector3;
				__m512 selector4;
				__m512 selector5;
				__m512 selector6;
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
					BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
					BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
					BF16_F32_BIAS_BCAST(selector6, bias_mask, 5);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 3 ) );
					selector5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 4 ) );
					selector6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 5 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

				// c[4, 16-31]
				c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector6, c_float_5p0 );

				// c[5, 16-31]
				c_float_5p1 = _mm512_add_ps( selector6, c_float_5p1 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6x32:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[2,16-31]
			c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[3,16-31]
			c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			// c[4,16-31]
			c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

			// c[5,0-15]
			c_float_5p0 = _mm512_max_ps( selector1, c_float_5p0 );

			// c[5,16-31]
			c_float_5p1 = _mm512_max_ps( selector1, c_float_5p1 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6x32:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[0, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_0p1)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_1p1)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_2p1)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[3, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_3p1)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			// c[4, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_4p1)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_5p0)

			// c[5, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_5p1)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_6x32:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 16-31]
			GELU_TANH_F32_AVX512(c_float_4p1, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512(c_float_5p0, r, r2, x, z, dn, x_tanh, q)

			// c[5, 16-31]
			GELU_TANH_F32_AVX512(c_float_5p1, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_6x32:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			// c[4, 16-31]
			GELU_ERF_F32_AVX512(c_float_4p1, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512(c_float_5p0, r, x, x_erf)

			// c[5, 16-31]
			GELU_ERF_F32_AVX512(c_float_5p1, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_6x32:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(c_float_2p1, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(c_float_3p1, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			// c[4, 16-31]
			CLIP_F32_AVX512(c_float_4p1, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(c_float_5p0, min, max)

			// c[5, 16-31]
			CLIP_F32_AVX512(c_float_5p1, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_6x32:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();
		__m512 selector5 = _mm512_setzero_ps();
		__m512 selector6 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();
		__m512 zero_point4 = _mm512_setzero_ps();
		__m512 zero_point5 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector6 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point4 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point5 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector2,zero_point1);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector1,zero_point0);

			// c[5, 16-31]
			SCL_MULRND_F32(c_float_5p1,selector2,zero_point1);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
				selector6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 5 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
				zero_point4 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
				zero_point5 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 5 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector5,zero_point4);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector5,zero_point4);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector6,zero_point5);

			// c[5, 16-31]
			SCL_MULRND_F32(c_float_5p1,selector6,zero_point5);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_6x32:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,5);
				}
				else
				{
					// c[0:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					BF16_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr6,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr2,5);
				}
				else
				{
					// c[0:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					F32_F32_MATRIX_ADD_2COL(selector1,selector2,scl_fctr6,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_6x32:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,5);
				}
				else
				{
					// c[0:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr6,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr2,5);
				}
				else
				{
					// c[0:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					F32_F32_MATRIX_MUL_2COL(selector1,selector2,scl_fctr6,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_6x32:
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SWISH_F32_AVX512_DEF(c_float_4p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(c_float_5p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 16-31]
			SWISH_F32_AVX512_DEF(c_float_5p1, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_6x32:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(c_float_1p1, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[2, 16-31]
			TANHF_AVX512(c_float_2p1, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(c_float_3p0, r, r2, x, z, dn, q)

			// c[3, 16-31]
			TANHF_AVX512(c_float_3p1, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(c_float_4p0, r, r2, x, z, dn, q)

			// c[4, 16-31]
			TANHF_AVX512(c_float_4p1, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(c_float_5p0, r, r2, x, z, dn, q)

			// c[5, 16-31]
			TANHF_AVX512(c_float_5p1, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_6x32:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_1p1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_2p1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_3p0, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_3p1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_4p0, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_4p1, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_5p0, al_in, r, r2, z, dn, ex_out);

			// c[5, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_5p1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_6x32_DISABLE:
		;
		if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Generate a mask16 of all 1's.
			__m512i selector_a = _mm512_setzero_epi32();
			__m512i selector_b = _mm512_set1_epi32( 10 );
			__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

			// c[0, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

			// c[1,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

			// c[1, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

			// c[2,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

			// c[2, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

			// c[3,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

			// c[3, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);

			// c[4,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);

			// c[4, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_4p1,4,1);

			// c[5,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_5p0,5,0);

			// c[5, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_5p1,5,1);
		}

		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_float_0p0 );

			// c[0, 16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ), c_float_0p1 );

			// c[1,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_float_1p0 );

			// c[1,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ), c_float_1p1 );

			// c[2,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_float_2p0 );

			// c[2,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ), c_float_2p1 );

			// c[3,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_float_3p0 );

			// c[3,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ), c_float_3p1 );

			// c[4,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_float_4p0 );

			// c[4,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ), c_float_4p1 );

			// c[5,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_float_5p0 );

			// c[5,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ), c_float_5p1 );
		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 5 );
			lpgemm_rowvar_bf16s4f32of32_5x32
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 4 );
			lpgemm_rowvar_bf16s4f32of32_4x32
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 3 );
			lpgemm_rowvar_bf16s4f32of32_3x32
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 2 );
			lpgemm_rowvar_bf16s4f32of32_2x32
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 1 );
			lpgemm_rowvar_bf16s4f32of32_1x32
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
	}

}

// 6x48 bf16 fringe kernel
LPGEMM_N_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x48m)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_6x48_DISABLE,
						  &&POST_OPS_BIAS_6x48,
						  &&POST_OPS_RELU_6x48,
						  &&POST_OPS_RELU_SCALE_6x48,
						  &&POST_OPS_GELU_TANH_6x48,
						  &&POST_OPS_GELU_ERF_6x48,
						  &&POST_OPS_CLIP_6x48,
						  &&POST_OPS_DOWNSCALE_6x48,
						  &&POST_OPS_MATRIX_ADD_6x48,
						  &&POST_OPS_SWISH_6x48,
						  &&POST_OPS_MATRIX_MUL_6x48,
						  &&POST_OPS_TANH_6x48,
						  &&POST_OPS_SIGMOID_6x48
						};

	dim_t MR = 6;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t group_size = pre_ops_attr.group_size;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	__m256i b0_s4;
	__m256i b1_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	__m512i shift_idx_64;
	MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
	__m512i sign_comp = _mm512_set1_epi8(0x08);

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8;
	__m512i b1_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1, scale2, scale3, scale4, scale5;

	/* Regs to store zero point values */
	__m512i zero_point, zero_point0, zero_point1;

	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

	/* Reg to store masks to interleave zero-point */
	__m512i mask_zp1, mask_zp2;

	mask_zp1 = _mm512_set_epi64( 0x5F1F5E1E5D1D5C1C, 0x5B1B5A1A59195818,
                                 0x5717561655155414, 0x5313521251115010,
                                 0x4F0F4E0E4D0D4C0C, 0x4B0B4A0A49094808,
                                 0x4707460645054404, 0x4303420241014000 );

    mask_zp2 = _mm512_set_epi64( 0x7F3F7E3E7D3D7C3C, 0x7B3B7A3A79397838,
                                 0x7737763675357434, 0x7333723271317030,
                                 0x6F2F6E2E6D2D6C2C, 0x6B2B6A2A69296828,
                                 0x6727662665256424, 0x6323622261216020 );

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		// Registers to use for accumulating C.
		__m512 c_float_0p0 = _mm512_setzero_ps();
		__m512 c_float_0p1 = _mm512_setzero_ps();
		__m512 c_float_0p2 = _mm512_setzero_ps();

		__m512 c_float_1p0 = _mm512_setzero_ps();
		__m512 c_float_1p1 = _mm512_setzero_ps();
		__m512 c_float_1p2 = _mm512_setzero_ps();

		__m512 c_float_2p0 = _mm512_setzero_ps();
		__m512 c_float_2p1 = _mm512_setzero_ps();
		__m512 c_float_2p2 = _mm512_setzero_ps();

		__m512 c_float_3p0 = _mm512_setzero_ps();
		__m512 c_float_3p1 = _mm512_setzero_ps();
		__m512 c_float_3p2 = _mm512_setzero_ps();

		__m512 c_float_4p0 = _mm512_setzero_ps();
		__m512 c_float_4p1 = _mm512_setzero_ps();
		__m512 c_float_4p2 = _mm512_setzero_ps();

		__m512 c_float_5p0 = _mm512_setzero_ps();
		__m512 c_float_5p1 = _mm512_setzero_ps();
		__m512 c_float_5p2 = _mm512_setzero_ps();

		dim_t group_start = pre_ops_attr.pre_op_b_i / group_size;
		dim_t group_end   = ( pre_ops_attr.pre_op_b_i + k0 - 1 ) / group_size;

		bfloat16* a_group = (bfloat16*) a;
		int8_t* b_group = (int8_t*)b;

		if( pre_ops_attr.zero_point_len > 0 )
		{
			dim_t pre_op_sf_off = 0;
			dim_t pre_op_zp_off = 0;

			for( dim_t group = group_start; group <= group_end; group++ )
			{

				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				if( pre_ops_attr.scale_factor_len > 1 )
				{
					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
						scale2 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 );
						scale4 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 32 );
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
						scale2 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 ));
						scale4 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 32 ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
					scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
					scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
					scale5 = _mm512_permutex2var_ps( scale4, mask_scale2, scale4 );
					scale4 = _mm512_permutex2var_ps( scale4, mask_scale1, scale4 );

				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *( ( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
					scale2 = scale0;
					scale3 = scale0;
					scale4 = scale0;
					scale5 = scale0;
				}

				if( pre_ops_attr.zero_point_len > 1 )
				{
					pre_op_zp_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;
					zero_point = _mm512_maskz_loadu_epi8( 0xFFFFFFFFFFFF,
					            ( pre_ops_attr.zero_point + pre_op_zp_off ) );
				}
				else
				{
					pre_op_zp_off = group;
					zero_point = _mm512_set1_epi8( *( ( int8_t* )
					                  pre_ops_attr.zero_point + pre_op_zp_off ) );
				}
				zero_point1 = _mm512_permutex2var_epi8( zero_point, mask_zp2, zero_point );
				zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_point );

				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{
					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group +
					                            ( rs_b * kr ) / 2 ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					b1_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )( b_group +
					                         ( ( rs_b * kr ) / 2 ) + 32 ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b1_s8 = _mm512_sub_epi8( b1_s8, zero_point1 );

					b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 1, scale5 ),
					                          CVT_INT8_F32_SCAL_16( b1_s8, 0, scale4 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
					c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
					c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
					c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
					c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
					c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-47] = a[5,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
					c_float_5p2 = _mm512_dpbf16_ps( c_float_5p2, a_bf16_0, b2 );

				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					__m512i zero_reg = _mm512_setzero_si512();

					/* Interleave zero_point values with zeroes */
					zero_point1 = _mm512_permutex2var_epi8( zero_point, mask_zp2, zero_reg );
					zero_point0 = _mm512_permutex2var_epi8( zero_point, mask_zp1, zero_reg );

					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
					                                    sign_comp, signed_upscale);

					b0_s8 = _mm512_sub_epi8( b0_s8, zero_point0 );

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					b1_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )( b_group + 32 ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_64, \
					                                    sign_comp, signed_upscale);

					b1_s8 = _mm512_sub_epi8( b1_s8, zero_point1 );

					b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 1, scale5 ),
					                          CVT_INT8_F32_SCAL_16( b1_s8, 0, scale4 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
					c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

					// Broadcast a[1,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
					c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
					c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
					c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
					c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-47] = a[5,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
					c_float_5p2 = _mm512_dpbf16_ps( c_float_5p2, a_bf16_0, b2 );
				} // k_partial_pieces
			} // group
		} // zero_point condition
		else
		{
			for( dim_t group = group_start; group <= group_end; group++ )
			{
				dim_t k_start = bli_max( group * group_size, pre_ops_attr.pre_op_b_i );
				dim_t k_end = bli_min( ( ( group + 1 ) * group_size - 1 ),
			                           pre_ops_attr.pre_op_b_i + k0 - 1);
				dim_t kg0 = k_end - k_start + 1;
				dim_t k_full_pieces = kg0 / 2;
				dim_t k_partial_pieces = kg0 % 2;

				int16_t a_kfringe_buf = 0;

				dim_t pre_op_sf_off = 0;
				if( pre_ops_attr.scale_factor_len > 1 )
				{
					pre_op_sf_off = ( group * pre_ops_attr.pre_op_ld ) +
					                pre_ops_attr.pre_op_b_j;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						// load scale factor vectors
						scale0 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off);
						scale2 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 );
						scale4 = _mm512_loadu_ps( (float*)( pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 32 );
					}
					else
					{
						// load and convert scale factor vectors to F32 type
						scale0 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off ));
						scale2 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 16 ));
						scale4 = CVT_BF16_F32_INT_SHIFT( (__m256i)_mm256_loadu_epi16(
						                        (bfloat16*)(pre_ops_attr.scale_factor ) +
						                                    pre_op_sf_off + 32 ));
					}

					// interleave scale factor vectors
					scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
					scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
					scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
					scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
					scale5 = _mm512_permutex2var_ps( scale4, mask_scale2, scale4 );
					scale4 = _mm512_permutex2var_ps( scale4, mask_scale1, scale4 );

				}
				else
				{
					pre_op_sf_off = group;

					if( pre_ops_attr.scale_factor_type == F32 )
					{
						scale0 = _mm512_set1_ps(
						               *( ( float* )pre_ops_attr.scale_factor +
						                            pre_op_sf_off ) );
					}
					else
					{
						scale0 = CVT_BF16_F32_INT_SHIFT( _mm256_set1_epi16(
						                *( ( bfloat16* )( pre_ops_attr.scale_factor ) +
						                                pre_op_sf_off ) ) );
					}

					scale1 = scale0;
					scale2 = scale0;
					scale3 = scale0;
					scale4 = scale0;
					scale5 = scale0;
				}
				for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
				{

					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group +
					                            ( rs_b * kr ) / 2 ) );


					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
											CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
											CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					b1_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )( b_group +
					                         ( ( rs_b * kr ) / 2 ) + 32 ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_64, \
														sign_comp, signed_upscale);

					b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 1, scale5 ),
											CVT_INT8_F32_SCAL_16( b1_s8, 0, scale4 ) );

					// Broadcast a[0,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 0 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
					c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

					// Broadcast a[1,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 1 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
					c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

					// Broadcast a[2,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 2 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
					c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

					// Broadcast a[3,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 3 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
					c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );

					// Broadcast a[4,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 4 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
					c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );

					// Broadcast a[5,kr:kr+2].
					a_bf16_0 = (__m512bh)_mm512_set1_epi32(
							*( int32_t* )( a_group + ( rs_a * 5 ) + ( cs_a * kr ) ) );

					// Perform column direction mat-mul with k = 2.
					// c[5,0-47] = a[5,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
					c_float_5p2 = _mm512_dpbf16_ps( c_float_5p2, a_bf16_0, b2 );
				} // k-loop

				a_group += k_full_pieces * cs_a;
				b_group += ( k_full_pieces * rs_b ) / 2;

				// Group_size is always even, so k_partial_pieces will always
				// appear in the last group. So, a_group and b_group pointers
				// need not be updated after handling k_partial pieces.
				if( k_partial_pieces )
				{
					b0_s4 = _mm256_loadu_si256( (__m256i const *)( b_group ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
					                                    sign_comp, signed_upscale);

					b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

					b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
					                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

					b1_s4 = _mm256_maskz_loadu_epi8( 0xFFFF, (__m256i const* )( b_group + 32 ) );

					CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_64, \
					                                    sign_comp, signed_upscale);

					b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 1, scale5 ),
					                          CVT_INT8_F32_SCAL_16( b1_s8, 0, scale4 ) );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 0) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
					c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
					c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

					// Broadcast a[0,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 1) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
					c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
					c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

					// Broadcast a[2,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 2) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
					c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
					c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

					// Broadcast a[3,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 3) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
					c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
					c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );

					// Broadcast a[4,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 4) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
					c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
					c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );

					// Broadcast a[5,kr:kr+2].
					a_kfringe_buf = *( a_group + (rs_a * 5) );
					a_bf16_0 = (__m512bh)_mm512_set1_epi16(( a_kfringe_buf ));

					// Perform column direction mat-mul with k = 2.
					// c[5,0-47] = a[5,kr:kr+2]*b[kr:kr+2,0-47]
					c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_0, b0 );
					c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_0, b1 );
					c_float_5p2 = _mm512_dpbf16_ps( c_float_5p2, a_bf16_0, b2 );
				} // k_partial_pieces
			} // group loop
		} // zero-point condition


		// Load alpha and beta
		__m512 selector1 = _mm512_set1_ps( alpha );
		__m512 selector2 = _mm512_set1_ps( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
			c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
			c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

			c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
			c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
			c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );

			c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
			c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
			c_float_2p2 = _mm512_mul_ps( selector1, c_float_2p2 );

			c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
			c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );
			c_float_3p2 = _mm512_mul_ps( selector1, c_float_3p2 );

			c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
			c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );
			c_float_4p2 = _mm512_mul_ps( selector1, c_float_4p2 );

			c_float_5p0 = _mm512_mul_ps( selector1, c_float_5p0 );
			c_float_5p1 = _mm512_mul_ps( selector1, c_float_5p1 );
			c_float_5p2 = _mm512_mul_ps( selector1, c_float_5p2 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				// c[0,0-15]
				BF16_F32_BETA_OP(c_float_0p0,ir,0,0,selector1,selector2)

				// c[0, 16-31]
				BF16_F32_BETA_OP(c_float_0p1,ir,0,1,selector1,selector2)

				// c[0,32-47]
				BF16_F32_BETA_OP(c_float_0p2,ir,0,2,selector1,selector2)

				// c[1,0-15]
				BF16_F32_BETA_OP(c_float_1p0,ir,1,0,selector1,selector2)

				// c[1,16-31]
				BF16_F32_BETA_OP(c_float_1p1,ir,1,1,selector1,selector2)

				// c[1,32-47]
				BF16_F32_BETA_OP(c_float_1p2,ir,1,2,selector1,selector2)

				// c[2,0-15]
				BF16_F32_BETA_OP(c_float_2p0,ir,2,0,selector1,selector2)

				// c[2,16-31]
				BF16_F32_BETA_OP(c_float_2p1,ir,2,1,selector1,selector2)

				// c[2,32-47]
				BF16_F32_BETA_OP(c_float_2p2,ir,2,2,selector1,selector2)

				// c[3,0-15]
				BF16_F32_BETA_OP(c_float_3p0,ir,3,0,selector1,selector2)

				// c[3,16-31]
				BF16_F32_BETA_OP(c_float_3p1,ir,3,1,selector1,selector2)

				// c[3,32-47]
				BF16_F32_BETA_OP(c_float_3p2,ir,3,2,selector1,selector2)

				// c[4,0-15]
				BF16_F32_BETA_OP(c_float_4p0,ir,4,0,selector1,selector2)

				// c[4,16-31]
				BF16_F32_BETA_OP(c_float_4p1,ir,4,1,selector1,selector2)

				// c[4,32-47]
				BF16_F32_BETA_OP(c_float_4p2,ir,4,2,selector1,selector2)

				// c[5,0-15]
				BF16_F32_BETA_OP(c_float_5p0,ir,5,0,selector1,selector2)

				// c[5,16-31]
				BF16_F32_BETA_OP(c_float_5p1,ir,5,1,selector1,selector2)

				// c[5,32-47]
				BF16_F32_BETA_OP(c_float_5p2,ir,5,2,selector1,selector2)
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0,ir,0,0,selector1,selector2)

				// c[0, 16-31]
				F32_F32_BETA_OP(c_float_0p1,ir,0,1,selector1,selector2)

				// c[0,32-47]
				F32_F32_BETA_OP(c_float_0p2,ir,0,2,selector1,selector2)

				// c[1,0-15]
				F32_F32_BETA_OP(c_float_1p0,ir,1,0,selector1,selector2)

				// c[1,16-31]
				F32_F32_BETA_OP(c_float_1p1,ir,1,1,selector1,selector2)

				// c[1,32-47]
				F32_F32_BETA_OP(c_float_1p2,ir,1,2,selector1,selector2)

				// c[2,0-15]
				F32_F32_BETA_OP(c_float_2p0,ir,2,0,selector1,selector2)

				// c[2,16-31]
				F32_F32_BETA_OP(c_float_2p1,ir,2,1,selector1,selector2)

				// c[2,32-47]
				F32_F32_BETA_OP(c_float_2p2,ir,2,2,selector1,selector2)

				// c[3,0-15]
				F32_F32_BETA_OP(c_float_3p0,ir,3,0,selector1,selector2)

				// c[3,16-31]
				F32_F32_BETA_OP(c_float_3p1,ir,3,1,selector1,selector2)

				// c[3,32-47]
				F32_F32_BETA_OP(c_float_3p2,ir,3,2,selector1,selector2)

				// c[4,0-15]
				F32_F32_BETA_OP(c_float_4p0,ir,4,0,selector1,selector2)

				// c[4,16-31]
				F32_F32_BETA_OP(c_float_4p1,ir,4,1,selector1,selector2)

				// c[4,32-47]
				F32_F32_BETA_OP(c_float_4p2,ir,4,2,selector1,selector2)

				// c[5,0-15]
				F32_F32_BETA_OP(c_float_5p0,ir,5,0,selector1,selector2)

				// c[5,16-31]
				F32_F32_BETA_OP(c_float_5p1,ir,5,1,selector1,selector2)

				// c[5,32-47]
				F32_F32_BETA_OP(c_float_5p2,ir,5,2,selector1,selector2)
			}
		}
		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6x48:
		{
			__m512 selector3;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
					BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
					BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
				}
				else
				{
					selector1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

				// c[3,32-47]
				c_float_3p2 = _mm512_add_ps( selector3, c_float_3p2 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

				// c[4, 16-31]
				c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );

				// c[4,32-47]
				c_float_4p2 = _mm512_add_ps( selector3, c_float_4p2 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector1, c_float_5p0 );

				// c[5, 16-31]
				c_float_5p1 = _mm512_add_ps( selector2, c_float_5p1 );

				// c[5,32-47]
				c_float_5p2 = _mm512_add_ps( selector3, c_float_5p2 );
			}
			else
			{
				__m512 selector4;
				__m512 selector5;
				__m512 selector6;
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
					BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
					BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
					BF16_F32_BIAS_BCAST(selector6, bias_mask, 5);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 3 ) );
					selector5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 4 ) );
					selector6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 5 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

				// c[3,32-47]
				c_float_3p2 = _mm512_add_ps( selector4, c_float_3p2 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

				// c[4, 16-31]
				c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );

				// c[4,32-47]
				c_float_4p2 = _mm512_add_ps( selector5, c_float_4p2 );

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector6, c_float_5p0 );

				// c[5, 16-31]
				c_float_5p1 = _mm512_add_ps( selector6, c_float_5p1 );

				// c[5,32-47]
				c_float_5p2 = _mm512_add_ps( selector6, c_float_5p2 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6x48:
		{
			//printf("relu\n");
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[2,16-31]
			c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[3,16-31]
			c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_max_ps( selector1, c_float_3p2 );

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			// c[4,16-31]
			c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

			// c[4,32-47]
			c_float_4p2 = _mm512_max_ps( selector1, c_float_4p2 );

			// c[5,0-15]
			c_float_5p0 = _mm512_max_ps( selector1, c_float_5p0 );

			// c[5,16-31]
			c_float_5p1 = _mm512_max_ps( selector1, c_float_5p1 );

			// c[5,32-47]
			c_float_5p2 = _mm512_max_ps( selector1, c_float_5p2 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6x48:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[0, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_0p1)

			// c[0, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_0p2)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_1p1)

			// c[1, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_1p2)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_2p1)

			// c[2, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_2p2)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[3, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_3p1)

			// c[3, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_3p2)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			// c[4, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_4p1)

			// c[4, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_4p2)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_5p0)

			// c[5, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_5p1)

			// c[5, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_5p2)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_6x48:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

			// c[3, 32-47]
			GELU_TANH_F32_AVX512(c_float_3p2, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 16-31]
			GELU_TANH_F32_AVX512(c_float_4p1, r, r2, x, z, dn, x_tanh, q)

			// c[4, 32-47]
			GELU_TANH_F32_AVX512(c_float_4p2, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512(c_float_5p0, r, r2, x, z, dn, x_tanh, q)

			// c[5, 16-31]
			GELU_TANH_F32_AVX512(c_float_5p1, r, r2, x, z, dn, x_tanh, q)

			// c[5, 32-47]
			GELU_TANH_F32_AVX512(c_float_5p2, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_6x48:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

			// c[3, 32-47]
			GELU_ERF_F32_AVX512(c_float_3p2, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			// c[4, 16-31]
			GELU_ERF_F32_AVX512(c_float_4p1, r, x, x_erf)

			// c[4, 32-47]
			GELU_ERF_F32_AVX512(c_float_4p2, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512(c_float_5p0, r, x, x_erf)

			// c[5, 16-31]
			GELU_ERF_F32_AVX512(c_float_5p1, r, x, x_erf)

			// c[5, 32-47]
			GELU_ERF_F32_AVX512(c_float_5p2, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_6x48:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(c_float_1p2, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(c_float_2p1, min, max)

			// c[2, 32-47]
			CLIP_F32_AVX512(c_float_2p2, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(c_float_3p1, min, max)

			// c[3, 32-47]
			CLIP_F32_AVX512(c_float_3p2, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			// c[4, 16-31]
			CLIP_F32_AVX512(c_float_4p1, min, max)

			// c[4, 32-47]
			CLIP_F32_AVX512(c_float_4p2, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(c_float_5p0, min, max)

			// c[5, 16-31]
			CLIP_F32_AVX512(c_float_5p1, min, max)

			// c[5, 32-47]
			CLIP_F32_AVX512(c_float_5p2, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_6x48:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

			// c[3, 32-47]
			SCL_MULRND_F32(c_float_3p2,selector3,zero_point2);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector2,zero_point1);

			// c[4, 32-47]
			SCL_MULRND_F32(c_float_4p2,selector3,zero_point2);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector1,zero_point0);

			// c[5, 16-31]
			SCL_MULRND_F32(c_float_5p1,selector2,zero_point1);

			// c[5, 32-47]
			SCL_MULRND_F32(c_float_5p2,selector3,zero_point2);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

			// c[3, 32-47]
			SCL_MULRND_F32(c_float_3p2,selector4,zero_point3);

			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 5 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 5 ) ) );
			}
			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector1,zero_point0);

			// c[4, 32-47]
			SCL_MULRND_F32(c_float_4p2,selector1,zero_point0);

			// c[5, 0-15]
			SCL_MULRND_F32(c_float_5p0,selector2,zero_point1);

			// c[5, 16-31]
			SCL_MULRND_F32(c_float_5p1,selector2,zero_point1);

			// c[5, 32-47]
			SCL_MULRND_F32(c_float_5p2,selector2,zero_point1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_6x48:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 selector3 = _mm512_setzero_ps();

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,0);

					// c[1:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,1);

					// c[2:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,3);

					// c[4:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,4);

					// c[5:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47]
					BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,0);

					// c[1:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,1);

					// c[2:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,3);

					// c[4:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,4);

					// c[5:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47]
					F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
							scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_6x48:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 selector3 = _mm512_setzero_ps();

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,0);

					// c[1:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,1);

					// c[2:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,3);

					// c[4:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,4);

					// c[5:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47]
					BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,0);

					// c[1:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,1);

					// c[2:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,3);

					// c[4:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,4);

					// c[5:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr2,scl_fctr3,5);
				}
				else
				{
					// c[0:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr5,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31,32-47]
					F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
							scl_fctr6,scl_fctr6,scl_fctr6,5);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_6x48:
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SWISH_F32_AVX512_DEF(c_float_0p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SWISH_F32_AVX512_DEF(c_float_3p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SWISH_F32_AVX512_DEF(c_float_4p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 32-47]
			SWISH_F32_AVX512_DEF(c_float_4p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(c_float_5p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 16-31]
			SWISH_F32_AVX512_DEF(c_float_5p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[5, 32-47]
			SWISH_F32_AVX512_DEF(c_float_5p2, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_6x48:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[0, 32-47]
			TANHF_AVX512(c_float_0p2, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(c_float_1p1, r, r2, x, z, dn, q)

			// c[1, 32-47]
			TANHF_AVX512(c_float_1p2, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[2, 16-31]
			TANHF_AVX512(c_float_2p1, r, r2, x, z, dn, q)

			// c[2, 32-47]
			TANHF_AVX512(c_float_2p2, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(c_float_3p0, r, r2, x, z, dn, q)

			// c[3, 16-31]
			TANHF_AVX512(c_float_3p1, r, r2, x, z, dn, q)

			// c[3, 32-47]
			TANHF_AVX512(c_float_3p2, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(c_float_4p0, r, r2, x, z, dn, q)

			// c[4, 16-31]
			TANHF_AVX512(c_float_4p1, r, r2, x, z, dn, q)

			// c[4, 32-47]
			TANHF_AVX512(c_float_4p2, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(c_float_5p0, r, r2, x, z, dn, q)

			// c[5, 16-31]
			TANHF_AVX512(c_float_5p1, r, r2, x, z, dn, q)

			// c[5, 32-47]
			TANHF_AVX512(c_float_5p2, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_6x48:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_0p2, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_1p1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_1p2, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_2p1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_2p2, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_3p0, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_3p1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_3p2, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_4p0, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_4p1, al_in, r, r2, z, dn, ex_out);

			// c[4, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_4p2, al_in, r, r2, z, dn, ex_out);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_5p0, al_in, r, r2, z, dn, ex_out);

			// c[5, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_5p1, al_in, r, r2, z, dn, ex_out);

			// c[5, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_5p2, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_6x48_DISABLE:
		;
		// Case where the output C matrix is bf16 (downscaled) and this is the
		// final write for a given block within C.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Generate a mask16 of all 1's.
			__m512i selector_a = _mm512_setzero_epi32();
			__m512i selector_b = _mm512_set1_epi32( 10 );
			__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

			// Store the results in downscaled type (bf16 instead of float).

			// c[0, 0-15]
			CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

			// c[0, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

			// c[0, 32-47]
			CVT_STORE_F32_BF16_MASK(c_float_0p2,0,2);

			// c[1, 0-15]
			CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

			// c[1, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

			// c[1, 32-47]
			CVT_STORE_F32_BF16_MASK(c_float_1p2,1,2);

			// c[2, 0-15]
			CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

			// c[2, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

			// c[2, 32-47]
			CVT_STORE_F32_BF16_MASK(c_float_2p2,2,2);

			// c[3, 0-15]
			CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

			// c[3, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);

			// c[3, 32-47]
			CVT_STORE_F32_BF16_MASK(c_float_3p2,3,2);

			// c[4, 0-15]
			CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);

			// c[4, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_4p1,4,1);

			// c[4, 32-47]
			CVT_STORE_F32_BF16_MASK(c_float_4p2,4,2);

			// c[5, 0-15]
			CVT_STORE_F32_BF16_MASK(c_float_5p0,5,0);

			// c[5, 16-31]
			CVT_STORE_F32_BF16_MASK(c_float_5p1,5,1);

			// c[5, 32-47]
			CVT_STORE_F32_BF16_MASK(c_float_5p2,5,2);
		}

		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_float_0p0 );

			// c[0, 16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ), c_float_0p1 );

			// c[0,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 2*16 ), c_float_0p2 );

			// c[1,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_float_1p0 );

			// c[1,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ), c_float_1p1 );

			// c[1,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 2*16 ), c_float_1p2 );

			// c[2,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_float_2p0 );

			// c[2,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ), c_float_2p1 );

			// c[2,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 2*16 ), c_float_2p2 );

			// c[3,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_float_3p0 );

			// c[3,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ), c_float_3p1 );

			// c[3,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 2*16 ), c_float_3p2 );

			// c[4,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_float_4p0 );

			// c[4,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ), c_float_4p1 );

			// c[4,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 2*16 ), c_float_4p2 );

			// c[5,0-15]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_float_5p0 );

			// c[5,16-31]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ), c_float_5p1 );

			// c[5,32-47]
			_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 2*16 ), c_float_5p2 );
		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;

	}

	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 5 );
			lpgemm_rowvar_bf16s4f32of32_5x48
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 4 );
			lpgemm_rowvar_bf16s4f32of32_4x48
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 3 );
			lpgemm_rowvar_bf16s4f32of32_3x48
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 2 );
			lpgemm_rowvar_bf16s4f32of32_2x48
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			int cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 1 );
			lpgemm_rowvar_bf16s4f32of32_1x48
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr, pre_ops_attr
			);
		}
	}

}
#endif
#endif
