/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_post_ops.h"

BLIS_INLINE void lpgemm_set_node_params
     (
       lpgemm_post_op* post_op_node,
       LPGEMM_POST_OP_CODE op_code,
       void* op1,
       void* op2,
       void* op3,
       void* scale_factor,
       dim_t scale_factor_len,
       bool is_power_of_2
     )
{
	post_op_node->op_code = op_code;
	post_op_node->op_args1 = op1;
	post_op_node->op_args2 = op2;
	post_op_node->op_args3 = op3;
	post_op_node->scale_factor = scale_factor;
	post_op_node->scale_factor_len = scale_factor_len;
	post_op_node->is_power_of_2 = is_power_of_2;
	post_op_node->next = NULL;
}

err_t lpgemm_translate_to_post_ops_list
     (
       aocl_post_op*   post_op_unparsed,
       lpgemm_post_op* post_op_list,
       void*           scale_buffer,
       void*           meta_arg
     )
{
	( void )( scale_buffer ); //Unused for now, potential to be used later.
	if ( ( post_op_unparsed == NULL ) || ( post_op_unparsed->seq_length <= 0 ) )
	{
		lpgemm_set_node_params
		(
		  post_op_list, POST_OPS_DISABLE,
		  NULL, NULL, NULL, NULL, 0, FALSE
		);

		return BLIS_SUCCESS;
	}

	if ( ( post_op_unparsed->seq_length > AOCL_MAX_POST_OPS ) )
	{
		lpgemm_set_node_params
		(
		  post_op_list, POST_OPS_DISABLE,
		  NULL, NULL, NULL, NULL, 0, FALSE
		);

		bli_print_msg(" Max supported post-ops is 5, supplied input post-ops" \
						" are more. Exiting..", __FILE__, __LINE__ );
		return BLIS_UNEXPECTED_VECTOR_DIM; //Error, seq length exceeds max post ops permitted.
	}

	dim_t e_i = 0; //Multiple eltwise supported.
	for ( dim_t i = 0; i < post_op_unparsed->seq_length; ++i )
	{
		// Dispatcher code
		switch ( *( post_op_unparsed->seq_vector + i ) )
		{
			case SUM:
					lpgemm_set_node_params
					(
					  ( post_op_list + i ), POST_OPS_SUM,
					  post_op_unparsed->sum.buff,
					  post_op_unparsed->sum.zero_point,
					  NULL,
					  post_op_unparsed->sum.scale_factor,
					  post_op_unparsed->sum.scale_factor_len,
					  post_op_unparsed->sum.is_power_of_2
					);
					break;
			case ELTWISE:
					{
						LPGEMM_POST_OP_CODE tmp_code = POST_OPS_DISABLE;
						// Eltwise algo dispatcher.
						switch ( ( post_op_unparsed->eltwise + e_i )->algo.algo_type )
						{
							case RELU:
									tmp_code = POST_OPS_RELU;
									break;
							case PRELU:
									if( ( post_op_unparsed->eltwise + e_i )->algo.alpha == NULL )
									{
										bli_print_msg(" Post_op.alpha is NULL. Exiting..", __FILE__, __LINE__ );
										return BLIS_NULL_POINTER;
									}
									tmp_code = POST_OPS_RELU_SCALE;
									break;
							case GELU_TANH:
									tmp_code = POST_OPS_GELU_TANH;
									break;
							case GELU_ERF:
									tmp_code = POST_OPS_GELU_ERF;
									break;
							case CLIP:
									if( ( ( post_op_unparsed->eltwise + e_i )->algo.alpha == NULL ) ||
									    ( ( post_op_unparsed->eltwise + e_i )->algo.beta  == NULL ) )
									{
										bli_print_msg(" Post_op.clip min or max value is NULL. Exiting..", __FILE__, __LINE__ );
										return BLIS_NULL_POINTER;
									}
									tmp_code = POST_OPS_CLIP;
									break;
							default:
									break;
						}
						lpgemm_set_node_params
						(
						  ( post_op_list + i ), tmp_code,
						  NULL,
						  ( post_op_unparsed->eltwise + e_i )->algo.alpha,
						  ( post_op_unparsed->eltwise + e_i )->algo.beta,
						  ( post_op_unparsed->eltwise + e_i )->scale_factor,
						  ( post_op_unparsed->eltwise + e_i )->scale_factor_len,
						  ( post_op_unparsed->eltwise + e_i )->is_power_of_2
						);
						e_i += 1;
					}
					break;
			case BIAS:
					if( post_op_unparsed->bias.bias == NULL )
					{
						bli_print_msg(" Post_op.bias is NULL. Exiting..", __FILE__, __LINE__ );
						return BLIS_NULL_POINTER;
					}
					lpgemm_set_node_params
					(
					  ( post_op_list + i ), POST_OPS_BIAS,
					  post_op_unparsed->bias.bias,
					  meta_arg, NULL, NULL, 0, FALSE
					);
					break;
			case SCALE:
					if ( ( post_op_unparsed->sum.scale_factor_len > 0 ) &&
						 ( post_op_unparsed->sum.scale_factor == NULL ) )
					{
						bli_print_msg(" Post_op.scale scale_factor is NULL. Exiting..",
										__FILE__, __LINE__ );
						return BLIS_NULL_POINTER;
					}
					if ( ( post_op_unparsed->sum.zero_point_len > 0 ) &&
						  ( post_op_unparsed->sum.zero_point == NULL ) )
					{
						bli_print_msg(" Post_op.scale zero_point is NULL. Exiting..",
										__FILE__, __LINE__ );
						return BLIS_NULL_POINTER;
					}

					lpgemm_set_node_params
					(
					  ( post_op_list + i ), POST_OPS_DOWNSCALE,
					  post_op_unparsed->sum.zero_point,
					  meta_arg, &( post_op_unparsed->sum.zero_point_len ),
					  post_op_unparsed->sum.scale_factor,
					  post_op_unparsed->sum.scale_factor_len,
					  FALSE
					);
					break;
			case MATRIX_ADD:
					if ( ( post_op_unparsed->matrix_add.matrix == NULL ) ||
						 ( post_op_unparsed->matrix_add.ldm <= 0 ) )
					{
						bli_print_msg(" Post_op.matrix_add attributes are invalid. Exiting..",
										__FILE__, __LINE__ );
						return BLIS_NULL_POINTER;
					}

					lpgemm_set_node_params
					(
					  ( post_op_list + i ), POST_OPS_MATRIX_ADD,
					  post_op_unparsed->matrix_add.matrix,
					  meta_arg, &( post_op_unparsed->matrix_add.ldm ),
					  NULL, 0, FALSE
					);
					break;
			default:
					break;
		}

		// Simulating linked link using an array.
		if ( i < ( post_op_unparsed->seq_length - 1 ) )
		{
			( post_op_list + i )->next = ( post_op_list + i + 1);
		}
	}
	return BLIS_SUCCESS;
}
