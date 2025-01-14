/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "bench_lpgemm_helpers.h"

CONVERT_TO_FLOAT(uint8_t)
CONVERT_TO_FLOAT(int8_t)
CONVERT_TO_FLOAT(int16_t)
CONVERT_TO_FLOAT(float)
CONVERT_TO_FLOAT(int32_t)

PRINT_MATRIX(uint8_t)
PRINT_MATRIX(int8_t)
PRINT_MATRIX(int16_t)
PRINT_MATRIX(float)
PRINT_MATRIX(int32_t)

GEN_FILL_ARRAY_FUNC(int8_t)
GEN_FILL_ARRAY_FUNC(int16_t)
GEN_FILL_ARRAY_FUNC(float)
GEN_FILL_ARRAY_FUNC(int32_t)

void fill_array_uint8_t ( void* arr, dim_t size )
{
    if( size < 0 ) return;
    uint8_t* temp_arr = ( uint8_t* ) arr;
    for ( dim_t i = 0; i < size; ++i )
    {
        temp_arr[i] = ( uint8_t )( rand() % 5 );
    }
}

void fill_array_int4_c_t( void* arr, dim_t size )
{
    int8_t int4_c_t_values[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };
    //int8_t int4_c_t_values[8] = { 0x01, 0x23, 0x45, 0x67, 0x01, 0x23, 0x45, 0x67 };
    dim_t int4_c_t_size = ( size + 1 ) / 2;
    if ( size < 0 ) return;
    // Fill in pairs for in4_t since 4 bits/half byte access is not
    // straight forward.
    int8_t* temp_arr = ( int8_t* )arr;
    for (dim_t i = 0; i < int4_c_t_size; ++i)
    {
        temp_arr[i] = int4_c_t_values[( rand() % 8 )];
    }
}

#define GEN_BLIS_MAT_MUL_FUNC(A_type,B_type,C_type,ACCUM_type,BLAS_SFX) \
void mat_mul_ ## BLAS_SFX \
     ( \
       char    stor_order, \
       char    transa, \
       char    transb,  \
       char    op_a, \
       char    op_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ACCUM_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       ACCUM_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       aocl_post_op*  post_op\
     ) \
{ \
    aocl_gemm_ ## BLAS_SFX( stor_order, transa, transb, m, n, k, \
                    alpha, \
                    a, lda, op_a, \
                    b, ldb, op_b, \
                    beta, \
                    c, ldc, post_op ); \
} \

GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16)
GEN_BLIS_MAT_MUL_FUNC(float,float,float,float,f32f32f32of32)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int16_t,int16_t,s8s8s16os16)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int8_t,int16_t,s8s8s16os8)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,int8_t,float,float,bf16s4f32of32)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,int8_t,bfloat16,float,bf16s4f32obf16)

double get_gflops
     (
       dim_t  m,
       dim_t  n,
       dim_t  k,
       double runtime
     )
{
    return ( ( 2.0 * m * n * k ) / ( runtime * 1.0e9 ) );
}

void print_result
     (
       const char* msg,
       int32_t     n_repeats,
       char        transa,
       char        transb,
       dim_t       m,
       dim_t       n,
       dim_t       k,
       dim_t       lda,
       dim_t       ldb,
       dim_t       ldc,
       double      gflops
     )
{
    //double gflops = get_gflops( m, n, k, runtime );
    printf("%s transa:%c, transb:%c, m: %ld, n: %ld, k: %ld, lda: %ld, ldb: %ld, ldc: %ld," \
                    " Gops: %f, n_repeats: %d\n",
            msg, transa, transb, m, n, k, lda, ldb, ldc, gflops, n_repeats);
}

#define GEN_MAT_MUL_BENCH_DRV_FUNC(A_type,B_type,C_type,ACCUM_type,BLAS_SFX) \
void mat_mul_bench_driver_ ## BLAS_SFX \
     ( \
       char    stor_order, \
       char    transa, \
       char    transb, \
       char    op_a, \
       char    op_b, \
       int32_t n_repeats, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ACCUM_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       ACCUM_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       aocl_post_op*  post_op\
     ) \
{ \
    double   dtime;                 \
    double   dtime_save = DBL_MAX;  \
\
    for ( int32_t nr = 0; nr < n_repeats; ++nr ) \
    { \
        dtime = bli_clock();            \
 \
        GEN_FUNC_NAME(mat_mul_,BLAS_SFX) \
        ( \
          stor_order, transa, transb, op_a, op_b, m, n, k, \
          alpha, \
          a, lda, \
          b, ldb, \
          beta, \
          c, ldc, \
          post_op \
        ); \
 \
        dtime_save = bli_clock_min_diff( dtime_save, dtime ); \
 \
    } \
    double gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 ); \
 \
    print_result( XSTR(BLAS_SFX), n_repeats, transa, transb, m, n, k, lda, ldb, ldc, gflops); \
} \

GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16)
GEN_MAT_MUL_BENCH_DRV_FUNC(float,float,float,float,f32f32f32of32)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int16_t,int16_t,s8s8s16os16)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int8_t,int16_t,s8s8s16os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,int8_t,float,float,bf16s4f32of32)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,int8_t,bfloat16,float,bf16s4f32obf16)

#define GEN_MAT_MUL_ACC_CHK_DOWNSCALE(C_type,ACCUM_type,SCALE_type,BLAS_DOWNSCALE_SFX) \
static inline ACCUM_type mat_mul_accuracy_check_downscale_ ## BLAS_DOWNSCALE_SFX \
     (\
       ACCUM_type temp_accum,\
       aocl_post_op*  post_op, \
       dim_t j \
     )\
{ \
    dim_t j_scale = j; \
    if ( ( post_op->sum )->scale_factor_len == 1 ) \
    { \
       j_scale = 0; \
    } \
 \
    dim_t j_zp = j; \
    if ( ( post_op->sum )->zero_point_len == 1 ) \
    { \
       j_zp = 0; \
    } \
 \
    ACCUM_type out_temp_accum = \
        ( ACCUM_type )min( \
                        max( nearbyintf( ( SCALE_type )( temp_accum ) * \
                            ( *( ( SCALE_type* )( post_op->sum )->scale_factor + j_scale ) ) ) + \
                            *( ( C_type* )( post_op->sum )->zero_point + j_zp ), \
                            DSCALE_CLIP_MIN ), \
                        DSCALE_CLIP_MAX ); \
    return out_temp_accum; \
}\

GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int16_t,float,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(uint8_t,int16_t,float,u8s8s16ou8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int32_t,float,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int32_t,float,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int16_t,float,s8s8s16os8)

static inline float mat_mul_accuracy_check_downscale_bf16bf16f32obf16
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }

    float zp_float = 0.0;
    bfloat16_to_float( *( ( bfloat16* )( post_op->sum )->zero_point + j_zp ),
                       &zp_float );
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                zp_float );
    return out_temp_accum;
}
static inline float mat_mul_accuracy_check_downscale_f32f32f32of32
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                 *( ( float* )( post_op->sum )->zero_point + j_zp ) );
    return out_temp_accum;
}
#define GEN_MAT_MUL_ACC_CHK_ACCUM(A_type, B_type, C_type,ACCUM_type,BLAS_SFX) \
static inline ACCUM_type mat_mul_accuracy_check_accum_ ## BLAS_SFX \
     (\
       A_type* a, \
       B_type* b, \
       C_type* c_ref, \
       ACCUM_type temp_accum,\
       ACCUM_type  alpha, \
       ACCUM_type beta, \
       dim_t rs_a, \
       dim_t rs_b, \
       dim_t cs_a, \
       dim_t cs_b, \
       dim_t rs_c_ref, \
       dim_t cs_c_ref, \
       dim_t i, \
       dim_t j, \
       dim_t k, \
       dim_t pre_op_ld, /* Ignored */ \
       bool int4_testing, /* Workaround to enable int4 B matrix testing. */\
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     ) \
{ \
    ( void )int4_testing; \
    ( void ) pre_op; \
    for ( dim_t p = 0; p < k; ++p) \
    { \
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) * \
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) ); \
    } \
 \
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) ) \
                         + ( alpha * temp_accum ); \
    return temp_accum; \
} \

GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_ACC_CHK_ACCUM(float,float,float,float,f32f32f32of32)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int8_t,int16_t,s8s8s16os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int16_t,int16_t,s8s8s16os16)

#define GEN_MAT_MUL_ACC_CHK_ACCUM_INT4(A_type, B_type, C_type,ACCUM_type,BLAS_SFX) \
static inline ACCUM_type mat_mul_accuracy_check_accum_ ## BLAS_SFX \
     (\
       A_type* a, \
       B_type* b, \
       C_type* c_ref, \
       ACCUM_type temp_accum,\
       ACCUM_type  alpha, \
       ACCUM_type beta, \
       dim_t rs_a, \
       dim_t rs_b, \
       dim_t cs_a, \
       dim_t cs_b, \
       dim_t rs_c_ref, \
       dim_t cs_c_ref, \
       dim_t i, \
       dim_t j, \
       dim_t k, \
       dim_t pre_op_ld, /* Ignored */ \
       bool int4_testing, /* Workaround to enable int4 B matrix testing. */\
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     ) \
{ \
    ( void ) pre_op; \
    if ( int4_testing == FALSE ) \
    { \
        for ( dim_t p = 0; p < k; ++p) \
        { \
            temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) * \
                             *( b + ( rs_b * p ) + ( cs_b * j ) ) ); \
        } \
    } \
    else \
    { \
        for ( dim_t p = 0; p < k; ++p) \
        { \
            /* Get B matrix int4_t value and upscale it to int8_t. */ \
            dim_t b_inc = ( rs_b * p ) + ( cs_b * j ); \
            int8_t b_val = 0; \
            /* Even index will have data at low 4 bits, and odd at hi 4 bits.
             * B matrix increments has to be halved to account for 4 bit
             * traversal. */ \
            if ( ( b_inc % 2 ) != 0 ) \
            { \
                b_val = ( ( *( b + ( b_inc / 2 ) ) ) >> 4 ) & 0x0F; \
            } \
            else \
            { \
                b_val = ( *( b + ( b_inc / 2 ) ) ) & 0x0F; \
            } \
            /* Signed scale. */ \
            if ( b_val & 0x08 ) \
            { \
                 b_val = b_val | 0xF0; \
            } \
            temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) * b_val ); \
        } \
    } \
 \
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) ) \
                         + ( alpha * temp_accum ); \
    return temp_accum; \
} \

GEN_MAT_MUL_ACC_CHK_ACCUM_INT4(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_ACCUM_INT4(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)

static inline float mat_mul_accuracy_check_accum_bf16bf16f32of32
     (
       bfloat16* a,
       bfloat16* b,
       float* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */ \
       bool int4_testing, /* Ignored for bf16 testing */\
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    ( void )int4_testing;
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i * rs_a + p * cs_a ) , &a_float);
        bfloat16_to_float( *( b + p * rs_b + j * cs_b ) , &b_float);
        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) )
                         + ( alpha * temp_accum );
    return temp_accum;
}

static inline float mat_mul_accuracy_check_accum_bf16bf16f32obf16
     (
       bfloat16* a,
       bfloat16* b,
       bfloat16* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */
       bool int4_testing, /* Ignored for bf16 testing */\
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    ( void )int4_testing;
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i*rs_a + p*cs_a ), &a_float );
        bfloat16_to_float( *( b + p*rs_b + j*cs_b ), &b_float );
        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    float c_ref_float;
    bfloat16_to_float( *( c_ref + i*rs_c_ref + j*cs_c_ref ), &c_ref_float );
    temp_accum = ( beta * ( c_ref_float ) ) + ( alpha * temp_accum );

    return temp_accum;
}

static inline float get_s4_to_f32_scale_val
     (
       int8_t* b,
       dim_t p,
       dim_t j,
       dim_t n,
       dim_t b_inc,
       aocl_pre_op* pre_op
     )
{
    float b_float = 0.0;
    int8_t b_val = 0;

    dim_t group_size = pre_op->group_size;

    /* Even index will have data at low 4 bits, and odd at hi 4 bits.
     * B matrix increments has to be halved to account for 4 bit
     * traversal. */
    if ( ( b_inc % 2 ) != 0 )
    {
        b_val = ( ( *( b + ( b_inc / 2 ) ) ) >> 4 ) & 0x0F;
    }
    else
    {
        b_val = ( *( b + ( b_inc / 2 ) ) ) & 0x0F;
    }

    /* Signed scale. */
    if ( b_val & 0x08 )
    {
         b_val = b_val | 0xF0;
    }

    if ( ( pre_op != NULL ) && ( pre_op->seq_length > 0 ) )
    {
        dim_t j_zp=0, j_scale=0;
        if(group_size!=0)
        {
            j_zp = ( ( p / group_size ) * n ) + j;
            if ( ( pre_op->b_zp != NULL ) &&
                ( ( pre_op->b_zp )->zero_point_len == 1 ) )
            {
                j_zp = p / group_size;
            }

            j_scale = ( ( p / group_size ) * n ) + j;
            if ( ( pre_op->b_scl != NULL ) &&
                ( ( pre_op->b_scl )->scale_factor_len == 1 ) )
            {
                j_scale = (p / group_size);
            }
        }

        // Assuming only 1 scale and zp.
        int8_t zp = 0;
        if ( ( pre_op->b_zp != NULL ) &&
             ( ( pre_op->b_zp )->zero_point != NULL ) )
        {
            zp = *( ( int8_t* )( pre_op->b_zp )->zero_point + j_zp );
        }

        float scale_factor = 1.0;
        if ( ( pre_op->b_scl != NULL ) &&
             ( ( pre_op->b_scl )->scale_factor != NULL ) )
        {
            if( pre_op->b_scl->scale_factor_type == AOCL_GEMM_F32 )
            {
                scale_factor = *( ( float* )( pre_op->b_scl )->scale_factor + j_scale );
            }
            else
            {
                bfloat16_to_float( *( ( bfloat16* )( pre_op->b_scl )->scale_factor + j_scale ) , &scale_factor);
            }

        }
        b_float = (float)( b_val - zp ) * scale_factor;
    }
    else
    {
        b_float = (float)( b_val);
    }

    return b_float;
}

static inline float mat_mul_accuracy_check_accum_bf16s4f32of32
     (
       bfloat16* a,
       int8_t* b,
       float* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld,
       bool int4_testing, /* Ignored s4 implies int4 testing. */\
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    ( void )int4_testing;
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i * rs_a + p * cs_a ) , &a_float);

        /* Get B matrix int4_t value and upscale it to float. */
        dim_t b_inc = ( rs_b * p ) + ( cs_b * j );
        b_float = get_s4_to_f32_scale_val( b, p, j, pre_op_ld, b_inc, pre_op );

        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) )
                         + ( alpha * temp_accum );
    return temp_accum;
}

static inline float mat_mul_accuracy_check_accum_bf16s4f32obf16
     (
       bfloat16* a,
       int8_t* b,
       bfloat16* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld,
       bool int4_testing, /* Ignored for bf16 testing */\
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    ( void )int4_testing;
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i*rs_a + p*cs_a ), &a_float );

        /* Get B matrix int4_t value and upscale it to float. */
        dim_t b_inc = ( rs_b * p ) + ( cs_b * j );
        b_float = get_s4_to_f32_scale_val( b, p, j, pre_op_ld, b_inc, pre_op );

        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    float c_ref_float;
    bfloat16_to_float( *( c_ref + i*rs_c_ref + j*cs_c_ref ), &c_ref_float );
    temp_accum = ( beta * ( c_ref_float ) ) + ( alpha * temp_accum );

    return temp_accum;
}

GEN_GELU_TANH_POSTOP_INT(int16_t,u8s8s16os8)
GEN_GELU_TANH_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_GELU_TANH_POSTOP_INT(int16_t,u8s8s16os16)
GEN_GELU_TANH_POSTOP_INT(int32_t,u8s8s32os8)
GEN_GELU_TANH_POSTOP_INT(int32_t,u8s8s32os32)
GEN_GELU_TANH_POSTOP_INT(int32_t,s8s8s32os8)
GEN_GELU_TANH_POSTOP_INT(int32_t,s8s8s32os32)
GEN_GELU_TANH_POSTOP_INT(int16_t,s8s8s16os8)
GEN_GELU_TANH_POSTOP_INT(int16_t,s8s8s16os16)

GEN_GELU_TANH_POSTOP_FLOAT(f32f32f32of32)
GEN_GELU_TANH_POSTOP_FLOAT(bf16bf16f32of32)
GEN_GELU_TANH_POSTOP_FLOAT(bf16bf16f32obf16)
GEN_GELU_TANH_POSTOP_FLOAT(bf16s4f32of32)
GEN_GELU_TANH_POSTOP_FLOAT(bf16s4f32obf16)

GEN_TANH_POSTOP_INT(int16_t,u8s8s16os8)
GEN_TANH_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_TANH_POSTOP_INT(int16_t,u8s8s16os16)
GEN_TANH_POSTOP_INT(int32_t,u8s8s32os8)
GEN_TANH_POSTOP_INT(int32_t,u8s8s32os32)
GEN_TANH_POSTOP_INT(int32_t,s8s8s32os8)
GEN_TANH_POSTOP_INT(int32_t,s8s8s32os32)
GEN_TANH_POSTOP_INT(int16_t,s8s8s16os8)
GEN_TANH_POSTOP_INT(int16_t,s8s8s16os16)

GEN_TANH_POSTOP_FLOAT(f32f32f32of32)
GEN_TANH_POSTOP_FLOAT(bf16bf16f32of32)
GEN_TANH_POSTOP_FLOAT(bf16bf16f32obf16)
GEN_TANH_POSTOP_FLOAT(bf16s4f32of32)
GEN_TANH_POSTOP_FLOAT(bf16s4f32obf16)

GEN_GELU_ERF_POSTOP_INT(int16_t,u8s8s16os8)
GEN_GELU_ERF_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_GELU_ERF_POSTOP_INT(int16_t,u8s8s16os16)
GEN_GELU_ERF_POSTOP_INT(int32_t,u8s8s32os8)
GEN_GELU_ERF_POSTOP_INT(int32_t,u8s8s32os32)
GEN_GELU_ERF_POSTOP_INT(int32_t,s8s8s32os8)
GEN_GELU_ERF_POSTOP_INT(int32_t,s8s8s32os32)
GEN_GELU_ERF_POSTOP_INT(int16_t,s8s8s16os8)
GEN_GELU_ERF_POSTOP_INT(int16_t,s8s8s16os16)

GEN_GELU_ERF_POSTOP_FLOAT(f32f32f32of32)
GEN_GELU_ERF_POSTOP_FLOAT(bf16bf16f32of32)
GEN_GELU_ERF_POSTOP_FLOAT(bf16bf16f32obf16)
GEN_GELU_ERF_POSTOP_FLOAT(bf16s4f32of32)
GEN_GELU_ERF_POSTOP_FLOAT(bf16s4f32obf16)

GEN_SWISH_POSTOP_INT(int16_t,u8s8s16os8)
GEN_SWISH_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_SWISH_POSTOP_INT(int16_t,u8s8s16os16)
GEN_SWISH_POSTOP_INT(int32_t,u8s8s32os8)
GEN_SWISH_POSTOP_INT(int32_t,u8s8s32os32)
GEN_SWISH_POSTOP_INT(int32_t,s8s8s32os8)
GEN_SWISH_POSTOP_INT(int32_t,s8s8s32os32)
GEN_SWISH_POSTOP_INT(int16_t,s8s8s16os8)
GEN_SWISH_POSTOP_INT(int16_t,s8s8s16os16)

GEN_SWISH_POSTOP_FLOAT(f32f32f32of32)
GEN_SWISH_POSTOP_FLOAT(bf16bf16f32of32)
GEN_SWISH_POSTOP_FLOAT(bf16bf16f32obf16)
GEN_SWISH_POSTOP_FLOAT(bf16s4f32of32)
GEN_SWISH_POSTOP_FLOAT(bf16s4f32obf16)

GEN_SIGMOID_POSTOP_INT(int16_t,u8s8s16os8)
GEN_SIGMOID_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_SIGMOID_POSTOP_INT(int16_t,u8s8s16os16)
GEN_SIGMOID_POSTOP_INT(int32_t,u8s8s32os8)
GEN_SIGMOID_POSTOP_INT(int32_t,u8s8s32os32)
GEN_SIGMOID_POSTOP_INT(int32_t,s8s8s32os8)
GEN_SIGMOID_POSTOP_INT(int32_t,s8s8s32os32)
GEN_SIGMOID_POSTOP_INT(int16_t,s8s8s16os8)
GEN_SIGMOID_POSTOP_INT(int16_t,s8s8s16os16)

GEN_SIGMOID_POSTOP_FLOAT(f32f32f32of32)
GEN_SIGMOID_POSTOP_FLOAT(bf16bf16f32of32)
GEN_SIGMOID_POSTOP_FLOAT(bf16bf16f32obf16)
GEN_SIGMOID_POSTOP_FLOAT(bf16s4f32of32)
GEN_SIGMOID_POSTOP_FLOAT(bf16s4f32obf16)

GEN_GET_MATRIX_ADD_POST_OP_VAL_BF16(bfloat16,bf16bf16f32obf16)
GEN_GET_MATRIX_ADD_POST_OP_VAL_BF16(bfloat16,bf16s4f32obf16)

GEN_GET_MATRIX_ADD_POST_OP_VAL(int8_t,int32_t,u8s8s32os8)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int32_t,int32_t,u8s8s32os32)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int8_t,int16_t,u8s8s16os8)
GEN_GET_MATRIX_ADD_POST_OP_VAL(uint8_t,int16_t,u8s8s16ou8)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int16_t,int16_t,u8s8s16os16)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int8_t,int32_t,s8s8s32os8)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int32_t,int32_t,s8s8s32os32)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int8_t,int16_t,s8s8s16os8)
GEN_GET_MATRIX_ADD_POST_OP_VAL(int16_t,int16_t,s8s8s16os16)
GEN_GET_MATRIX_ADD_POST_OP_VAL(float,float,f32f32f32of32)
GEN_GET_MATRIX_ADD_POST_OP_VAL(float,float,bf16bf16f32of32)
GEN_GET_MATRIX_ADD_POST_OP_VAL(float,float,bf16s4f32of32)

GEN_GET_MATRIX_MUL_POST_OP_VAL_BF16(bfloat16,bf16bf16f32obf16)
GEN_GET_MATRIX_MUL_POST_OP_VAL_BF16(bfloat16,bf16s4f32obf16)

GEN_GET_MATRIX_MUL_POST_OP_VAL(int8_t,int32_t,u8s8s32os8)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int32_t,int32_t,u8s8s32os32)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int8_t,int16_t,u8s8s16os8)
GEN_GET_MATRIX_MUL_POST_OP_VAL(uint8_t,int16_t,u8s8s16ou8)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int16_t,int16_t,u8s8s16os16)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int8_t,int32_t,s8s8s32os8)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int32_t,int32_t,s8s8s32os32)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int8_t,int16_t,s8s8s16os8)
GEN_GET_MATRIX_MUL_POST_OP_VAL(int16_t,int16_t,s8s8s16os16)
GEN_GET_MATRIX_MUL_POST_OP_VAL(float,float,f32f32f32of32)
GEN_GET_MATRIX_MUL_POST_OP_VAL(float,float,bf16bf16f32of32)
GEN_GET_MATRIX_MUL_POST_OP_VAL(float,float,bf16s4f32of32)

GEN_GET_BIAS_POST_OP_VAL_BF16(bf16bf16f32obf16)
GEN_GET_BIAS_POST_OP_VAL_BF16(bf16s4f32obf16)

GEN_GET_BIAS_POST_OP_VAL(int32_t,u8s8s32os8)
GEN_GET_BIAS_POST_OP_VAL(int32_t,u8s8s32os32)
GEN_GET_BIAS_POST_OP_VAL(int16_t,u8s8s16os8)
GEN_GET_BIAS_POST_OP_VAL(int16_t,u8s8s16ou8)
GEN_GET_BIAS_POST_OP_VAL(int16_t,u8s8s16os16)
GEN_GET_BIAS_POST_OP_VAL(int32_t,s8s8s32os8)
GEN_GET_BIAS_POST_OP_VAL(int32_t,s8s8s32os32)
GEN_GET_BIAS_POST_OP_VAL(int16_t,s8s8s16os8)
GEN_GET_BIAS_POST_OP_VAL(int16_t,s8s8s16os16)
GEN_GET_BIAS_POST_OP_VAL_f32(f32f32f32of32)
GEN_GET_BIAS_POST_OP_VAL_f32(bf16bf16f32of32)
GEN_GET_BIAS_POST_OP_VAL_f32(bf16s4f32of32)

GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int32_t,int32_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int8_t,int32_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int16_t,int16_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int8_t,int16_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(uint8_t,int16_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(float,float)

#define GEN_MAT_MUL_ACC_CHK_DRV_FUNC(A_type,B_type,C_type,ACCUM_type,SCALE_type,BLAS_SFX,BLAS_DOWNSCALE_SFX) \
void mat_mul_accuracy_check_driver_ ## BLAS_SFX \
     ( \
       FILE*   fout, \
       const char stor_order, \
       char    transa, \
       char    transb, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ACCUM_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       ACCUM_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       C_type* c_ref, \
       dim_t   ldc_ref, \
       aocl_post_op*  post_op, \
       bool    int4_testing /* Workaround to enable int4 B matrix testing. */ \
     ) \
{ \
    dim_t rs_a, cs_a; \
    if( ( transa == 'n' ) || ( transa == 'N' ) ) \
    { \
        rs_a = lda; \
        cs_a = 1; \
    } \
    else \
    { \
        rs_a = 1; \
        cs_a = lda; \
    } \
    dim_t rs_b, cs_b; \
    if( ( transb == 'n' ) || ( transb == 'N' ) ) \
    { \
        rs_b = ldb; \
        cs_b = 1; \
    } \
    else \
    { \
        rs_b = 1; \
        cs_b = ldb; \
    } \
    dim_t rs_c = ldc; \
    dim_t cs_c = 1; \
    dim_t rs_c_ref = ldc_ref; \
    dim_t cs_c_ref = 1; \
 \
    if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
    { \
        if( transa == 'n' || transa == 'N') \
        { \
            rs_a = 1; \
            cs_a = lda; \
        } \
        else \
        { \
            rs_a = lda; \
            cs_a = 1; \
        } \
        if( ( transb == 'n' ) || ( transb == 'N' ) ) \
        { \
            rs_b = 1; \
            cs_b = ldb; \
        } \
        else \
        { \
            rs_b = ldb; \
            cs_b = 1; \
        } \
        rs_c = 1; \
        cs_c = ldc; \
        rs_c_ref = 1; \
        cs_c_ref = ldc_ref; \
    } \
 \
    aocl_pre_op* a_pre_op = NULL; \
    if ( post_op != NULL ) \
    { \
        a_pre_op = post_op->pre_ops; \
    } \
    for ( dim_t i = 0; i < m; ++i ) \
    { \
        for ( dim_t j = 0; j < n; ++j ) \
        { \
            ACCUM_type temp_accum = 0; \
            C_type out_temp_accum = 0; \
 \
            temp_accum = GEN_FUNC_NAME(mat_mul_accuracy_check_accum_,BLAS_SFX) \
                (a, b, c_ref, temp_accum, alpha, beta,\
                 rs_a, rs_b, cs_a, cs_b, rs_c_ref, cs_c_ref, i, j, k, n, \
                 int4_testing, a_pre_op); \
\
            if ( post_op != NULL ) \
            { \
                dim_t ele_i = 0; \
                for ( dim_t op_id = 0; op_id < post_op->seq_length; ++op_id ) \
                { \
                    if ( post_op->seq_vector[op_id] == BIAS ) \
                    { \
                        temp_accum += GEN_FUNC_NAME(get_bias_post_op_val_,BLAS_SFX) \
                            ( ( post_op->bias )->bias, j, ( post_op->bias )->stor_type ); \
                    } \
                    else if ( post_op->seq_vector[op_id] == ELTWISE ) \
                    { \
                        if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                PRELU ) /* PReLU*/ \
                        { \
                            temp_accum = ( temp_accum > 0 ) ? \
                                temp_accum : \
                                ( temp_accum * \
                                *( ( ACCUM_type* ) ( post_op->eltwise + ele_i )->algo.alpha ) ); \
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                GELU_TANH ) /* TANH GeLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(GELU_TANH_post_op_,BLAS_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                GELU_ERF ) /* ERF GeLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(GELU_ERF_post_op_,BLAS_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                SWISH ) /* SiLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(SWISH_post_op_,BLAS_SFX) \
                                (temp_accum, \
                                 *( ( ACCUM_type* ) \
                                    ( post_op->eltwise + ele_i )->algo.alpha ) );\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                RELU ) /* ReLU*/ \
                        { \
                            temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ; \
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                TANH ) /* TANH*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(TANH_post_op_,BLAS_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                SIGMOID ) /* Sigmoid*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(SIGMOID_post_op_,BLAS_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                CLIP ) /* CLIP*/ \
                        { \
                            temp_accum = \
                                min \
                                ( \
                                  max \
                                  ( \
                                    temp_accum, \
                                    *( ( ACCUM_type* ) \
                                       ( post_op->eltwise + ele_i )->algo.alpha ) \
                                  ), \
                                  *( ( ACCUM_type* ) \
                                     ( post_op->eltwise + ele_i )->algo.beta) \
                                ); \
                            ele_i += 1; \
                        } \
                        else \
                        {} \
                    } \
                    else if ( post_op->seq_vector[op_id] == SCALE ) \
                    { \
                        temp_accum = GEN_FUNC_NAME(mat_mul_accuracy_check_downscale_,BLAS_DOWNSCALE_SFX) \
                            (temp_accum, post_op, j); \
                    } \
                    else if ( post_op->seq_vector[op_id] == MATRIX_ADD ) \
                    { \
                        dim_t rs_m = ( post_op->matrix_add )->ldm; \
                        dim_t cs_m = 1; \
                        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
                        { \
                            cs_m = rs_m; \
                            rs_m = 1; \
                        } \
                        float* scl_fctr = ( float* )( ( post_op->matrix_add )->scale_factor ); \
                        dim_t scl_fctr_len = ( post_op->matrix_add )->scale_factor_len; \
                        temp_accum += GEN_FUNC_NAME(get_matrix_add_post_op_val_,BLAS_SFX) \
                                    ( *( ( C_type* )( post_op->matrix_add )->matrix + \
                                         ( i * rs_m ) + ( j * cs_m ) ), \
                                      j, scl_fctr, scl_fctr_len ); \
                    } \
                    else if ( post_op->seq_vector[op_id] == MATRIX_MUL ) \
                    { \
                        dim_t rs_m = ( post_op->matrix_mul )->ldm; \
                        dim_t cs_m = 1; \
                        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
                        { \
                            cs_m = rs_m; \
                            rs_m = 1; \
                        } \
                        float* scl_fctr = ( float* )( ( post_op->matrix_mul )->scale_factor ); \
                        dim_t scl_fctr_len = ( post_op->matrix_mul )->scale_factor_len; \
                        temp_accum *= GEN_FUNC_NAME(get_matrix_mul_post_op_val_,BLAS_SFX) \
                                    ( *( ( C_type* )( post_op->matrix_mul )->matrix + \
                                           ( i * rs_m ) + ( j * cs_m ) ), \
                                      j, scl_fctr, scl_fctr_len ); \
                    } \
                    else \
                    {} \
                } \
            } \
            /* Need to convert to downscaled type if required.*/ \
            mat_mul_get_output_type_val ## ACCUM_type ## C_type \
            ( \
              &out_temp_accum, &temp_accum \
            ); \
 \
            float comp_float, ref_float; \
            GEN_FUNC_NAME(C_type,_to_float)(*( c + ( rs_c * i ) + ( cs_c * j ) ), &comp_float); \
            GEN_FUNC_NAME(C_type,_to_float)(out_temp_accum, &ref_float); \
 \
            if ( ( ( comp_float - ref_float ) > 1.0E-5 ) || \
                 ( ( ref_float - comp_float ) > 1.0E-5 ) ) \
            { \
                if ( fout ) \
                { \
                    fprintf( fout, "%s Failure input m: %ld, n: %ld, k: %ld," \
                                    " lda: %ld, ldb: %ld, ldc: %ld, computed:%f, ref:%f, diff:%f\n", \
                                    XSTR(BLAS_SFX), m, n, k, lda, ldb, ldc, comp_float, \
                                    ref_float, comp_float - ref_float); \
                    fflush( fout ); \
                } \
                    printf("failure, m_index: %ld, n_index: %ld, k: %ld, computed:%f, ref:%f," \
                            "diff:%f\n", i, j, k, comp_float, ref_float, comp_float-ref_float); \
                goto cleanup_acc; \
            } \
        } \
    } \
cleanup_acc: \
    return; \
} \

GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,float,u8s8s16os16,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,float,u8s8s16os8,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,uint8_t,int16_t,float,u8s8s16ou8,u8s8s16ou8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,float,u8s8s32os32,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,float,u8s8s32os8,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,float,float,float,bf16bf16f32of32,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,bfloat16,float,float,bf16bf16f32obf16,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(float,float,float,float,float,f32f32f32of32,f32f32f32of32)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int32_t,int32_t,float,s8s8s32os32,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int8_t,int32_t,float,s8s8s32os8,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int16_t,int16_t,float,s8s8s16os16,s8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int8_t,int16_t,float,s8s8s16os8,s8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,int8_t,float,float,float,bf16s4f32of32,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,int8_t,bfloat16,float,float,bf16s4f32obf16,bf16bf16f32obf16)

GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int16_t,float,int16_t,u8s8s16os16)
GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int32_t,float,int32_t,u8s8s32os32)
GEN_MAT_MUL_POST_OPS_CREATOR(bfloat16,float,float,bfloat16,bf16bf16f32of32)
GEN_MAT_MUL_POST_OPS_CREATOR(float,float,float,float,f32f32f32of32)
GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int32_t,float,int32_t,s8s8s32os32)
GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int16_t,float,int16_t,s8s8s16os16)

// Hack to fix compiler errors.
#define GET_B_TYPE_bf16bf16f32of32 bfloat16
#define GET_B_TYPE_u8s8s16os16 int8_t
#define GET_B_TYPE_u8s8s32os32 int8_t
#define GET_B_TYPE_f32f32f32of32 float
#define GET_B_TYPE_s8s8s32os32 int8_t
#define GET_B_TYPE_s8s8s16os16 int8_t

#define GEN_MAT_MUL_BENCH_MAIN_FUNC(A_type, B_type, C_type, Sum_type, BLAS_SFX, REORDER_SFX, INT4_REORDER_SFX) \
void mat_mul_bench_main_ ## BLAS_SFX \
     ( \
       FILE*   fin, \
       FILE*   fout, \
       char    stor_order, \
       char    transa, \
       char    transb, \
       char    op_a, \
       char    op_b, \
       dim_t m, \
       dim_t n, \
       dim_t k, \
       dim_t stride_a, \
       dim_t stride_b, \
       dim_t stride_c, \
       char*   post_ops_str, \
       bool    int4_testing /* Workaround to enable int4 B matrix testing. */\
     ) \
{ \
    int32_t n_repeats = bli_max( 30, bli_min(( 3e10 / ( ( int64_t )m * n * k )), 1000 )); \
    if ( global_n_repeat > 0 ) \
    { \
        n_repeats = global_n_repeat; \
    } \
 \
    dim_t size_A = 0; \
    dim_t size_B = 0; \
    dim_t size_C = 0; \
    if( ( stor_order == 'r' ) || ( stor_order == 'R' ) ) \
    { \
        size_A = ( ( transa == 'n' ) || ( transa == 'N' ) ) ? m * stride_a : k * stride_a; \
        size_B = ( ( transb == 'n' ) || ( transb == 'N' ) ) ? k * stride_b : n * stride_b; \
        size_C = m * stride_c; \
    } \
    else \
    { \
        size_A = ( ( transa == 'n' ) || ( transa == 'N' ) ) ? k * stride_a : m * stride_a; \
        size_B = ( ( transb == 'n' ) || ( transb == 'N' ) ) ? n * stride_b : k * stride_b; \
        size_C = n * stride_c; \
    } \
    A_type* a = ( A_type* ) lpgemm_malloc( sizeof( A_type ) * size_A ); \
    GEN_FUNC_NAME(fill_array_,A_type)(a, size_A ); \
 \
    B_type* b = ( B_type* ) lpgemm_malloc( sizeof( B_type ) * size_B ); \
    if ( int4_testing == FALSE ) \
    { \
        GEN_FUNC_NAME(fill_array_,B_type)(b, size_B ); \
    } \
    else \
    { \
        GEN_FUNC_NAME(fill_array_,int4_c_t)(b, size_B); \
    } \
 \
    C_type* c = ( C_type* ) lpgemm_malloc( sizeof( C_type ) * size_C ); \
 \
    C_type* c_ref = ( C_type* ) lpgemm_malloc( sizeof( C_type ) * size_C ); \
 \
    if ( bench_mode == 'a' ) \
    { \
        GEN_FUNC_NAME(fill_array_,C_type)( c, ( size_C ) ); \
        memcpy(c_ref, c , (size_C * sizeof(C_type))); \
    } \
    else \
    { \
        memset( ( void* ) c, 0, sizeof( C_type ) * size_C ); \
        memset( ( void* ) c_ref, 0, sizeof( C_type ) * size_C ); \
    } \
 \
    Sum_type alpha = 0; \
    Sum_type beta = 0; \
    if ( bench_mode == 'p' ) \
    { \
        alpha = 2; \
        beta = 9; \
    } \
    else if ( bench_mode == 'a' ) \
    { \
        n_repeats = 1; \
        alpha = 2; \
        beta = 9; \
    } \
 \
    aocl_post_op* post_op = NULL; \
    if ( ( ( post_ops_str != NULL ) && \
           ( strcmp( post_ops_str, "none" ) != 0 ) ) || \
         ( global_dscale_out == 'y' ) || ( global_pre_op == 'y' ) ) \
    { \
        post_op = GEN_FUNC_NAME(lpgemm_create_post_ops_struct_,REORDER_SFX)( m, n, k, post_ops_str, stor_order ); \
        if ( post_op == NULL ) \
        { \
            printf(" post op struct allocation failure, returning.\n"); \
            return; \
        } \
    } \
 \
    if ( ( op_b == 'p' ) || ( op_b == 'P' ) || ( op_b == 'n' ) || ( op_b == 'N' ) )  \
    { \
        /* No reordering of B.*/ \
        GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
        ( \
          stor_order, transa, transb, op_a, op_b, n_repeats, m, n, k, \
          alpha, \
          a, stride_a, \
          b, stride_b, \
          beta, \
          c, stride_c, \
          post_op \
        ); \
    } \
    else if ( ( op_b == 'r' ) || ( op_b == 'R' ) ) \
    { \
        B_type* b_reorder = NULL; \
        /* Reorder B.*/ \
        if ( int4_testing == FALSE ) \
        { \
            siz_t b_reorder_buf_siz_req = \
                GEN_FUNC_NAME(aocl_get_reorder_buf_size_,REORDER_SFX)( stor_order, transb, 'B', k, n ); \
 \
            b_reorder = ( B_type* ) lpgemm_malloc( b_reorder_buf_siz_req ); \
            GEN_FUNC_NAME(aocl_reorder_,REORDER_SFX)( stor_order, transb, 'B', \
                    ( GET_B_TYPE_ ## REORDER_SFX * )b, \
                    ( GET_B_TYPE_ ## REORDER_SFX * )b_reorder, \
                    k, n, stride_b ); \
        } \
        /* It has to be ensured, for now, only int4 testing takes else path. */ \
        else \
        { \
            siz_t b_reorder_buf_siz_req = \
                GEN_FUNC_NAME(aocl_get_reorder_buf_size_,INT4_REORDER_SFX)( stor_order, transb, 'B', k, n ); \
 \
            b_reorder = ( B_type* ) lpgemm_malloc( b_reorder_buf_siz_req ); \
            GEN_FUNC_NAME(aocl_reorder_,INT4_REORDER_SFX)( stor_order, transb, 'B', \
                        ( int8_t* )b, ( int8_t* )b_reorder, k, n, stride_b ); \
        } \
 \
         GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
        ( \
          stor_order, transa, transb, op_a, op_b, n_repeats, m, n, k, \
          alpha, \
          a, stride_a, \
          b_reorder, stride_b, \
          beta, \
          c, stride_c, \
          post_op \
        ); \
 \
        /* Free the reorder buffer */ \
        lpgemm_free( b_reorder ); \
    } \
 \
    if ( bench_mode == 'a' ) \
    { \
        printf(" Running accuracy check.\n"); \
        GEN_FUNC_NAME(mat_mul_accuracy_check_driver_,BLAS_SFX) \
        ( \
          fout, stor_order, transa, transb, m, n, k, \
          alpha, \
          a, stride_a, \
          b, stride_b, \
          beta, \
          c, stride_c, \
          c_ref, stride_c, \
          post_op, int4_testing \
        ); \
    } \
 \
    lpgemm_destroy_post_ops_struct( post_op ); \
 \
    lpgemm_free( a ); \
    lpgemm_free( b ); \
    lpgemm_free( c ); \
    lpgemm_free( c_ref ); \
} \

GEN_MAT_MUL_BENCH_MAIN_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32,bf16bf16f32of32,bf16s4f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16,bf16bf16f32of32,bf16s4f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16,u8s8s16os16,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8,u8s8s16os16,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8,u8s8s16os16,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32,u8s8s32os32,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8,u8s8s32os32,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(float,float,float,float,f32f32f32of32,f32f32f32of32,bf16s4f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int32_t,int32_t,s8s8s32os32,s8s8s32os32,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int8_t,int32_t,s8s8s32os8,s8s8s32os32,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int16_t,int16_t,s8s8s16os16,s8s8s16os16,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int8_t,int16_t,s8s8s16os8,s8s8s16os16,u8s4s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(bfloat16,int8_t,float,float,bf16s4f32of32,bf16bf16f32of32,bf16s4f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(bfloat16,int8_t,bfloat16,float,bf16s4f32obf16,bf16bf16f32of32,bf16s4f32of32)

int main( int argc, char** argv )
{
    FILE* fin  = NULL;
    if ( argc < 5 )
    {
        printf
        (
          "Usage: ./bench_lpgemm -i input.txt -m mode < -n 100 -o op1,op2 >\n" \
          "--Mode is either a or p.\n" \
          "\ta is used for accuracy testing.\n" \
          "\tp is used for performance benchmarking.\n" \
          "--n_repeats can be set optionally using -n arg.\n" \
          "--Post ops can be executed optionaly by providing a coma separated\n" \
          "  list of post-ops after -o arg. Following post-ops are supported:\n" \
          "    1. bias\n" \
          "    2. 4 activators\n" \
          "      a. relu\n" \
          "      b. prelu\n" \
          "      c. gelu_tanh\n" \
          "      d. gelu_erf\n" \
          "    3.clip\n" \
          "  Atleast one post-op needs to be specified if the -o arg is used.\n" \
          "  eg: -o gelu_tanh; -o bias,relu ; -o clip,prelu,bias.\n" \
          "  It is to be noted only one activator can be used at a time.\n" \
          "  If more than one activator is used, only the first activator is\n" \
          "  applied and the other activators are ignored.\n" \
          "--Downscaled version of an API is enabled by using -d arg followed\n" \
          "  by the datatype that needs to be downscaled to"
          "  Downscaled api's are used to enable quantization workflows.\n" \
          "  Following downscaled api's are supported:\n" \
          "    1. u8s8s32os32 -d s8 = u8s8s32os8.\n" \
          "    2. u8s8s16os16 -d s8 = u8s8s16os8.\n" \
          "    3. u8s8s16os16 -d u8 = u8s8s16ou8.\n" \
          "    4. bf16bf16f32of32 -d bf16 = bf16bf16f32obf16.\n" \
          "    5. s8s8s32os32 -d s8 = s8s8s32os8.\n" \
          "    6. s8s8s16os16 -d s8 = s8s8s16os8.\n" \
          "  Example: ./bench_lpgemm -m a -n 2 -o bias,relu -d bf16 -i input.txt\n" \
        );
        exit( 1 );
    }

    char* file_name = NULL;

#define GEMM_TYPE_STR_LEN 24
    char gemm_type_str[GEMM_TYPE_STR_LEN];

#define POST_OPS_STR_LEN 104
    char post_ops_str[POST_OPS_STR_LEN];
    char post_ops_str_dest[POST_OPS_STR_LEN]; //Strtok is used to parse, need to maintain a copy.

#define OPS_INPUT_STR_LEN 128
    char ops_input_str[OPS_INPUT_STR_LEN];

    // Parse CLI arguments.
     getopt_t state;
     // Initialize the state for running bli_getopt(). Here, 0 is the
     // initial value for opterr, which suppresses error messages.
     bli_getopt_init_state( 0, &state );

     int opt;
     // Process all option arguments until we get a -1, which means we're done.
     while( (opt = bli_getopt( argc, argv, "i:m:n:", &state )) != -1 )
    {
        char opt_ch = ( char )opt;
        switch( opt_ch )
        {
            case 'i':
                    file_name = state.optarg;
                    break;
            case 'm':
                    bench_mode = ( ( ( *state.optarg ) == 'a' ) || ( ( *state.optarg ) == 'p' ) ) ? ( *state.optarg ) : 'p';
                    break;
            case 'n':
                    global_n_repeat = ( atoi( state.optarg ) > 0 ) ? atoi( state.optarg ) : 0;
                    break;
            default:
                    break;
        }
    }

    if ( bench_mode == 'p' )
    {
        printf( "Running bench in performance benchmarking mode.\n" );
    }
    else if ( bench_mode == 'a' )
    {
        printf( "Running bench in accuracy/correctness testing mode.\n" );
    }

    if ( file_name == NULL )
    {
        printf( " File name provided is invalid.\n" );
        exit( 1 );
    }

    fin = fopen( file_name, "r" );
    if (fin == NULL)
    {
        printf( "Error opening the file %s\n", argv[1] );
        exit( 1 );
    }

    FILE* fout = NULL;

    fout = fopen( "lpgemm_accuracy_test_failures.txt", "w" );

    char op_a, op_b;
    char stor_order;
    char transa, transb;
    dim_t m, n, k;
    dim_t stride_a, stride_b, stride_c;

    const dim_t len_list_omp_cores_for_testing = 2;
    const dim_t list_omp_cores_for_testing[2] = { 1, 64 };

    dim_t core_index = 0;
    bool can_run = TRUE;
    while ( ( can_run == TRUE ) && ( fseek( fin, 0L, SEEK_SET ) == 0 ) )
    {
        if ( bench_mode == 'p' )
        {
            can_run = FALSE;
        }
        else if ( bench_mode == 'a' )
        {
            // For accuracy testing, we test accuracy using multiple different
            // number of cores. This helps uncover any bugs related to over
            // subscription or varying thread factorizations.
            // Set current number of cores.
#ifdef BLIS_ENABLE_OPENMP
            omp_set_num_threads( list_omp_cores_for_testing[core_index] );
#endif
            printf( "Accuracy test using %ld threads.\n",
                            list_omp_cores_for_testing[core_index] );

            core_index++;
            if ( core_index < len_list_omp_cores_for_testing )
            {
                can_run = TRUE;
            }
            else
            {
                can_run = FALSE;
            }
        }

        // Input format: data_type stor_type pack/reorder m n k lda ldb ldc
        while ( fscanf( fin, "%c %c %c %c %c " INT_FS INT_FS INT_FS
                INT_FS INT_FS INT_FS " %s\n", &stor_order, &transa,
                &transb, &op_a, &op_b, &m, &n, &k, &stride_a,
                &stride_b, &stride_c, ops_input_str ) == 12 )
        {
            char* ops_tok = strtok( ops_input_str, ":" );
            strncpy( gemm_type_str, ops_tok, GEMM_TYPE_STR_LEN - 1 );
            str_tolower( gemm_type_str ); \

            ops_tok = strtok( NULL, "" );
            if ( ops_tok != NULL )
            {
                strncpy( post_ops_str, ops_tok, POST_OPS_STR_LEN - 1 );
            }
            else
            {
                strncpy( post_ops_str, "none", POST_OPS_STR_LEN - 1 );
            }

            stor_order = ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ||
                            ( stor_order == 'c' ) || ( stor_order == 'C' ) ) ?
                            stor_order : 'r';

            if ( ( strcmp( gemm_type_str, "u8s8s32os32" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                // Copy the original post op str to a temp string buffer.
                // Done so that strtok can be applied on the same (strtok
                // is a destructive parser.
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = INT_MIN;
                DSCALE_CLIP_MAX = INT_MAX;
                GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os32)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "u8s8s32os8" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = -128;
                DSCALE_CLIP_MAX = +127;
                GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os8)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "u8s4s32os32" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                // Copy the original post op str to a temp string buffer.
                // Done so that strtok can be applied on the same (strtok
                // is a destructive parser.
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = INT_MIN;
                DSCALE_CLIP_MAX = INT_MAX;

                if ( ( op_b != 'r' ) && ( op_b != 'R' ) )
                {
                    printf("Int4 B matrix only permitted if B reodering "
                                  "is enabled.\n");
                }
                else
                {
                    GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os32)
                    (
                      fin, fout, stor_order, transa, transb, op_a, op_b,
                      m, n, k, stride_a, stride_b, stride_c,
                      post_ops_str_dest, TRUE
                    );
                }
            }
            if ( ( strcmp( gemm_type_str, "f32f32f32of32" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_can_dscale = 'y';
                global_dscale_out = 'n';
                global_pre_op = 'n';
                GEN_FUNC_NAME(mat_mul_bench_main_,f32f32f32of32)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
#if 0
            if ( ( strcmp( gemm_type_str, "u8s8s16os16" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = SHRT_MIN;
                DSCALE_CLIP_MAX = SHRT_MAX;
                GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os16)
                (
                    fin, fout, stor_order, transa, transb, op_a, op_b,
                    m, n, k, stride_a, stride_b, stride_c,
                    post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "u8s8s16os8" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = -128;
                DSCALE_CLIP_MAX = +127;
                GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os8)
                (
                    fin, fout, stor_order, transa, transb, op_a, op_b,
                    m, n, k, stride_a, stride_b, stride_c,
                    post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "u8s8s16ou8" ) == 0 ) ||
                      ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = 0;
                DSCALE_CLIP_MAX = +255;
                GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16ou8)
                (
                    fin, fout, stor_order, transa, transb, op_a, op_b,
                    m, n, k, stride_a, stride_b, stride_c,
                    post_ops_str_dest, FALSE
                );
            }
#endif
            if ( ( strcmp( gemm_type_str, "bf16bf16f32of32" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'n';
                GEN_FUNC_NAME(mat_mul_bench_main_, bf16bf16f32of32)
                (
                    fin, fout, stor_order, transa, transb, op_a, op_b,
                    m, n, k, stride_a, stride_b, stride_c,
                    post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "bf16bf16f32obf16" ) == 0 ) ||
                      ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'n';
                GEN_FUNC_NAME(mat_mul_bench_main_, bf16bf16f32obf16)
                (
                    fin, fout, stor_order, transa, transb, op_a, op_b,
                    m, n, k, stride_a, stride_b, stride_c,
                    post_ops_str_dest, FALSE
                );
            }
            if ( strcmp( gemm_type_str, "bf16s4f32of32" ) == 0 )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'y';

                if ( ( op_b != 'r' ) && ( op_b != 'R' ) )
                {
                    printf("Int4 B matrix only permitted if B reodering "
                                  "is enabled.\n");
                }
                else
                {
                    GEN_FUNC_NAME(mat_mul_bench_main_, bf16s4f32of32)
                    (
                        fin, fout, stor_order, transa, transb, op_a, op_b,
                        m, n, k, stride_a, stride_b, stride_c,
                        post_ops_str_dest, TRUE
                    );
                }
            }
            if ( strcmp( gemm_type_str, "bf16s4f32obf16" ) == 0 )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'y';

                if ( ( op_b != 'r' ) && ( op_b != 'R' ) )
                {
                    printf("Int4 B matrix only permitted if B reodering "
                                  "is enabled.\n");
                }
                else
                {
                    GEN_FUNC_NAME(mat_mul_bench_main_, bf16s4f32obf16)
                    (
                        fin, fout, stor_order, transa, transb, op_a, op_b,
                        m, n, k, stride_a, stride_b, stride_c,
                        post_ops_str_dest, TRUE
                    );
                }
            }
            if ( ( strcmp( gemm_type_str, "s8s8s32os32" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = INT_MIN;
                DSCALE_CLIP_MAX = INT_MAX;
                GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s32os32)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "s8s8s32os8" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = -128;
                DSCALE_CLIP_MAX = +127;
                GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s32os8)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
#if 0
            if ( ( strcmp( gemm_type_str, "s8s8s16os16" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = SHRT_MIN;
                DSCALE_CLIP_MAX = SHRT_MAX;
                GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s16os16)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
            if ( ( strcmp( gemm_type_str, "s8s8s16os8" ) == 0 ) ||
                 ( strcmp( gemm_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                global_pre_op = 'n';
                DSCALE_CLIP_MIN = -128;
                DSCALE_CLIP_MAX = +127;
                GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s16os8)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest, FALSE
                );
            }
#endif
        }
    }

    if ( fin )
    {
        fclose( fin );
    }
    if ( fout )
    {
        fclose( fout );
    }
    return 0;
}
