/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "immintrin.h"

#define D_MR_ 24
#define D_NR_ 8

#if defined __clang__
    #define UNROLL_LOOP()      _Pragma("clang loop unroll_count(4)")
    /*
    *   in clang, unroll_count(4) generates inefficient
    *   code compared to unroll(full) when loopCount = 4.
    */
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
    #define UNROLL_LOOP_N(n)   UNROLL_LOOP_FULL() // for clang,
    //full unroll is always more performant
#elif defined __GNUC__
    #define UNROLL_LOOP()      _Pragma("GCC unroll 4")
    #define UNROLL_LOOP_FULL() _Pragma("GCC unroll 24")

    #define STRINGIFY(x) #x
    #define TOSTRING(x) STRINGIFY(x)
    #define UNROLL_LOOP_N(n) _Pragma(TOSTRING(GCC unroll n))
#else // unknown compiler
    #define UNROLL_LOOP()      // no unroll if compiler is not known
    #define UNROLL_LOOP_FULL()
    #define UNROLL_LOOP_N(n)
#endif

#define ENABLE_PACK_A                    // enable pack for A, comment out this line to disable packing
                                         // removing pack A will remove support for Left variants.

#define ENABLE_ALT_N_REM                 // clear stack frame for N remainder code, this removes false
                                         // dependencies in gcc compiler.

#define ENABLE_PACK_A_FOR_UPPER false    // enable pack A for upper variants


// define division of multiplication instruction for diagonal element
// depending on if preinversion is enabled or not
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
  #define DIAG_BROADCAST(a) (1 / a)       // if preinversion enabled, divide before broadcast
  #define DIAG_DIV_OR_MUL _mm512_mul_pd
#endif //BLIS_ENABLE_TRSM_PREINVERSION

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
  #define DIAG_BROADCAST(a) (a)
  #define DIAG_DIV_OR_MUL _mm512_div_pd   //  if preinversion disabled, divide during compute
#endif //BLIS_DISABLE_TRSM_PREINVERSION

// initialize common variables used among all right kernels
#define INIT_R() \
    double minus_one = -1; /* used as alpha in gemm kernel */               \
    auxinfo_t auxinfo;     /* needed for gemm kernel      */                \
    __m512d t_reg[1];      /*temporary registers*/                          \
    __m512d c_reg[D_MR_]; /*registers to hold GEMM accumulation*/           \
                                                                            \
    __mmask8 mask_m_0 = 0b11111111; /*register to hold mask for load/store*/\
    __mmask8 mask_m_1 = 0b11111111; /*register to hold mask for load/store*/\
    __mmask8 mask_m_2 = 0b11111111; /*register to hold mask for load/store*/\
                                                                            \
    dim_t m = bli_obj_length( b );                                          \
    dim_t n = bli_obj_width( b );                                           \
    dim_t cs_a = bli_obj_col_stride( a );                                   \
    dim_t rs_a = bli_obj_row_stride( a );                                   \
    dim_t cs_b = bli_obj_col_stride( b );                                   \
    dim_t cs_a_ = cs_a;                                                     \
    dim_t rs_a_ = rs_a;                                                     \
                                                                            \
    bool transa = bli_obj_has_trans( a );                                   \
    bool is_unitdiag = bli_obj_has_unit_diag( a );                          \
    double AlphaVal = *(double *)AlphaObj->buffer;                          \
                                                                            \
    dim_t d_mr = D_MR_;                                                     \
    dim_t d_nr = D_NR_;                                                     \
    dim_t i, j;                                                             \
    dim_t k_iter;                                                           \
                                                                            \
    double* restrict L = bli_obj_buffer_at_off( a );                        \
    double* restrict B = bli_obj_buffer_at_off( b );                        \


// Generate load/store masks
#define GENERATE_MASK(M)                                                    \
    if(M > 16)                                                              \
    {                                                                       \
        mask_m_0 = 0b11111111;                                              \
        mask_m_1 = 0b11111111;                                              \
        mask_m_2 = (__mmask8)(1 << (M-16)) - 1;                             \
    }                                                                       \
    else if(M > 8)                                                          \
    {                                                                       \
        mask_m_0 = 0b11111111;                                              \
        mask_m_1 = (__mmask8)(1 << (M-8)) - 1;                              \
        mask_m_2 = 0;                                                       \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        mask_m_0 = (__mmask8)(1 << M) - 1;                                  \
        mask_m_1 = 0;                                                       \
        mask_m_2 = 0;                                                       \
    }                                                                       \

/*
*  Perform TRSM computation for Right Upper
*  NonTranpose variant.
*  n is compile time constant.
*  M <= 24 and N <= 8
*
*  c_reg array contains alpha*B11 - A01*B10
*  let  alpha*B11 - A01*B10 = C
*/
#define TRSM_MAIN_RUN_NxM(M)                                                       \
                                                                                   \
    UNROLL_LOOP_FULL()                                                             \
    for ( dim_t ii = 0; ii < M; ++ii )                                             \
    {                                                                              \
        if( !is_unitdiag )                                                         \
        {                                                                          \
            t_reg[0] = _mm512_set1_pd( DIAG_BROADCAST( *(a11 + ii*cs_a) ) );       \
            c_reg[ii+ 0] = DIAG_DIV_OR_MUL(c_reg[ii+ 0], t_reg[0]);                \
            c_reg[ii+ 8] = DIAG_DIV_OR_MUL(c_reg[ii+ 8], t_reg[0]);                \
            c_reg[ii+16] = DIAG_DIV_OR_MUL(c_reg[ii+16], t_reg[0]);                \
        }                                                                          \
        UNROLL_LOOP_FULL()                                                         \
        for( dim_t jj = ii+1; jj < M; ++jj ) /* C[next_col] -= C[curr_col] * A11 */\
        {                                                                          \
            t_reg[0] = _mm512_set1_pd(*(a11 + jj*cs_a));                           \
            c_reg[jj+ 0] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii+ 0], c_reg[jj+ 0]); \
            c_reg[jj+ 8] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii+ 8], c_reg[jj+ 8]); \
            c_reg[jj+16] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii+16], c_reg[jj+16]); \
        }                                                                          \
        a11 += rs_a;                                                               \
    }                                                                              \

/*
*  Perform TRSM computation for Right Lower
*  NonTranpose variant.
*  N is compile time constant.
*/
#define TRSM_MAIN_RLN_NxM(M)                                                       \
                                                                                   \
    a11 += rs_a * (M-1);                                                           \
    UNROLL_LOOP_FULL()                                                             \
    for( dim_t ii = (M-1); ii >= 0; --ii )                                         \
    {                                                                              \
        if( !is_unitdiag )                                                         \
        {                                                                          \
            t_reg[0] = _mm512_set1_pd( DIAG_BROADCAST( *(a11 + ii*cs_a) ) );       \
            c_reg[ii+ 0] = DIAG_DIV_OR_MUL(c_reg[ii+ 0], t_reg[0]);                \
            c_reg[ii+ 8] = DIAG_DIV_OR_MUL(c_reg[ii+ 8], t_reg[0]);                \
            c_reg[ii+16] = DIAG_DIV_OR_MUL(c_reg[ii+16], t_reg[0]);                \
        }                                                                          \
        UNROLL_LOOP_FULL()                                                         \
        for( dim_t jj = (ii-1); jj >= 0; --jj )                                    \
        {                                                                          \
            t_reg[0] = _mm512_set1_pd(*(a11 + jj*cs_a));                           \
            c_reg[jj+ 0] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii+ 0], c_reg[jj+ 0]); \
            c_reg[jj+ 8] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii+ 8], c_reg[jj+ 8]); \
            c_reg[jj+16] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii+16], c_reg[jj+16]); \
        }                                                                          \
        a11 -= rs_a;                                                               \
    }                                                                              \

/*
* load N columns of C (24xN) into registers
*  n is a compile time constant.
*/
#define LOAD_C( N )                                                                    \
    UNROLL_LOOP_FULL()                                                                 \
    for ( dim_t ii=0; ii < N; ++ii )                                                   \
    {                                                                                  \
        c_reg[ii+ 0] = _mm512_maskz_loadu_pd(mask_m_0, b11 + (cs_b*ii+ 0)); /*load B*/ \
        c_reg[ii+ 8] = _mm512_maskz_loadu_pd(mask_m_1, b11 + (cs_b*ii+ 8)); /*load B*/ \
        c_reg[ii+16] = _mm512_maskz_loadu_pd(mask_m_2, b11 + (cs_b*ii+16)); /*load B*/ \
    }                                                                                  \

/*
*  Stores output from registers(c_reg) to memory(B)
*  n is a compile time constant.
*/
#define STORE_RIGHT_C( N )                                                      \
    UNROLL_LOOP_FULL()                                                          \
    for ( dim_t ii=0; ii < N; ++ii )                                            \
    {                                                                           \
        _mm512_mask_storeu_pd((b11 + (ii * cs_b) + 0), mask_m_0, c_reg[ii+ 0]); \
        _mm512_mask_storeu_pd((b11 + (ii * cs_b) + 8), mask_m_1, c_reg[ii+ 8]); \
        _mm512_mask_storeu_pd((b11 + (ii * cs_b) +16), mask_m_2, c_reg[ii+16]); \
    }                                                                           \

/*
* Perform GEMM + TRSM computation for Right Upper NonTranpose  
*/
#define RUNN_FRINGE( M, N )        \
    GENERATE_MASK(M)               \
    a01 = L_;                      \
    a11 = L + j*cs_a + j*rs_a;     \
    b10 = B + i;                   \
    b11 = B + i + j*cs_b;          \
    k_iter = j;                    \
    bli_dgemmsup_rv_zen5_asm_24x8m \
    (                              \
        BLIS_NO_CONJUGATE,         \
        BLIS_NO_CONJUGATE,         \
        M,                         \
        N,                         \
        k_iter,                    \
        &minus_one,                \
        b10,                       \
        1,                         \
        cs_b,                      \
        a01,                       \
        rs_a_,                     \
        cs_a_,                     \
        &AlphaVal,                 \
        b11, 1, cs_b,              \
        &auxinfo,                  \
        NULL                       \
    );                             \
    LOAD_C( N )                    \
    TRSM_MAIN_RUN_NxM( N )         \
    STORE_RIGHT_C( N )             \

/*
* Perform GEMM + TRSM computation for Right Lower NonTranpose
*/
#define RLNN_FRINGE( M, N )                                  \
    GENERATE_MASK(M)                                         \
    a01 = L_;                                                \
    a11 = L + (j - N + d_nr) * rs_a + (j - N + d_nr) * cs_a; \
    b10 = B + (i - M + d_mr) + (j + d_nr) * cs_b;            \
    b11 = B + (i - M + d_mr) + (j - N + d_nr) * cs_b;        \
    k_iter = (n - j - d_nr);                                 \
    bli_dgemmsup_rv_zen5_asm_24x8m                           \
    (                                                        \
        BLIS_NO_CONJUGATE,                                   \
        BLIS_NO_CONJUGATE,                                   \
        M,                                                   \
        N,                                                   \
        k_iter,                                              \
        &minus_one,                                          \
        b10,                                                 \
        1,                                                   \
        cs_b,                                                \
        a01,                                                 \
        rs_a_,                                               \
        cs_a_,                                               \
        &AlphaVal,                                           \
        b11, 1, cs_b,                                        \
        &auxinfo,                                            \
        NULL                                                 \
    );                                                       \
    LOAD_C( N )                                              \
    TRSM_MAIN_RLN_NxM( N )                                   \
    STORE_RIGHT_C( N )                                       \


/*
   declaration of trsm small kernels function pointer
*/
typedef err_t (*trsmsmall_ker_ft)
    (
      obj_t*   AlphaObj,
      obj_t*   a,
      obj_t*   b,
      cntx_t*  cntx,
      cntl_t*  cntl
    );

trsmsmall_ker_ft ker_fps_zen5[4][8] =
  {
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
    {bli_dtrsm_small_AutXB_AlXB_ZEN5,
     bli_dtrsm_small_AltXB_AuXB_ZEN5,
     bli_dtrsm_small_AltXB_AuXB_ZEN5,
     bli_dtrsm_small_AutXB_AlXB_ZEN5,
     bli_dtrsm_small_XAutB_XAlB_ZEN5,
     bli_dtrsm_small_XAltB_XAuB_ZEN5,
     bli_dtrsm_small_XAltB_XAuB_ZEN5,
     bli_dtrsm_small_XAutB_XAlB_ZEN5},
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
};

err_t bli_trsm_small_ZEN5
     (
       side_t   side,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl,
       bool     is_parallel
     )
{
    err_t err;
    bool uplo = bli_obj_is_upper(a);
    bool transa = bli_obj_has_trans(a);
    num_t dt = bli_obj_dt(a);

    /* If alpha is zero, B matrix will become zero after scaling
       hence solution is also zero matrix */
    if (bli_obj_equals(alpha, &BLIS_ZERO))
    {
        return BLIS_NOT_YET_IMPLEMENTED; // scale B by alpha
    }

    // A is expected to be triangular in trsm
    if (!bli_obj_is_upper_or_lower(a))
    {
        return BLIS_EXPECTED_TRIANGULAR_OBJECT;
    }

    /*
     *  Compose kernel index based on inputs
     */
    dim_t keridx = (((side & 0x1) << 2) |
          ((uplo & 0x1) << 1) |
          (transa & 0x1));
    trsmsmall_ker_ft ker_fp = ker_fps_zen5[dt][keridx];
    /*Call the kernel*/
    err = ker_fp
          (
            alpha,
            a,
            b,
            cntx,
            cntl
           );
    return err;
}

#ifdef BLIS_ENABLE_OPENMP
/*
 * Parallelized dtrsm_small across m-dimension or n-dimension based on side(Left/Right)
 */
err_t bli_trsm_small_mt_ZEN5
     (
       side_t   side,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl,
       bool     is_parallel
     )
{
    gint_t m = bli_obj_length(b); // number of rows of matrix b
    gint_t n = bli_obj_width(b);  // number of columns of Matrix b
    dim_t d_mr = D_MR_, d_nr = D_NR_;

    // num_t dt = bli_obj_dt(a);
    rntm_t rntm;
    bli_rntm_init_from_global(&rntm);
#ifdef AOCL_DYNAMIC
    // If dynamic-threading is enabled, calculate optimum number
    //  of threads.
    //  rntm will be updated with optimum number of threads.
    if (bli_obj_is_double(b) || bli_obj_is_dcomplex(b) )
    {
        bli_nthreads_optimum(a, b, b, BLIS_TRSM, &rntm);
    }
#endif
    // Query the total number of threads from the rntm_t object.
    dim_t n_threads = bli_rntm_num_threads(&rntm);
    if (n_threads < 0)
        n_threads = 1;
    err_t status = BLIS_SUCCESS;
    _Pragma("omp parallel num_threads(n_threads)")
    {
        // Query the thread's id from OpenMP.
        const dim_t tid = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();

        // if num threads requested and num thread available
        // is not same then use single thread small
        if(nt_real != n_threads)
        {
            if(tid == 0)
            {
                bli_trsm_small_ZEN5
                (
                side,
                alpha,
                a,
                b,
                cntx,
                cntl,
                is_parallel
                );
            }
        }
        else
        {
            obj_t b_t;
            dim_t start; // Each thread start Index
            dim_t end;   // Each thread end Index
            thrinfo_t thread;

            thread.n_way = n_threads;
            thread.work_id = tid;
            thread.ocomm_id = tid;

            // Compute start and end indexes of matrix partitioning for each thread
            if (bli_is_right(side))
            {
                bli_thread_range_sub
                (
                  &thread,
                  m,
                  d_mr, // Need to decide based on type
                  FALSE,
                  &start,
                  &end
                );
                // For each thread acquire matrix block on which they operate
                // Data-based parallelism

                bli_acquire_mpart_mdim(BLIS_FWD, BLIS_SUBPART1, start, end - start, b, &b_t);
            }
            else
            {
                bli_thread_range_sub
                (
                  &thread,
                  n,
                  d_nr,// Need to decide based on type
                  FALSE,
                  &start,
                  &end
                );
                // For each thread acquire matrix block on which they operate
                // Data-based parallelism

                bli_acquire_mpart_ndim(BLIS_FWD, BLIS_SUBPART1, start, end - start, b, &b_t);
            }

            // Parallelism is only across m-dimension/n-dimension - therefore matrix a is common to
            // all threads
            err_t status_l = BLIS_SUCCESS;

            status_l = bli_trsm_small_ZEN5
              (
                side,
                alpha,
                a,
                &b_t,
                NULL,
                NULL,
                is_parallel
              );
            // To capture the error populated from any of the threads
            if ( status_l != BLIS_SUCCESS )
            {
                _Pragma("omp critical")
                status = (status != BLIS_NOT_YET_IMPLEMENTED) ? status_l : status;
            }
        }
    }
    return status;
}
#endif


/*
* Solve Right Upper NonTranspose TRSM when N < 8
*/
BLIS_INLINE void runn_n_rem
(
    dim_t i,
    dim_t j,
    dim_t cs_a,
    dim_t rs_a,
    dim_t cs_a_,
    dim_t rs_a_,
    dim_t cs_b,
    dim_t m,
    dim_t n,
    double* L,
    double* L_,
    double* p,
    double* B,
    dim_t k_iter,
    bool transa,
    bool bPackedA,
    double AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p; // avoid warning if pack not enabled

    dim_t d_mr = D_MR_;
    double minus_one = -1; // alpha for gemm

#ifdef ENABLE_PACK_A
    double one = 1;
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[1]; /*temporary registers*/
    __m512d c_reg[D_MR_]; /*registers to hold GEMM accumulation*/
    for(dim_t i = 0; i < D_MR_; ++i)
    {
        c_reg[i] = _mm512_setzero_pd(); // initialize c_reg to zero
    }

    __mmask8 mask_m_0, mask_m_1, mask_m_2;

    double *a01, *a11, *b10, *b11;
    dim_t m_rem;
    dim_t n_rem = n - j;
    L_ = L + j*cs_a;

#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_8xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                n_rem,
                j,
                j,
                &one,
                L + j*cs_a,
                cs_a,
                rs_a,
                p,
                D_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = D_NR_;
            L_ = p;
        }
#endif
    for( i = 0; (i+d_mr-1) < m; i += d_mr )
    {
        RUNN_FRINGE(D_MR_, n_rem);
    }
    m_rem = m - i;
    if( m_rem > 0 )
    {
        RUNN_FRINGE( m_rem, n_rem );
    }
}

/*
* Solve Right Upper NonTranspose TRSM when N < 8
*/
BLIS_INLINE void rlnn_n_rem
(
    dim_t i,
    dim_t j,
    dim_t cs_a,
    dim_t rs_a,
    dim_t cs_a_,
    dim_t rs_a_,
    dim_t cs_b,
    dim_t m,
    dim_t n,
    double* L,
    double* L_,
    double* p,
    double* B,
    dim_t k_iter,
    bool transa,
    bool bPackedA,
    double AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p;
    dim_t d_mr = D_MR_;
    dim_t d_nr = D_NR_;
    double minus_one = -1;
#ifdef ENABLE_PACK_A
    double one = 1;
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[1]; /*temporary registers*/
    __m512d c_reg[D_MR_]; /*registers to hold GEMM accumulation*/

    __mmask8 mask_m_0, mask_m_1, mask_m_2;
    double *a01, *a11, *b10, *b11;
    dim_t m_rem;
    dim_t n_rem = j + d_nr;
    L_ = L + ((j - n_rem + d_nr) * cs_a) + (j + d_nr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_8xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                n_rem,
                (n - j - d_nr),
                (n - j - d_nr),
                &one,
                L + ((j - n_rem + d_nr) * cs_a) + (j + d_nr) * rs_a,
                cs_a,
                rs_a,
                p,
                D_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = D_NR_;
            L_ = p;
        }
#endif
    for( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
    {
        RLNN_FRINGE(D_MR_, n_rem);
    }
    m_rem = i + d_mr;
    if( m_rem > 0 )
    {
        RLNN_FRINGE( m_rem, n_rem );
    }
}


// RUNN - RLTN
err_t __attribute__((target("tune=znver3")))
    bli_dtrsm_small_XAltB_XAuB_ZEN5
      (
        obj_t* AlphaObj,
        obj_t* a,
        obj_t* b,
        cntx_t* cntx,
        cntl_t* cntl
      )
{
    INIT_R();
    if( transa )
    {
        /*
        * If variants being solved is RLTN
        * then after swapping rs_a and cs_a,
        * problem will become same as RUNN
        */
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
        cs_a_ = cs_a;
        rs_a_ = rs_a;
    }
    double *a01, *a11, *b10, *b11;
    double* restrict L_ = L;

#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;
    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    double* p = NULL;
    double one = 1;
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (D_NR_*n*sizeof(double)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
#endif

    for( j = 0; (j+d_nr-1) < n; j += d_nr )
    {
        L_ = L + j*cs_a;

#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_8xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                D_NR_,
                j,
                j,
                &one,
                L + j*cs_a,
                cs_a,
                rs_a,
                p,
                D_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = D_NR_;
            L_ = p;
        }
#endif
        for( i = 0; (i+d_mr-1) < m; i += d_mr )
        {
            RUNN_FRINGE( D_MR_, D_NR_ );
        }
        dim_t m_rem = m - i;
        if ( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem , D_NR_ );
        }
    }

    dim_t n_rem = n - j;
    if( n_rem > 0 )
    {
#ifdef ENABLE_ALT_N_REM
#ifndef ENABLE_PACK_A
        double* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        runn_n_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
#else //ENABLE_ALT_N_REM
        L_ = L + j*cs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_8xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                n_rem,
                j,
                j,
                &one,
                L + j*cs_a,
                cs_a,
                rs_a,
                p,
                D_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = D_NR_;
            L_ = p;
        }
#endif //ENABLE_PACK_A
        for( i = 0; (i+d_mr-1) < m; i += d_mr )
        {
            RUNN_FRINGE(D_MR_, n_rem);
        }
        dim_t m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, n_rem );
        }
#endif //ENABLE_ALT_N_REM
    }

#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
    return BLIS_SUCCESS;
}

// RLNN - RUTN
err_t bli_dtrsm_small_XAutB_XAlB_ZEN5
      (
        obj_t* AlphaObj,
        obj_t* a,
        obj_t* b,
        cntx_t* cntx,
        cntl_t* cntl
      )
{
    INIT_R();
    if( transa )
    {
        /*
        * If variants being solved is RUTN
        * then after swapping rs_a and cs_a,
        * problem will become same as RLNN
        */
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
        cs_a_ = cs_a;
        rs_a_ = rs_a;
    }
    double *a01, *a11, *b10, *b11;
    double* restrict L_ = L;

#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;

    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    double* p = NULL;
    double one = 1;
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (D_NR_*n*sizeof(double)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
#endif

    for ( j = (n - d_nr); j > -1; j -= d_nr )
    {
        L_ = L + ((j - D_NR_ + d_nr) * cs_a) + (j + d_nr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_8xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                D_NR_,
                (n - j - d_nr),
                (n - j - d_nr),
                &one,
                L_ = L + ((j - D_NR_ + d_nr) * cs_a) + (j + d_nr) * rs_a,
                cs_a,
                rs_a,
                p,
                D_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = D_NR_;
            L_ = p;
        }
#endif
        for ( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
        {
            RLNN_FRINGE( D_MR_, D_NR_ );
        }
        dim_t m_rem = i + d_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, D_NR_ );
        }
    }
    dim_t n_rem = j + d_nr;
    if( n_rem > 0 )
    {
#ifndef ENABLE_PACK_A
        double* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        rlnn_n_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
    }
#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
    return BLIS_SUCCESS;
}

// shuffles for 8x8 transpose
// transpose upper or lower half of 8 zmm registers
// inputs are taken from T_0 to T_3 and output is stored
// in O_1 to O_4
#define SHUFFLE_TRANSPOSE(T_0, T_1, T_2, T_3, O_1, O_2, O_3, O_4)  \
    t_reg[0] = _mm512_shuffle_f64x2(T_0, T_1, 0b10001000);         \
    t_reg[1] = _mm512_shuffle_f64x2(T_2, T_3, 0b10001000);         \
    O_1    = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b10001000); \
    O_2    = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b11011101); \
                                                                   \
    t_reg[0] = _mm512_shuffle_f64x2(T_0, T_1, 0b11011101);         \
    t_reg[1] = _mm512_shuffle_f64x2(T_2, T_3, 0b11011101);         \
    O_3    = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b10001000); \
    O_4    = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b11011101); \

// transpose 8x8 matrix, input is taken from
// c_reg[0+OFFSET] to c_reg[7+OFFSET] and output is stored
// back into same registers
#define TRANSPOSE_8x8(OFFSET) \
    t_reg[2] = _mm512_unpacklo_pd(c_reg[0+OFFSET], c_reg[1+OFFSET]);     \
    t_reg[3] = _mm512_unpacklo_pd(c_reg[2+OFFSET], c_reg[3+OFFSET]);     \
    t_reg[4] = _mm512_unpacklo_pd(c_reg[4+OFFSET], c_reg[5+OFFSET]);     \
    t_reg[5] = _mm512_unpacklo_pd(c_reg[6+OFFSET], c_reg[7+OFFSET]);     \
                                                                         \
    t_reg[6] = _mm512_unpackhi_pd(c_reg[0+OFFSET], c_reg[1+OFFSET]);     \
    t_reg[7] = _mm512_unpackhi_pd(c_reg[2+OFFSET], c_reg[3+OFFSET]);     \
    t_reg[8] = _mm512_unpackhi_pd(c_reg[4+OFFSET], c_reg[5+OFFSET]);     \
    t_reg[9] = _mm512_unpackhi_pd(c_reg[6+OFFSET], c_reg[7+OFFSET]);     \
                                                                         \
    SHUFFLE_TRANSPOSE( t_reg[2], t_reg[3], t_reg[4], t_reg[5],           \
     c_reg[0+OFFSET], c_reg[4+OFFSET], c_reg[2+OFFSET], c_reg[6+OFFSET]) \
                                                                         \
    SHUFFLE_TRANSPOSE( t_reg[6], t_reg[7], t_reg[8], t_reg[9],           \
     c_reg[1+OFFSET], c_reg[5+OFFSET], c_reg[3+OFFSET], c_reg[7+OFFSET]) \

// transpose 24x8 matrix stored in
// c_reg[0] to c_reg[23]
#define TRANSPOSE_24x8() \
    TRANSPOSE_8x8(0)     \
    TRANSPOSE_8x8(8)     \
    TRANSPOSE_8x8(16)    \

// initialize common variables used among left N left kernels
#define INIT_L_N_LEFT()                                                     \
    double minus_one = -1; /* used as alpha in gemm kernel */               \
    auxinfo_t auxinfo;     /* for dgemm kernel*/                            \
    __m512d t_reg[10];     /*temporary registers*/                          \
    __m512d c_reg[D_MR_]; /*registers to hold GEMM accumulation*/           \
    for(dim_t i = 0; i < D_MR_; ++i)                                        \
    {                                                                       \
        c_reg[i] = _mm512_setzero_pd(); /*initialize c_reg to zero*/        \
    }                                                                       \
                                                                            \
    __mmask8 mask_m_0 = 0b11111111; /*register to hold mask for load/store*/\
    __mmask8 mask_m_1 = 0b11111111; /*register to hold mask for load/store*/\
    __mmask8 mask_m_2 = 0b11111111; /*register to hold mask for load/store*/\

// initialize common variables used among all left kernels
#define INIT_L() \
    INIT_L_N_LEFT()                                                         \
    dim_t m = bli_obj_length( b );                                          \
    dim_t n = bli_obj_width( b );                                           \
    dim_t cs_a = bli_obj_col_stride( a );                                   \
    dim_t rs_a = bli_obj_row_stride( a );                                   \
    dim_t cs_b = bli_obj_col_stride( b );                                   \
    dim_t cs_a_ = cs_a;                                                     \
    dim_t rs_a_ = rs_a;                                                     \
                                                                            \
    bool transa = bli_obj_has_trans( a );                                   \
    bool is_unitdiag = bli_obj_has_unit_diag( a );                          \
    double AlphaVal = *(double *)AlphaObj->buffer;                          \
                                                                            \
    dim_t d_mr =D_MR_;                                                      \
    dim_t d_nr = D_NR_;                                                     \
    dim_t i, j;                                                             \
    dim_t k_iter;                                                           \
                                                                            \
    double* restrict L = bli_obj_buffer_at_off( a );                        \
    double* restrict B = bli_obj_buffer_at_off( b );                        \

/*
*  Perform TRSM computation for Left Lower
*  NonTranpose variant.
*  n is compile time constant.
*  M <= 24 and N <= 8
*
*  c_reg array contains alpha*B11 - A01*B10
*  let  alpha*B11 - A01*B10 = C
*/
#define TRSM_MAIN_LLN_NxM(M)                                                       \
                                                                                   \
    UNROLL_LOOP_FULL()                                                             \
    for ( dim_t ii = 0; ii < M; ++ii )                                             \
    {                                                                              \
        if( !is_unitdiag )                                                         \
        {                                                                          \
            t_reg[0] = _mm512_set1_pd( DIAG_BROADCAST( *(a11 + ii*cs_a) ));                       \
            c_reg[ii] = DIAG_DIV_OR_MUL(c_reg[ii], t_reg[0]);                        \
        }                                                                          \
        UNROLL_LOOP_FULL()                                                         \
        for( dim_t jj = ii+1; jj < M; ++jj ) /* C[next_col] -= C[curr_col] * A11 */\
        {                                                                          \
            t_reg[0] = _mm512_set1_pd(*(a11 + jj*cs_a));                           \
            c_reg[jj] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii], c_reg[jj]);          \
        }                                                                          \
        a11 += rs_a;                                                               \
    }                                                                              \


/*
*  Perform TRSM computation for Left Upper
*  NonTranpose variant.
*  n is compile time constant.
*  M <= 24 and N <= 8
*
*  c_reg array contains alpha*B11 - A01*B10
*  let  alpha*B11 - A01*B10 = C
*/
#define TRSM_MAIN_LUN_NxM(M)                                              \
                                                                          \
    a11 += rs_a * (M-1);                                                  \
    UNROLL_LOOP_FULL()                                                    \
    for( dim_t ii = (M-1); ii >= 0; --ii )                                \
    {                                                                     \
        if( !is_unitdiag )                                                \
        {                                                                 \
            t_reg[0] = _mm512_set1_pd( DIAG_BROADCAST( *(a11 + ii*cs_a) ));              \
            c_reg[ii] = DIAG_DIV_OR_MUL(c_reg[ii], t_reg[0]);               \
        }                                                                 \
        UNROLL_LOOP_N(23) /*unroll loop 24 is generating warning in gcc*/ \
        for( dim_t jj = (ii-1); jj >= 0; --jj )                           \
        {                                                                 \
            t_reg[0] = _mm512_set1_pd(*(a11 + jj*cs_a));                  \
            c_reg[jj] = _mm512_fnmadd_pd(t_reg[0], c_reg[ii], c_reg[jj]); \
        }                                                                 \
        a11 -= rs_a;                                                      \
    }                                                                     \

/*
* Perform GEMM + TRSM computation for Left Lower NonTranpose  
*/
#define LLNN_FRINGE( M, N )            \
    GENERATE_MASK(M)                   \
    a01 = L_;                          \
    a11 = L + (i * rs_a) + (i * cs_a); \
    b01 = B + j * cs_b;                \
    b11 = B + i + j * cs_b;            \
    k_iter = i;                        \
    bli_dgemmsup_rv_zen5_asm_24x8m     \
    (                                  \
        BLIS_NO_CONJUGATE,             \
        BLIS_NO_CONJUGATE,             \
        M,                             \
        N,                             \
        k_iter,                        \
        &minus_one,                    \
        a01,                           \
        rs_a_,                         \
        cs_a_,                         \
        b01,                           \
        1,                             \
        cs_b,                          \
        &AlphaVal,                     \
        b11, 1, cs_b,                  \
        &auxinfo,                      \
        NULL                           \
    );                                 \
    LOAD_C( N )                        \
    TRANSPOSE_24x8()                   \
    TRSM_MAIN_LLN_NxM( M )             \
    TRANSPOSE_24x8()                   \
    STORE_RIGHT_C( N )                 \

/*
* Perform GEMM + TRSM computation for Left Upper NonTranpose  
*/
#define LUNN_FRINGE( M, N )                                 \
    GENERATE_MASK(M)                                        \
    a01 = L_;                                               \
    a11 = L + (i - M + d_mr) * rs_a + (i - M + d_mr) * cs_a;\
    b01 = B + (i + d_mr) + (j - N + d_nr) * cs_b;           \
    b11 = B + (i - M + d_mr) + (j - N + d_nr) * cs_b;       \
    k_iter = ( m - i - d_mr );                              \
    bli_dgemmsup_rv_zen5_asm_24x8m                          \
    (                                                       \
        BLIS_NO_CONJUGATE,                                  \
        BLIS_NO_CONJUGATE,                                  \
        M,                                                  \
        N,                                                  \
        k_iter,                                             \
        &minus_one,                                         \
        a01,                                                \
        rs_a_,                                              \
        cs_a_,                                              \
        b01,                                                \
        1,                                                  \
        cs_b,                                               \
        &AlphaVal,                                          \
        b11, 1, cs_b,                                       \
        &auxinfo,                                           \
        NULL                                                \
    );                                                      \
    LOAD_C( N )                                             \
    TRANSPOSE_24x8()                                        \
    TRSM_MAIN_LUN_NxM( M )                                  \
    TRANSPOSE_24x8()                                        \
    STORE_RIGHT_C( N )                                      \

/*
* Solve Left Lower NonTranspose TRSM when m < 24
*/
BLIS_INLINE void llnn_m_rem
(
    dim_t i,
    dim_t j,
    dim_t cs_a,
    dim_t rs_a,
    dim_t cs_a_,
    dim_t rs_a_,
    dim_t cs_b,
    dim_t m,
    dim_t n,
    double* L,
    double* L_,
    double* p,
    double* B,
    dim_t k_iter,
    bool transa,
    bool bPackedA,
    double AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p;
    // dim_t d_mr = D_MR_;
    dim_t d_nr = D_NR_;
    double minus_one = -1;
#ifdef ENABLE_PACK_A
    double one = 1;
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[10]; /*temporary registers*/
    __m512d c_reg[D_MR_]; /*registers to hold GEMM accumulation*/

    __mmask8 mask_m_0, mask_m_1, mask_m_2;
    double *a01, *a11, *b01, *b11;
    dim_t m_rem = m - i;
    dim_t n_rem;
    L_ = L + (i * cs_a);
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_24xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                m_rem,
                i,
                i,
                &one,
                L + (i*cs_a),
                cs_a,
                rs_a,
                p,
                D_MR_,
                cntx
            );
            cs_a_ = D_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
    for( j = 0; (j + d_nr - 1) < n; j += d_nr )
    {
        LLNN_FRINGE(m_rem, D_NR_);
    }
    n_rem = n - j;
    if( n_rem > 0 )
    {
        LLNN_FRINGE( m_rem, n_rem );
    }
}

/*
* Solve Left Upper NonTranspose TRSM when m < 24
*/
BLIS_INLINE void lunn_m_rem
(
    dim_t i,
    dim_t j,
    dim_t cs_a,
    dim_t rs_a,
    dim_t cs_a_,
    dim_t rs_a_,
    dim_t cs_b,
    dim_t m,
    dim_t n,
    double* L,
    double* L_,
    double* p,
    double* B,
    dim_t k_iter,
    bool transa,
    bool bPackedA,
    double AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p;
    dim_t d_mr = D_MR_;
    dim_t d_nr = D_NR_;
    double minus_one = -1;
#ifdef ENABLE_PACK_A
    double one = 1;
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[10]; /*temporary registers*/
    __m512d c_reg[D_MR_]; /*registers to hold GEMM accumulation*/
    for(dim_t i = 0; i < D_MR_; ++i)
    {
        c_reg[i] = _mm512_setzero_pd(); /*initialize c_reg to zero*/
    }

    __mmask8 mask_m_0, mask_m_1, mask_m_2;
    double *a01, *a11, *b01, *b11;
    dim_t m_rem = i + d_mr;
    dim_t n_rem;
    L_ = L + ((i - m_rem + d_mr) * cs_a) + (i + d_mr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_24xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                m_rem,
                ( m - i - d_mr ),
                ( m - i - d_mr ),
                &one,
                L + ((i - m_rem + d_mr) * cs_a) + (i + d_mr) * rs_a,
                cs_a,
                rs_a,
                p,
                D_MR_,
                cntx
            );
            cs_a_ = D_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
    for( j = (n - d_nr); (j + 1) > 0; j -= d_nr )
    {
        LUNN_FRINGE(m_rem, D_NR_);
    }
    n_rem = j + d_nr;
    if( n_rem > 0 )
    {
        LUNN_FRINGE( m_rem, n_rem );
    }
}


/*
* Solve Left Lower NonTranspose TRSM when N < 8
*/
BLIS_INLINE void llnn_n_rem
(
    dim_t i,
    dim_t j,
    dim_t cs_a,
    dim_t rs_a,
    dim_t cs_a_,
    dim_t rs_a_,
    dim_t cs_b,
    dim_t m,
    dim_t n,
    double* L,
    double* L_,
    double* B,
    dim_t k_iter,
    double AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    INIT_L_N_LEFT()
    double *a01, *a11, *b01, *b11;
    dim_t n_rem = n - j;
    LLNN_FRINGE( D_MR_, n_rem );
}


/*
* Solve Left Lower NonTranspose TRSM when N < 8
*/
BLIS_INLINE void lunn_n_rem
(
    dim_t i,
    dim_t j,
    dim_t cs_a,
    dim_t rs_a,
    dim_t cs_a_,
    dim_t rs_a_,
    dim_t cs_b,
    dim_t m,
    dim_t n,
    double* L,
    double* L_,
    double* B,
    dim_t k_iter,
    double AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    INIT_L_N_LEFT()
    c_reg[6] = _mm512_setzero_pd(); // zerod to avoid warning in GCC
    c_reg[7] = _mm512_setzero_pd();
    double *a01, *a11, *b01, *b11;
    dim_t d_mr = D_MR_;
    dim_t d_nr = D_NR_;
    dim_t n_rem = j + d_nr;
    LUNN_FRINGE( D_MR_, n_rem );
}


// LLNN - LUTN
err_t
__attribute__((target("tune=skylake-avx512")))
bli_dtrsm_small_AutXB_AlXB_ZEN5
      (
        obj_t*   AlphaObj,
        obj_t*   a,
        obj_t*   b,
        cntx_t*  cntx,
        cntl_t*  cntl
      )
{
    INIT_L()
    if( !transa )
    {
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    double *a01, *a11, *b01, *b11;
    double* restrict L_ = L;
#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;
    if ( transa )
    {
        bPackedA = true;
    }
    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    double* p = NULL;
    double one = 1;
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (D_MR_*m*sizeof(double)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
    if (!bPackedA)
#endif
    {
        if (transa)
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
    }
    for( i = 0; (i + d_mr - 1) < m; i += d_mr )
    {
        L_ = L + i*cs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_24xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                D_MR_,
                i,
                i,
                &one,
                L + i*cs_a,
                cs_a,
                rs_a,
                p,
                D_MR_,
                cntx
            );
            cs_a_ = D_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
        for( j = 0; j < n - d_nr + 1; j += d_nr )
        {
            LLNN_FRINGE( D_MR_, D_NR_ );
        }
        dim_t n_rem = n - j;
        if( n_rem > 0 )
        {
            llnn_n_rem
            (
                i, j,
                cs_a, rs_a,
                cs_a_, rs_a_, cs_b,
                m, n, L, L_, B, k_iter, AlphaVal,
                is_unitdiag, cntx
            );
        }
    }
    dim_t m_rem = m - i;
    if( m_rem > 0 )
    {
#ifndef ENABLE_PACK_A
        double* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        llnn_m_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
    }
#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
    return BLIS_SUCCESS;
}

// LUNN - LLTN
err_t bli_dtrsm_small_AltXB_AuXB_ZEN5
      (
        obj_t*   AlphaObj,
        obj_t*   a,
        obj_t*   b,
        cntx_t*  cntx,
        cntl_t*  cntl
      )
{
    INIT_L()
    if( !transa )
    {
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    double *a01, *a11, *b01, *b11;
    double* restrict L_ = L;
#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;
    if ( transa )
    {
        bPackedA = true;
    }
    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    double* p = NULL;
    double one = 1;
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (D_MR_*m*sizeof(double)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
    if (!bPackedA)
#endif
    {
        if (transa)
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
    }
    for( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
    {
        L_ = L + ((i - D_MR_ + d_mr) * cs_a) + (i + d_mr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_dpackm_zen4_asm_24xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                D_MR_,
                ( m - i - d_mr ),
                ( m - i - d_mr ),
                &one,
                L + ((i - D_MR_ + d_mr) * cs_a) + (i + d_mr) * rs_a,
                cs_a,
                rs_a,
                p,
                D_MR_,
                cntx
            );
            cs_a_ = D_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
        for( j = (n - d_nr); (j + 1) > 0; j -= d_nr )
        {
            LUNN_FRINGE( D_MR_, D_NR_ );
        }
        dim_t n_rem = j + d_nr;
        if( n_rem > 0 )
        {
            lunn_n_rem
            (
                i, j,
                cs_a, rs_a,
                cs_a_, rs_a_, cs_b,
                m, n, L, L_, B, k_iter, AlphaVal,
                is_unitdiag, cntx
            );
        }
    }
    dim_t m_rem = i + d_mr;
    if( m_rem > 0 )
    {
#ifndef ENABLE_PACK_A
         double* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        lunn_m_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
    }
#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
return BLIS_SUCCESS;
}
