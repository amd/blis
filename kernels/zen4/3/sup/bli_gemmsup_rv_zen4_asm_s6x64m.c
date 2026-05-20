/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

#include "bli_gemmsup_rv_zen4_asm_s6x64.h"

#define NR 64

/*
   rrr:
     --------        ------        --------
     --------        ------        --------
     --------   +=   ------ ...    --------
     --------        ------        --------
     --------        ------            :
     --------        ------            :
   Assumptions:
   - B is row-stored;
   - A is row-stored;
   - m0 and n0 are at most MR (6) and NR (64), respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.
*/
void bli_sgemmsup_rv_zen4_asm_6x64m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t n_left = n0 % NR;                // n0 is expected to be n0<=NR

    // First check whether this is an edge case in the n dimension.
    // If so, dispatch other 6x?m kernels, as needed.
    if ( n_left )
    {
        float* cij = c;
        float* bj  = b;
        float* ai  = a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;
            bli_sgemmsup_rv_zen4_asm_6x48m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            bli_sgemmsup_rv_zen4_asm_6x32m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;
            bli_sgemmsup_rv_zen4_asm_6x16m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += nr_cur * cs_c0;
            bj  += nr_cur * cs_b0;
            n_left -= nr_cur;
        }

        // Previously, for cases where n_left was less than 16, separate function calls
        // were made to kernels that handled n_left of 8, 4, 2 and 1. However, this was inefficient.
        // Consider the case where n_left == 15, this would require function calls to 8 + 4 + 2 + 1 kernels.
        // This has now been replaced with masked kernels, which handles the following cases:
        // when n_left = [9, 15], this is handled by the 6x16m_mask kernel
        // which uses masked operations on the zmm register, thus the case where n_left == 15
        // is now handled by using just one function call (compared to four function calls previously).
        // Also for cases where n_left = [5, 8], this is handled by the 6x8m_mask kernel
        // This is because of double pumping on zen4, it is more efficient to use masked ymm
        // operations as opposed to using masked zmm operations.
        // Also for cases where n_left = [2, 4], this is handled by the 6x4m_mask kernel
        // This is because it was observed that for certain cases using masked loads on ymm
        // caused unnecessary cache misses on zen4, this is probably due to some peculiarities
        // of the zen4 prefetcher, but this has to be investigated further.
        // Also for the case where n_left = 1, this is handled directly by the sgemm kernel
        // as was done previously.
        if ( n_left > 8 )
        {
            bli_sgemmsup_rv_zen4_asm_6x16m_mask
            (
              conja, conjb, m0, n_left, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
        }
        else if ( n_left > 4 )
        {
            bli_sgemmsup_rv_zen4_asm_6x8m_mask
            (
              conja, conjb, m0, n_left, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
        }
        else if ( n_left > 1 )
        {
            bli_sgemmsup_rv_zen4_asm_6x4m_mask
            (
              conja, conjb, m0, n_left, k0,
              alpha, ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0, beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
        }
        else if ( n_left > 0 )
        {
          dim_t ps_a0 = bli_auxinfo_ps_a ( data);
          if ( ps_a0 == 6 * rs_a0 )
          {
            bli_sgemv_ex (
                BLIS_NO_TRANSPOSE, conjb, m0, k0,
                alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                beta, cij, rs_c0, cntx, NULL);
          }
          else
          {
            const dim_t mr = 6;

            // Since A is packed into row panels,
            // we must use a loop over gemv.
            dim_t m_iter = ( m0 + mr - 1 ) / mr;
            dim_t m_left = m0 % mr;

            float *restrict ai_ii = ai;
            float *restrict cij_ii = cij;

            for ( dim_t ii = 0; ii < m_iter; ii += 1 )
            {
              dim_t mr_cur = ( bli_is_not_edge_f ( ii, m_iter, m_left )
                                  ? mr
                                  : m_left);

              bli_sgemv_ex (
                  BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                  alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                  beta, cij_ii, rs_c0, cntx, NULL);
              cij_ii += mr_cur * rs_c0;
              ai_ii += ps_a0;
            }
          }
        }

        if ( n0 / NR == 0 )
        {
            return;
        }
    }

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    // C Prefetch
    cmp ( imm ( 4 ), rdi )
    jz ( .SPOSTPFETCH )                       // haven't added col-prefetch cases

    label ( .SROWPFETCH )
    lea ( mem ( rcx, rdi, 2 ), rdx )
    lea ( mem ( rdx, rdi, 1 ), rdx )

    prefetch ( 0, mem ( rcx,         7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 2, 7*8 ) )
    prefetch ( 0, mem ( rdx,         7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 2, 7*8 ) )

    label ( .SPOSTPFETCH )

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE4 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA4 ( 5, 24, 25, 26, 27 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA4 ( 6, 28, 29, 30, 31 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA4 ( 5, 24, 25, 26, 27 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA4 ( 6, 28, 29, 30, 31 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA4 ( 5, 24, 25, 26, 27 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA4 ( 6, 28, 29, 30, 31 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA4 ( 5, 24, 25, 26, 27 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA4 ( 6, 28, 29, 30, 31 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                              // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA4 ( 5, 24, 25, 26, 27 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA4 ( 6, 28, 29, 30, 31 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )                               // i -= 1;
    jne ( .K_LEFT_LOOP )                      // iterate again if i != 0.

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4 ( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4 ( 7, 12, 13, 14, 15 )
    ALPHA_SCALE4 ( 7, 16, 17, 18, 19 )
    ALPHA_SCALE4 ( 7, 20, 21, 22, 23 )
    ALPHA_SCALE4 ( 7, 24, 25, 26, 27 )
    ALPHA_SCALE4 ( 7, 28, 29, 30, 31 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C4 ( 4,  8,  9, 10, 11 )
    UPDATE_C4 ( 4, 12, 13, 14, 15 )
    UPDATE_C4 ( 4, 16, 17, 18, 19 )
    UPDATE_C4 ( 4, 20, 21, 22, 23 )
    UPDATE_C4 ( 4, 24, 25, 26, 27 )
    UPDATE_C4 ( 4, 28, 29, 30, 31 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     *
     * |-----------------------------------|       |------------------|--------|
     * |        |        |        |        |       |                  |        |
     * |        |        |        |        |       |       16x4       |  16x2  |
     * |  4x16  |  4x16  |  4x16  |  4x16  |       |                  |        |
     * |        |        |        |        |       |------------------|--------|
     * |        |        |        |        |       |                  |        |
     * |-----------------------------------|  ->   |       16x4       |  16x2  |
     * |        |        |        |        |       |                  |        |
     * |  2x16  |  2x16  |  2x16  |  2x16  |       |------------------|--------|
     * |        |        |        |        |       |                  |        |
     * |-----------------------------------|       |       16x4       |  16x2  |
     *                                             |                  |        |
     *                                             |------------------|--------|
     *                                             |                  |        |
     *                                             |       16x4       |  16x2  |
     *                                             |                  |        |
     *                                             |------------------|--------|
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 11, 15, 19, 23 )
    add ( rdi, rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16 ( 24, 28 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 25, 29 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 26, 30 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 27, 31 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C4_BZ ( 8, 9, 10, 11 )
    UPDATE_C4_BZ ( 12, 13, 14, 15 )
    UPDATE_C4_BZ ( 16, 17, 18, 19 )
    UPDATE_C4_BZ ( 20, 21, 22, 23 )
    UPDATE_C4_BZ ( 24, 25, 26, 27 )
    UPDATE_C4_BZ ( 28, 29, 30, 31 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 11, 15, 19, 23 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ ( 24, 28 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 25, 29 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 26, 30 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 27, 31 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a; rdx = ps_a4
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge * rs_c;
        float* restrict ai  = a + m_iter * ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen4_asm_4x64
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen4_asm_2x64
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen4_asm_1x64
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen4_asm_6x48m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    // C Prefetch
    lea ( mem ( rcx, rdi, 2 ), rdx )
    lea ( mem ( rdx, rdi, 1 ), rdx )

    cmp ( imm ( 4 ), rdi )
    jz ( .SPOSTPFETCH )                       // haven't added col-prefetch cases

    label ( .SROWPFETCH )
    prefetch ( 0, mem ( rcx,         7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 2, 7*8 ) )
    prefetch ( 0, mem ( rdx,         7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 2, 7*8 ) )

    label ( .SPOSTPFETCH )

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE3 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA3 ( 6, 28, 29, 30 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA3 ( 6, 28, 29, 30 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA3 ( 6, 28, 29, 30 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA3 ( 6, 28, 29, 30 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                              // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA3 ( 6, 28, 29, 30 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )
    // Scaling A * B with alpha.
    ALPHA_SCALE3 ( 7, 8, 9, 10 )
    ALPHA_SCALE3 ( 7, 12, 13, 14 )
    ALPHA_SCALE3 ( 7, 16, 17, 18 )
    ALPHA_SCALE3 ( 7, 20, 21, 22 )
    ALPHA_SCALE3 ( 7, 24, 25, 26 )
    ALPHA_SCALE3 ( 7, 28, 29, 30 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF of ( 4*rs_c ) == 4.
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C3 ( 4, 8, 9, 10 )
    UPDATE_C3 ( 4, 12, 13, 14 )
    UPDATE_C3 ( 4, 16, 17, 18 )
    UPDATE_C3 ( 4, 20, 21, 22 )
    UPDATE_C3 ( 4, 24, 25, 26 )
    UPDATE_C3 ( 4, 28, 29, 30 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x48 tile is split into 3 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split
     * into two tiles of 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x48 tile and are stored as 48x6 tile.
     */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16 ( 24, 28 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 25, 29 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 26, 30 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C3_BZ ( 8, 9, 10 )
    UPDATE_C3_BZ ( 12, 13, 14 )
    UPDATE_C3_BZ ( 16, 17, 18 )
    UPDATE_C3_BZ ( 20, 21, 22 )
    UPDATE_C3_BZ ( 24, 25, 26 )
    UPDATE_C3_BZ ( 28, 29, 30 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /**
     * 6x48 tile is split into 3 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into
     * two tiles of 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x48 tile and are stored as 48x6 tile.
     */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    /* Transposing 4x16 tiles to 16x4 tiles */
    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 10, 14, 18, 22 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ ( 24, 28 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 25, 29 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 26, 30 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter * ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen4_asm_4x48
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen4_asm_2x48
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen4_asm_1x48
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen4_asm_6x32m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    // C Prefetch
    lea ( mem ( rcx, rdi, 2 ), rdx )
    lea ( mem ( rdx, rdi, 1 ), rdx )

    cmp ( imm ( 4 ), rdi )
    jz ( .SPOSTPFETCH )                       // haven't added col-prefetch cases

    label ( .SROWPFETCH )
    prefetch ( 0, mem ( rcx,         7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 2, 7*8 ) )
    prefetch ( 0, mem ( rdx,         7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 2, 7*8 ) )

    label ( .SPOSTPFETCH )

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE2 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi,rsi )                          // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA2 ( 6, 28, 29 )

    add ( r9, rbx )
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA2 ( 6, 28, 29 )

    add ( r9, rbx )
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA2 ( 6, 28, 29 )

    add ( r9, rbx )
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA2 ( 6, 28, 29 )

    add ( r9, rbx )
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                    // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax, r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )
    vbroadcastss ( mem ( rax, r15, 1 ), zmm6 )
    VFMA2 ( 6, 28, 29 )

    add ( r9, rbx )
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )
    // Scaling A * B with alpha.
    ALPHA_SCALE2 ( 7, 8, 9 )
    ALPHA_SCALE2 ( 7, 12, 13 )
    ALPHA_SCALE2 ( 7, 16, 17 )
    ALPHA_SCALE2 ( 7, 20, 21 )
    ALPHA_SCALE2 ( 7, 24, 25 )
    ALPHA_SCALE2 ( 7, 28, 29 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C2 ( 4, 8, 9 )
    UPDATE_C2 ( 4, 12, 13 )
    UPDATE_C2 ( 4, 16, 17 )
    UPDATE_C2 ( 4, 20, 21 )
    UPDATE_C2 ( 4, 24, 25 )
    UPDATE_C2 ( 4, 28, 29 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x32 tile is split into 2 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into
     * two tiles of 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x32 tile and are stored as 32x6 tile.
     */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    /* Transposing 4x16 tiles to 16x4 tiles */
    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16 ( 24, 28 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 25, 29 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C2_BZ ( 8, 9 )
    UPDATE_C2_BZ ( 12, 13 )
    UPDATE_C2_BZ ( 16, 17 )
    UPDATE_C2_BZ ( 20, 21 )
    UPDATE_C2_BZ ( 24, 25 )
    UPDATE_C2_BZ ( 28, 29 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /**
     * 6x32 tile is split into 2 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split
     * into two tiles of 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x32 tile and are stored as 32x6 tile.
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ ( 24, 28 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 25, 29 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter * ps_a;
        float* restrict bj  = b;

        if ( 4 <= m_left )
        {
            const dim_t mr_cur = 4;
            bli_sgemmsup_rv_zen4_asm_4x32
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 2 <= m_left )
        {
            const dim_t mr_cur = 2;
            bli_sgemmsup_rv_zen4_asm_2x32
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }

        if ( 1 <= m_left )
        {
            const dim_t mr_cur = 1;
            bli_sgemmsup_rv_zen4_asm_1x32
            (
              conja, conjb, mr_cur, n0, k0, alpha,
              ai, rs_a0, cs_a0,
              bj, rs_b0, cs_b0,
              beta,
              cij, rs_c0, cs_c0,
              data, cntx
            );
            cij += mr_cur * rs_c;
            ai  += mr_cur * rs_a;
            m_left -= mr_cur;
        }
    }
}

void bli_sgemmsup_rv_zen4_asm_6x16m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    // C Prefetch
    lea ( mem ( rcx, rdi, 2 ), rdx )
    lea ( mem ( rdx, rdi, 1 ), rdx )

    cmp ( imm ( 4 ), rdi )
    jz ( .SPOSTPFETCH )                       // haven't added col-prefetch cases

    label ( .SROWPFETCH )
    prefetch ( 0, mem ( rcx,         7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rcx, rdi, 2, 7*8 ) )
    prefetch ( 0, mem ( rdx,         7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 1, 7*8 ) )
    prefetch ( 0, mem ( rdx, rdi, 2, 7*8 ) )

    label ( .SPOSTPFETCH )

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE1 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load a row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )
    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load a row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load a row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load a row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                              // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load a row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )
    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7, 8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )
    ALPHA_SCALE1 ( 7, 20 )
    ALPHA_SCALE1 ( 7, 24 )
    ALPHA_SCALE1 ( 7, 28 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1 ( 4, 8 )
    UPDATE_C1 ( 4, 12 )
    UPDATE_C1 ( 4, 16 )
    UPDATE_C1 ( 4, 20 )
    UPDATE_C1 ( 4, 24 )
    UPDATE_C1 ( 4, 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x16 tile is split into one 6x16 tile.
     * This 6x16 tiles is further split into
     * two tiles of 4x16 & 2x16.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x16 tile and are stored as 16x6 tile.
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16 ( 24, 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ ( 8 )
    UPDATE_C1_BZ ( 12 )
    UPDATE_C1_BZ ( 16 )
    UPDATE_C1_BZ ( 20 )
    UPDATE_C1_BZ ( 24 )
    UPDATE_C1_BZ ( 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /**
     * 6x16 tile is split into 1 equal 6x16 tiles.
     * This 6x16 tiles is further split
     * into two tiles of 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x16 tile and are stored as 16x6 tile.
     */
    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c

    TRANSPOSE_2X16_BZ ( 24, 28 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter*ps_a;
        float* restrict bj  = b;

        switch ( m_left )
        {
          case 5:
                  bli_sgemmsup_rv_zen4_asm_5x16
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );

                    break;

          case 4:
                  bli_sgemmsup_rv_zen4_asm_4x16
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );

                    break;

          case 3:
                  bli_sgemmsup_rv_zen4_asm_3x16
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );

                    break;

          case 2:
                  bli_sgemmsup_rv_zen4_asm_2x16
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );

                    break;

          case 1:
                  bli_sgemmsup_rv_zen4_asm_1x16
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );

                    break;
        }
    }
}

void bli_sgemmsup_rv_zen4_asm_6x16m_mask
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t n_left = n0 % 16;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )          // Load mask values for the last loop
    kmovw ( esi, K ( 1 ) )

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE1 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                              // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=16)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], zmm_x ) -> zmm_b     Latency: 9 cycles, Throughput: 1
    //   vfmadd231ps ( zmm_b, zmm_v, zmm_accum )         Latency: 4 cycles, Throughput: 1
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to16([A_scalar]), zmm_v, zmm_accum )  Latency: 11 cycles, Throughput: 1
    vfmadd231ps ( mem_1to16 ( rax ), zmm0, zmm8 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 1 ), zmm0, zmm12 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 2 ), zmm0, zmm16 )
    vfmadd231ps ( mem_1to16 ( rax, r13, 1 ), zmm0, zmm20 )
    vfmadd231ps ( mem_1to16 ( rax, r8, 4 ), zmm0, zmm24 )
    vfmadd231ps ( mem_1to16 ( rax, r15, 1 ), zmm0, zmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )
    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7, 8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )
    ALPHA_SCALE1 ( 7, 20 )
    ALPHA_SCALE1 ( 7, 24 )
    ALPHA_SCALE1 ( 7, 28 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK ( 4, 8 )
    UPDATE_C1_MASK ( 4, 12 )
    UPDATE_C1_MASK ( 4, 16 )
    UPDATE_C1_MASK ( 4, 20 )
    UPDATE_C1_MASK ( 4, 24 )
    UPDATE_C1_MASK ( 4, 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 6x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 6xA tile size, this is further split
    * into 4xA and 2xA tiles which are transposed
    * to Ax4 and Ax2 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                             // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_6X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                   // Compute entry address
    add ( rdx, rax )                                   // Convert offset to absolute address
    jmpi ( rax )                                       // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_6X16M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X16M_15 ) )

    label ( .SCOLSTORED_LEFT_6X16M_9 )
    TRANSPOSE_4X9 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X9 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X16M_10 )
    TRANSPOSE_4X10 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X10 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X16M_11 )
    TRANSPOSE_4X11 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X11 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X16M_12 )
    TRANSPOSE_4X12 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X12 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X16M_13 )
    TRANSPOSE_4X13 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X13 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X16M_14 )
    TRANSPOSE_4X14 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X14 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X16M_15 )
    TRANSPOSE_4X15 ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X15 ( 24, 28 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK ( 8 )
    UPDATE_C1_BZ_MASK ( 12 )
    UPDATE_C1_BZ_MASK ( 16 )
    UPDATE_C1_BZ_MASK ( 20 )
    UPDATE_C1_BZ_MASK ( 24 )
    UPDATE_C1_BZ_MASK ( 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 6x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 6xA tile size, this is further split
    * into 4xA and 2xA tiles which are transposed
    * to Ax4 and Ax2 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                                // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_6X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                   // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                      // Compute entry address
    add ( rdx, rax )                                      // Convert offset to absolute address
    jmpi ( rax )                                          // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_6X16M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X16M_15 ) )

    label ( .SCOLSTORED_BZ_LEFT_6X16M_9 )
    TRANSPOSE_4X9_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X9_BZ ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X16M_10 )
    TRANSPOSE_4X10_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X10_BZ ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X16M_11 )
    TRANSPOSE_4X11_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X11_BZ ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X16M_12 )
    TRANSPOSE_4X12_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X12_BZ ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X16M_13 )
    TRANSPOSE_4X13_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X13_BZ ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X16M_14 )
    TRANSPOSE_4X14_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X14_BZ ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X16M_15 )
    TRANSPOSE_4X15_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X15_BZ ( 24, 28 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter*ps_a;
        float* restrict bj  = b;

        switch ( m_left )
        {
          case 5:
                  bli_sgemmsup_rv_zen4_asm_5x16_mask
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );
                    break;

          case 4:
                  bli_sgemmsup_rv_zen4_asm_4x16_mask
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );
                    break;

          case 3:
                  bli_sgemmsup_rv_zen4_asm_3x16_mask
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );
                    break;

          case 2:
                  bli_sgemmsup_rv_zen4_asm_2x16_mask
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );
                    break;

          case 1:
                  bli_sgemmsup_rv_zen4_asm_1x16_mask
                    (
                      conja, conjb, m_left, n0, k0, alpha,
                      ai, rs_a0, cs_a0,
                      bj, rs_b0, cs_b0,
                      beta,
                      cij, rs_c0, cs_c0,
                      data, cntx
                    );
                    break;
        }
    }
}

void bli_sgemmsup_rv_zen4_asm_6x8m_mask
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // This function should only be called when n0 <= 8
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )          // Load mask values for the last loop
    kmovw ( esi, K ( 1 ) )

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), ymm7 )

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <= 8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=8)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], ymm_x ) -> ymm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( ymm_b, ymm_v, ymm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to8([A_scalar]), ymm_v, ymm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to8 ( rax ), ymm0, ymm8 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 1 ), ymm0, ymm12 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 2 ), ymm0, ymm16 )
    vfmadd231ps ( mem_1to8 ( rax, r13, 1 ), ymm0, ymm20 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 4 ), ymm0, ymm24 )
    vfmadd231ps ( mem_1to8 ( rax, r15, 1 ), ymm0, ymm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <= 8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=8)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], ymm_x ) -> ymm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( ymm_b, ymm_v, ymm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to8([A_scalar]), ymm_v, ymm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to8 ( rax ), ymm0, ymm8 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 1 ), ymm0, ymm12 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 2 ), ymm0, ymm16 )
    vfmadd231ps ( mem_1to8 ( rax, r13, 1 ), ymm0, ymm20 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 4 ), ymm0, ymm24 )
    vfmadd231ps ( mem_1to8 ( rax, r15, 1 ), ymm0, ymm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <= 8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=8)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], ymm_x ) -> ymm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( ymm_b, ymm_v, ymm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to8([A_scalar]), ymm_v, ymm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to8 ( rax ), ymm0, ymm8 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 1 ), ymm0, ymm12 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 2 ), ymm0, ymm16 )
    vfmadd231ps ( mem_1to8 ( rax, r13, 1 ), ymm0, ymm20 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 4 ), ymm0, ymm24 )
    vfmadd231ps ( mem_1to8 ( rax, r15, 1 ), ymm0, ymm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <= 8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=8)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], ymm_x ) -> ymm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( ymm_b, ymm_v, ymm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to8([A_scalar]), ymm_v, ymm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to8 ( rax ), ymm0, ymm8 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 1 ), ymm0, ymm12 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 2 ), ymm0, ymm16 )
    vfmadd231ps ( mem_1to8 ( rax, r13, 1 ), ymm0, ymm20 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 4 ), ymm0, ymm24 )
    vfmadd231ps ( mem_1to8 ( rax, r15, 1 ), ymm0, ymm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                              // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <= 8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=8)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], ymm_x ) -> ymm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( ymm_b, ymm_v, ymm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to8([A_scalar]), ymm_v, ymm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to8 ( rax ), ymm0, ymm8 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 1 ), ymm0, ymm12 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 2 ), ymm0, ymm16 )
    vfmadd231ps ( mem_1to8 ( rax, r13, 1 ), ymm0, ymm20 )
    vfmadd231ps ( mem_1to8 ( rax, r8, 4 ), ymm0, ymm24 )
    vfmadd231ps ( mem_1to8 ( rax, r15, 1 ), ymm0, ymm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )
    // Scaling A * B with alpha.
    ALPHA_SCALE1_YMM ( 7, 8 )
    ALPHA_SCALE1_YMM ( 7, 12 )
    ALPHA_SCALE1_YMM ( 7, 16 )
    ALPHA_SCALE1_YMM ( 7, 20 )
    ALPHA_SCALE1_YMM ( 7, 24 )
    ALPHA_SCALE1_YMM ( 7, 28 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), ymm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_YMM ( 4, 8 )
    UPDATE_C1_MASK_YMM ( 4, 12 )
    UPDATE_C1_MASK_YMM ( 4, 16 )
    UPDATE_C1_MASK_YMM ( 4, 20 )
    UPDATE_C1_MASK_YMM ( 4, 24 )
    UPDATE_C1_MASK_YMM ( 4, 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 6x[5-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 6xA tile size, this is further split
    * into 4xA and 2xA tiles which are transposed
    * to Ax4 and Ax2 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 5 ), rsi )                            // convert n_left ( >= 5 ) to [0, 3] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_6X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_6X8M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X8M_8 ) )

    label ( .SCOLSTORED_LEFT_6X8M_5 )
    TRANSPOSE_4X5_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X5_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X8M_6 )
    TRANSPOSE_4X6_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X6_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X8M_7 )
    TRANSPOSE_4X7_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X7_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X8M_8 )
    TRANSPOSE_4X8_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X8_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_YMM ( 8 )
    UPDATE_C1_BZ_MASK_YMM ( 12 )
    UPDATE_C1_BZ_MASK_YMM ( 16 )
    UPDATE_C1_BZ_MASK_YMM ( 20 )
    UPDATE_C1_BZ_MASK_YMM ( 24 )
    UPDATE_C1_BZ_MASK_YMM ( 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 6x[5-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 6xA tile size, this is further split
    * into 4xA and 2xA tiles which are transposed
    * to Ax4 and Ax2 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 5 ), rsi )                               // convert n_left ( >= 5 ) to [0, 3] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_6X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_6X8M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X8M_8 ) )

    label ( .SCOLSTORED_BZ_LEFT_6X8M_5 )
    TRANSPOSE_4X5_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X5_BZ_YMM ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X8M_6 )
    TRANSPOSE_4X6_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X6_BZ_YMM ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X8M_7 )
    TRANSPOSE_4X7_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X7_BZ_YMM ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X8M_8 )
    TRANSPOSE_4X8_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X8_BZ_YMM ( 24, 28 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter*ps_a;
        float* restrict bj  = b;

        switch ( m_left )
        {
          case 5:
                  bli_sgemmsup_rv_zen4_asm_5x8_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 4:
                  bli_sgemmsup_rv_zen4_asm_4x8_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 3:
                  bli_sgemmsup_rv_zen4_asm_3x8_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 2:
                  bli_sgemmsup_rv_zen4_asm_2x8_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 1:
                  bli_sgemmsup_rv_zen4_asm_1x8_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;
        }
    }
}

void bli_sgemmsup_rv_zen4_asm_6x4m_mask
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,     inc_t rs_a0, inc_t cs_a0,
       float*     restrict b,     inc_t rs_b0, inc_t cs_b0,
       float*     restrict beta,
       float*     restrict c,     inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // This function should only be called when n0 <= 8
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a ( data );
    uint64_t ps_a4  = ps_a * sizeof ( float );

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )          // Load mask values for the last loop
    kmovw ( esi, K ( 1 ) )

    mov ( var ( rs_a ), r8 )                  // load rs_a
    lea ( mem ( , r8, 4 ), r8 )               // rs_a *= sizeof ( dt ) => rs_a *= 4
    mov ( var ( rs_b ), r9 )                  // load rs_b
    lea ( mem ( , r9, 4 ), r9 )               // rs_b *= sizeof ( dt ) => rs_b *= 4
    mov ( var ( cs_a ), r10 )                 // load cs_a
    lea ( mem ( , r10, 4 ), r10 )             // cs_a *= sizeof ( dt ) => cs_a *= 4
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float ) => rs_c *= 4
    lea ( mem ( r8, r8, 2 ), r13 )            // r13 = 3 * rs_a
    lea ( mem ( r8, r8, 4 ), r15 )            // r15 = 5 * rs_a

    mov ( var ( m_iter ), r11 )               // load m_iter

    label ( .M_LOOP_ITER )

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), xmm7 )

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <= 4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=4)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], xmm_x ) -> xmm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( xmm_b, xmm_v, xmm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to4([A_scalar]), xmm_v, xmm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to4 ( rax ), xmm0, xmm8 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 1 ), xmm0, xmm12 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 2 ), xmm0, xmm16 )
    vfmadd231ps ( mem_1to4 ( rax, r13, 1 ), xmm0, xmm20 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 4 ), xmm0, xmm24 )
    vfmadd231ps ( mem_1to4 ( rax, r15, 1 ), xmm0, xmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <= 4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=4)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], xmm_x ) -> xmm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( xmm_b, xmm_v, xmm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to4([A_scalar]), xmm_v, xmm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to4 ( rax ), xmm0, xmm8 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 1 ), xmm0, xmm12 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 2 ), xmm0, xmm16 )
    vfmadd231ps ( mem_1to4 ( rax, r13, 1 ), xmm0, xmm20 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 4 ), xmm0, xmm24 )
    vfmadd231ps ( mem_1to4 ( rax, r15, 1 ), xmm0, xmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <= 4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=4)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], xmm_x ) -> xmm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( xmm_b, xmm_v, xmm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to4([A_scalar]), xmm_v, xmm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to4 ( rax ), xmm0, xmm8 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 1 ), xmm0, xmm12 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 2 ), xmm0, xmm16 )
    vfmadd231ps ( mem_1to4 ( rax, r13, 1 ), xmm0, xmm20 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 4 ), xmm0, xmm24 )
    vfmadd231ps ( mem_1to4 ( rax, r15, 1 ), xmm0, xmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <= 4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=4)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], xmm_x ) -> xmm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( xmm_b, xmm_v, xmm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to4([A_scalar]), xmm_v, xmm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to4 ( rax ), xmm0, xmm8 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 1 ), xmm0, xmm12 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 2 ), xmm0, xmm16 )
    vfmadd231ps ( mem_1to4 ( rax, r13, 1 ), xmm0, xmm20 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 4 ), xmm0, xmm24 )
    vfmadd231ps ( mem_1to4 ( rax, r15, 1 ), xmm0, xmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                              // else, we prepare to enter k_left loop.
    label ( .K_LEFT_LOOP )
    // Load a row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <= 4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,6), j:[0,<=4)
    // Broadcast(A[:,k]), load(B[k,:])
    //
    // There is no reuse of A values, so replace [broadcast(reg) -> FMA] with a single [FMA with memory operand].
    //
    // BEFORE (two instructions per A scalar):
    //   vbroadcastss ( [A_scalar], xmm_x ) -> xmm_b     Latency: 8 cycles, Throughput: 2
    //   vfmadd231ps ( xmm_b, xmm_v, xmm_accum )         Latency: 4 cycles, Throughput: 2
    //
    // AFTER (single fused instruction):
    //   vfmadd231ps ( mem_1to4([A_scalar]), xmm_v, xmm_accum )  Latency: 11 cycles, Throughput: 2
    vfmadd231ps ( mem_1to4 ( rax ), xmm0, xmm8 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 1 ), xmm0, xmm12 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 2 ), xmm0, xmm16 )
    vfmadd231ps ( mem_1to4 ( rax, r13, 1 ), xmm0, xmm20 )
    vfmadd231ps ( mem_1to4 ( rax, r8, 4 ), xmm0, xmm24 )
    vfmadd231ps ( mem_1to4 ( rax, r15, 1 ), xmm0, xmm28 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )
    // Scaling A * B with alpha.
    ALPHA_SCALE1_XMM ( 7, 8 )
    ALPHA_SCALE1_XMM ( 7, 12 )
    ALPHA_SCALE1_XMM ( 7, 16 )
    ALPHA_SCALE1_XMM ( 7, 20 )
    ALPHA_SCALE1_XMM ( 7, 24 )
    ALPHA_SCALE1_XMM ( 7, 28 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), xmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_XMM ( 4, 8 )
    UPDATE_C1_MASK_XMM ( 4, 12 )
    UPDATE_C1_MASK_XMM ( 4, 16 )
    UPDATE_C1_MASK_XMM ( 4, 20 )
    UPDATE_C1_MASK_XMM ( 4, 24 )
    UPDATE_C1_MASK_XMM ( 4, 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 6x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 6xA tile size, this is further split
    * into 4xA and 2xA tiles which are transposed
    * to Ax4 and Ax2 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_6X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_6X4M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_6X4M_4 ) )

    label ( .SCOLSTORED_LEFT_6X4M_1 )
    TRANSPOSE_4X1_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X1_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X4M_2 )
    TRANSPOSE_4X2_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X2_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X4M_3 )
    TRANSPOSE_4X3_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X3_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_6X4M_4 )
    TRANSPOSE_4X4_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X4_YMM ( 24, 28 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_XMM ( 8 )
    UPDATE_C1_BZ_MASK_XMM ( 12 )
    UPDATE_C1_BZ_MASK_XMM ( 16 )
    UPDATE_C1_BZ_MASK_XMM ( 20 )
    UPDATE_C1_BZ_MASK_XMM ( 24 )
    UPDATE_C1_BZ_MASK_XMM ( 28 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 6x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 6xA tile size, this is further split
    * into 4xA and 2xA tiles which are transposed
    * to Ax4 and Ax2 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load cs_c; rdi = cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c*sizeof ( dt ) => rdi = cs_c*4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * cs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_6X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_6X4M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_6X4M_4 ) )

    label ( .SCOLSTORED_BZ_LEFT_6X4M_1 )
    TRANSPOSE_4X1_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X1_BZ_YMM ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X4M_2 )
    TRANSPOSE_4X2_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X2_BZ_YMM ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X4M_3 )
    TRANSPOSE_4X3_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X3_BZ_YMM ( 24, 28 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_6X4M_4 )
    TRANSPOSE_4X4_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), r12 )                 // load rs_c; r12 = rs_c
    lea ( mem ( , r12, 4 ), r12 )             // r12 = rs_c*sizeof ( dt ) => r12 = rs_c*4
    lea ( mem ( rcx, r12, 4 ), rcx )          // rcx += 4 * r12 => rcx = 4 * rs_c
    TRANSPOSE_2X4_BZ_YMM ( 24, 28 )

    label ( .SDONE )

    mov ( var ( ps_a4 ), rdx )                // load panel stride of a
    mov ( var ( abuf ), rax )                 // load address of a
    add ( rdx, rax )                          // a += ps_a4
    mov ( rax, var ( abuf ) )                 // store updated a

    mov ( var ( rs_c ), rdi )                 // load rs_c; rdi = rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem (    , rdi, 2 ), rdx )          // rdx = rs_c * 2
    lea ( mem ( rdx, rdi, 4 ), rdx )          // rdx = rdi * 4 => rdx = rs_c * 6
    mov ( var ( cbuf ), rcx )                 // load address of c
    add ( rdx, rcx )                          // c += rs_c * 6 ( MR )
    mov ( rcx, var ( cbuf ) )                 // store updated c

    dec ( r11 )
    jne ( .M_LOOP_ITER )

    end_asm (
    : // output operands ( none )
    : // input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
      [ps_a4]  "m" ( ps_a4 ),
      [b]      "m" ( b ),
      [rs_b]   "m" ( rs_b ),
      [cs_b]   "m" ( cs_b ),
      [alpha]  "m" ( alpha ),
      [beta]   "m" ( beta ),
      [c]      "m" ( c ),
      [rs_c]   "m" ( rs_c ),
      [cs_c]   "m" ( cs_c ),
      [n0]     "m" ( n0 ),
      [m0]     "m" ( m0 ),
      [m_iter] "m" ( m_iter ),
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict ai  = a + m_iter*ps_a;
        float* restrict bj  = b;

        switch ( m_left )
        {
          case 5:
                  bli_sgemmsup_rv_zen4_asm_5x4_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 4:
                  bli_sgemmsup_rv_zen4_asm_4x4_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 3:
                  bli_sgemmsup_rv_zen4_asm_3x4_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 2:
                  bli_sgemmsup_rv_zen4_asm_2x4_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;

          case 1:
                  bli_sgemmsup_rv_zen4_asm_1x4_mask
                  (
                    conja, conjb, m_left, n0, k0, alpha,
                    ai, rs_a0, cs_a0,
                    bj, rs_b0, cs_b0,
                    beta,
                    cij, rs_c0, cs_c0,
                    data, cntx
                  );
                  break;
        }
    }
}

