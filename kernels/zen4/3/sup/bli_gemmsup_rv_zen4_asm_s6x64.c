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

void bli_sgemmsup_rv_zen4_asm_5x48
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA3 ( 5, 24, 25, 26 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3 ( 7,  8,  9, 10 )
    ALPHA_SCALE3 ( 7, 12, 13, 14 )
    ALPHA_SCALE3 ( 7, 16, 17, 18 )
    ALPHA_SCALE3 ( 7, 20, 21, 22 )
    ALPHA_SCALE3 ( 7, 24, 25, 26 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C3 ( 4,  8,  9, 10 )
    UPDATE_C3 ( 4, 12, 13, 14 )
    UPDATE_C3 ( 4, 16, 17, 18 )
    UPDATE_C3 ( 4, 20, 21, 22 )
    UPDATE_C3 ( 4, 24, 25, 26 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 (  8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 (  9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16 ( 24 )
    UPDATE_C_1X16 ( 25 )
    UPDATE_C_1X16 ( 26 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C3_BZ (  8,  9, 10 )
    UPDATE_C3_BZ ( 12, 13, 14 )
    UPDATE_C3_BZ ( 16, 17, 18 )
    UPDATE_C3_BZ ( 20, 21, 22 )
    UPDATE_C3_BZ ( 24, 25, 26 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ (  8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ (  9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 24 )
    UPDATE_C_1X16_BZ ( 25 )
    UPDATE_C_1X16_BZ ( 26 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_5x32
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE2 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )

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

    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax,  r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax,  r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )
    vbroadcastss ( mem ( rax,  r8, 4 ), zmm5 )
    VFMA2 ( 5, 24, 25 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2 ( 7,  8,  9 )
    ALPHA_SCALE2 ( 7, 12, 13 )
    ALPHA_SCALE2 ( 7, 16, 17 )
    ALPHA_SCALE2 ( 7, 20, 21 )
    ALPHA_SCALE2 ( 7, 24, 25 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C2 ( 4,  8,  9 )
    UPDATE_C2 ( 4, 12, 13 )
    UPDATE_C2 ( 4, 16, 17 )
    UPDATE_C2 ( 4, 20, 21 )
    UPDATE_C2 ( 4, 24, 25 )
    jmp ( .SDONE )

    label ( .SCOLSTORED )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 9, 13, 17, 21 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16 ( 24 )
    UPDATE_C_1X16 ( 25 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C2_BZ (  8,  9 )
    UPDATE_C2_BZ ( 12, 13 )
    UPDATE_C2_BZ ( 16, 17 )
    UPDATE_C2_BZ ( 20, 21 )
    UPDATE_C2_BZ ( 24, 25 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem (    , rdi, 4 ), rdi )          // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 24 )
    UPDATE_C_1X16_BZ ( 25 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_5x16
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )
    ALPHA_SCALE1 ( 7, 20 )
    ALPHA_SCALE1 ( 7, 24 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1 ( 4,  8 )
    UPDATE_C1 ( 4, 12 )
    UPDATE_C1 ( 4, 16 )
    UPDATE_C1 ( 4, 20 )
    UPDATE_C1 ( 4, 24 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 (  8, 12, 16, 20 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16 ( 24 )
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

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 4x16 tiles to 16x4 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 24 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_5x16_mask
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

    uint64_t n_left = n0 % 16;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )
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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )
    ALPHA_SCALE1 ( 7, 20 )
    ALPHA_SCALE1 ( 7, 24 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK ( 4,  8 )
    UPDATE_C1_MASK ( 4, 12 )
    UPDATE_C1_MASK ( 4, 16 )
    UPDATE_C1_MASK ( 4, 20 )
    UPDATE_C1_MASK ( 4, 24 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 5x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 5xA tile size, this is further split
    * into 4xA and 1xA tiles which are transposed
    * to Ax4 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                             // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_5X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                   // Compute entry address
    add ( rdx, rax )                                   // Convert offset to absolute address
    jmpi ( rax )                                       // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_5X16M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X16M_15 ) )

    label ( .SCOLSTORED_LEFT_5X16M_9 )
    TRANSPOSE_4X9 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X9 ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X16M_10 )
    TRANSPOSE_4X10 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X10 ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X16M_11 )
    TRANSPOSE_4X11 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X11 ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X16M_12 )
    TRANSPOSE_4X12 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X12 ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X16M_13 )
    TRANSPOSE_4X13 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X13 ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X16M_14 )
    TRANSPOSE_4X14 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X14 ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X16M_15 )
    TRANSPOSE_4X15 (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X15 ( 24 )
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
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 5x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 5xA tile size, this is further split
    * into 4xA and 1xA tiles which are transposed
    * to Ax4 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                                // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_5X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                   // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                      // Compute entry address
    add ( rdx, rax )                                      // Convert offset to absolute address
    jmpi ( rax )                                          // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_5X16M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X16M_15 ) )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_9 )
    TRANSPOSE_4X9_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X9_BZ ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_10 )
    TRANSPOSE_4X10_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X10_BZ ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_11 )
    TRANSPOSE_4X11_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X11_BZ ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_12 )
    TRANSPOSE_4X12_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X12_BZ ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_13 )
    TRANSPOSE_4X13_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X13_BZ ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_14 )
    TRANSPOSE_4X14_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X14_BZ ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X16M_15 )
    TRANSPOSE_4X15_BZ ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X15_BZ ( 24 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_5x8_mask
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

    // This function should only be called when n0 <= 8
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )
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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=8)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_YMM ( 7,  8 )
    ALPHA_SCALE1_YMM ( 7, 12 )
    ALPHA_SCALE1_YMM ( 7, 16 )
    ALPHA_SCALE1_YMM ( 7, 20 )
    ALPHA_SCALE1_YMM ( 7, 24 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), ymm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_YMM ( 4,  8 )
    UPDATE_C1_MASK_YMM ( 4, 12 )
    UPDATE_C1_MASK_YMM ( 4, 16 )
    UPDATE_C1_MASK_YMM ( 4, 20 )
    UPDATE_C1_MASK_YMM ( 4, 24 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 5x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 5xA tile size, this is further split
    * into 4xA and 1xA tiles which are transposed
    * to Ax4 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_5X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_5X8M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X8M_8 ) )

    label ( .SCOLSTORED_LEFT_5X8M_1 )
    TRANSPOSE_4X1_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_2 )
    TRANSPOSE_4X2_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_3 )
    TRANSPOSE_4X3_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_4 )
    TRANSPOSE_4X4_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_5 )
    TRANSPOSE_4X5_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X5_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_6 )
    TRANSPOSE_4X6_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X6_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_7 )
    TRANSPOSE_4X7_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X7_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X8M_8 )
    TRANSPOSE_4X8_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X8_YMM ( 24 )
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
    jmp ( .SDONE )

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 5x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 5xA tile size, this is further split
    * into 4xA and 1xA tiles which are transposed
    * to Ax4 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                         // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_5X8M, rax ) // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                 // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                    // Compute entry address
    add ( rdx, rax )                                    // Convert offset to absolute address
    jmpi ( rax )                                        // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_5X8M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X8M_8 ) )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_1 )
    TRANSPOSE_4X1_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_2 )
    TRANSPOSE_4X2_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_3 )
    TRANSPOSE_4X3_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_4 )
    TRANSPOSE_4X4_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_5 )
    TRANSPOSE_4X5_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X5_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_6 )
    TRANSPOSE_4X6_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X6_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_7 )
    TRANSPOSE_4X7_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X7_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X8M_8 )
    TRANSPOSE_4X8_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X8_BZ_YMM ( 24 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_4x8_mask
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

    // In this function, n0 <= 8
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=8)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_YMM ( 7,  8 )
    ALPHA_SCALE1_YMM ( 7, 12 )
    ALPHA_SCALE1_YMM ( 7, 16 )
    ALPHA_SCALE1_YMM ( 7, 20 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), ymm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_YMM ( 4,  8 )
    UPDATE_C1_MASK_YMM ( 4, 12 )
    UPDATE_C1_MASK_YMM ( 4, 16 )
    UPDATE_C1_MASK_YMM ( 4, 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 4x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 4xA tile size, is directly transposed to Ax4
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_4X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_4X8M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X8M_8 ) )

    label ( .SCOLSTORED_LEFT_4X8M_1 )
    TRANSPOSE_4X1_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_2 )
    TRANSPOSE_4X2_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_3 )
    TRANSPOSE_4X3_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_4 )
    TRANSPOSE_4X4_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_5 )
    TRANSPOSE_4X5_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_6 )
    TRANSPOSE_4X6_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_7 )
    TRANSPOSE_4X7_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X8M_8 )
    TRANSPOSE_4X8_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_YMM ( 8 )
    UPDATE_C1_BZ_MASK_YMM ( 12 )
    UPDATE_C1_BZ_MASK_YMM ( 16 )
    UPDATE_C1_BZ_MASK_YMM ( 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 4x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 4xA tile size, is directly transposed to Ax4
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_4X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_4X8M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X8M_8 ) )

    label ( .SCOLSTORED_BZ_LEFT_4X8M_1 )
    TRANSPOSE_4X1_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_2 )
    TRANSPOSE_4X2_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_3 )
    TRANSPOSE_4X3_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_4 )
    TRANSPOSE_4X4_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_5 )
    TRANSPOSE_4X5_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_6 )
    TRANSPOSE_4X6_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_7 )
    TRANSPOSE_4X7_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X8M_8 )
    TRANSPOSE_4X8_BZ_YMM ( 8, 12, 16, 20 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_4x4_mask
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

    // In this function, n0 <= 4
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=4)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_XMM ( 7,  8 )
    ALPHA_SCALE1_XMM ( 7, 12 )
    ALPHA_SCALE1_XMM ( 7, 16 )
    ALPHA_SCALE1_XMM ( 7, 20 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), xmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_XMM ( 4,  8 )
    UPDATE_C1_MASK_XMM ( 4, 12 )
    UPDATE_C1_MASK_XMM ( 4, 16 )
    UPDATE_C1_MASK_XMM ( 4, 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 4x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 4xA tile size, is directly transposed to Ax4
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                      // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_4X4M, rax ) // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )              // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                 // Compute entry address
    add ( rdx, rax )                                 // Convert offset to absolute address
    jmpi ( rax )                                     // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_4X4M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X4M_4 ) )

    label ( .SCOLSTORED_LEFT_4X4M_1 )
    TRANSPOSE_4X1_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X4M_2 )
    TRANSPOSE_4X2_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X4M_3 )
    TRANSPOSE_4X3_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X4M_4 )
    TRANSPOSE_4X4_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_XMM ( 8 )
    UPDATE_C1_BZ_MASK_XMM ( 12 )
    UPDATE_C1_BZ_MASK_XMM ( 16 )
    UPDATE_C1_BZ_MASK_XMM ( 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 4x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 4xA tile size, is directly transposed to Ax4
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_4X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_4X4M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X4M_4 ) )

    label ( .SCOLSTORED_BZ_LEFT_4X4M_1 )
    TRANSPOSE_4X1_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X4M_2 )
    TRANSPOSE_4X2_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X4M_3 )
    TRANSPOSE_4X3_BZ_YMM ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X4M_4 )
    TRANSPOSE_4X4_BZ_YMM ( 8, 12, 16, 20 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_5x4_mask
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

    // This function should only be called when n0 <= 4
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )
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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=4)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,5), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_XMM ( 7,  8 )
    ALPHA_SCALE1_XMM ( 7, 12 )
    ALPHA_SCALE1_XMM ( 7, 16 )
    ALPHA_SCALE1_XMM ( 7, 20 )
    ALPHA_SCALE1_XMM ( 7, 24 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), xmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_XMM ( 4,  8 )
    UPDATE_C1_MASK_XMM ( 4, 12 )
    UPDATE_C1_MASK_XMM ( 4, 16 )
    UPDATE_C1_MASK_XMM ( 4, 20 )
    UPDATE_C1_MASK_XMM ( 4, 24 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 5x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 5xA tile size, this is further split
    * into 4xA and 1xA tiles which are transposed
    * to Ax4 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_5X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_5X4M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_5X4M_4 ) )

    label ( .SCOLSTORED_LEFT_5X4M_1 )
    TRANSPOSE_4X1_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X4M_2 )
    TRANSPOSE_4X2_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X4M_3 )
    TRANSPOSE_4X3_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_5X4M_4 )
    TRANSPOSE_4X4_YMM (  8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_YMM ( 24 )
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
    jmp ( .SDONE )

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 5x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 5xA tile size, this is further split
    * into 4xA and 1xA tiles which are transposed
    * to Ax4 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_5X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_5X4M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_5X4M_4 ) )

    label ( .SCOLSTORED_BZ_LEFT_5X4M_1 )
    TRANSPOSE_4X1_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X4M_2 )
    TRANSPOSE_4X2_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X4M_3 )
    TRANSPOSE_4X3_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_BZ_YMM ( 24 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_5X4M_4 )
    TRANSPOSE_4X4_BZ_YMM ( 8, 12, 16, 20 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 4 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_BZ_YMM ( 24 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_3x48
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3 ( 7,  8,  9, 10 )
    ALPHA_SCALE3 ( 7, 12, 13, 14 )
    ALPHA_SCALE3 ( 7, 16, 17, 18 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C3 ( 4,  8,  9, 10 )
    UPDATE_C3 ( 4, 12, 13, 14 )
    UPDATE_C3 ( 4, 16, 17, 18 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 (  8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 (  9, 13 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 10, 14 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16 ( 16 )
    UPDATE_C_1X16 ( 17 )
    UPDATE_C_1X16 ( 18 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C3_BZ (  8,  9, 10 )
    UPDATE_C3_BZ ( 12, 13, 14 )
    UPDATE_C3_BZ ( 16, 17, 18 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ (  8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ (  9, 13 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 10, 14 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 16 )
    UPDATE_C_1X16_BZ ( 17 )
    UPDATE_C_1X16_BZ ( 18 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_3x32
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE2 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )

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

    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2 ( 7,  8,  9 )
    ALPHA_SCALE2 ( 7, 12, 13 )
    ALPHA_SCALE2 ( 7, 16, 17 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C2 ( 4,  8,  9 )
    UPDATE_C2 ( 4, 12, 13 )
    UPDATE_C2 ( 4, 16, 17 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 ( 8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 9, 13 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16 ( 16 )
    UPDATE_C_1X16 ( 17 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C2_BZ (  8,  9 )
    UPDATE_C2_BZ ( 12, 13 )
    UPDATE_C2_BZ ( 16, 17 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ ( 8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 9, 13 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 16 )
    UPDATE_C_1X16_BZ ( 17 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_3x16
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1 ( 4,  8 )
    UPDATE_C1 ( 4, 12 )
    UPDATE_C1 ( 4, 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 ( 8, 12 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16 ( 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ (  8 )
    UPDATE_C1_BZ ( 12 )
    UPDATE_C1_BZ ( 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ ( 8, 12 )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 16 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_3x16_mask
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

    uint64_t n_left = n0 % 16;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )
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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )         // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK ( 4,  8 )
    UPDATE_C1_MASK ( 4, 12 )
    UPDATE_C1_MASK ( 4, 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 3x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 3xA tile size, this is further split
    * into 2xA and 1xA tiles which are transposed
    * to Ax2 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                             // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_3X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                   // Compute entry address
    add ( rdx, rax )                                   // Convert offset to absolute address
    jmpi ( rax )                                       // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_3X16M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X16M_15 ) )

    label ( .SCOLSTORED_LEFT_3X16M_9 )
    TRANSPOSE_2X9 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X9 ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X16M_10 )
    TRANSPOSE_2X10 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X10 ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X16M_11 )
    TRANSPOSE_2X11 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X11 ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X16M_12 )
    TRANSPOSE_2X12 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X12 ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X16M_13 )
    TRANSPOSE_2X13 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X13 ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X16M_14 )
    TRANSPOSE_2X14 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X14 ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X16M_15 )
    TRANSPOSE_2X15 ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X15 ( 16 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK (  8 )
    UPDATE_C1_BZ_MASK ( 12 )
    UPDATE_C1_BZ_MASK ( 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 3x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 3xA tile size, this is further split
    * into 2xA and 1xA tiles which are transposed
    * to Ax2 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                                // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_3X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                   // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                      // Compute entry address
    add ( rdx, rax )                                      // Convert offset to absolute address
    jmpi ( rax )                                          // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_3X16M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X16M_15 ) )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_9 )
    TRANSPOSE_2X9_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X9_BZ ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_10 )
    TRANSPOSE_2X10_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X10_BZ ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_11 )
    TRANSPOSE_2X11_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X11_BZ ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_12 )
    TRANSPOSE_2X12_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X12_BZ ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_13 )
    TRANSPOSE_2X13_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X13_BZ ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_14 )
    TRANSPOSE_2X14_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X14_BZ ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X16M_15 )
    TRANSPOSE_2X15_BZ ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X15_BZ ( 16 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_3x8_mask
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

    // This function should only be called when n0 <= 8
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )
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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=8)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )        // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_YMM ( 7,  8 )
    ALPHA_SCALE1_YMM ( 7, 12 )
    ALPHA_SCALE1_YMM ( 7, 16 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), ymm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_YMM ( 4,  8 )
    UPDATE_C1_MASK_YMM ( 4, 12 )
    UPDATE_C1_MASK_YMM ( 4, 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 3x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 3xA tile size, this is further split
    * into 2xA and 1xA tiles which are transposed
    * to Ax2 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_3X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_3X8M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X8M_8 ) )

    label ( .SCOLSTORED_LEFT_3X8M_1 )
    TRANSPOSE_2X1_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_2 )
    TRANSPOSE_2X2_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_3 )
    TRANSPOSE_2X3_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_4 )
    TRANSPOSE_2X4_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_5 )
    TRANSPOSE_2X5_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X5_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_6 )
    TRANSPOSE_2X6_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X6_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_7 )
    TRANSPOSE_2X7_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X7_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X8M_8 )
    TRANSPOSE_2X8_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X8_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_YMM (  8 )
    UPDATE_C1_BZ_MASK_YMM ( 12 )
    UPDATE_C1_BZ_MASK_YMM ( 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 3x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 3xA tile size, this is further split
    * into 2xA and 1xA tiles which are transposed
    * to Ax2 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_3X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_3X8M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X8M_8 ) )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_1 )
    TRANSPOSE_2X1_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_2 )
    TRANSPOSE_2X2_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_3 )
    TRANSPOSE_2X3_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_4 )
    TRANSPOSE_2X4_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_5 )
    TRANSPOSE_2X5_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X5_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_6 )
    TRANSPOSE_2X6_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X6_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_7 )
    TRANSPOSE_2X7_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X7_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X8M_8 )
    TRANSPOSE_2X8_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X8_BZ_YMM ( 16 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_3x4_mask
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

    // This function should only be called when n0 <= 4
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    /*Produce MRXNR outputs */
    // -------------------------------------------------------------------------
    begin_asm()

    mov ( var ( n_load_mask ), esi )
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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=4)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )         // ymm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,3), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_XMM ( 7,  8 )
    ALPHA_SCALE1_XMM ( 7, 12 )
    ALPHA_SCALE1_XMM ( 7, 16 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), xmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_XMM ( 4,  8 )
    UPDATE_C1_MASK_XMM ( 4, 12 )
    UPDATE_C1_MASK_XMM ( 4, 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 3x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 3xA tile size, this is further split
    * into 2xA and 1xA tiles which are transposed
    * to Ax2 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_3X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_3X4M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_3X4M_4 ) )

    label ( .SCOLSTORED_LEFT_3X4M_1 )
    TRANSPOSE_2X1_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X4M_2 )
    TRANSPOSE_2X2_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X4M_3 )
    TRANSPOSE_2X3_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_3X4M_4 )
    TRANSPOSE_2X4_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_XMM (  8 )
    UPDATE_C1_BZ_MASK_XMM ( 12 )
    UPDATE_C1_BZ_MASK_XMM ( 16 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 3x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, for every 3xA tile size, this is further split
    * into 2xA and 1xA tiles which are transposed
    * to Ax2 and Ax1 tiles respectively.
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_3X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_3X4M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_3X4M_4 ) )

    label ( .SCOLSTORED_BZ_LEFT_3X4M_1 )
    TRANSPOSE_2X1_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X1_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X4M_2 )
    TRANSPOSE_2X2_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X2_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X4M_3 )
    TRANSPOSE_2X3_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X3_BZ_YMM ( 16 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_BZ_LEFT_3X4M_4 )
    TRANSPOSE_2X4_BZ_YMM ( 8, 12 )
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( rs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rcx, rdi, 2 ), rcx )
    mov ( var ( cs_c ), rdi )                 // load cs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = cs_c *= sizeof ( dt ) => cs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )
    UPDATE_C_1X4_BZ_YMM ( 16 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_2x8_mask
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

    // n <= 8 in this function
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=8)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=8)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_YMM ( 7,  8 )
    ALPHA_SCALE1_YMM ( 7, 12 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), ymm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_YMM ( 4,  8 )
    UPDATE_C1_MASK_YMM ( 4, 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 2x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 2xA tile size, is directly transposed to Ax2
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_2X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )		          // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )						          // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_2X8M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X8M_8 ) )

    label ( .SCOLSTORED_LEFT_2X8M_1 )
    TRANSPOSE_2X1_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_2 )
    TRANSPOSE_2X2_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_3 )
    TRANSPOSE_2X3_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_4 )
    TRANSPOSE_2X4_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_5 )
    TRANSPOSE_2X5_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_6 )
    TRANSPOSE_2X6_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_7 )
    TRANSPOSE_2X7_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X8M_8 )
    TRANSPOSE_2X8_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_YMM (  8 )
    UPDATE_C1_BZ_MASK_YMM ( 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 2x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 2xA tile size, is directly transposed to Ax2
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_2X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_2X8M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X8M_8 ) )

    label ( .SCOLSTORED_BZ_LEFT_2X8M_1 )
    TRANSPOSE_2X1_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_2 )
    TRANSPOSE_2X2_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_3 )
    TRANSPOSE_2X3_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_4 )
    TRANSPOSE_2X4_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_5 )
    TRANSPOSE_2X5_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_6 )
    TRANSPOSE_2X6_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_7 )
    TRANSPOSE_2X7_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X8M_8 )
    TRANSPOSE_2X8_BZ_YMM ( 8, 12 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_2x4_mask
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

    // n <= 4 in this function
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=4)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_XMM ( 7,  8 )
    ALPHA_SCALE1_XMM ( 7, 12 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), xmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_XMM ( 4,  8 )
    UPDATE_C1_MASK_XMM ( 4, 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 2x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 2xA tile size, is directly transposed to Ax2
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_2X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_2X4M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X4M_4 ) )

    label ( .SCOLSTORED_LEFT_2X4M_1 )
    TRANSPOSE_2X1_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X4M_2 )
    TRANSPOSE_2X2_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X4M_3 )
    TRANSPOSE_2X3_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X4M_4 )
    TRANSPOSE_2X4_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_XMM (  8 )
    UPDATE_C1_BZ_MASK_XMM ( 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 2x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 2xA tile size, is directly transposed to Ax2
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_2X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_2X4M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X4M_4 ) )

    label ( .SCOLSTORED_BZ_LEFT_2X4M_1 )
    TRANSPOSE_2X1_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X4M_2 )
    TRANSPOSE_2X2_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X4M_3 )
    TRANSPOSE_2X3_BZ_YMM ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X4M_4 )
    TRANSPOSE_2X4_BZ_YMM ( 8, 12 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_1x8_mask
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

    // In this function n0 <= 8
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=1)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=1)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=1)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=1)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), YMM ( 0 MASK_KZ ( 1 ) ) )     // ymm0 <- B[k, <=8)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=1)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_YMM ( 7, 8 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), ymm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_YMM ( 4, 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 1x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 1xA tile size, is directly transposed to Ax1
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_1X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_1X8M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X8M_8 ) )

    label ( .SCOLSTORED_LEFT_1X8M_1 )
    UPDATE_C_1X1_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_2 )
    UPDATE_C_1X2_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_3 )
    UPDATE_C_1X3_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_4 )
    UPDATE_C_1X4_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_5 )
    UPDATE_C_1X5_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_6 )
    UPDATE_C_1X6_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_7 )
    UPDATE_C_1X7_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X8M_8 )
    UPDATE_C_1X8_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_YMM ( 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 1x[1-8] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 1xA tile size, is directly transposed to Ax1
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-8 to 0-7 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_1X8M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_1X8M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_4 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_5 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_6 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_7 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X8M_8 ) )

    label ( .SCOLSTORED_BZ_LEFT_1X8M_1 )
    UPDATE_C_1X1_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_2 )
    UPDATE_C_1X2_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_3 )
    UPDATE_C_1X3_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_4 )
    UPDATE_C_1X4_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_5 )
    UPDATE_C_1X5_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_6 )
    UPDATE_C_1X6_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_7 )
    UPDATE_C_1X7_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X8M_8 )
    UPDATE_C_1X8_BZ_YMM ( 8 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_1x4_mask
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

    // In this function n0 <= 4
    uint64_t n_left = n0;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=4)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), XMM ( 0 MASK_KZ ( 1 ) ) )     // xmm0 <- B[k, <=4)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=4)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1_XMM ( 7, 8 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), xmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK_XMM ( 4, 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 1x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 1xA tile size, is directly transposed to Ax1
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                       // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_1X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )               // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                  // Compute entry address
    add ( rdx, rax )                                  // Convert offset to absolute address
    jmpi ( rax )                                      // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_1X4M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X4M_4 ) )

    label ( .SCOLSTORED_LEFT_1X4M_1 )
    UPDATE_C_1X1_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X4M_2 )
    UPDATE_C_1X2_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X4M_3 )
    UPDATE_C_1X3_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X4M_4 )
    UPDATE_C_1X4_YMM ( 8 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK_XMM ( 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 1x[1-4] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 1xA tile size, is directly transposed to Ax1
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    dec ( rsi )                                          // Convert 1-4 to 0-3 as the jump table is 0 indexed
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_1X4M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                  // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                     // Compute entry address
    add ( rdx, rax )                                     // Convert offset to absolute address
    jmpi ( rax )                                         // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_1X4M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X4M_1 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X4M_2 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X4M_3 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X4M_4 ) )

    label ( .SCOLSTORED_BZ_LEFT_1X4M_1 )
    UPDATE_C_1X1_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X4M_2 )
    UPDATE_C_1X2_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X4M_3 )
    UPDATE_C_1X3_BZ_YMM ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X4M_4 )
    UPDATE_C_1X4_BZ_YMM ( 8 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
      "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
      "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_4x64
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    dec ( rsi )
    jne ( .K_LOOP_ITER )                      // if rsi != 0, repeat k-loop

    label ( .CONSID_K_LEFT )

    mov ( var ( k_left ), rsi )               // i = k_left;
    test ( rsi, rsi )                         // check i via logical AND.
    je ( .SPOSTACCUM )                        // if i == 0, we're done; jump to end.
                                  // else, we prepare to enter k_left loop.
    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LEFT_LOOP )

    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA4 ( 6, 16, 17, 18, 19 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA4 ( 4, 20, 21, 22, 23 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4 ( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4 ( 7, 12, 13, 14, 15 )
    ALPHA_SCALE4 ( 7, 16, 17, 18, 19 )
    ALPHA_SCALE4 ( 7, 20, 21, 22, 23 )

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

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

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
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 (  8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 (  9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 11, 15, 19, 23 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C4_BZ (  8,  9, 10, 11 )
    UPDATE_C4_BZ ( 12, 13, 14, 15 )
    UPDATE_C4_BZ ( 16, 17, 18, 19 )
    UPDATE_C4_BZ ( 20, 21, 22, 23 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 11, 15, 19, 23 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_4x48
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA3 ( 6, 16, 17, 18 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA3 ( 4, 20, 21, 22 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3 ( 7,  8,  9, 10 )
    ALPHA_SCALE3 ( 7, 12, 13, 14 )
    ALPHA_SCALE3 ( 7, 16, 17, 18 )
    ALPHA_SCALE3 ( 7, 20, 21, 22 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C3 ( 4, 8, 9, 10 )
    UPDATE_C3 ( 4, 12, 13, 14 )
    UPDATE_C3 ( 4, 16, 17, 18 )
    UPDATE_C3 ( 4, 20, 21, 22 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

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
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 10, 14, 18, 22 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C3_BZ ( 8, 9, 10 )
    UPDATE_C3_BZ ( 12, 13, 14 )
    UPDATE_C3_BZ ( 16, 17, 18 )
    UPDATE_C3_BZ ( 20, 21, 22 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 10, 14, 18, 22 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_1x16_mask
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

    uint64_t n_left = n0 % 16;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=16)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7, 8 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK ( 4, 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 1x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 1xA tile size, is directly transposed to Ax1
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                             // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_1X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                   // Compute entry address
    add ( rdx, rax )                                   // Convert offset to absolute address
    jmpi ( rax )                                       // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_1X16M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_1X16M_15 ) )

    label ( .SCOLSTORED_LEFT_1X16M_9 )
    UPDATE_C_1X9 ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X16M_10 )
    UPDATE_C_1X10 ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X16M_11 )
    UPDATE_C_1X11 ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X16M_12 )
    UPDATE_C_1X12 ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X16M_13 )
    UPDATE_C_1X13 ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X16M_14 )
    UPDATE_C_1X14 ( 8 )
    jmp ( .SDONE )

    label ( .SCOLSTORED_LEFT_1X16M_15 )
    UPDATE_C_1X15 ( 8 )
    jmp ( .SDONE )

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK ( 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 1x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 1xA tile size, is directly transposed to Ax1
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                                // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_1X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                   // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                      // Compute entry address
    add ( rdx, rax )                                      // Convert offset to absolute address
    jmpi ( rax )                                          // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_1X16M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_1X16M_15 ) )

    label ( .SCOLSTORED_BZ_LEFT_1X16M_9 )
    UPDATE_C_1X9_BZ ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X16M_10 )
    UPDATE_C_1X10_BZ ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X16M_11 )
    UPDATE_C_1X11_BZ ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X16M_12 )
    UPDATE_C_1X12_BZ ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X16M_13 )
    UPDATE_C_1X13_BZ ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X16M_14 )
    UPDATE_C_1X14_BZ ( 8 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_1X16M_15 )
    UPDATE_C_1X15_BZ ( 8 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_1x16
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7, 8 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1 ( 4, 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16 ( 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ ( 8 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 8 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_1x32
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE2 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )

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

    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2 ( 7, 8, 9 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C2 ( 4, 8, 9 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16 ( 8 )
    UPDATE_C_1X16 ( 9 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C2_BZ ( 8, 9 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ ( 8 )
    UPDATE_C_1X16_BZ ( 9 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_1x48
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4, 8, 9, 10 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3 ( 7, 8, 9, 10 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C3 ( 4, 8, 9, 10 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16 (  8 )
    UPDATE_C_1X16 (  9 )
    UPDATE_C_1X16 ( 10 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C3_BZ ( 8, 9, 10 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ (  8 )
    UPDATE_C_1X16_BZ (  9 )
    UPDATE_C_1X16_BZ ( 10 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_1x64
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4, 8, 9, 10, 11 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4, 8, 9, 10, 11 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4, 8, 9, 10, 11 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4, 8, 9, 10, 11 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,1), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4, 8, 9, 10, 11 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4 ( 7, 8, 9, 10, 11 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C4 ( 4, 8, 9, 10, 11 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    UPDATE_C_1X16 (  8 )
    UPDATE_C_1X16 (  9 )
    UPDATE_C_1X16 ( 10 )
    UPDATE_C_1X16 ( 11 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C4_BZ ( 8, 9, 10, 11 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )

    /* Transposing 1x16 tiles to 16x1 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )

    UPDATE_C_1X16_BZ (  8 )
    UPDATE_C_1X16_BZ (  9 )
    UPDATE_C_1X16_BZ ( 10 )
    UPDATE_C_1X16_BZ ( 11 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_2x16_mask
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

    uint64_t n_left = n0 % 16;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=16)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK ( 4,  8 )
    UPDATE_C1_MASK ( 4, 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 2x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 2xA tile size, is directly transposed to Ax2
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                             // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_2X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                   // Compute entry address
    add ( rdx, rax )                                   // Convert offset to absolute address
    jmpi ( rax )                                       // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_2X16M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_2X16M_15 ) )

    label ( .SCOLSTORED_LEFT_2X16M_9 )
    TRANSPOSE_2X9 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X16M_10 )
    TRANSPOSE_2X10 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X16M_11 )
    TRANSPOSE_2X11 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X16M_12 )
    TRANSPOSE_2X12 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X16M_13 )
    TRANSPOSE_2X13 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X16M_14 )
    TRANSPOSE_2X14 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_2X16M_15 )
    TRANSPOSE_2X15 ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK (  8 )
    UPDATE_C1_BZ_MASK ( 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 2x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 2xA tile size, is directly transposed to Ax2
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                                // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_2X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                   // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                      // Compute entry address
    add ( rdx, rax )                                      // Convert offset to absolute address
    jmpi ( rax )                                          // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_2X16M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_2X16M_15 ) )

    label ( .SCOLSTORED_BZ_LEFT_2X16M_9 )
    TRANSPOSE_2X9_BZ ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X16M_10 )
    TRANSPOSE_2X10_BZ ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X16M_11 )
    TRANSPOSE_2X11_BZ ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X16M_12 )
    TRANSPOSE_2X12_BZ ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X16M_13 )
    TRANSPOSE_2X13_BZ ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X16M_14 )
    TRANSPOSE_2X14_BZ ( 8, 12 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_2X16M_15 )
    TRANSPOSE_2X15_BZ ( 8, 12 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_2x16
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1 ( 4,  8 )
    UPDATE_C1 ( 4, 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 ( 8, 12 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ (  8 )
    UPDATE_C1_BZ ( 12 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ ( 8, 12 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_2x32
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE2 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )

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

    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4,  8,  9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2 ( 7,  8,  9 )
    ALPHA_SCALE2 ( 7, 12, 13 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C2 ( 4,  8,  9 )
    UPDATE_C2 ( 4, 12, 13 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 ( 8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 9, 13 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C2_BZ (  8,  9 )
    UPDATE_C2_BZ ( 12, 13 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ ( 8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 9, 13 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_2x48
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 3 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,48)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA3 ( 4,  8,  9, 10 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA3 ( 5, 12, 13, 14 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE3 ( 7,  8,  9, 10 )
    ALPHA_SCALE3 ( 7, 12, 13, 14 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C3 ( 4,  8,  9, 10 )
    UPDATE_C3 ( 4, 12, 13, 14 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 (  8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 (  9, 13 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 10, 14 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C3_BZ (  8, 9, 10 )
    UPDATE_C3_BZ ( 12, 13, 14 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ (  8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ (  9, 13 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 10, 14 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_2x64
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 4 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)
    vmovups ( 0x80 ( rbx ), zmm2 )            // zmm2 <- B[k, 32:48)
    vmovups ( 0xc0 ( rbx ), zmm3 )            // zmm3 <- B[k, 48:64)

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )

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

    // Outer product: compute A[i,k] * B[k,j], i:[0,2), j:[0,64)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA4 ( 4,  8,  9, 10, 11 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA4 ( 5, 12, 13, 14, 15 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE4 ( 7,  8,  9, 10, 11 )
    ALPHA_SCALE4 ( 7, 12, 13, 14, 15 )

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

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

    /**
     * 6x64 tile is split into 4 equal 6x16 tiles.
     * Each of these 6x16 tiles is further split into two tiles of
     * 4x16 & 2x16 each.
     * These smaller 4x16 & 2x16 tiles are transposed to 16x4 & 16x2 tiles,
     * to get the transpose of 6x64 tile and are stored as 64x6 tile.
     */
    /* Transposing 2x16 tiles to 16x2 tiles */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_2X16 (  8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 (  9, 13 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 10, 14 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16 ( 11, 15 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C4_BZ (  8,  9, 10, 11 )
    UPDATE_C4_BZ ( 12, 13, 14, 15 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_2X16_BZ (  8, 12 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ (  9, 13 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 10, 14 )
    lea ( mem ( rcx, rdi, 2 ), rcx )
    TRANSPOSE_2X16_BZ ( 11, 15 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
}

void bli_sgemmsup_rv_zen4_asm_4x16_mask
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

    uint64_t n_left = n0 % 16;
    int32_t n_load_mask = ( 1 << n_left ) - 1;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=16)
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

    // Load 1 row from B matrix.
    vmovups ( mem ( rbx ), ZMM ( 0 MASK_KZ ( 1 ) ) )     // zmm0 <- B[k, <=16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,<=16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )
    ALPHA_SCALE1 ( 7, 20 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1_MASK ( 4,  8 )
    UPDATE_C1_MASK ( 4, 12 )
    UPDATE_C1_MASK ( 4, 16 )
    UPDATE_C1_MASK ( 4, 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 4x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 4xA tile size, is directly transposed to Ax4
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                             // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_LEFT_4X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                   // Compute entry address
    add ( rdx, rax )                                   // Convert offset to absolute address
    jmpi ( rax )                                       // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_LEFT_4X16M,
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_LEFT_4X16M_15 ) )

    label ( .SCOLSTORED_LEFT_4X16M_9 )
    TRANSPOSE_4X9 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X16M_10 )
    TRANSPOSE_4X10 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X16M_11 )
    TRANSPOSE_4X11 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X16M_12 )
    TRANSPOSE_4X12 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X16M_13 )
    TRANSPOSE_4X13 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X16M_14 )
    TRANSPOSE_4X14 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_LEFT_4X16M_15 )
    TRANSPOSE_4X15 ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ_MASK ( 8 )
    UPDATE_C1_BZ_MASK ( 12 )
    UPDATE_C1_BZ_MASK ( 16 )
    UPDATE_C1_BZ_MASK ( 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORBZ )
    /*
    * For the masked case the transpose need to handle cases
    * where the tile can be of the range 4x[9-15] ( kernel range )
    * The most efficient way to handle this was to create code
    * snippets handling the various cases for n_left and a jump table
    * is used to avoid unnecessary branching using if-else statements
    *
    * Here, every 4xA tile size, is directly transposed to Ax4
    */
    mov ( var ( cbuf ), rcx )                 // load address of c
    mov ( var ( cs_c ), rdi )                 // load rs_c
    lea ( mem ( , rdi, 4 ), rdi )             // rs_c *= sizeof ( float )
    lea ( mem ( rdi, rdi, 2 ), r12 )
    mov ( var ( n_left ), rsi )

    sub ( imm ( 9 ), rsi )                                // convert n_left ( >= 9 ) to [0, 6] to index into the jump-table
    lea_rip ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_4X16M, rax )  // Load table base (RIP-relative)
    movslq ( mem ( rax, rsi, 4 ), rdx )                   // Load 4-byte relative offset
    lea ( mem ( rax, rsi, 4 ), rax )                      // Compute entry address
    add ( rdx, rax )                                      // Convert offset to absolute address
    jmpi ( rax )                                          // Indirect jump

    jump_table ( LJUMPTABLE_SCOLSTORED_BZ_LEFT_4X16M,
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_9 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_10 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_11 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_12 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_13 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_14 )
               TABLE_ENTRY ( SCOLSTORED_BZ_LEFT_4X16M_15 ) )

    label ( .SCOLSTORED_BZ_LEFT_4X16M_9 )
    TRANSPOSE_4X9_BZ ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X16M_10 )
    TRANSPOSE_4X10_BZ ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X16M_11 )
    TRANSPOSE_4X11_BZ ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X16M_12 )
    TRANSPOSE_4X12_BZ ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X16M_13 )
    TRANSPOSE_4X13_BZ ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X16M_14 )
    TRANSPOSE_4X14_BZ ( 8, 12, 16, 20 )
    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED_BZ_LEFT_4X16M_15 )
    TRANSPOSE_4X15_BZ ( 8, 12, 16, 20 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf ),
      [n_load_mask]   "m" ( n_load_mask ),
      [n_left]   "m" ( n_left )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory", "k1"
    )
}

void bli_sgemmsup_rv_zen4_asm_4x16
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,16)
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

    // Load 1 row from B matrix.
    vmovups ( ( rbx ), zmm0 )                  // zmm0 <- B[k, 0:16)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,16)
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

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE1 ( 7,  8 )
    ALPHA_SCALE1 ( 7, 12 )
    ALPHA_SCALE1 ( 7, 16 )
    ALPHA_SCALE1 ( 7, 20 )

    mov ( var ( beta ), rdx )                 // load address of beta
    vbroadcastss ( ( rdx ), zmm4 )

    vxorps ( xmm1, xmm1, xmm1 )
    vucomiss ( xmm1, xmm4 )                   // check if beta = 0
    je ( .SBETAZERO )                         // jump to beta = 0 case

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4
    jz ( .SCOLSTORED )                        // jump to column storage case

    label ( .SROWSTORED )

    UPDATE_C1 ( 4,  8 )
    UPDATE_C1 ( 4, 12 )
    UPDATE_C1 ( 4, 16 )
    UPDATE_C1 ( 4, 20 )

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

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
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C1_BZ ( 8 )
    UPDATE_C1_BZ ( 12 )
    UPDATE_C1_BZ ( 16 )
    UPDATE_C1_BZ ( 20 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rv_zen4_asm_4x32
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

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

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

    INIT_REG                                  // zero out the SIMD registers

    mov ( var ( abuf ), rax )                 // load address of a
    mov ( var ( bbuf ), rbx )                 // load address of b
    mov ( var ( cbuf ), rcx )                 // load address of c

    mov ( var ( alpha ), rdx )                // load address of alpha
    vbroadcastss ( ( rdx ), zmm7 )            // broadcast alpha in zmm, which is later used in the ALPHA_SCALE2 macro

    mov ( var ( k_iter ), rsi )               // load k_iter
    test ( rsi, rsi )                         // if there are no full k iterations, jump to the code that handles edge cases
    je ( .CONSID_K_LEFT )

    // The k-loop iterates over 4 rows of B, and broadcasts from each row of A.
    label ( .K_LOOP_ITER )
    // ITER 0
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 1
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 2
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A

    // ITER 3
    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )

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

    // Load 2 rows from B matrix.
    vmovups (      ( rbx ), zmm0 )            // zmm0 <- B[k, 0:16)
    vmovups ( 0x40 ( rbx ), zmm1 )            // zmm1 <- B[k, 16:32)

    // Outer product: compute A[i,k] * B[k,j], i:[0,4), j:[0,32)
    // Broadcast(A[:,k]), load(B[k,:])
    vbroadcastss ( ( rax ), zmm4 )
    VFMA2 ( 4, 8, 9 )
    vbroadcastss ( mem ( rax, r8, 1 ), zmm5 )
    VFMA2 ( 5, 12, 13 )
    vbroadcastss ( mem ( rax, r8, 2 ), zmm6 )
    VFMA2 ( 6, 16, 17 )
    vbroadcastss ( mem ( rax, r13, 1 ), zmm4 )
    VFMA2 ( 4, 20, 21 )

    add (  r9, rbx )                          // advance rbx pointer to the next row of B
    add ( r10, rax )                          // advance rax pointer to the next column of A
    dec ( rsi )
    jne ( .K_LEFT_LOOP )                      // if rsi != 0, repeat k-loop

    label ( .SPOSTACCUM )

    // Scaling A * B with alpha.
    ALPHA_SCALE2 ( 7,  8,  9 )
    ALPHA_SCALE2 ( 7, 12, 13 )
    ALPHA_SCALE2 ( 7, 16, 17 )
    ALPHA_SCALE2 ( 7, 20, 21 )

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

    jmp ( .SDONE )                            // jump to the end

    label ( .SCOLSTORED )

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
    lea ( mem ( , rdi, 4 ), rdi )             // rdi = rs_c *= sizeof ( dt ) => rs_c *= 4
    lea ( mem ( rdi, rdi, 2 ), r12 )          // rdi += rdi * 2 => rdi = 3 * rs_c

    TRANSPOSE_4X16 ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16 ( 9, 13, 17, 21 )
    lea ( mem ( rcx, r12, 4 ), rcx )

    jmp ( .SDONE )                            // jump to the end

    label ( .SBETAZERO )

    cmp ( imm ( 4 ), rdi )                    // set ZF if ( 4*rs_c ) == 4.
    jz ( .SCOLSTORBZ )                        // jump to column storage case

    label ( .SROWSTORBZ )

    UPDATE_C2_BZ ( 8, 9 )
    UPDATE_C2_BZ ( 12, 13 )
    UPDATE_C2_BZ ( 16, 17 )
    UPDATE_C2_BZ ( 20, 21 )

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
    lea ( mem ( rdi, rdi, 2 ), r12 )

    TRANSPOSE_4X16_BZ ( 8, 12, 16, 20 )
    lea ( mem ( rcx, r12, 4 ), rcx )
    TRANSPOSE_4X16_BZ ( 9, 13, 17, 21 )

    label ( .SDONE )

    end_asm (
    :// output operands ( none )
    :// input operands
      [k_iter] "m" ( k_iter ),
      [k_left] "m" ( k_left ),
      [a]      "m" ( a ),
      [rs_a]   "m" ( rs_a ),
      [cs_a]   "m" ( cs_a ),
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
      [abuf]   "m" ( abuf ),
      [bbuf]   "m" ( bbuf ),
      [cbuf]   "m" ( cbuf )
    :// register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm1", "xmm4",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}
