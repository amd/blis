/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "bli_gemmsup_rd_zen4_asm_s6x64.h"

void bli_sgemmsup_rd_zen4_asm_5x64
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // rs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov(imm(0), r15)                    // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea(mem(   , r15, 1), rsi)          // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem( r8, r8, 2 ), r10 )        // r10 = 3 * rs_a
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_a

    INIT_REG

    mov( var( k_iter64 ), rsi )         // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load rows from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load columns from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )

    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), ZMM(3 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 4), ZMM(4 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 17, 18, 19, 29, 30 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), YMM(3 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 4), YMM(4 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA5(  8,  9, 10, 20, 21 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA5( 17, 18, 19, 29, 30 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    // The horizontal sum of each ZMM register has the result for a single
    // element of the C Matrix.
    // ZMM_TO_YMM adds the upper half of ZMM registers to the lower half of
    // the respective ZMM registers, thus having the result in the lower half of
    // ZMM registers which is equivalent to its respective YMM counterpart.
    // ymm = lo(zmm) + hi(zmm)
    // zmm8 = z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15
    // ymm0 = z8 z9 z10 z11 z12 z13 z14 z15
    // ymm8 = z0 z1  z2  z3  z4  z5  z6  z7
    // ymm0 = ymm0 + ymm8
    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    // Accumulates the results by horizontally adding the YMM registers,
    // and having the final result in xmm registers.
    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR                          // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )
    ZMM_TO_YMM( 21, 24, 27, 30,  8,  9, 10, 11 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 8, 9, 10, 11, 5 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR2                         // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ                       // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )
    ZMM_TO_YMM( 21, 24, 27, 30,  8,  9, 10, 11 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 8, 9, 10, 11, 5 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ2                      // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 6*rs_c

    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 6*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 64 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm23", "ymm24", "ymm26", "ymm27",
      "ymm29", "ymm30",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_4x64
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))

    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem( , r15, 1 ), rsi )         // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )    // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )   // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )

    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )

    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )

    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), ZMM(3 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA4(  8,  9, 10, 20 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA4( 11, 12, 13, 23 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA4( 14, 15, 16, 26 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA4( 17, 18, 19, 29 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), YMM(3 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA4(  8,  9, 10, 20 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA4( 11, 12, 13, 23 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA4( 14, 15, 16, 26 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA4( 17, 18, 19, 29 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )


    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )

    label( .POST_ACCUM_STOR )       // Accumulating & storing the results when beta != 0

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     // Scaling the result of A*B with alpha

    C_STOR       // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )

    ALPHA_SCALE     // Scaling the result of A*B with alpha

    C_STOR1

    jmp( .SDONE )

    label( .POST_ACCUM_STOR_BZ )  // Accumulating & storing the results when beta == 0

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     // Scaling the result of A*B with alpha

    C_STOR_BZ       // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE     // Scaling the result of A*B with alpha

    C_STOR_BZ1       // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )         // load rs_c
    lea( mem( , rdi, 4 ), rdi )     // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )         //
    lea( mem( r12, rdi, 4 ), r12 )         // c_ii = r12 += 3*rs_c

    lea( mem( r14, r8,  2 ), r14 )         //
    lea( mem( r14, r8,  4 ), r14 )         // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 64 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm23", "ymm26", "ymm29",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_3x64
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )    // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))

    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )  // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA3(  8,  9, 10 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA3( 11, 12, 13 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA3( 14, 15, 16 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA3( 17, 18, 19 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA3(  8,  9, 10 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA3( 11, 12, 13 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA3( 14, 15, 16 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA3( 17, 18, 19 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR                          // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                     // Scaling the result of A*B with alpha

    C_STOR_BZ                       // Storing result to C


    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 3*rs_c
    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 64 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_2x64
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

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

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )    // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )   // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 8, 9 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 11, 12 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA2( 14, 15 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 17, 18 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA2( 8, 9 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA2( 11, 12 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA2( 14, 15 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA2( 17, 18 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 11, 12,  4,  5,  7,  8 )
    ZMM_TO_YMM( 14, 15, 17, 18, 10, 11, 13, 14 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )

    ALPHA_SCALE2                // Scaling the result of A*B with alpha

    C_STOR2                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 11, 12,  4,  5,  7,  8 )
    ZMM_TO_YMM( 14, 15, 17, 18, 10, 11, 13, 14 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )

    ALPHA_SCALE2                // Scaling the result of A*B with alpha

    C_STOR_BZ2                  // Storing result to C

    label( .SDONE )

    add( imm(  4 ), r15 )
    cmp( imm( 64 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm17", "ymm18",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_1x64
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

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

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))

    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b


    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )

    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 8 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 11 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA1( 14 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 17 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA1( 8 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA1( 11 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA1( 14 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA1( 17 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM( 8, 11, 14, 17, 4, 7, 10, 13 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )

    ALPHA_SCALE1                // Scaling the result of A*B with alpha

    C_STOR1                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM( 8, 11, 14, 17, 4, 7, 10, 13 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )

    ALPHA_SCALE1                // Scaling the result of A*B with alpha

    C_STOR_BZ1                  // Storing result to C


    label( .SDONE )

    add( imm(  4 ), r15 )
    cmp( imm( 64 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8", "ymm10", "ymm11", "ymm13",
      "ymm14", "ymm17",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_5x48
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), ZMM(3 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 4), ZMM(4 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 17, 18, 19, 29, 30 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), YMM(3 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 4), YMM(4 MASK_KZ(1) ) )
    
    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA5( 17, 18, 19, 29, 30 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR                      // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )
    ZMM_TO_YMM( 21, 24, 27, 30,  8,  9, 10, 11 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 8, 9, 10, 11, 5 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR2                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ                   // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )
    ZMM_TO_YMM( 21, 24, 27, 30,  8,  9, 10, 11 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 8, 9, 10, 11, 5 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ2                  // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 3*rs_c
    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 48 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm23", "ymm24", "ymm26", "ymm27",
      "ymm29", "ymm30",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_4x48
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))

    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )

    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), ZMM(3 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA4(  8,  9, 10, 20 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA4( 11, 12, 13, 23 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA4( 14, 15, 16, 26 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA4( 17, 18, 19, 29 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), YMM(3 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA4(  8,  9, 10, 20 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA4( 11, 12, 13, 23 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA4( 14, 15, 16, 26 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA4( 17, 18, 19, 29 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE             // Scaling the result of A*B with alpha

    C_STOR                  // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )

    ALPHA_SCALE             // Scaling the result of A*B with alpha

    C_STOR1                 // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE             // Scaling the result of A*B with alpha

    C_STOR_BZ               // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE             // Scaling the result of A*B with alpha

    C_STOR_BZ1              // Storing result to C


    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 3*rs_c

    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 48 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm23", "ymm26", "ymm29",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_3x48
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )         // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA3(  8,  9, 10 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA3( 11, 12, 13 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA3( 14, 15, 16 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA3( 17, 18, 19 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA3(  8,  9, 10 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA3( 11, 12, 13 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA3( 14, 15, 16 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA3( 17, 18, 19 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR                      // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ                   // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )                 // load rs_c
    lea( mem( , rdi, 4 ), rdi )             // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )          // c_ii = r12 += 3*rs_c
    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )          // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 48 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_2x48
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

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

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )    // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )   // rdi = 5 * rs_b


    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 8, 9 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 11, 12 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA2( 14, 15 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 17, 18 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA2( 8, 9 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA2( 11, 12 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA2( 14, 15 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA2( 17, 18 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 11, 12,  4,  5,  7,  8 )
    ZMM_TO_YMM( 14, 15, 17, 18, 10, 11, 13, 14 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )

    ALPHA_SCALE2                // Scaling the result of A*B with alpha

    C_STOR2                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 11, 12,  4,  5,  7,  8 )
    ZMM_TO_YMM( 14, 15, 17, 18, 10, 11, 13, 14 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )

    ALPHA_SCALE2                // Scaling the result of A*B with alpha

    C_STOR_BZ2                  // Storing result to C


    label( .SDONE )

    add( imm(  4 ), r15 )
    cmp( imm( 48 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm17", "ymm18",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_1x48
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

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

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 8 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 11 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA1( 14 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 17 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA1( 8 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA1( 11 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA1( 14 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA1( 17 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM( 8, 11, 14, 17, 4, 7, 10, 13 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )

    ALPHA_SCALE1                // Scaling the result of A*B with alpha

    C_STOR1                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM( 8, 11, 14, 17, 4, 7, 10, 13 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )

    ALPHA_SCALE1                // Scaling the result of A*B with alpha

    C_STOR_BZ1                  // Storing result to C


    label( .SDONE )

    add( imm(  4 ), r15 )
    cmp( imm( 48 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8", "ymm10", "ymm11", "ymm13", "ymm14", "ymm17",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_5x32
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )    // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )   // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )

    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax, r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    vmovups( ( rax,  r8, 4 ), zmm4 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA5( 17, 18, 19, 29, 30 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), ZMM(3 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 4), ZMM(4 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA5( 17, 18, 19, 29, 30 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), YMM(3 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 4), YMM(4 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA5( 8, 9, 10, 20, 21 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA5( 11, 12, 13, 23, 24 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA5( 14, 15, 16, 26, 27 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA5( 17, 18, 19, 29, 30 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR                      // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )
    ZMM_TO_YMM( 21, 24, 27, 30,  8,  9, 10, 11 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 8, 9, 10, 11, 5 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR2                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ                   // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )
    ZMM_TO_YMM( 21, 24, 27, 30,  8,  9, 10, 11 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )
    ACCUM_YMM( 8, 9, 10, 11, 5 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ2                  // Storing result to C

    label( .SDONE )

    mov( var( rs_c ), rdi )                 // load rs_c
    lea( mem( , rdi, 4 ), rdi )             // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )          // c_ii = r12 += 3*rs_c
    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )          // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 32 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm21", "ymm23", "ymm24", "ymm26", "ymm27",
      "ymm29", "ymm30",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_4x32
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    vmovups( ( rax, r10, 1 ), zmm3 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA4(  8,  9, 10, 20 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA4( 11, 12, 13, 23 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA4( 14, 15, 16, 26 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA4( 17, 18, 19, 29 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), ZMM(3 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA4(  8,  9, 10, 20 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA4( 11, 12, 13, 23 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA4( 14, 15, 16, 26 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA4( 17, 18, 19, 29 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )
    vmovups( mem(rax, r10, 1), YMM(3 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA4(  8,  9, 10, 20 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA4( 11, 12, 13, 23 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA4( 14, 15, 16, 26 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA4( 17, 18, 19, 29 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR                      // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR1                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ                   // Storing result to C

    ZMM_TO_YMM( 20, 23, 26, 29,  4,  5,  6,  7 )

    ACCUM_YMM( 4, 5, 6, 7, 4 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ1                  // Storing result to C


    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 3*rs_c
    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 32 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "ymm20", "ymm23", "ymm26", "ymm29",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_3x32
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    float *abuf = a;
    float *bbuf = b;
    float *cbuf = c;

    begin_asm()

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    vmovups( ( rax,  r8, 2 ), zmm2 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA3(  8,  9, 10 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA3( 11, 12, 13 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA3( 14, 15, 16 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA3( 17, 18, 19 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), ZMM(2 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA3(  8,  9, 10 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA3( 11, 12, 13 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA3( 14, 15, 16 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA3( 17, 18, 19 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 2), YMM(2 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA3(  8,  9, 10 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA3( 11, 12, 13 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA3( 14, 15, 16 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA3( 17, 18, 19 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR                      // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 10, 11,  4,  5,  6,  7 )
    ZMM_TO_YMM( 12, 13, 14, 15,  8,  9, 10, 11 )
    ZMM_TO_YMM( 16, 17, 18, 19, 12, 13, 14, 15 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )
    ACCUM_YMM( 6, 9, 12, 15, 6 )

    ALPHA_SCALE                 // Scaling the result of A*B with alpha

    C_STOR_BZ                   // Storing result to C


    label( .SDONE )

    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r12, rdi, 2 ), r12 )
    lea( mem( r12, rdi, 4 ), r12 )      // c_ii = r12 += 3*rs_c
    lea( mem( r14, r8,  2 ), r14 )
    lea( mem( r14, r8,  4 ), r14 )      // a_ii = r14 += 3*rs_a

    add( imm(  4 ), r15 )
    cmp( imm( 32 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [m_iter]   "m" (m_iter),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_2x32
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

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

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))


    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b

    INIT_REG

    mov( var( k_iter64 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )


    

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    vmovups( ( rax,  r8, 1 ), zmm1 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA2( 8, 9 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA2( 11, 12 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA2( 14, 15 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA2( 17, 18 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), ZMM(1 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 8, 9 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 11, 12 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA2( 14, 15 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA2( 17, 18 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )
    vmovups(  mem(rax, r8, 1), YMM(1 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA2( 8, 9 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA2( 11, 12 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA2( 14, 15 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA2( 17, 18 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM(  8,  9, 11, 12,  4,  5,  7,  8 )
    ZMM_TO_YMM( 14, 15, 17, 18, 10, 11, 13, 14 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )

    ALPHA_SCALE2                // Scaling the result of A*B with alpha

    C_STOR2                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM(  8,  9, 11, 12,  4,  5,  7,  8 )
    ZMM_TO_YMM( 14, 15, 17, 18, 10, 11, 13, 14 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )
    ACCUM_YMM( 5, 8, 11, 14, 5 )

    ALPHA_SCALE2                // Scaling the result of A*B with alpha

    C_STOR_BZ2                  // Storing result to C


    label( .SDONE )

    add( imm(  4 ), r15 )
    cmp( imm( 32 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm5", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
      "ymm14", "ymm15", "ymm17", "ymm18",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}

void bli_sgemmsup_rd_zen4_asm_1x32
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
    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left1  = k_left32 % 16;
    int32_t iter_1_mask = (1 << k_left1) - 1;

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

    mov( var( rs_a ), r8 )              // load rs_a
    lea( mem( , r8, 4 ), r8 )           // rs_a *= sizeof(dt) => rs_a *= 4
    mov( var( cs_b ), r9 )              // load cs_b
    lea( mem( , r9, 4 ), r9 )           // cs_b *= sizeof(dt) => cs_b *= 4
    mov( var( cs_a ), r10 )             // load cs_a
    lea( mem( , r10, 4 ), r10 )         // cs_a *= sizeof(dt) => cs_a *= 4
    mov( var( rs_c ), rdi )             // load rs_c
    lea( mem( , rdi, 4 ), rdi )         // rs_c *= sizeof(float) => rs_c *= 4
    lea( mem( r9, r9, 2 ), r13 )        // r13 = 3 * rs_b

    mov(var(iter_1_mask), esi)          // Load mask values for the last loop
    kmovw(esi, K(1))

    mov( imm( 0 ), r15 )                // jj = 0;
    label( .SLOOP3X4J )                 // LOOP OVER jj = [ 0 1 ... ]

    mov( var( abuf ), r14 )             // load address of a
    mov( var( bbuf ), rdx )             // load address of b
    mov( var( cbuf ), r12 )             // load address of c

    lea( mem( , r15, 1 ), rsi )
    imul( imm( 1*4 ), rsi )
    lea( mem( r12, rsi, 1 ), r12 )      // c += r15 * cs_c

    lea( mem(  , r15, 1 ), rsi )        // rsi = r15 = 4*jj;
    imul( r9, rsi )                     // rsi *= cs_b;
    lea( mem( rdx, rsi, 1 ), rdx )      // rbx = b + 4*jj*cs_b;

    lea( mem( r12 ), rcx )              // load c to rcx
    lea( mem( r14 ), rax )              // load a to rax
    lea( mem( rdx ), rbx )              // load b to rbx

    lea( mem(  r8, r8, 2 ), r10 )       // r10 = 3 * rs_b
    lea( mem( r10, r8, 2 ), rdi )       // rdi = 5 * rs_b


    INIT_REG

    mov( var( k_iter64 ), rsi )         // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_32 )


    label( .K_LOOP_ITER64 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 2
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 3
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    dec( rsi )
    jne( .K_LOOP_ITER64 )


    label( .CONSIDER_K_ITER_32 )

    mov( var( k_iter32 ), rsi )       // load k_iter
    test( rsi, rsi )
    je( .CONSIDER_K_ITER_16 )

    // ITER 0
    // load row from A
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    // ITER 1
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    // load column from B
    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )


    label( .CONSIDER_K_ITER_16 )
    mov( var( k_iter16 ), rsi )
    test( rsi, rsi )
    je( .CONSIDER_K_LEFT_1)

    // If the k-loop decomposition uses iterations of 64, 32, and 8 elements, which is inefficient for k values below 32.
    // For example, when k=31, the current implementation requires three 8-element loops (processing 24 elements) followed 
    // by seven scalar 1-element loops (processing the remaining 7 elements), totaling 10 loop iterations.
    // By redesigning the decomposition to use 64, 32, 16, and masked operations instead, the same k=31 case would 
    // require only one 16-element loop followed by a single masked operation to process the remaining 15 elements. 
    // This reduces the total iterations from 10 down to 2, significantly improving efficiency for k values in the range of 16-31.
    
    // ITER 0
    vmovups(         ( rax ), zmm0 )
    add( imm( 16*4 ), rax )

    vmovups(        ( rbx ), zmm6 )
    VFMA1( 8 )

    vmovups( ( rbx, r9, 1 ), zmm6 )
    VFMA1( 11 )

    vmovups( ( rbx, r9, 2 ), zmm6 )
    VFMA1( 14 )

    vmovups( ( rbx, r13, 1 ), zmm6 )
    VFMA1( 17 )

    add( imm( 16*4 ), rbx )

    label( .CONSIDER_K_LEFT_1 )
    mov( var(k_left1), rsi )
    test( rsi, rsi )
    je( .POST_ACCUM )

    // In the case where we need to only compute on floats
    // which fit in the ymm register, it is better to 
    // operate on masked ymm registers in this case
    // because on Zen4, the throughput of masked loads
    // on zmm is 0.5 while on ymm/xmm is 1
    cmp( imm(8), rsi )
    jle( .K_FLOATS_LEFT_LE_8 )

    label( .K_FLOATS_LEFT_GT_8 )
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), ZMM(0 MASK_KZ(1) ) )

    vmovups(         mem(rbx), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 8 )

    vmovups(  mem(rbx, r9, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 11 )

    vmovups( mem(rbx, r9, 2),  ZMM(6 MASK_KZ(1) ) )
    VFMA1( 14 )

    vmovups( mem(rbx, r13, 1), ZMM(6 MASK_KZ(1) ) )
    VFMA1( 17 )

    // unconditional branch to end of the loop after 
    // the computation of the case processing >8 floats
    jmp( .POST_ACCUM ) 

    label( .K_FLOATS_LEFT_LE_8 )
    // When operating on elements <= 8, it is better to operate
    // on masked YMM registers on Zen4 because vmovups on 
    // masked YMM registers has a throughput of 1 while 
    // the same operation on ZMM has a throughput of 0.5
    // Instead of looping over element by element and performing
    // VFMAs on for every element which is wasteful. 
    // Perform a masked FMA operation on the remaining elements
    vmovups(         mem(rax), YMM(0 MASK_KZ(1) ) )

    vmovups(         mem(rbx), YMM(6 MASK_KZ(1) ) )
    VFMA1( 8 )

    vmovups(  mem(rbx, r9, 1), YMM(6 MASK_KZ(1) ) )
    VFMA1( 11 )

    vmovups( mem(rbx, r9, 2),  YMM(6 MASK_KZ(1) ) )
    VFMA1( 14 )

    vmovups( mem(rbx, r13, 1), YMM(6 MASK_KZ(1) ) )
    VFMA1( 17 )

    label( .POST_ACCUM )

    mov( var( beta ), rax )         // load address of beta
    vbroadcastss( ( rax ), xmm0 )
    vxorps( xmm1, xmm1, xmm1 )
    vucomiss( xmm1, xmm0 )          // check if beta = 0
    je( .POST_ACCUM_STOR_BZ )


    // Accumulating & storing the results when beta != 0
    label( .POST_ACCUM_STOR )

    ZMM_TO_YMM( 8, 11, 14, 17, 4, 7, 10, 13 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )

    ALPHA_SCALE1                // Scaling the result of A*B with alpha

    C_STOR1                     // Storing result to C

    jmp( .SDONE )


    // Accumulating & storing the results when beta == 0
    label( .POST_ACCUM_STOR_BZ )

    ZMM_TO_YMM( 8, 11, 14, 17, 4, 7, 10, 13 )

    ACCUM_YMM( 4, 7, 10, 13, 4 )

    ALPHA_SCALE1                // Scaling the result of A*B with alpha

    C_STOR_BZ1                  // Storing result to C


    label( .SDONE )

    add( imm(  4 ), r15 )
    cmp( imm( 32 ), r15 )
    jl( .SLOOP3X4J )

    end_asm(
    : // output operands (none)
    : // input operands
      [iter_1_mask] "m" (iter_1_mask),
      [k_iter64] "m" (k_iter64),
      [k_left64] "m" (k_left64),
      [k_iter32] "m" (k_iter32),
      [k_left32] "m" (k_left32),
      [k_iter16] "m" (k_iter16),
      [k_left1]  "m" (k_left1),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [rs_c]     "m" (rs_c),
      [cs_c]     "m" (cs_c),
      [n0]       "m" (n0),
      [m0]       "m" (m0),
      [abuf]     "m" (abuf),
      [bbuf]     "m" (bbuf),
      [cbuf]     "m" (cbuf)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm4", "xmm6",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8", "ymm10", "ymm11", "ymm13",
      "ymm14", "ymm17",
      "zmm0", "zmm1", "zmm2", "zmm3",
      "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
      "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
      "zmm16", "zmm17", "zmm18", "zmm19",
      "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "memory"
    )
}
