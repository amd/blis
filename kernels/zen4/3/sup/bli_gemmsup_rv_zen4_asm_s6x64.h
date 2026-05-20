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

#define INIT_REG \
    vxorps( zmm0,zmm0,zmm0 ) \
    vxorps( zmm1,zmm1,zmm1 ) \
    vxorps( zmm2,zmm2,zmm2 ) \
    vxorps( zmm3,zmm3,zmm3 ) \
    vxorps( zmm4,zmm4,zmm4 ) \
    vxorps( zmm5,zmm5,zmm5 ) \
    vxorps( zmm6,zmm6,zmm6 ) \
    vxorps( zmm7,zmm7,zmm7 ) \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 ) \
    vxorps( zmm19,zmm19,zmm19 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm22,zmm22,zmm22 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm25,zmm25,zmm25 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm27,zmm27,zmm27 ) \
    vxorps( zmm28,zmm28,zmm28 ) \
    vxorps( zmm29,zmm29,zmm29 ) \
    vxorps( zmm30,zmm30,zmm30 ) \
    vxorps( zmm31,zmm31,zmm31 )

/**
 * VFMA4 - performs 4 VFMAs for k-loop
 * zmm0-3 - contains 4 rows of B
 * R0 - register containing A broadcast
 * R1-4 - registers to store intermediate result
 */
#define VFMA4( R0, R1, R2, R3, R4) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) ) \
    vfmadd231ps( zmm1,zmm(R0),zmm(R2) ) \
    vfmadd231ps( zmm2,zmm(R0),zmm(R3) ) \
    vfmadd231ps( zmm3,zmm(R0),zmm(R4) )

#define VFMA3( R0, R1, R2, R3) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) ) \
    vfmadd231ps( zmm1,zmm(R0),zmm(R2) ) \
    vfmadd231ps( zmm2,zmm(R0),zmm(R3) )

#define VFMA2( R0, R1, R2 ) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) ) \
    vfmadd231ps( zmm1,zmm(R0),zmm(R2) )

#define VFMA1( R0, R1 ) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) )

#define VFMA1_YMM( R0, R1 ) \
    vfmadd231ps( ymm0,ymm(R0),ymm(R1) )

/**
 * ALPHA_SCALE4 - scales 4 zmm registers by alpha
 * R0 - register having alpha
 * R1-4 - registers to be scaled
 */
#define ALPHA_SCALE4( R0, R1, R2, R3, R4 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) ) \
    vmulps( zmm(R0), zmm(R2), zmm(R2) ) \
    vmulps( zmm(R0), zmm(R3), zmm(R3) ) \
    vmulps( zmm(R0), zmm(R4), zmm(R4) )

#define ALPHA_SCALE3( R0, R1, R2, R3 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) ) \
    vmulps( zmm(R0), zmm(R2), zmm(R2) ) \
    vmulps( zmm(R0), zmm(R3), zmm(R3) )

#define ALPHA_SCALE2( R0, R1, R2 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) ) \
    vmulps( zmm(R0), zmm(R2), zmm(R2) )

#define ALPHA_SCALE1( R0, R1 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) )

#define ALPHA_SCALE1_YMM( R0, R1 ) \
    vmulps( ymm(R0), ymm(R1), ymm(R1) )

#define ALPHA_SCALE1_XMM( R0, R1 ) \
    vmulps( xmm(R0), xmm(R1), xmm(R1) )

/**
 * UPDATE_C4 - loads 4 C rows, performs 4 VFMAs (scaling by beta), stores to buffer & increments C ptr
 * R0 -> register having beta
 * R1-4 -> registers having intermediate results ( alpha * A * B )
 */
#define UPDATE_C4( R0, R1, R2, R3, R4 ) \
    vfmadd231ps((rcx), zmm(R0), zmm(R1)) \
    vmovups( zmm(R1),(rcx) ) \
    vfmadd231ps(0x40(rcx), zmm(R0), zmm(R2)) \
    vmovups( zmm(R2),0x40(rcx) ) \
    vfmadd231ps(0x80(rcx), zmm(R0), zmm(R3)) \
    vmovups( zmm(R3),0x80(rcx) ) \
    vfmadd231ps(0xc0(rcx), zmm(R0), zmm(R4)) \
    vmovups( zmm(R4),0xc0(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C3( R0, R1, R2, R3 ) \
    vfmadd231ps((rcx), zmm(R0), zmm(R1)) \
    vmovups( zmm(R1),(rcx) ) \
    vfmadd231ps(0x40(rcx), zmm(R0), zmm(R2)) \
    vmovups( zmm(R2),0x40(rcx) ) \
    vfmadd231ps(0x80(rcx), zmm(R0), zmm(R3)) \
    vmovups( zmm(R3),0x80(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C2( R0, R1, R2 ) \
    vfmadd231ps((rcx), zmm(R0), zmm(R1)) \
    vmovups( zmm(R1), (rcx) ) \
    vfmadd231ps(0x40(rcx), zmm(R0), zmm(R2)) \
    vmovups( zmm(R2), 0x40(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C1_MASK_YMM( R0, R1 ) \
    vmovups( mem( rcx ), ymm(1 MASK_KZ(1) ) ) \
    vfmadd231ps( ymm(R0), ymm1, ymm(R1) ) \
    vmovups( ymm(R1 ), mem(rcx) MASK_K(1) ) \
    add( rdi, rcx )

#define UPDATE_C1_MASK_XMM( R0, R1 ) \
    vmovups( mem( rcx ), xmm(1 MASK_KZ(1) ) ) \
    vfmadd231ps( xmm(R0), xmm1, xmm(R1) ) \
    vmovups( xmm(R1 ), mem(rcx) MASK_K(1) ) \
    add( rdi, rcx )

#define UPDATE_C1_MASK( R0, R1 ) \
    vmovups( mem( rcx ), zmm(1 MASK_KZ(1) ) ) \
    vfmadd231ps( zmm(R0), zmm1, zmm(R1) ) \
    vmovups( zmm(R1 ), mem(rcx) MASK_K(1) ) \
    add( rdi, rcx )

#define UPDATE_C1( R0, R1 ) \
    vfmadd231ps((rcx), zmm(R0), zmm(R1)) \
    vmovups( zmm(R1), (rcx) ) \
    add( rdi, rcx )

/**
 * UPDATE_C4_BZ - stores result to buffer & increments C ptr
 * R0-3 -> registers having intermediate results ( alpha * A * B )
 */
#define UPDATE_C4_BZ( R0, R1, R2, R3 ) \
    vmovups( zmm(R0),(rcx) ) \
    vmovups( zmm(R1),0x40(rcx) ) \
    vmovups( zmm(R2),0x80(rcx) ) \
    vmovups( zmm(R3),0xc0(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C3_BZ( R0, R1, R2 ) \
    vmovups( zmm(R0),(rcx) ) \
    vmovups( zmm(R1),0x40(rcx) ) \
    vmovups( zmm(R2),0x80(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C2_BZ( R0, R1 ) \
    vmovups( zmm(R0),(rcx) ) \
    vmovups( zmm(R1),0x40(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C1_BZ( R0 ) \
    vmovups( zmm(R0),(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C1_BZ_MASK_YMM( R0 ) \
    vmovups( ymm(R0 ), mem(rcx) MASK_K(1) ) \
    add( rdi, rcx )

#define UPDATE_C1_BZ_MASK_XMM( R0 ) \
    vmovups( xmm(R0 ), mem(rcx) MASK_K(1) ) \
    add( rdi, rcx )

#define UPDATE_C1_BZ_MASK( R0 ) \
    vmovups( zmm(R0 ), mem(rcx) MASK_K(1) ) \
    add( rdi, rcx )

#define TRANSPOSE_4X16( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_10_11_14_15( R0, R1, R2, R3 )

// Only operate on cols [0, 14]
#define TRANSPOSE_4X15( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_10_11_14( R0, R1, R2, R3 )

// Only operate on cols [0, 13]
#define TRANSPOSE_4X14( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_10_11( R0, R1, R2, R3 )

// Only operate on cols [0, 12]
#define TRANSPOSE_4X13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9_12( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_10_11( R0, R1, R2, R3 )

// Only operate on cols [0, 11]
#define TRANSPOSE_4X12( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_10_11( R0, R1, R2, R3 )

// Only operate on cols [0, 10]
#define TRANSPOSE_4X11( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_10( R0, R1, R2, R3 )

// Only operate on cols [0, 9]
#define TRANSPOSE_4X10( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 8]
#define TRANSPOSE_4X9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_8( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 7]
#define TRANSPOSE_4X8_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_7_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 6]
#define TRANSPOSE_4X7_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_6_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 5]
#define TRANSPOSE_4X6_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 4]
#define TRANSPOSE_4X5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_4_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 3]
#define TRANSPOSE_4X4_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_3_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 2]
#define TRANSPOSE_4X3_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_ST_2_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 1]
#define TRANSPOSE_4X2_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_1_YMM( R0, R1, R2, R3 )

// Only operate on cols [0]
#define TRANSPOSE_4X1_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_ST_0_YMM( R0, R1, R2, R3 )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0, 4, 8, 12 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// Store: [a12 b12 c12 d12] in col 12
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Load: Col 1, 5, 9, 13 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
// Store: [a9 b9 c9 d9] in col 9
// Store: [a13 b13 c13 d13] in col 13
#define TRANSPOSE_4X16L_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )


// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0, 4, 8, 12 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// Store: [a12 b12 c12 d12] in col 12
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Load: Col 1, 5, 9 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
// Store: [a9 b9 c9 d9] in col 9
#define TRANSPOSE_4X16L_ST_0_1_4_5_8_9_12( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx )


// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0, 4, 8 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Load: Col 1, 5, 9 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
// Store: [a9 b9 c9 d9] in col 9
#define TRANSPOSE_4X16L_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx )


// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0, 4, 8 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Load: Col 1, 5 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
#define TRANSPOSE_4X16L_ST_0_1_4_5_8( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx )

// // R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// // R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// // R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// // R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// // zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// // zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// // --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// // zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// // Load: Col 0, 4 into consecutive lanes of zmm0 using vinsertf32x4
// // Store: [a0 b0 c0 d0] in col 0
// // Store: [a4 b4 c4 d4] in col 4
// // --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// // zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// // Load: Col 1, 5 into consecutive lanes of zmm0 using vinsertf32x4
// // Store: [a1 b1 c1 d1] in col 1
// // Store: [a5 b5 c5 d5] in col 5
#define TRANSPOSE_4X16L_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), ymm0, ymm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), ymm0, ymm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0, 4 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Load: Col 1 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a1 b1 c1 d1] in col 1
#define TRANSPOSE_4X16L_ST_0_1_4_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), ymm0, ymm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Load: Col 1 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a1 b1 c1 d1] in col 1
#define TRANSPOSE_4X16L_ST_0_1_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Load: Col 0 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a0 b0 c0 d0] in col 0
#define TRANSPOSE_4X16L_ST_0_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2, 6, 10, 14 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// Store: [a14 b14 c14 d14] in col 14
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7, 11, 15 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
// Store: [a11 b11 c11 d11] in col 11
// Store: [a15 b15 c15 d15] in col 15
#define TRANSPOSE_4X16H_ST_2_3_6_7_10_11_14_15( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2, 6, 10, 14 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// Store: [a14 b14 c14 d14] in col 14
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7, 11 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
// Store: [a11 b11 c11 d11] in col 11
#define TRANSPOSE_4X16H_ST_2_3_6_7_10_11_14( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2, 6, 10 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7, 11 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
// Store: [a11 b11 c11 d11] in col 11
#define TRANSPOSE_4X16H_ST_2_3_6_7_10_11( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2, 6, 10 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
#define TRANSPOSE_4X16H_ST_2_3_6_7_10( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vinsertf32x4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2, 6 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
#define TRANSPOSE_4X16H_ST_2_3_6_7_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), ymm0, ymm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), ymm0, ymm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx )

// // R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// // R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// // R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// // R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// // zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// // zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// // --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// // zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// // Load: Col 2, 6 into consecutive lanes of zmm0 using vinsertf32x4
// // Store: [a2 b2 c2 d2] in col 2
// // Store: [a6 b6 c6 d6] in col 6
// // --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// // zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// // Load: Col 3, 7 into consecutive lanes of zmm0 using vinsertf32x4
// // Store: [a3 b3 c3 d3] in col 3
#define TRANSPOSE_4X16H_ST_2_3_6_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vinsertf32x4( imm(0x1), mem(rcx, rdi, 4), ymm0, ymm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a3 b3 c3 d3] in col 3
#define TRANSPOSE_4X16H_ST_2_3_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Load: Col 2 into consecutive lanes of zmm0 using vinsertf32x4
// Store: [a2 b2 c2 d2] in col 2
// --- Second set of vinsertf32x4 and vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Load: Col 3, 7 into consecutive lanes of zmm0 using vinsertf32x4
#define TRANSPOSE_4X16H_ST_2_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups ( mem(rcx), xmm0 ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovups( xmm5, mem(rcx) )

#define TRANSPOSE_4X16_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11_14_15( R0, R1, R2, R3 )

// Only operate on cols [0, 14]
#define TRANSPOSE_4X15_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11_14( R0, R1, R2, R3 )

// Only operate on cols [0, 13]
#define TRANSPOSE_4X14_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11( R0, R1, R2, R3 )

// Only operate on cols [0, 12]
#define TRANSPOSE_4X13_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9_12( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11( R0, R1, R2, R3 )

// Only operate on cols [0, 11]
#define TRANSPOSE_4X12_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11( R0, R1, R2, R3 )

// Only operate on cols [0, 10]
#define TRANSPOSE_4X11_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10( R0, R1, R2, R3 )

// Only operate on cols [0, 9]
#define TRANSPOSE_4X10_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 8]
#define TRANSPOSE_4X9_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 7]
#define TRANSPOSE_4X8_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_7_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 6]
#define TRANSPOSE_4X7_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_6_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 5]
#define TRANSPOSE_4X6_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 4]
#define TRANSPOSE_4X5_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_4_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 3]
#define TRANSPOSE_4X4_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_3_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 2]
#define TRANSPOSE_4X3_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ_ST_2_YMM( R0, R1, R2, R3 )

// Only operate on cols [0, 1]
#define TRANSPOSE_4X2_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_1_YMM( R0, R1, R2, R3 )

// Only operate on cols [0]
#define TRANSPOSE_4X1_BZ_YMM( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ_ST_0_YMM( R0, R1, R2, R3 )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// Store: [a12 b12 c12 d12] in col 12
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
// Store: [a9 b9 c9 d9] in col 9
// Store: [a13 b13 c13 d13] in col 13
#define TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// Store: [a12 b12 c12 d12] in col 12
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
// Store: [a9 b9 c9 d9] in col 9
#define TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9_12( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
// Store: [a9 b9 c9 d9] in col 9
#define TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8_9( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// Store: [a8 b8 c8 d8] in col 8
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
#define TRANSPOSE_4X16L_BZ_ST_0_1_4_5_8( R0, R1, R2, R3 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm6 ) \
    vunpcklps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups ( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups ( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
// Store: [a5 b5 c5 d5] in col 5
#define TRANSPOSE_4X16L_BZ_ST_0_1_4_5_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// Store: [a4 b4 c4 d4] in col 4
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
#define TRANSPOSE_4X16L_BZ_ST_0_1_4_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a1 b1 c1 d1 | a5 b5 c5 d5 | a9 b9 c9 d9 | a13 b13 c13 d13]
// Store: [a1 b1 c1 d1] in col 1
#define TRANSPOSE_4X16L_BZ_ST_0_1_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// zmm7: [c0 d0 c1 d1 | c4 d4 c5 d5 | c8 d8 c9 d9 | c12 d12 c13 d13]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a0 b0 c0 d0 | a4 b4 c4 d4 | a8 b8 c8 d8 | a12 b12 c12 d12]
// Store: [a0 b0 c0 d0] in col 0
#define TRANSPOSE_4X16L_BZ_ST_0_YMM( R0, R1, R2, R3 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm6 ) \
    vunpcklps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// Store: [a14 b14 c14 d14] in col 14
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
// Store: [a11 b11 c11 d11] in col 11
// Store: [a15 b15 c15 d15] in col 15
#define TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11_14_15( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// Store: [a14 b14 c14 d14] in col 14
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
// Store: [a11 b11 c11 d11] in col 11
#define TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11_14( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    vextractf32x4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
// Store: [a11 b11 c11 d11] in col 11
#define TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10_11( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm3 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm4 ) \
    vshufps( imm(0X44), zmm4, zmm3, zmm2 ) \
    vmovups( xmm2, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm2, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm2, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm4, zmm3, zmm2 ) \
    vmovups( xmm2, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm2, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm2, mem(rcx, rdi, 8) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// Store: [a10 b10 c10 d10] in col 10
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
#define TRANSPOSE_4X16H_BZ_ST_2_3_6_7_10( R0, R1, R2, R3 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm6 ) \
    vunpckhps( zmm(R3), zmm(R2), zmm7 ) \
    vshufps( imm(0X44), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    vextractf32x4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), zmm7, zmm6, zmm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), zmm5, mem(rcx, rdi, 4) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
// Store: [a7 b7 c7 d7] in col 7
#define TRANSPOSE_4X16H_BZ_ST_2_3_6_7_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// Store: [a6 b6 c6 d6] in col 6
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
#define TRANSPOSE_4X16H_BZ_ST_2_3_6_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    vextractf32x4( imm(0x01), ymm5, mem(rcx, rdi, 4) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
// --- Second set of vextractf32x4 using 0xEE vshufps ---
// zmm5: [a3 b3 c3 d3 | a7 b7 c7 d7 | a11 b11 c11 d11 | a15 b15 c15 d15]
// Store: [a3 b3 c3 d3] in col 3
#define TRANSPOSE_4X16H_BZ_ST_2_3_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx ) \
    vshufps( imm(0XEE), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx ) \

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// R2 -> c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
// R3 -> d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15
// zmm6: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// zmm7: [c2 d2 c3 d3 | c6 d6 c7 d7 | c10 d10 c11 d11 | c14 d14 c15 d15]
// --- First set of vextractf32x4 using 0x44 vshufps ---
// zmm5: [a2 b2 c2 d2 | a6 b6 c6 d6 | a10 b10 c10 d10 | a14 b14 c14 d14]
// Store: [a2 b2 c2 d2] in col 2
#define TRANSPOSE_4X16H_BZ_ST_2_YMM( R0, R1, R2, R3 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm6 ) \
    vunpckhps( ymm(R3), ymm(R2), ymm7 ) \
    vshufps( imm(0X44), ymm7, ymm6, ymm5 ) \
    vmovups( xmm5, mem(rcx) ) \
    add( rdi, rcx )

#define TRANSPOSE_2X16( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_10_11_14_15( R0, R1 )

// Only operate on cols [0, 14]
#define TRANSPOSE_2X15( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_10_11_14( R0, R1 )

// Only operate on cols [0, 13]
#define TRANSPOSE_2X14( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_10_11( R0, R1 )

// Only operate on cols [0, 12]
#define TRANSPOSE_2X13( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9_12( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_10_11( R0, R1 )

// Only operate on cols [0, 11]
#define TRANSPOSE_2X12( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_10_11( R0, R1 )

// Only operate on cols [0, 10]
#define TRANSPOSE_2X11( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_10( R0, R1 )

// Only operate on cols [0, 9]
#define TRANSPOSE_2X10( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8_9( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_YMM( R0, R1 )

// Only operate on cols [0, 8]
#define TRANSPOSE_2X9( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_8( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_YMM( R0, R1 )

// Only operate on cols [0, 7]
#define TRANSPOSE_2X8_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_7_YMM( R0, R1 )

// Only operate on cols [0, 6]
#define TRANSPOSE_2X7_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_6_YMM( R0, R1 )

// Only operate on cols [0, 5]
#define TRANSPOSE_2X6_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_5_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_YMM( R0, R1 )

// Only operate on cols [0, 4]
#define TRANSPOSE_2X5_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_4_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_YMM( R0, R1 )

// Only operate on cols [0, 3]
#define TRANSPOSE_2X4_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_3_YMM( R0, R1 )

// Only operate on cols [0, 2]
#define TRANSPOSE_2X3_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16H_ST_2_YMM( R0, R1 )

// Only operate on cols [0, 1]
#define TRANSPOSE_2X2_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_1_YMM( R0, R1 )

// Only operate on cols [0]
#define TRANSPOSE_2X1_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_ST_0_YMM( R0, R1 )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 4 (to xmm1)
// Load: from col 5 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 8 (to xmm1)
// Load: from col 9 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// Load: from col 12 (to xmm1)
// Load: from col 13 (to xmm1)
// vinsertf32x4 ( imm(0x3)...)
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
// Store: [a9 b9] in col 9
// vextractf32x4( imm(0x3)...)
// Store: [a12 b12] in col 12
// Store: [a13 b13] in col 13
#define TRANSPOSE_2X16L_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x3), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) ) \
    vmovhpd( xmm3, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 4 (to xmm1)
// Load: from col 5 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 8 (to xmm1)
// Load: from col 9 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// Load: from col 12 (to xmm1)
// vinsertf32x4 ( imm(0x3)...)
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
// Store: [a9 b9] in col 9
// vextractf32x4( imm(0x3)...)
// Store: [a12 b12] in col 12
#define TRANSPOSE_2X16L_ST_0_1_4_5_8_9_12( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x3), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 4 (to xmm1)
// Load: from col 5 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 8 (to xmm1)
// Load: from col 9 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
// Store: [a9 b9] in col 9
#define TRANSPOSE_2X16L_ST_0_1_4_5_8_9( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 4 (to xmm1)
// Load: from col 5 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 8 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
#define TRANSPOSE_2X16L_ST_0_1_4_5_8( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 4 (to xmm1)
// Load: from col 5 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
#define TRANSPOSE_2X16L_ST_0_1_4_5_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, ymm0, ymm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 4 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
#define TRANSPOSE_2X16L_ST_0_1_4_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, ymm0, ymm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// Load: from col 1 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// ...
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
#define TRANSPOSE_2X16L_ST_0_1_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 0 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// ...
// Store: [a0 b0] in col 0
#define TRANSPOSE_2X16L_ST_0_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 6 (to xmm1)
// Load: from col 7 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 10 (to xmm1)
// Load: from col 11 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// Load: from col 14 (to xmm1)
// Load: from col 15 (to xmm1)
// vinsertf32x4 ( imm(0x3)...)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
// Store: [a11 b11] in col 11
// vextractf32x4( imm(0x3)...)
// Store: [a14 b14] in col 14
// Store: [a15 b15] in col 15
#define TRANSPOSE_2X16H_ST_2_3_6_7_10_11_14_15( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x3), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) ) \
    vmovhpd( xmm3, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 6 (to xmm1)
// Load: from col 7 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 10 (to xmm1)
// Load: from col 11 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// Load: from col 14 (to xmm1)
// vinsertf32x4 ( imm(0x3)...)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
// Store: [a11 b11] in col 11
// vextractf32x4( imm(0x3)...)
// Store: [a14 b14] in col 14
#define TRANSPOSE_2X16H_ST_2_3_6_7_10_11_14( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x3), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 6 (to xmm1)
// Load: from col 7 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 10 (to xmm1)
// Load: from col 11 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
// Store: [a11 b11] in col 11
#define TRANSPOSE_2X16H_ST_2_3_6_7_10_11( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 6 (to xmm1)
// Load: from col 7 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// Load: from col 10 (to xmm1)
// vinsertf32x4 ( imm(0x2)...)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
#define TRANSPOSE_2X16H_ST_2_3_6_7_10( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, zmm0, zmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x2), xmm1, zmm0, zmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( zmm0, zmm4, zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 6 (to xmm1)
// Load: from col 7 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
#define TRANSPOSE_2X16H_ST_2_3_6_7_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, ymm0, ymm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// lea( mem(rcx, rdi, 4), rcx )
// Load: from col 6 (to xmm1)
// vinsertf32x4 ( imm(0x1)...)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
#define TRANSPOSE_2X16H_ST_2_3_6_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( mem(rcx), xmm1, xmm1 ) \
    vinsertf32x4( imm(0x1), xmm1, ymm0, ymm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// Load: from col 3 (to xmm0)
// ...
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
#define TRANSPOSE_2X16H_ST_2_3_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    vmovhpd( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// Load: from col 2 (to xmm0)
// ...
// Store: [a2 b2] in col 2
#define TRANSPOSE_2X16H_ST_2_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( mem(rcx), xmm0, xmm0 ) \
    mov( r12, rcx ) \
    vfmadd231ps( ymm0, ymm4, ymm5 ) \
    vmovlpd( xmm5, mem(rcx) )

#define TRANSPOSE_2X16_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11_14_15( R0, R1 )

// Only operate on cols [0, 14]
#define TRANSPOSE_2X15_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11_14( R0, R1 )

// Only operate on cols [0, 13]
#define TRANSPOSE_2X14_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11( R0, R1 )

// Only operate on cols [0, 12]
#define TRANSPOSE_2X13_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9_12( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11( R0, R1 )

// Only operate on cols [0, 11]
#define TRANSPOSE_2X12_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11( R0, R1 )

// Only operate on cols [0, 10]
#define TRANSPOSE_2X11_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10( R0, R1 )

// Only operate on cols [0, 9]
#define TRANSPOSE_2X10_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_YMM( R0, R1 )

// Only operate on cols [0, 8]
#define TRANSPOSE_2X9_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_YMM( R0, R1 )

// Only operate on cols [0, 7]
#define TRANSPOSE_2X8_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_7_YMM( R0, R1 )

// Only operate on cols [0, 6]
#define TRANSPOSE_2X7_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_6_YMM( R0, R1 )

// Only operate on cols [0, 5]
#define TRANSPOSE_2X6_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_5_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_YMM( R0, R1 )

// Only operate on cols [0, 4]
#define TRANSPOSE_2X5_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_4_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_YMM( R0, R1 )

// Only operate on cols [0, 3]
#define TRANSPOSE_2X4_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_3_YMM( R0, R1 )

// Only operate on cols [0, 2]
#define TRANSPOSE_2X3_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_YMM( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ_ST_2_YMM( R0, R1 )

// Only operate on cols [0, 1]
#define TRANSPOSE_2X2_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_1_YMM( R0, R1 )

// Only operate on cols [0]
#define TRANSPOSE_2X1_BZ_YMM( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ_ST_0_YMM( R0, R1 )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
// Store: [a9 b9] in col 9
// vextractf32x4( imm(0x3)...)
// Store: [a12 b12] in col 12
// Store: [a13 b13] in col 13
#define TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9_12_13( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) ) \
    vmovhpd( xmm3, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
// Store: [a9 b9] in col 9
// vextractf32x4( imm(0x3)...)
// Store: [a12 b12] in col 12
#define TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9_12( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
// Store: [a9 b9] in col 9
#define TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8_9( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
// vextractf32x4( imm(0x2)...)
// Store: [a8 b8] in col 8
#define TRANSPOSE_2X16L_BZ_ST_0_1_4_5_8( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
// Store: [a5 b5] in col 5
#define TRANSPOSE_2X16L_BZ_ST_0_1_4_5_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
// vextractf32x4( imm(0x1)...)
// Store: [a4 b4] in col 4
#define TRANSPOSE_2X16L_BZ_ST_0_1_4_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
// Store: [a1 b1] in col 1
#define TRANSPOSE_2X16L_BZ_ST_0_1_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a0 b0 a1 b1 | a4 b4 a5 b5 | a8 b8 a9 b9 | a12 b12 a13 b13]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a0 b0] in col 0
#define TRANSPOSE_2X16L_BZ_ST_0_YMM( R0, R1 ) \
    vunpcklps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
// Store: [a11 b11] in col 11
// vextractf32x4( imm(0x3)...)
// Store: [a14 b14] in col 14
// Store: [a15 b15] in col 15
#define TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11_14_15( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) ) \
    vmovhpd( xmm3, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
// Store: [a11 b11] in col 11
// vextractf32x4( imm(0x3)...)
// Store: [a14 b14] in col 14
#define TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11_14( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x3), zmm(5), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
// Store: [a11 b11] in col 11
#define TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10_11( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
// vextractf32x4( imm(0x2)...)
// Store: [a10 b10] in col 10
#define TRANSPOSE_2X16H_BZ_ST_2_3_6_7_10( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), zmm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    vextractf32x4( imm(0x2), zmm(5), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
// Store: [a7 b7] in col 7
#define TRANSPOSE_2X16H_BZ_ST_2_3_6_7_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
// vextractf32x4( imm(0x1)...)
// Store: [a6 b6] in col 6
#define TRANSPOSE_2X16H_BZ_ST_2_3_6_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vextractf32x4( imm(0x1), ymm(5), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
// Store: [a3 b3] in col 3
#define TRANSPOSE_2X16H_BZ_ST_2_3_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) ) \
    vmovhpd( xmm5, mem(rcx, rdi, 1) )

// R0 -> a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
// R1 -> b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
// zmm5: [a2 b2 a3 b3 | a6 b6 a7 b7 | a10 b10 a11 b11 | a14 b14 a15 b15]
// --- First set of vinsertf32x4 and vextractf32x4 using 0x44 vshufps ---
// The vmovlpd and vmovhpd store instructions in the following order
// Store: [a2 b2] in col 2
#define TRANSPOSE_2X16H_BZ_ST_2_YMM( R0, R1 ) \
    vunpckhps( ymm(R1), ymm(R0), ymm5 ) \
    vmovlpd( xmm5, mem(rcx) )

#define UPDATE_C_1X16_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x02, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x03, R0 )

#define UPDATE_C_1X15_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x02, R0 ) \
    UPDATE_C_1X15_BZ_UTIL( 0x03, R0 )

#define UPDATE_C_1X14_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x02, R0 ) \
    UPDATE_C_1X14_BZ_UTIL( 0x03, R0 )

#define UPDATE_C_1X13_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x02, R0 ) \
    UPDATE_C_1X13_BZ_UTIL( 0x03, R0 )

#define UPDATE_C_1X12_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x02, R0 )

#define UPDATE_C_1X11_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X15_BZ_UTIL( 0x02, R0 )

#define UPDATE_C_1X10_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X14_BZ_UTIL( 0x02, R0 )

#define UPDATE_C_1X9_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X13_BZ_UTIL( 0x02, R0 )


#define UPDATE_C_1X8_BZ_YMM(R0) \
    UPDATE_C_1X16_BZ_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X7_BZ_YMM(R0) \
    UPDATE_C_1X16_BZ_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X15_BZ_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X6_BZ_YMM(R0) \
    UPDATE_C_1X16_BZ_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X14_BZ_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X5_BZ_YMM(R0) \
    UPDATE_C_1X16_BZ_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X13_BZ_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X4_BZ_YMM(R0) \
    UPDATE_C_1X16_BZ_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X3_BZ_YMM(R0) \
    UPDATE_C_1X15_BZ_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X2_BZ_YMM(R0) \
    UPDATE_C_1X14_BZ_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X1_BZ_YMM(R0) \
    UPDATE_C_1X13_BZ_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X16_BZ_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vshufps( imm(0x02), xmm0, xmm0, xmm2 ) \
    vshufps( imm(0x03), xmm0, xmm0, xmm3 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    vmovss( xmm2, (rcx, rdi, 2) ) \
    vmovss( xmm3, (rcx, r12, 1) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X16_BZ_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vshufps( imm(0x02), xmm0, xmm0, xmm2 ) \
    vshufps( imm(0x03), xmm0, xmm0, xmm3 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    vmovss( xmm2, (rcx, rdi, 2) ) \
    vmovss( xmm3, (rcx, r12, 1) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X15_BZ_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vshufps( imm(0x02), xmm0, xmm0, xmm2 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    vmovss( xmm2, (rcx, rdi, 2) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X15_BZ_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vshufps( imm(0x02), xmm0, xmm0, xmm2 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    vmovss( xmm2, (rcx, rdi, 2) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X14_BZ_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X14_BZ_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X13_BZ_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) \
    vmovss( xmm0, (rcx) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X13_BZ_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) \
    vmovss( xmm0, (rcx) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X16( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_UTIL( 0x02, R0 ) \
    UPDATE_C_1X16_UTIL( 0x03, R0 )

#define UPDATE_C_1X15( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_UTIL( 0x02, R0 ) \
    UPDATE_C_1X15_UTIL( 0x03, R0 )

#define UPDATE_C_1X14( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_UTIL( 0x02, R0 ) \
    UPDATE_C_1X14_UTIL( 0x03, R0 )

#define UPDATE_C_1X13( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_UTIL( 0x02, R0 ) \
    UPDATE_C_1X13_UTIL( 0x03, R0 )

#define UPDATE_C_1X12( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_UTIL( 0x02, R0 )

#define UPDATE_C_1X11( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X15_UTIL( 0x02, R0 )

#define UPDATE_C_1X10( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X14_UTIL( 0x02, R0 )

#define UPDATE_C_1X9( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X13_UTIL( 0x02, R0 )


#define UPDATE_C_1X8_YMM( R0 ) \
    UPDATE_C_1X16_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X7_YMM( R0 ) \
    UPDATE_C_1X16_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X15_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X6_YMM( R0 ) \
    UPDATE_C_1X16_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X14_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X5_YMM( R0 ) \
    UPDATE_C_1X16_UTIL_YMM( 0x00, R0 ) \
    UPDATE_C_1X13_UTIL_YMM( 0x01, R0 )

#define UPDATE_C_1X4_YMM( R0 ) \
    UPDATE_C_1X16_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X3_YMM( R0 ) \
    UPDATE_C_1X15_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X2_YMM( R0 ) \
    UPDATE_C_1X14_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X1_YMM( R0 ) \
    UPDATE_C_1X13_UTIL_YMM( 0x00, R0 )

#define UPDATE_C_1X16_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 )   /* xmm6 <- [a1, a0, a0, a0] */ \
    vshufps( imm(0x02), xmm0, xmm0, xmm7 )   /* xmm7 <- [a2, a0, a0, a0] */ \
    vshufps( imm(0x03), xmm0, xmm0, xmm12 )  /* xmm12 <- [a3, a0, a0, a0] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )           /* xmm0[0] += C[i, 0]  * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 1), xmm4, xmm6 )    /* xmm6[0] += C[i, 1]  * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 2 ), xmm4, xmm7 )   /* xmm7[0] += C[i, 2]  * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, r12, 1 ), xmm4, xmm12 )  /* xmm12[0] += C[i, 3] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )           /* C[i, 0] <- xmm0[0] */ \
    vmovss( xmm6, (rcx, rdi, 1) )   /* C[i, 1] <- xmm6[0] */ \
    vmovss( xmm7, (rcx, rdi, 2) )   /* C[i, 2] <- xmm7[0] */ \
    vmovss( xmm12, (rcx, r12, 1) )  /* C[i, 3] <- xmm12[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X16_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 )   /* xmm6 <- [a1, a0, a0, a0] */ \
    vshufps( imm(0x02), xmm0, xmm0, xmm7 )   /* xmm7 <- [a2, a0, a0, a0] */ \
    vshufps( imm(0x03), xmm0, xmm0, xmm12 )  /* xmm12 <- [a3, a0, a0, a0] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )           /* xmm0[0] += C[i, 0]  * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 1 ), xmm4, xmm6 )   /* xmm6[0] += C[i, 1]  * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 2 ), xmm4, xmm7 )   /* xmm7[0] += C[i, 2]  * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, r12, 1 ), xmm4, xmm12 )  /* xmm12[0] += C[i, 3] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )           /* C[i, 0] <- xmm0[0] */ \
    vmovss( xmm6, (rcx, rdi, 1) )   /* C[i, 1] <- xmm6[0] */ \
    vmovss( xmm7, (rcx, rdi, 2) )   /* C[i, 2] <- xmm7[0] */ \
    vmovss( xmm12, (rcx, r12, 1) )  /* C[i, 3] <- xmm12[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X15_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 )   /* xmm6 <- [a1, a0, a0, a0] */ \
    vshufps( imm(0x02), xmm0, xmm0, xmm7 )   /* xmm7 <- [a2, a0, a0, a0] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )         /* xmm0[0] += C[i, 0] * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 1 ), xmm4, xmm6 ) /* xmm6[0] += C[i, 1] * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 2 ), xmm4, xmm7 ) /* xmm7[0] += C[i, 2] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )          /* C[i, 0] <- xmm0[0] */   \
    vmovss( xmm6, (rcx, rdi, 1) )  /* C[i, 1] <- xmm6[0] */ \
    vmovss( xmm7, (rcx, rdi, 2) )  /* C[i, 2] <- xmm7[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X15_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 )   /* xmm6 <- [a1, a0, a0, a0] */ \
    vshufps( imm(0x02), xmm0, xmm0, xmm7 )   /* xmm7 <- [a2, a0, a0, a0] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )         /* xmm0[0] += C[i, 0] * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 1), xmm4, xmm6 )  /* xmm6[0] += C[i, 1] * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 2 ), xmm4, xmm7 ) /* xmm7[0] += C[i, 2] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )         /* C[i, 0] <- xmm0[0] */ \
    vmovss( xmm6, (rcx, rdi, 1) ) /* C[i, 1] <- xmm6[0] */ \
    vmovss( xmm7, (rcx, rdi, 2) ) /* C[i, 2] <- xmm7[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X14_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 )   /* xmm6 <- [a1, a0, a0, a0] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )         /* xmm0[0] += C[i, 0] * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 1), xmm4, xmm6 )  /* xmm6[0] += C[i, 1] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )         /* C[i, 0] <- xmm0[0] */ \
    vmovss( xmm6, (rcx, rdi, 1) ) /* C[i, 1] <- xmm6[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X14_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 )   /* xmm6 <- [a1, a0, a0, a0] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )         /* xmm0[0] += C[i, 0] * xmm4[0] (beta) */ \
    vfmadd231ps( mem_1to4( rcx, rdi, 1), xmm4, xmm6 )  /* xmm6[0] += C[i, 1] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )          /* C[i, 0] <- xmm0[0] */ \
    vmovss( xmm6, (rcx, rdi, 1) )  /* C[i, 1] <- xmm6[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X13_UTIL( IMM, R0 ) \
    vextractf32x4( imm(IMM), zmm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )   /* xmm0[0] += C[i, 0] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )     /* C[i, 0] <- xmm0[0] */ \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X13_UTIL_YMM( IMM, R0 ) \
    vextractf32x4( imm(IMM), ymm(R0), xmm0 ) /* xmm0 <- [a0, a1, a2, a3] */ \
    vfmadd231ps( mem_1to4( rcx ), xmm4, xmm0 )   /* xmm0[0] += C[i, 0] * xmm4[0] (beta) */ \
    vmovss( xmm0, (rcx) )    /* C[i, 0] <- xmm0[0] */ \
    lea( (rcx, rdi, 4), rcx )
