/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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
#include <immintrin.h>

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

// vshuff32x4 immediate encoder. Each output half (lo, hi) is built from two
// 128-bit lanes drawn from the 4 available across both sources (A, B):
//   0 = A.lane0, 1 = A.lane1, 2 = B.lane0, 3 = B.lane1
// lo0/lo1 select the two lanes placed in the result's low 256 bits,
// hi0/hi1 select the two lanes placed in the result's high 256 bits.
// (Similar in spirit to _MM_SHUFFLE, but for 128-bit lanes instead of
// 32-bit elements, and with 4 independent fields instead of a fixed
// A/B split.)
#define _SHUFF32X4_IMM(hi1, hi0, lo1, lo0) \
	( ( (hi1) << 6 ) | ( (hi0) << 4 ) | ( (lo1) << 2 ) | (lo0) )

static void bli_spackm_zen5_transpose_16x16( __m512 v[16] )
{
	__m512 t[16];

	/*
	 Consider the 16 zmm registers v[0..15], each holding one "row" of the
	 source (16 contiguous k-values). A 512-bit zmm has four 128-bit lanes,
	 so each register's 16 elements split into 4 groups of 4. Using letters
	 a..p for rows v[0]..v[15] (element i of row a is written a_i), and
	 writing each register as its 4 lane-groups:
	   v[0]=a  <- [ {a0,a1,a2,a3},   {a4,a5,a6,a7},   {a8,a9,a10,a11},   {a12,a13,a14,a15}   ]
	   v[1]=b  <- [ {b0,b1,b2,b3},   {b4,b5,b6,b7},   {b8,b9,b10,b11},   {b12,b13,b14,b15}   ]
	   v[2]=c  <- [ {c0,c1,c2,c3},   {c4,c5,c6,c7},   {c8,c9,c10,c11},   {c12,c13,c14,c15}   ]
	   ...                                                     (similarly d=v[3] .. p=v[15])

	 STAGE 1: unpacklo/unpackhi interleave elements within each 128-bit lane
	 only (never across lanes) -- exactly the same operation as the 8x8
	 case's Stage 1, just repeated across 4 lanes instead of 2:
	   _mm512_unpacklo_ps(a,b): t[0] <- [ {a0,b0,a1,b1}, {a4,b4,a5,b5}, {a8,b8,a9,b9},   {a12,b12,a13,b13} ]
	   _mm512_unpackhi_ps(a,b): t[1] <- [ {a2,b2,a3,b3}, {a6,b6,a7,b7}, {a10,b10,a11,b11},{a14,b14,a15,b15} ]
	   Similarly (rows c/d): t[2],t[3]     Similarly (rows e/f): t[4],t[5]
	   Similarly (rows g/h): t[6],t[7]     Similarly (rows i/j): t[8],t[9]
	   Similarly (rows k/l): t[10],t[11]   Similarly (rows m/n): t[12],t[13]
	   Similarly (rows o/p): t[14],t[15]

	 STAGE 2: shuffle_ps combines two t-regs into 4-row groups (abcd, efgh,
	 ijkl, mnop), still split across all 4 128-bit lanes -- each lane now
	 holds a different element index for the same 4 rows (again, the same
	 pattern as the 8x8 case's Stage 2, just repeated 4 times instead of 2):
	   _mm512_shuffle_ps(t[0],t[2],_MM_SHUFFLE(1,0,1,0)):
	     v[0] <- [ {a0,b0,c0,d0}, {a4,b4,c4,d4}, {a8,b8,c8,d8},   {a12,b12,c12,d12} ]
	   _mm512_shuffle_ps(t[0],t[2],_MM_SHUFFLE(3,2,3,2)):
	     v[1] <- [ {a1,b1,c1,d1}, {a5,b5,c5,d5}, {a9,b9,c9,d9},   {a13,b13,c13,d13} ]
	   _mm512_shuffle_ps(t[1],t[3],_MM_SHUFFLE(1,0,1,0)):
	     v[2] <- [ {a2,b2,c2,d2}, {a6,b6,c6,d6}, {a10,b10,c10,d10},{a14,b14,c14,d14} ]
	   _mm512_shuffle_ps(t[1],t[3],_MM_SHUFFLE(3,2,3,2)):
	     v[3] <- [ {a3,b3,c3,d3}, {a7,b7,c7,d7}, {a11,b11,c11,d11},{a15,b15,c15,d15} ]
	   Similarly from t[4]/t[6] and t[5]/t[7] (rows efgh):     v[4..7]
	   Similarly from t[8]/t[10] and t[9]/t[11] (rows ijkl):   v[8..11]
	   Similarly from t[12]/t[14] and t[13]/t[15] (rows mnop): v[12..15]

	   After Stage 2, v[s] (s=0..15) holds element s for its own 4-row
	   group, with its 4 lanes holding elements s, s+4, s+8, s+12
	   respectively (e.g. v[0]'s lanes = elem0, elem4, elem8, elem12 of
	   rows a,b,c,d). No data has crossed a 128-bit lane boundary yet.

	 STAGE 3 & 4: _mm512_shuffle_f32x4 is the 512-bit analogue of
	 permute2f128 -- the only instruction here that moves data across
	 128-bit lanes. Each call draws from just 2 source registers and can
	 place at most 2 of its 4 destination lanes from each source, so
	 (unlike the 2-lane 256-bit case, which fully resolves in 1 shuffle)
	 wiring all 4 lanes together requires 2 rounds:

	   Stage 3 merges the abcd group with the efgh group (the ijkl/mnop
	   merge, and the final elem8/elem12 lane placement, are still left
	   for Stage 4):
	     _mm512_shuffle_f32x4(v[0],v[4],0x88):
	       t[0] <- [ {a0,b0,c0,d0}, {a8,b8,c8,d8}, {e0,f0,g0,h0}, {e8,f8,g8,h8} ]
	     _mm512_shuffle_f32x4(v[0],v[4],0xdd):
	       t[4] <- [ {a4,b4,c4,d4}, {a12,b12,c12,d12}, {e4,f4,g4,h4}, {e12,f12,g12,h12} ]
	     Similarly: v[1]/v[5]->t[1],t[5]   v[2]/v[6]->t[2],t[6]   v[3]/v[7]->t[3],t[7]
	     and for the ijkl/mnop groups:
	       v[8]/v[12]->t[8],t[12]   v[9]/v[13]->t[9],t[13]
	       v[10]/v[14]->t[10],t[14] v[11]/v[15]->t[11],t[15]

	   Stage 4 merges the (abcd+efgh) result with the (ijkl+mnop) result,
	   completing the transpose -- each destination lane now comes from a
	   different one of the 4 row-groups, giving a full 16-row column:
	     _mm512_shuffle_f32x4(t[0],t[8],0x88):
	       v[0] <- [ {a0,b0,c0,d0}, {e0,f0,g0,h0}, {i0,j0,k0,l0}, {m0,n0,o0,p0} ]
	             = full column 0 (all 16 rows a..p)
	     _mm512_shuffle_f32x4(t[0],t[8],0xdd):
	       v[8] <- [ {a8,b8,c8,d8}, {e8,f8,g8,h8}, {i8,j8,k8,l8}, {m8,n8,o8,p8} ]
	             = full column 8
	     Similarly: t[1]/t[9]->v[1](col 1),v[9](col 9)
	                t[2]/t[10]->v[2](col 2),v[10](col 10)
	                t[3]/t[11]->v[3](col 3),v[11](col 11)
	                t[4]/t[12]->v[4](col 4),v[12](col 12)
	                t[5]/t[13]->v[5](col 5),v[13](col 13)
	                t[6]/t[14]->v[6](col 6),v[14](col 14)
	                t[7]/t[15]->v[7](col 7),v[15](col 15)

	 Result: v[0]..v[15] each hold one complete transposed row (element s
	 of every one of the 16 source rows a..p), i.e. the 16x16 in-register
	 transpose that converts the source's row-major (per-k-contiguous)
	 micropanel into P's column-major (per-m-contiguous) layout.
	*/

	// STAGE 1
	t[0]  = _mm512_unpacklo_ps( v[0],  v[1]  );
	t[1]  = _mm512_unpackhi_ps( v[0],  v[1]  );
	t[2]  = _mm512_unpacklo_ps( v[2],  v[3]  );
	t[3]  = _mm512_unpackhi_ps( v[2],  v[3]  );
	t[4]  = _mm512_unpacklo_ps( v[4],  v[5]  );
	t[5]  = _mm512_unpackhi_ps( v[4],  v[5]  );
	t[6]  = _mm512_unpacklo_ps( v[6],  v[7]  );
	t[7]  = _mm512_unpackhi_ps( v[6],  v[7]  );
	t[8]  = _mm512_unpacklo_ps( v[8],  v[9]  );
	t[9]  = _mm512_unpackhi_ps( v[8],  v[9]  );
	t[10] = _mm512_unpacklo_ps( v[10], v[11] );
	t[11] = _mm512_unpackhi_ps( v[10], v[11] );
	t[12] = _mm512_unpacklo_ps( v[12], v[13] );
	t[13] = _mm512_unpackhi_ps( v[12], v[13] );
	t[14] = _mm512_unpacklo_ps( v[14], v[15] );
	t[15] = _mm512_unpackhi_ps( v[14], v[15] );

	// STAGE 2
	v[0]  = _mm512_shuffle_ps( t[0],  t[2],  _MM_SHUFFLE(1,0,1,0) );
	v[1]  = _mm512_shuffle_ps( t[0],  t[2],  _MM_SHUFFLE(3,2,3,2) );
	v[2]  = _mm512_shuffle_ps( t[1],  t[3],  _MM_SHUFFLE(1,0,1,0) );
	v[3]  = _mm512_shuffle_ps( t[1],  t[3],  _MM_SHUFFLE(3,2,3,2) );
	v[4]  = _mm512_shuffle_ps( t[4],  t[6],  _MM_SHUFFLE(1,0,1,0) );
	v[5]  = _mm512_shuffle_ps( t[4],  t[6],  _MM_SHUFFLE(3,2,3,2) );
	v[6]  = _mm512_shuffle_ps( t[5],  t[7],  _MM_SHUFFLE(1,0,1,0) );
	v[7]  = _mm512_shuffle_ps( t[5],  t[7],  _MM_SHUFFLE(3,2,3,2) );
	v[8]  = _mm512_shuffle_ps( t[8],  t[10], _MM_SHUFFLE(1,0,1,0) );
	v[9]  = _mm512_shuffle_ps( t[8],  t[10], _MM_SHUFFLE(3,2,3,2) );
	v[10] = _mm512_shuffle_ps( t[9],  t[11], _MM_SHUFFLE(1,0,1,0) );
	v[11] = _mm512_shuffle_ps( t[9],  t[11], _MM_SHUFFLE(3,2,3,2) );
	v[12] = _mm512_shuffle_ps( t[12], t[14], _MM_SHUFFLE(1,0,1,0) );
	v[13] = _mm512_shuffle_ps( t[12], t[14], _MM_SHUFFLE(3,2,3,2) );
	v[14] = _mm512_shuffle_ps( t[13], t[15], _MM_SHUFFLE(1,0,1,0) );
	v[15] = _mm512_shuffle_ps( t[13], t[15], _MM_SHUFFLE(3,2,3,2) );

	// STAGE 3
	t[0]  = _mm512_shuffle_f32x4( v[0],  v[4],  _SHUFF32X4_IMM(2,0,2,0) );
	t[1]  = _mm512_shuffle_f32x4( v[1],  v[5],  _SHUFF32X4_IMM(2,0,2,0) );
	t[2]  = _mm512_shuffle_f32x4( v[2],  v[6],  _SHUFF32X4_IMM(2,0,2,0) );
	t[3]  = _mm512_shuffle_f32x4( v[3],  v[7],  _SHUFF32X4_IMM(2,0,2,0) );
	t[4]  = _mm512_shuffle_f32x4( v[0],  v[4],  _SHUFF32X4_IMM(3,1,3,1) );
	t[5]  = _mm512_shuffle_f32x4( v[1],  v[5],  _SHUFF32X4_IMM(3,1,3,1) );
	t[6]  = _mm512_shuffle_f32x4( v[2],  v[6],  _SHUFF32X4_IMM(3,1,3,1) );
	t[7]  = _mm512_shuffle_f32x4( v[3],  v[7],  _SHUFF32X4_IMM(3,1,3,1) );
	t[8]  = _mm512_shuffle_f32x4( v[8],  v[12], _SHUFF32X4_IMM(2,0,2,0) );
	t[9]  = _mm512_shuffle_f32x4( v[9],  v[13], _SHUFF32X4_IMM(2,0,2,0) );
	t[10] = _mm512_shuffle_f32x4( v[10], v[14], _SHUFF32X4_IMM(2,0,2,0) );
	t[11] = _mm512_shuffle_f32x4( v[11], v[15], _SHUFF32X4_IMM(2,0,2,0) );
	t[12] = _mm512_shuffle_f32x4( v[8],  v[12], _SHUFF32X4_IMM(3,1,3,1) );
	t[13] = _mm512_shuffle_f32x4( v[9],  v[13], _SHUFF32X4_IMM(3,1,3,1) );
	t[14] = _mm512_shuffle_f32x4( v[10], v[14], _SHUFF32X4_IMM(3,1,3,1) );
	t[15] = _mm512_shuffle_f32x4( v[11], v[15], _SHUFF32X4_IMM(3,1,3,1) );

	// STAGE 4
	v[0]  = _mm512_shuffle_f32x4( t[0],  t[8],  _SHUFF32X4_IMM(2,0,2,0) );
	v[1]  = _mm512_shuffle_f32x4( t[1],  t[9],  _SHUFF32X4_IMM(2,0,2,0) );
	v[2]  = _mm512_shuffle_f32x4( t[2],  t[10], _SHUFF32X4_IMM(2,0,2,0) );
	v[3]  = _mm512_shuffle_f32x4( t[3],  t[11], _SHUFF32X4_IMM(2,0,2,0) );
	v[4]  = _mm512_shuffle_f32x4( t[4],  t[12], _SHUFF32X4_IMM(2,0,2,0) );
	v[5]  = _mm512_shuffle_f32x4( t[5],  t[13], _SHUFF32X4_IMM(2,0,2,0) );
	v[6]  = _mm512_shuffle_f32x4( t[6],  t[14], _SHUFF32X4_IMM(2,0,2,0) );
	v[7]  = _mm512_shuffle_f32x4( t[7],  t[15], _SHUFF32X4_IMM(2,0,2,0) );
	v[8]  = _mm512_shuffle_f32x4( t[0],  t[8],  _SHUFF32X4_IMM(3,1,3,1) );
	v[9]  = _mm512_shuffle_f32x4( t[1],  t[9],  _SHUFF32X4_IMM(3,1,3,1) );
	v[10] = _mm512_shuffle_f32x4( t[2],  t[10], _SHUFF32X4_IMM(3,1,3,1) );
	v[11] = _mm512_shuffle_f32x4( t[3],  t[11], _SHUFF32X4_IMM(3,1,3,1) );
	v[12] = _mm512_shuffle_f32x4( t[4],  t[12], _SHUFF32X4_IMM(3,1,3,1) );
	v[13] = _mm512_shuffle_f32x4( t[5],  t[13], _SHUFF32X4_IMM(3,1,3,1) );
	v[14] = _mm512_shuffle_f32x4( t[6],  t[14], _SHUFF32X4_IMM(3,1,3,1) );
	v[15] = _mm512_shuffle_f32x4( t[7],  t[15], _SHUFF32X4_IMM(3,1,3,1) );
}

void bli_spackm_zen5_asm_48xk
     (
       conj_t              conja,
       pack_t              schema,
       dim_t               cdim0,
       dim_t               k0,
       dim_t               k0_max,
       float*     restrict kappa,
       float*     restrict a, inc_t inca0, inc_t lda0,
       float*     restrict p,              inc_t ldp0,
       cntx_t*    restrict cntx
     )
{
	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mnr   = 48;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	const uint64_t k_iter = k0 / 8;
	const uint64_t k_left = k0 % 8;

	// NOTE: For the purposes of the comments in this packm kernel, we
    // interpret inca and lda as rs_a and cs_a, respectively, and similarly
    // interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
    // this packm kernel, you should think of the operation as packing an
    // m x n micropanel, where m and n are tiny and large, respectively, and
    // where elements of each column of the packed matrix P are contiguous.
    // (This packm kernel can still be used to pack micropanels of matrix B
    // in a gemm operation.)
    //
    // All "row"/"column" and "row-major"/"column-major" language below refers
    // to this m x n view (m = the 48-wide packed dimension = rows; n = k =
    // columns), NOT to B's native K x N storage. rs_a is the stride along m
    // (between adjacent rows) and cs_a is the stride along n (between adjacent
    // columns); thus rs_a == 1 means each column is contiguous (source stored
    // column-major) and cs_a == 1 means each row is contiguous (row-major).
    const uint64_t cs_a   = lda0;
    const uint64_t cs_p   = ldp0;
	const uint64_t rs_a   = inca0;

	const bool     gs     = ( inca0 != 1 && lda0 != 1 );

	// NOTE: If/when this kernel ever supports scaling by kappa within the
	// assembly region, this constraint should be lifted.
	const bool     unitk  = bli_seq1( *kappa );

	// -------------------------------------------------------------------------

	// Handles the case where rs_a == 1 (inca0 == 1), i.e. adjacent m-values
	// (rows, within the same column/k) are contiguous in memory -- i.e. each
	// column of the source micropanel is contiguous (source stored
	// column-major). Since P is likewise column-contiguous, this is a straight
	// copy with no transpose.
	if ( cdim0 == mnr && rs_a == 1 && !gs && unitk )
	{
		begin_asm()

		mov(var(a), rax)                   // load address of a.
		mov(var(cs_a), r10)                // load cs_a
		lea(mem(, r10, 4), r10)            // r10 <- cs_a*sizeof(float)
		mov(var(p), rbx)                   // load address of p.
		mov(var(cs_p), r8)                 // load cs_p
		lea(mem(, r8,  4), r8)             // r8  <- cs_p*sizeof(float)

		lea(mem(r10, r10, 2), r11)         // r11 <- 3*cs_a
		lea(mem(r11, r10, 2), r12)         // r12 <- 5*cs_a
		lea(mem(r11, r10, 4), r13)         // r13 <- 7*cs_a

		lea(mem(r8,  r8,  2), r14)         // r14 <- 3*cs_p
		lea(mem(r14, r8,  2), r15)         // r15 <- 5*cs_p
		lea(mem(r14, r8,  4), r9)          // r9  <- 7*cs_p

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SCONKLEFTCOLU)                 // if i == 0, jump to k_left loop.

		label(.SKITERCOLU)                 // MAIN LOOP (k_iter), 8 columns/iter

		// col0:
		vmovups(mem(rax, 0), zmm0) 			// zmm0 <- a[0:16][col0]
		vmovups(mem(rax, 64), zmm1) 		// zmm1 <- a[16:32][col0]
		vmovups(mem(rax, 128), zmm2) 		// zmm2 <- a[32:48][col0]
		vmovups(zmm0, mem(rbx, 0)) 			// p[0:16][col0]  <- zmm0
		vmovups(zmm1, mem(rbx, 64)) 		// p[16:32][col0] <- zmm1
		vmovups(zmm2, mem(rbx, 128)) 		// p[32:48][col0] <- zmm2

		// col1:
		vmovups(mem(rax, r10, 1, 0), zmm3)     // zmm3 <- a[0:16][col1]
		vmovups(mem(rax, r10, 1, 64), zmm4)    // zmm4 <- a[16:32][col1]
		vmovups(mem(rax, r10, 1, 128), zmm5)   // zmm5 <- a[32:48][col1]
		vmovups(zmm3, mem(rbx, r8,  1, 0))     // p[0:16][col1]  <- zmm3
		vmovups(zmm4, mem(rbx, r8,  1, 64))    // p[16:32][col1] <- zmm4
		vmovups(zmm5, mem(rbx, r8,  1, 128))   // p[32:48][col1] <- zmm5

		// col2:
		vmovups(mem(rax, r10, 2, 0), zmm6)     // zmm6 <- a[0:16][col2]
		vmovups(mem(rax, r10, 2, 64), zmm7)    // zmm7 <- a[16:32][col2]
		vmovups(mem(rax, r10, 2, 128), zmm8)   // zmm8 <- a[32:48][col2]
		vmovups(zmm6, mem(rbx, r8,  2, 0))     // p[0:16][col2]  <- zmm6
		vmovups(zmm7, mem(rbx, r8,  2, 64))    // p[16:32][col2] <- zmm7
		vmovups(zmm8, mem(rbx, r8,  2, 128))   // p[32:48][col2] <- zmm8

		// col3:
		vmovups(mem(rax, r11, 1, 0), zmm9)     // zmm9  <- a[0:16][col3]
		vmovups(mem(rax, r11, 1, 64), zmm10)   // zmm10 <- a[16:32][col3]
		vmovups(mem(rax, r11, 1, 128), zmm11)  // zmm11 <- a[32:48][col3]
		vmovups(zmm9, mem(rbx, r14, 1, 0))     // p[0:16][col3]  <- zmm9
		vmovups(zmm10, mem(rbx, r14, 1, 64))   // p[16:32][col3] <- zmm10
		vmovups(zmm11, mem(rbx, r14, 1, 128))  // p[32:48][col3] <- zmm11

		// col4: 
		vmovups(mem(rax, r10, 4, 0), zmm12)    // zmm12 <- a[0:16][col4]
		vmovups(mem(rax, r10, 4, 64), zmm13)   // zmm13 <- a[16:32][col4]
		vmovups(mem(rax, r10, 4, 128), zmm14)  // zmm14 <- a[32:48][col4]
		vmovups(zmm12, mem(rbx, r8,  4, 0))    // p[0:16][col4]  <- zmm12
		vmovups(zmm13, mem(rbx, r8,  4, 64))   // p[16:32][col4] <- zmm13
		vmovups(zmm14, mem(rbx, r8,  4, 128))  // p[32:48][col4] <- zmm14

		// col5:
		vmovups(mem(rax, r12, 1, 0), zmm15)    // zmm15 <- a[0:16][col5]
		vmovups(mem(rax, r12, 1, 64), zmm16)   // zmm16 <- a[16:32][col5]
		vmovups(mem(rax, r12, 1, 128), zmm17)  // zmm17 <- a[32:48][col5]
		vmovups(zmm15, mem(rbx, r15, 1, 0))    // p[0:16][col5]  <- zmm15
		vmovups(zmm16, mem(rbx, r15, 1, 64))   // p[16:32][col5] <- zmm16
		vmovups(zmm17, mem(rbx, r15, 1, 128))  // p[32:48][col5] <- zmm17

		// col6:
		vmovups(mem(rax, r11, 2, 0), zmm18)    // zmm18 <- a[0:16][col6]
		vmovups(mem(rax, r11, 2, 64), zmm19)   // zmm19 <- a[16:32][col6]
		vmovups(mem(rax, r11, 2, 128), zmm20)  // zmm20 <- a[32:48][col6]
		vmovups(zmm18, mem(rbx, r14, 2, 0))    // p[0:16][col6]  <- zmm18
		vmovups(zmm19, mem(rbx, r14, 2, 64))   // p[16:32][col6] <- zmm19
		vmovups(zmm20, mem(rbx, r14, 2, 128))  // p[32:48][col6] <- zmm20

		// col7: 
		vmovups(mem(rax, r13, 1, 0), zmm21)    // zmm21 <- a[0:16][col7]
		vmovups(mem(rax, r13, 1, 64), zmm22)   // zmm22 <- a[16:32][col7]
		vmovups(mem(rax, r13, 1, 128), zmm23)  // zmm23 <- a[32:48][col7]
		vmovups(zmm21, mem(rbx, r9,  1, 0))    // p[0:16][col7]  <- zmm21
		vmovups(zmm22, mem(rbx, r9,  1, 64))   // p[16:32][col7] <- zmm22
		vmovups(zmm23, mem(rbx, r9,  1, 128))  // p[32:48][col7] <- zmm23

		lea(mem(rax, r10, 8), rax)         // rax <- rax + 8*cs_a
		lea(mem(rbx, r8,  8), rbx)         // rbx <- rbx + 8*cs_p

		dec(rsi)                           // i -= 1;
		jne(.SKITERCOLU)                   // iterate again if i != 0.

		label(.SCONKLEFTCOLU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SDONE)                         // if i == 0, we're done.

		label(.SKLEFTCOLU)                 // EDGE LOOP (k_left)

		vmovups(mem(rax,   0), zmm0)
		vmovups(mem(rax,  64), zmm1)
		vmovups(mem(rax, 128), zmm2)
		vmovups(zmm0, mem(rbx,   0))
		vmovups(zmm1, mem(rbx,  64))
		vmovups(zmm2, mem(rbx, 128))
		add(r10, rax)
		add(r8,  rbx)

		dec(rsi)                           // i -= 1;
		jne(.SKLEFTCOLU)                   // iterate again if i != 0.

		label(.SDONE)

		end_asm(
		: // output operands (none)
		: // input operands
		  [k_iter] "m" (k_iter),
		  [k_left] "m" (k_left),
		  [a]      "m" (a),
		  [cs_a]   "m" (cs_a),
		  [p]      "m" (p),
		  [cs_p]   "m" (cs_p)
		: // register clobber list
		  "rax", "rbx", "rsi", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15",
		  "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", 
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
		  "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
		  "zmm16", "zmm17", "zmm18", "zmm19", "zmm20",
		  "zmm21", "zmm22", "zmm23", "memory"
		)
	}
	else if ( cdim0 == mnr && cs_a == 1 && !gs && unitk )
	{
		// Handles the case where cs_a == 1, i.e. adjacent n-values (columns/k,
		// within the same row m) are contiguous -- i.e. each row of the source
		// micropanel is contiguous (source stored row-major). A 16x16 transpose
		// is needed to produce P's column-contiguous layout.
		const dim_t k_unroll = k0 - ( k0 % 16 );
		dim_t k_base;
		__m512 v[16];

		for ( k_base = 0; k_base < k_unroll; k_base += 16 )
		{
			dim_t row_base;
			for ( row_base = 0; row_base < mnr; row_base += 16 )
			{
				dim_t  row_offset;

				// Load: each of 16 rows (m = row_base + row_offset) supplies 16
				// contiguous columns (k = k_base .. k_base + 15).
				for ( row_offset = 0; row_offset < 16; ++row_offset )
				{
					v[row_offset] = _mm512_loadu_ps( a + (row_base + row_offset)*rs_a + k_base );
				}

				bli_spackm_zen5_transpose_16x16( v );

				// After the transpose, v[i] holds the 16 row-values (m) for a
				// single column k = k_base + i; store each as a contiguous
				// column of P.
				for ( row_offset = 0; row_offset < 16; ++row_offset )
					_mm512_storeu_ps( p + (k_base + row_offset)*cs_p + row_base, v[row_offset] );
			}
		}

		// Handle the k_left tail (fewer than 16 remaining k values).
		for ( ; k_base < k0; ++k_base )
		{
			dim_t j;
			for ( j = 0; j < mnr; ++j )
				p[ k_base*cs_p + j ] = a[ j*rs_a + k_base ];
		}
	}
	else // if ( cdim0 < mnr || general storage || !unitk )
	{
		PASTEMAC(sscal2m,BLIS_TAPI_EX_SUF)
		(
		  0,
		  BLIS_NONUNIT_DIAG,
		  BLIS_DENSE,
		  ( trans_t )conja,
		  cdim0,
		  k0,
		  kappa,
		  a, rs_a, cs_a,
		  p,     1, cs_p,
		  cntx,
		  NULL
		);

		if ( cdim0 < mnr )
		{
			// Handle zero-filling along the "long" edge of the micropanel.
			const dim_t      i      = cdim0;
			const dim_t      m_edge = mnr - cdim0;
			const dim_t      n_edge = k0_max;
			float*  restrict p_edge = p + (i  )*1;

			bli_sset0s_mxn
			(
			  m_edge,
			  n_edge,
			  p_edge, 1, cs_p
			);
		}
	}

	if ( k0 < k0_max )
	{
		// Handle zero-filling along the "short" (far) edge of the micropanel.
		const dim_t      j      = k0;
		const dim_t      m_edge = mnr;
		const dim_t      n_edge = k0_max - k0;
		float*  restrict p_edge = p + (j  )*cs_p;

		bli_sset0s_mxn
		(
		  m_edge,
		  n_edge,
		  p_edge, 1, cs_p
		);
	}
}
