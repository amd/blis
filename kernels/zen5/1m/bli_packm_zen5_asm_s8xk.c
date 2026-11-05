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

// Used in the _mm256_permute2f128_ps(A, B, _VPERM2F128_IMM()) which 
// has two ymm registers as inputs A and B. These register can be
// thought of as having 4 128 bit lanes: A.low, A.high, B.low, B.high
// which corresponds to indexes [0, 1, 2, 3] respectively.
// Use this selector helper to select which 128 bit portion of 
// the inputs (A, B) must be stitched together in the output
// The input args are hi, lo (similar to _MM_SHUFFLE)
// So if you want to copy C <- [A.low, B.high]
// The call should be _VPERM2F128_IMM(3,0)
#define _VPERM2F128_IMM(hi, lo) ( ( (hi) << 4 ) | (lo) )

void bli_spackm_zen5_asm_8xk
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
    const dim_t      mnr   = 8;

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
    const uint64_t rs_a   = inca0;
    const uint64_t cs_a   = lda0;
    const uint64_t cs_p   = ldp0;

    // NOTE: If/when this kernel ever supports scaling by kappa within the
    // assembly region, this constraint should be lifted.
    const bool     unitk  = bli_seq1( *kappa );

    // -------------------------------------------------------------------------

    // Handles the case where rs_a == 1 (each column of A is contiguous) 
    // cs_a may be any stride. Basically A is Col-stored
    if ( cdim0 == mnr && rs_a == 1 && unitk )
    {
        begin_asm()

        mov(var(a), rax)                   // rax <- address of a.
        mov(var(cs_a), r10)                // r10 <- cs_a
        lea(mem(, r10, 4), r10)            // r10 <- r10*sizeof(float)
        mov(var(p), rbx)                   // rbx <- address of p
        mov(var(cs_p), r8)                 // r8 <- cs_p
        lea(mem(, r8,  4), r8)             // r8 <- r8*sizeof(float)

        lea(mem(r10, r10, 2), r11)         // r11 <- 3*lda
        lea(mem(r11, r10, 2), r12)         // r12 <- 5*lda
        lea(mem(r11, r10, 4), r13)         // r13 <- 7*lda

        lea(mem(r8,  r8,  2), r14)         // r14 <- 3*cs_p
        lea(mem(r14, r8,  2), r15)         // r15 <- 5*cs_p
        lea(mem(r14, r8,  4), r9)          // r9  <- 7*cs_p

        mov(var(k_iter), rsi)              // rsi <- k_iter;
        test(rsi, rsi)                     
        je(.SCONKLEFTCOLU)                 // if i == 0, jump to k_left loop.

        label(.SKITERCOLU)                 // MAIN LOOP (k_iter), 8 columns/iter

        // One of the main disadvantages of this kind of kernel is that
        // We waste a lot of memory since loads and stores are half 
        // cache line sizes. To mitigate this, we need to set KC
        // to a low value in the config so that A and P reside in 
        // the L1 cache when we finish one row and begin packing 
        // the next row. But it is observed that from large matrices
        // on large number of threads, this causes a regression
        // so setting to the normal higher value of KC using dynamic
        // block sizes in that case
        vmovups(mem(rax,       0), ymm0)   // ymm0 <- a[0:8][col0]
        vmovups(ymm0, mem(rbx,       0))   // p[0:8][col0] <- ymm0

        vmovups(mem(rax, r10, 1, 0), ymm1) // ymm1 <- a[0:8][col1]
        vmovups(ymm1, mem(rbx, r8,  1, 0)) // p[0:8][col1] <- ymm1

        vmovups(mem(rax, r10, 2, 0), ymm2) // ymm2 <- a[0:8][col2]
        vmovups(ymm2, mem(rbx, r8,  2, 0)) // p[0:8][col2] <- ymm2

        vmovups(mem(rax, r11, 1, 0), ymm3) // ymm3 <- a[0:8][col3]
        vmovups(ymm3, mem(rbx, r14, 1, 0)) // p[0:8][col3] <- ymm3

        vmovups(mem(rax, r10, 4, 0), ymm4) // ymm4 <- a[0:8][col4]
        vmovups(ymm4, mem(rbx, r8,  4, 0)) // p[0:8][col4] <- ymm4

        vmovups(mem(rax, r12, 1, 0), ymm5) // ymm5 <- a[0:8][col5]
        vmovups(ymm5, mem(rbx, r15, 1, 0)) // p[0:8][col5] <- ymm5

        vmovups(mem(rax, r11, 2, 0), ymm6) // ymm6 <- a[0:8][col6]
        vmovups(ymm6, mem(rbx, r14, 2, 0)) // p[0:8][col6] <- ymm6

        vmovups(mem(rax, r13, 1, 0), ymm7) // ymm7 <- a[0:8][col7]
        vmovups(ymm7, mem(rbx, r9,  1, 0)) // p[0:8][col7] <- ymm7

        lea(mem(rax, r10, 8), rax)         // rax <- rax + 8*cs_a
        lea(mem(rbx, r8,  8), rbx)         // rbx <- rbx + 8*cs_p

        dec(rsi)                           // i -= 1;
        jne(.SKITERCOLU)                   // iterate again if i != 0.

        label(.SCONKLEFTCOLU)

        mov(var(k_left), rsi)              // i = k_left;
        test(rsi, rsi)                     
        je(.SDONE)                         // if i == 0, we're done.

        label(.SKLEFTCOLU)                 // EDGE LOOP (k_left)

        vmovups(mem(rax, 0), ymm8)         // ymm8 <- a[:, col] (8 contiguous floats)
        vmovups(ymm8, mem(rbx))            // p[:, col] <- ymm8
        add(r10, rax)                      // rax <- rax + cs_a (advance to next column)
        add(r8,  rbx)                      // rbx <- rbx + cs_p (advance to next column)

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
          "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
          "ymm5", "ymm6", "ymm7", "ymm8", "memory"
        )
    }
    else if ( cdim0 == mnr && cs_a == 1 && rs_a != 1 && unitk )
    {
        // Handles the case where cs_a == 1 (each row of A is contiguous) 
        // rs_a may be any stride. Basically A is Row-stored
        float* restrict ap     = a;
        float* restrict pp     = p;

        const dim_t     p_iter = k0 / 8;
        const dim_t     p_left = k0 % 8;

        for ( dim_t kk = 0; kk < p_iter; ++kk )
        {
            // Load 8 rows, 8 consecutive k-values each.
            __m256 r0 = _mm256_loadu_ps( ap + 0*rs_a );   // r0 <- a[row0][0:8]
            __m256 r1 = _mm256_loadu_ps( ap + 1*rs_a );   // r0 <- a[row1][0:8]
            __m256 r2 = _mm256_loadu_ps( ap + 2*rs_a );   // r0 <- a[row2][0:8]
            __m256 r3 = _mm256_loadu_ps( ap + 3*rs_a );   // r0 <- a[row3][0:8]
            __m256 r4 = _mm256_loadu_ps( ap + 4*rs_a );   // r0 <- a[row4][0:8]
            __m256 r5 = _mm256_loadu_ps( ap + 5*rs_a );   // r0 <- a[row5][0:8]
            __m256 r6 = _mm256_loadu_ps( ap + 6*rs_a );   // r0 <- a[row6][0:8]
            __m256 r7 = _mm256_loadu_ps( ap + 7*rs_a );   // r0 <- a[row7][0:8]

            /*
             Consider the 8 ymm registers above, each holding one row of A
             (8 k-values; a 256-bit ymm has two 128-bit lanes, low = k0..k3,
             high = k4..k7). Using a/b/c/../h for rows 0..7:
               r0 <- [ {a0,a1,a2,a3}, {a4,a5,a6,a7} ]
               r1 <- [ {b0,b1,b2,b3}, {b4,b5,b6,b7} ]
               r2 <- [ {c0,c1,c2,c3}, {c4,c5,c6,c7} ]
               r3 <- [ {d0,d1,d2,d3}, {d4,d5,d6,d7} ]
               r4 <- [ {e0,e1,e2,e3}, {e4,e5,e6,e7} ]
               r5 <- [ {f0,f1,f2,f3}, {f4,f5,f6,f7} ]
               r6 <- [ {g0,g1,g2,g3}, {g4,g5,g6,g7} ]
               r7 <- [ {h0,h1,h2,h3}, {h4,h5,h6,h7} ]

             STAGE 1: unpacklo/unpackhi interleave elements within each 128-bit lane:
               _mm256_unpacklo_ps(r0,r1): t0 <- [ {a0,b0,a1,b1}, {a4,b4,a5,b5} ]
               _mm256_unpackhi_ps(r0,r1): t1 <- [ {a2,b2,a3,b3}, {a6,b6,a7,b7} ]
               Similarly (rows c/d):      t2 <- [ {c0,d0,c1,d1}, {c4,d4,c5,d5} ]
                                          t3 <- [ {c2,d2,c3,d3}, {c6,d6,c7,d7} ]
               Similarly (rows e/f):      t4 <- [ {e0,f0,e1,f1}, {e4,f4,e5,f5} ]
                                          t5 <- [ {e2,f2,e3,f3}, {e6,f6,e7,f7} ]
               Similarly (rows g/h):      t6 <- [ {g0,h0,g1,h1}, {g4,h4,g5,h5} ]
                                          t7 <- [ {g2,h2,g3,h3}, {g6,h6,g7,h7} ]

             STAGE 2: shuffle_ps combines two t-regs into 4-row groups, still
             split across the two 128-bit lanes (low lane = one k-column,
             high lane = a different k-column, for the same 4 rows):
               _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0)):
                 s0 <- [ {a0,b0,c0,d0}, {a4,b4,c4,d4} ]  // col k0 | col k4 
               _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2)):
                 s1 <- [ {a1,b1,c1,d1}, {a5,b5,c5,d5} ]  // col k1 | col k5
               Similarly for the other rows:
                 s2 <- [ {a2,b2,c2,d2}, {a6,b6,c6,d6} ]  // col k2 | col k6
                 s3 <- [ {a3,b3,c3,d3}, {a7,b7,c7,d7} ]  // col k3 | col k7
                 s4 <- [ {e0,f0,g0,h0}, {e4,f4,g4,h4} }  // col k0 | col k4 
                 s5 <- [ {e1,f1,g1,h1}, {e5,f5,g5,h5} }  // col k1 | col k5
                 s6 <- [ {e2,f2,g2,h2}, {e6,f6,g6,h6} }  // col k2 | col k6
                 s7 <- [ {e3,f3,g3,h3}, {e7,f7,g7,h7} }  // col k3 | col k7

             STAGE 3: permute2f128 stitches the row a-d half and row e-h half of
             matching k-columns together into full 8-row columns:
               _mm256_permute2f128_ps(s0,s4,0x20):  // low<-s0.low, high<-s4.low
                 c0 <- [ {a0,b0,c0,d0}, {e0,f0,g0,h0} ]  = full column k0, rows 0..7
               _mm256_permute2f128_ps(s0,s4,0x31):  // low<-s0.high, high<-s4.high
                 c4 <- [ {a4,b4,c4,d4}, {e4,f4,g4,h4} ]  = full column k4, rows 0..7
               Similarly:
                 c1 <- full column k1 (s1,s5,0x20)   c5 <- full column k5 (s1,s5,0x31)
                 c2 <- full column k2 (s2,s6,0x20)   c6 <- full column k6 (s2,s6,0x31)
                 c3 <- full column k3 (s3,s7,0x20)   c7 <- full column k7 (s3,s7,0x31)

             Result: c0..c7 each hold one complete packed column (8 rows),
             i.e. the 8x8 in-register transpose of A's row-stored micropanel
             into P's column-major layout.
            */
            
            // STAGE 1:
            __m256 t0 = _mm256_unpacklo_ps( r0, r1 );
            __m256 t1 = _mm256_unpackhi_ps( r0, r1 );
            __m256 t2 = _mm256_unpacklo_ps( r2, r3 );
            __m256 t3 = _mm256_unpackhi_ps( r2, r3 );
            __m256 t4 = _mm256_unpacklo_ps( r4, r5 );
            __m256 t5 = _mm256_unpackhi_ps( r4, r5 );
            __m256 t6 = _mm256_unpacklo_ps( r6, r7 );
            __m256 t7 = _mm256_unpackhi_ps( r6, r7 );
            
            // STAGE 2:
            __m256 s0 = _mm256_shuffle_ps( t0, t2, _MM_SHUFFLE(1,0,1,0) );
            __m256 s1 = _mm256_shuffle_ps( t0, t2, _MM_SHUFFLE(3,2,3,2) );
            __m256 s2 = _mm256_shuffle_ps( t1, t3, _MM_SHUFFLE(1,0,1,0) );
            __m256 s3 = _mm256_shuffle_ps( t1, t3, _MM_SHUFFLE(3,2,3,2) );
            __m256 s4 = _mm256_shuffle_ps( t4, t6, _MM_SHUFFLE(1,0,1,0) );
            __m256 s5 = _mm256_shuffle_ps( t4, t6, _MM_SHUFFLE(3,2,3,2) );
            __m256 s6 = _mm256_shuffle_ps( t5, t7, _MM_SHUFFLE(1,0,1,0) );
            __m256 s7 = _mm256_shuffle_ps( t5, t7, _MM_SHUFFLE(3,2,3,2) );
            
            // STAGE 3:
            __m256 c0 = _mm256_permute2f128_ps( s0, s4, _VPERM2F128_IMM(2, 0) );
            __m256 c1 = _mm256_permute2f128_ps( s1, s5, _VPERM2F128_IMM(2, 0) );
            __m256 c2 = _mm256_permute2f128_ps( s2, s6, _VPERM2F128_IMM(2, 0) );
            __m256 c3 = _mm256_permute2f128_ps( s3, s7, _VPERM2F128_IMM(2, 0) );
            __m256 c4 = _mm256_permute2f128_ps( s0, s4, _VPERM2F128_IMM(3, 1) );
            __m256 c5 = _mm256_permute2f128_ps( s1, s5, _VPERM2F128_IMM(3, 1) );
            __m256 c6 = _mm256_permute2f128_ps( s2, s6, _VPERM2F128_IMM(3, 1) );
            __m256 c7 = _mm256_permute2f128_ps( s3, s7, _VPERM2F128_IMM(3, 1) );

            // Final Store to P
            _mm256_storeu_ps( pp + 0*cs_p, c0 );
            _mm256_storeu_ps( pp + 1*cs_p, c1 );
            _mm256_storeu_ps( pp + 2*cs_p, c2 );
            _mm256_storeu_ps( pp + 3*cs_p, c3 );
            _mm256_storeu_ps( pp + 4*cs_p, c4 );
            _mm256_storeu_ps( pp + 5*cs_p, c5 );
            _mm256_storeu_ps( pp + 6*cs_p, c6 );
            _mm256_storeu_ps( pp + 7*cs_p, c7 );

            ap += 8*cs_a;     // advance A by 8 along k (contiguous dimension).
            pp += 8*cs_p;     // advance P by 8 packed columns.
        }
        
        // Tail Processing
        const __m256i vindex = _mm256_setr_epi32
        (
          0,        1*(int)rs_a, 2*(int)rs_a, 3*(int)rs_a,
          4*(int)rs_a, 5*(int)rs_a, 6*(int)rs_a, 7*(int)rs_a
        );

        for ( dim_t kk = 0; kk < p_left; ++kk )
        {
          // Perform one gather operation instead of multiple scalar loads
            __m256 col = _mm256_i32gather_ps( ap, vindex, 4 );
            _mm256_storeu_ps( pp, col );

            ap += cs_a;
            pp += cs_p;
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
