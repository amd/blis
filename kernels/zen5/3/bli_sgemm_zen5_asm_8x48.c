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
#include "bli_x86_asm_macros.h"


#define BETA_OPTIMIZATION
#define ENABLE_COL_GEN_STORE

#define LOOP_ALIGN ALIGN32

// The max displacement for A is in SUBITER_1(3)
// which is (8*3+ 9)*4 -> 132
// Similarly, the range for B is (from SUBITER_1(3))
// [(48*4*2) , (48*3+32)*4 + 48*4*2] -> [384, 1088]
// Which means that the theoretical offsets should be
// 66 and 736 respectively, but setting this to the
// nearest power of 2 for now
#define A_ADDITION (64)
#define B_ADDITION (512)

#define TAIL_NITER 24

// Zero all the scratch registers at the beginning of computation
// These are the registers that are used to store the accumulated
// values of the C-matrix
#define ZERO_REGISTERS() \
    VXORPS(ZMM(8) , ZMM(8),  ZMM(8))  \
    VXORPS(ZMM(9) , ZMM(9),  ZMM(9))  \
    VXORPS(ZMM(10), ZMM(10), ZMM(10)) \
    VXORPS(ZMM(11), ZMM(11), ZMM(11)) \
    VXORPS(ZMM(12), ZMM(12), ZMM(12)) \
    VXORPS(ZMM(13), ZMM(13), ZMM(13)) \
    VXORPS(ZMM(14), ZMM(14), ZMM(14)) \
    VXORPS(ZMM(15), ZMM(15), ZMM(15)) \
    VXORPS(ZMM(16), ZMM(16), ZMM(16)) \
    VXORPS(ZMM(17), ZMM(17), ZMM(17)) \
    VXORPS(ZMM(18), ZMM(18), ZMM(18)) \
    VXORPS(ZMM(19), ZMM(19), ZMM(19)) \
    VXORPS(ZMM(20), ZMM(20), ZMM(20)) \
    VXORPS(ZMM(21), ZMM(21), ZMM(21)) \
    VXORPS(ZMM(22), ZMM(22), ZMM(22)) \
    VXORPS(ZMM(23), ZMM(23), ZMM(23)) \
    VXORPS(ZMM(24), ZMM(24), ZMM(24)) \
    VXORPS(ZMM(25), ZMM(25), ZMM(25)) \
    VXORPS(ZMM(26), ZMM(26), ZMM(26)) \
    VXORPS(ZMM(27), ZMM(27), ZMM(27)) \
    VXORPS(ZMM(28), ZMM(28), ZMM(28)) \
    VXORPS(ZMM(29), ZMM(29), ZMM(29)) \
    VXORPS(ZMM(30), ZMM(30), ZMM(30)) \
    VXORPS(ZMM(31), ZMM(31), ZMM(31))


#define SUBITER_0(n) \
                                                                                                   \
    VFMADD231PS(ZMM( 8), ZMM(0), ZMM(6))                        /* zmm8 <- zmm8 + zmm0*zmm6 */     \
    VFMADD231PS(ZMM( 9), ZMM(1), ZMM(6))                        /* zmm9 <- zmm9 + zmm1*zmm6 */     \
    VFMADD231PS(ZMM(10), ZMM(2), ZMM(6))                        /* zmm10 <- zmm10 + zmm2*zmm6 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 2)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n][2]) */   \
    VFMADD231PS(ZMM(11), ZMM(0), ZMM(7))                        /* zmm11 <- zmm11 + zmm0*zmm7 */   \
    VFMADD231PS(ZMM(12), ZMM(1), ZMM(7))                        /* zmm12 <- zmm12 + zmm1*zmm7 */   \
    VFMADD231PS(ZMM(13), ZMM(2), ZMM(7))                        /* zmm13 <- zmm13 + zmm2*zmm7 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 3)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n][3]) */   \
    VFMADD231PS(ZMM(14), ZMM(0), ZMM(6))                        /* zmm14 <- zmm14 + zmm0*zmm6 */   \
    VFMADD231PS(ZMM(15), ZMM(1), ZMM(6))                        /* zmm15 <- zmm15 + zmm1*zmm6 */   \
    VFMADD231PS(ZMM(16), ZMM(2), ZMM(6))                        /* zmm16 <- zmm16 + zmm2*zmm6 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 4)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n][4]) */   \
    VFMADD231PS(ZMM(17), ZMM(0), ZMM(7))                        /* zmm17 <- zmm17 + zmm0*zmm7 */   \
    VFMADD231PS(ZMM(18), ZMM(1), ZMM(7))                        /* zmm18 <- zmm18 + zmm1*zmm7 */   \
    VFMADD231PS(ZMM(19), ZMM(2), ZMM(7))                        /* zmm19 <- zmm19 + zmm2*zmm7 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 5)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n][5]) */   \
    VFMADD231PS(ZMM(20), ZMM(0), ZMM(6))                        /* zmm20 <- zmm20 + zmm0*zmm6 */   \
    VFMADD231PS(ZMM(21), ZMM(1), ZMM(6))                        /* zmm21 <- zmm21 + zmm1*zmm6 */   \
    VFMADD231PS(ZMM(22), ZMM(2), ZMM(6))                        /* zmm22 <- zmm22 + zmm2*zmm6 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 6)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n][6]) */   \
    VFMADD231PS(ZMM(23), ZMM(0), ZMM(7))                        /* zmm23 <- zmm23 + zmm0*zmm7 */   \
    VFMADD231PS(ZMM(24), ZMM(1), ZMM(7))                        /* zmm24 <- zmm24 + zmm1*zmm7 */   \
    VFMADD231PS(ZMM(25), ZMM(2), ZMM(7))                        /* zmm25 <- zmm25 + zmm2*zmm7 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 7)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n][7]) */   \
    VFMADD231PS(ZMM(26), ZMM(0), ZMM(6))                        /* zmm26 <- zmm26 + zmm0*zmm6 */   \
    VFMADD231PS(ZMM(27), ZMM(1), ZMM(6))                        /* zmm27 <- zmm27 + zmm1*zmm6 */   \
    VFMADD231PS(ZMM(28), ZMM(2), ZMM(6))                        /* zmm28 <- zmm28 + zmm2*zmm6 */   \
                                                                                                   \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 8)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n+1][0]) */ \
    VFMADD231PS(ZMM(29), ZMM(0), ZMM(7))                        /* zmm29 <- zmm29 + zmm0*zmm7 */   \
    VFMADD231PS(ZMM(30), ZMM(1), ZMM(7))                        /* zmm30 <- zmm30 + zmm1*zmm7 */   \
    VFMADD231PS(ZMM(31), ZMM(2), ZMM(7))                        /* zmm31 <- zmm31 + zmm2*zmm7 */   \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 9)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n+1][1]) */ \
    VMOVAPS(ZMM(0), MEM(RBX,(48*n+0 )*4 - B_ADDITION + 48*4*2)) /* zmm0 <- b[n+2][0:15] */         \
    VMOVAPS(ZMM(1), MEM(RBX,(48*n+16)*4 - B_ADDITION + 48*4*2)) /* zmm1 <- b[n+2][16:31] */        \
    VMOVAPS(ZMM(2), MEM(RBX,(48*n+32)*4 - B_ADDITION + 48*4*2)) /* zmm2 <- b[n+2][32:47] */        \


#define SUBITER_1(n) \
                                                                                                                       \
    VFMADD231PS(ZMM( 8), ZMM(3), ZMM(6))                        /* zmm8 <- zmm8 + zmm3*zmm6 */                         \
    VFMADD231PS(ZMM( 9), ZMM(4), ZMM(6))                        /* zmm9 <- zmm9 + zmm4*zmm6 */                         \
    VFMADD231PS(ZMM(10), ZMM(5), ZMM(6))                        /* zmm10 <- zmm10 + zmm5*zmm6 */                       \
                                                                                                                       \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 2)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n][2]) */                       \
    VFMADD231PS(ZMM(11), ZMM(3), ZMM(7))                        /* zmm11 <- zmm11 + zmm3*zmm7 */                       \
    VFMADD231PS(ZMM(12), ZMM(4), ZMM(7))                        /* zmm12 <- zmm12 + zmm4*zmm7 */                       \
    VFMADD231PS(ZMM(13), ZMM(5), ZMM(7))                        /* zmm13 <- zmm13 + zmm5*zmm7 */                       \
                                                                                                                       \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 3)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n][3]) */                       \
    VFMADD231PS(ZMM(14), ZMM(3), ZMM(6))                        /* zmm14 <- zmm14 + zmm3*zmm6 */                       \
    VFMADD231PS(ZMM(15), ZMM(4), ZMM(6))                        /* zmm15 <- zmm15 + zmm4*zmm6 */                       \
    VFMADD231PS(ZMM(16), ZMM(5), ZMM(6))                        /* zmm16 <- zmm16 + zmm5*zmm6 */                       \
                                                                                                                       \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 4)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n][4]) */                       \
    VFMADD231PS(ZMM(17), ZMM(3), ZMM(7))                        /* zmm17 <- zmm17 + zmm3*zmm7 */                       \
    VFMADD231PS(ZMM(18), ZMM(4), ZMM(7))                        /* zmm18 <- zmm18 + zmm4*zmm7 */                       \
    VFMADD231PS(ZMM(19), ZMM(5), ZMM(7))                        /* zmm19 <- zmm19 + zmm5*zmm7 */                       \
                                                                                                                       \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 5)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n][5]) */                       \
    VFMADD231PS(ZMM(20), ZMM(3), ZMM(6))                        /* zmm20 <- zmm20 + zmm3*zmm6 */                       \
    VFMADD231PS(ZMM(21), ZMM(4), ZMM(6))                        /* zmm21 <- zmm21 + zmm4*zmm6 */                       \
    VFMADD231PS(ZMM(22), ZMM(5), ZMM(6))                        /* zmm22 <- zmm22 + zmm5*zmm6 */                       \
                                                                                                                       \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 6)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n][6]) */                       \
    VFMADD231PS(ZMM(23), ZMM(3), ZMM(7))                        /* zmm23 <- zmm23 + zmm3*zmm7 */                       \
    VFMADD231PS(ZMM(24), ZMM(4), ZMM(7))                        /* zmm24 <- zmm24 + zmm4*zmm7 */                       \
    VFMADD231PS(ZMM(25), ZMM(5), ZMM(7))                        /* zmm25 <- zmm25 + zmm5*zmm7 */                       \
                                                                                                                       \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 7)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n][7]) */                       \
    VFMADD231PS(ZMM(26), ZMM(3), ZMM(6))                        /* zmm26 <- zmm26 + zmm3*zmm6 */                       \
    VFMADD231PS(ZMM(27), ZMM(4), ZMM(6))                        /* zmm27 <- zmm27 + zmm4*zmm6 */                       \
    VFMADD231PS(ZMM(28), ZMM(5), ZMM(6))                        /* zmm28 <- zmm28 + zmm5*zmm6 */                       \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*n+ 8)*4 - A_ADDITION))      /* zmm6 <- broadcast(a[n+1][0]) */                     \
                                                                                                                       \
    VFMADD231PS(ZMM(29), ZMM(3), ZMM(7))                        /* zmm29 <- zmm29 + zmm3*zmm7 */                       \
    VFMADD231PS(ZMM(30), ZMM(4), ZMM(7))                        /* zmm30 <- zmm30 + zmm4*zmm7 */                       \
    VFMADD231PS(ZMM(31), ZMM(5), ZMM(7))                        /* zmm31 <- zmm31 + zmm5*zmm7 */                       \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*n+ 9)*4 - A_ADDITION))      /* zmm7 <- broadcast(a[n+1][1]) */                     \
    VMOVAPS(ZMM(3), MEM(RBX,(48*n+0 )*4 - B_ADDITION + 48*4*2)) /* zmm3 <- b[n+2][0:15] */                             \
    VMOVAPS(ZMM(4), MEM(RBX,(48*n+16)*4 - B_ADDITION + 48*4*2)) /* zmm4 <- b[n+2][16:31] */                            \
    VMOVAPS(ZMM(5), MEM(RBX,(48*n+32)*4 - B_ADDITION + 48*4*2)) /* zmm5 <- b[n+2][32:47] */                            \
                                                                /*48*4*2 is preload offset compensated for B preload*/ \


#define K_LOOP() \
    /* pre-load two rows of B */ \
    VMOVAPS(ZMM(0), MEM(RBX,(48*0+0 )*4)) /* zmm0 <- b[0][0:15] */  \
    VMOVAPS(ZMM(1), MEM(RBX,(48*0+16)*4)) /* zmm1 <- b[0][16:31] */ \
    VMOVAPS(ZMM(2), MEM(RBX,(48*0+32)*4)) /* zmm2 <- b[0][32:47] */ \
                                                                    \
    VMOVAPS(ZMM(3), MEM(RBX,(48*1+0 )*4)) /* zmm3 <- b[1][0:15] */  \
    VMOVAPS(ZMM(4), MEM(RBX,(48*1+16)*4)) /* zmm3 <- b[1][16:31] */ \
    VMOVAPS(ZMM(5), MEM(RBX,(48*1+32)*4)) /* zmm3 <- b[1][32:47] */ \
                                              /* pre-load A */ \
    VBROADCASTSS(ZMM(6), MEM(RAX,(8*0+0)*4)) /* zmm6 <- broadcast(a[0][0]) */ \
    VBROADCASTSS(ZMM(7), MEM(RAX,(8*0+1)*4)) /* zmm7 <- broadcast(a[0][1]) */ \
                                                                              \
    /* offset address of A and B forward so that negative addresses */ \
    /* can be used inside the loop */                                  \
    ADD(RBX, IMM( 0+B_ADDITION ))                                                            \
    ADD(RAX, IMM( 0+A_ADDITION ))                                                            \
                                                                                             \
    MOV(R13, VAR(k))                                                                         \
    MOV(R14, R13)                                                                            \
    AND(R14, IMM(3))  /* R14 <- k % 4 */                                                     \
    SAR(R13, IMM(2))  /* R13 <- k / 4 (k_iters) */                                           \
                                                                                             \
    MOV(RDX, IMM(8))  /* For row stride, prefect iters <- 8, needed for the 8 rows of C */   \
    MOV(RDI, IMM(64)) /* For row stride, we need to offest by 16 floats at each iteration */ \
                                                                                             \
    CMP(R10, IMM(4))                                                                         \
    JNZ(POST_STRIDE)  /* Jump if rs_c != 4*/                                                 \
        MOV(RDX, IMM(48)) /* For col stride, prefect iters <- 48, needed for the 48 cols of C */                         \
        MOV(RDI, IMM(0))  /* For col stride, we need to offest 0 since the loop itself fetches the required cols of C */ \
        MOV(R10, var(cs_c))                                                                                              \
    LABEL(POST_STRIDE)                                             \
	                                                               \
    SUB(R13, RDX)               /* R13 <- R13 - prefetch iters */  \
    SUB(R13, IMM(0+TAIL_NITER)) /* R13 <- R13 - tail iterations */ \
    JLE(K_PREFETCH)                                                \
                                                                   \
        LOOP_ALIGN /*Main loop*/ \
        LABEL(LOOP1)             \
                                 \
            SUBITER_0(0) /* k=0 */    \
            SUBITER_1(1) /* k=1 */    \
            SUBITER_0(2) /* k=2 */    \
            SUBITER_1(3) /* k=3 */    \
                                      \
            LEA(RAX, MEM(RAX,4*8*4))  \
            LEA(RBX, MEM(RBX,4*48*4)) \
            DEC(R13)                  \
                                      \
        JNZ(LOOP1) \
                   \
    LABEL(K_PREFETCH)                                 \
                                                      \
    ADD(R13, RDX) /* R13(k_iter) += prefetch_iters */ \
    JLE(K_TAIL)                                       \
                                                      \
        LOOP_ALIGN                     \
        LABEL(LOOP2) /*Prefetch loop*/ \
                                       \
            PREFETCHW0(MEM(R12))       \
            SUBITER_0(0)               \
            PREFETCHW0(MEM(R12,RDI,1)) \
            SUBITER_1(1)               \
            PREFETCHW0(MEM(R12,RDI,2)) \
            SUBITER_0(2)               \
            SUBITER_1(3)               \
                                       \
            LEA(RAX, MEM(RAX,4*8*4))   \
            LEA(RBX, MEM(RBX,4*48*4))  \
            LEA(R12, MEM(R12,R10,1))   \
            DEC(R13)                   \
                                       \
        JNZ(LOOP2) \
                   \
    LABEL(K_TAIL)                                                      \
                                                                       \
    ADD(R13, IMM(0+TAIL_NITER)) /* R13(k_iter) += TAIL_ITER */         \
    JLE(POST_K)                 /* jump to TAIL loop if k_iter <= 0 */ \
                                                                       \
        LOOP_ALIGN                     \
        LABEL(LOOP3) /*Leftover loop*/ \
                                       \
            SUBITER_0(0) /* k=0 */    \
            SUBITER_1(1) /* k=1 */    \
            SUBITER_0(2) /* k=2 */    \
            SUBITER_1(3) /* k=3 */    \
                                      \
            LEA(RAX, MEM(RAX,4*8*4))  \
            LEA(RBX, MEM(RBX,4*48*4)) \
            DEC(R13)                  \
                                      \
        JNZ(LOOP3) \
                   \
    LABEL(POST_K)  \
                   \
    TEST(R14, R14) \
    JZ(POSTACCUM)  \
        /* Only SUBITER_0 is used in this loop, */         \
        /* therefore negative offset is done for 1 iter */ \
        /* of K only(48*4) */                              \
        SUB(RBX, IMM(48*4)) /* rbx -> prev 4th row of b */                             \
        LOOP_ALIGN                                                                     \
        LABEL(LOOP4)        /* Handles the remainder (when k is not divisible by 4) */ \
                                                                                       \
            SUBITER_0(0) /*k=0 */   \
                                    \
            LEA(RAX, MEM(RAX,8*4))  \
            LEA(RBX, MEM(RBX,48*4)) \
            DEC(R14)                \
                                    \
        JNZ(LOOP4) \
    LABEL(POSTACCUM) \
    MOV(R10, VAR(rs_c)) /* load ldc into R10*/


 /*
 Consider two zmm vectors as follows (has four lanes of 4 floats each)
 R0 <- [ {a0, a1, a2, a3}, {a4, a5, a6, a7}, {a8, a9, a10, a11}, {a12, a13, a14, a15} ]
 R1 <- [ {b0, b1, b2, b3}, {b4, b5, b6, b7}, {b8, b9, b10, b11}, {b12, b13, b14, b15} ]
 R2 <- [ {c0, c1, c2, c3}, {c4, c5, c6, c7}, {c8, c9, c10, c11}, {c12, c13, c14, c15} ]
 R3 <- [ {d0, d1, d2, d3}, {d4, d5, d6, d7}, {d8, d9, d10, d11}, {d12, d13, d14, d15} ]

STAGE 1:
    Interleave low 32 bits (independently per-lane)
    VUNPCKLPS(T0, R0, R1):
        T0 <- [ {a0, b0, a1, b1}, {a4, b4, a5, b5}, {a8, b8, a9, b9}, {a12, b12, a13, b13} ]
    Interleave high 32 bits (independently per-lane)
    VUNPCKHPS(T1, R0, R1):
        T1 <- [ {a2, b2, a3, b3}, {a6, b6, a7, b7}, {a10, b10, a11, b11}, {a14, b14, a15, b15} ]

    Similarly:
        T2 <- [ {c0, d0, c1, d1}, {c4, d4, c5, d5}, {c8, d8, c9, d9}, {c12, d12, c13, d13} ]
        T3 <- [ {c2, d2, c3, d3}, {c6, d6, c7, d7}, {c10, d10, c11, d11}, {c14, d14, c15, d15} ]

    Note: data in Rx is no longer needed after this stage as they are interleaved in Tx

STAGE 2:
    Same interleave pattern, but based on 64 bit chunks instead of 32.

    From the previous stage:
        T0 <- [ {a0, b0, a1, b1}, {a4, b4, a5, b5}, {a8, b8, a9, b9}, {a12, b12, a13, b13} ]
        T2 <- [ {c0, d0, c1, d1}, {c4, d4, c5, d5}, {c8, d8, c9, d9}, {c12, d12, c13, d13} ]
    Each 128-bit lane is now viewed as two 64-bit (double precision) halves

    VUNPCKLPD(R0, T0, T2):
        R0 <- [ {a0, b0, c0, d0}, {a4, b4, c4, d4}, {a8, b8, c8, d8}, {a12, b12, c12, d12} ]
    VUNPCKHPD(R1, T0, T2):
        R1 <- [ {a1, b1, c1, d1}, {a5, b5, c5, d5}, {a9, b9, c9, d9}, {a13, b13, c13, d13} ]

    Similarly:
        R4 <- [ {e0, f0, g0, h0}, {e4, f4, g4, h4}, {e8, f8, g8, h8}, {e12, f12, g12, h12} ]
        R5 <- [ {e1, f1, g1, h1}, {e5, f5, g5, h5}, {e9, f9, g9, h9}, {e13, f13, g13, h13} ]

STAGE 3/4:
    Unlike VUNPCKL/HPS/PD (which only riffle elements *within* a lane),
    VSHUFF32X4(dst, SRC1, SRC2, imm) moves whole 128-bit lanes and can pull
    from either source: dst's low two lanes come from SRC1 (imm bits [1:0]
    and [3:2] each pick one of SRC1's 4 lanes), dst's high two lanes come
    from SRC2 (imm bits [5:4] and [7:6] pick from SRC2).

    VSHUFF32X4(T0, R0, R4, 0x88):   // imm 0x88 -> SRC1.lane0, SRC1.lane2, SRC2.lane0, SRC2.lane2
        T0 <- [ {a0, b0, c0, d0}, {a8, b8, c8, d8}, {e0, f0, g0, h0}, {e8, f8, g8, h8} ]

    VSHUFF32X4(R0, T0, T0, 0xD8):   // imm 0xD8 -> SRC1.lane0, SRC1.lane2, SRC2.lane1, SRC2.lane3 (SRC1==SRC2==T0 here)
        R0 <- [ {a0, b0, c0, d0}, {e0, f0, g0, h0}, {a8, b8, c8, d8}, {e8, f8, g8, h8} ]
        i.e. R0 = [ col0 | col8 ]

    Finally, On exit each output register holds two transposed C cols, one in each 256-bit half:
      R0 = [ col0  | col8  ]     R4 = [ col4  | col12 ]
      R1 = [ col1  | col9  ]     R5 = [ col5  | col13 ]
      R2 = [ col2  | col10 ]     R6 = [ col6  | col14 ]
      R3 = [ col3  | col11 ]     R7 = [ col7  | col15 ]

    Remember, we are trying to achive the following result the input is 8 zmm registers
    that holds 8x16 floats in row-major order. We want to transpose this so that we end up
    with 16 ymm registers in column major order for the same 8x16 storage. This is done
    by having an interleaved columns of C in the zmm registers which we will de-interleave
    when doing the final storage to C
 */
#define TRANSPOSE_8x16(R0, R1, R2, R3, R4, R5, R6, R7, T0, T1, T2, T3, T4, T5, T6, T7) \
    /* STAGE 1 */ \
    VUNPCKLPS(ZMM(T0), ZMM(R0), ZMM(R1)) \
    VUNPCKHPS(ZMM(T1), ZMM(R0), ZMM(R1)) \
    VUNPCKLPS(ZMM(T2), ZMM(R2), ZMM(R3)) \
    VUNPCKHPS(ZMM(T3), ZMM(R2), ZMM(R3)) \
    VUNPCKLPS(ZMM(T4), ZMM(R4), ZMM(R5)) \
    VUNPCKHPS(ZMM(T5), ZMM(R4), ZMM(R5)) \
    VUNPCKLPS(ZMM(T6), ZMM(R6), ZMM(R7)) \
    VUNPCKHPS(ZMM(T7), ZMM(R6), ZMM(R7)) \
    /* STAGE 2 */ \
    VUNPCKLPD(ZMM(R0), ZMM(T0), ZMM(T2)) \
    VUNPCKHPD(ZMM(R1), ZMM(T0), ZMM(T2)) \
    VUNPCKLPD(ZMM(R2), ZMM(T1), ZMM(T3)) \
    VUNPCKHPD(ZMM(R3), ZMM(T1), ZMM(T3)) \
    VUNPCKLPD(ZMM(R4), ZMM(T4), ZMM(T6)) \
    VUNPCKHPD(ZMM(R5), ZMM(T4), ZMM(T6)) \
    VUNPCKLPD(ZMM(R6), ZMM(T5), ZMM(T7)) \
    VUNPCKHPD(ZMM(R7), ZMM(T5), ZMM(T7)) \
    /* STAGE 3 */ \
    VSHUFF32X4(ZMM(T0), ZMM(R0), ZMM(R4), IMM(0x88)) \
    VSHUFF32X4(ZMM(T1), ZMM(R0), ZMM(R4), IMM(0xDD)) \
    VSHUFF32X4(ZMM(R0), ZMM(T0), ZMM(T0), IMM(0xD8)) \
    VSHUFF32X4(ZMM(R4), ZMM(T1), ZMM(T1), IMM(0xD8)) \
    VSHUFF32X4(ZMM(T2), ZMM(R1), ZMM(R5), IMM(0x88)) \
    VSHUFF32X4(ZMM(T3), ZMM(R1), ZMM(R5), IMM(0xDD)) \
    VSHUFF32X4(ZMM(R1), ZMM(T2), ZMM(T2), IMM(0xD8)) \
    VSHUFF32X4(ZMM(R5), ZMM(T3), ZMM(T3), IMM(0xD8)) \
    VSHUFF32X4(ZMM(T4), ZMM(R2), ZMM(R6), IMM(0x88)) \
    VSHUFF32X4(ZMM(T5), ZMM(R2), ZMM(R6), IMM(0xDD)) \
    VSHUFF32X4(ZMM(R2), ZMM(T4), ZMM(T4), IMM(0xD8)) \
    VSHUFF32X4(ZMM(R6), ZMM(T5), ZMM(T5), IMM(0xD8)) \
    VSHUFF32X4(ZMM(T6), ZMM(R3), ZMM(R7), IMM(0x88)) \
    VSHUFF32X4(ZMM(T7), ZMM(R3), ZMM(R7), IMM(0xDD)) \
    VSHUFF32X4(ZMM(R3), ZMM(T6), ZMM(T6), IMM(0xD8)) \
    VSHUFF32X4(ZMM(R7), ZMM(T7), ZMM(T7), IMM(0xD8))


/*
beta != 0 case
The input to this macro is the result of TRANSPOSE_8x16:
      R0 = [ col0  | col8  ]     R4 = [ col4  | col12 ]
      R1 = [ col1  | col9  ]     R5 = [ col5  | col13 ]
      R2 = [ col2  | col10 ]     R6 = [ col6  | col14 ]
      R3 = [ col3  | col11 ]     R7 = [ col7  | col15 ]
*/
#define STORE_COL_16(R0, R1, R2, R3, R4, R5, R6, R7, T0, Rbeta) \
    LEA(RAX, MEM(RCX, R12, 8))                          /* RAX <- RCX + 8*cs_c (col offset by 8 since this is how the registers store data)*/  \
    VEXTRACTF32X8(YMM(T0), ZMM(R0), IMM(1))             /* ymm_T0 <- R0[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R0), YMM(Rbeta), MEM(RCX))          /* ymm_R0 <- ymm_R0 + beta*C[0:8][0] */                                                \
    VMOVUPS(MEM(RCX), YMM(R0))                          /* C[0:8][0] <- ymm_R0 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX))          /* ymm_T0 <- ymm_T0 + beta*C[0:8][8] */                                                \
    VMOVUPS(MEM(RAX), YMM(T0))                          /* C[0:8][8] <- ymm_T0 */                                                              \
    VEXTRACTF32X8(YMM(T0), ZMM(R1), IMM(1))             /* ymm_T0 <- R1[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R1), YMM(Rbeta), MEM(RCX, R12, 1))  /* ymm_R1 <- ymm_R1 + beta*C[0:8][1] */                                                \
    VMOVUPS(MEM(RCX, R12, 1), YMM(R1))                  /* C[0:8][1] <- ymm_R1 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, R12, 1))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][9] */                                                \
    VMOVUPS(MEM(RAX, R12, 1), YMM(T0))                  /* C[0:8][9] <- ymm_T0 */                                                              \
    VEXTRACTF32X8(YMM(T0), ZMM(R2), IMM(1))             /* ymm_T0 <- R2[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R2), YMM(Rbeta), MEM(RCX, R12, 2))  /* ymm_R2 <- ymm_R2 + beta*C[0:8][2] */                                                \
    VMOVUPS(MEM(RCX, R12, 2), YMM(R2))                  /* C[0:8][2] <- ymm_R2 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, R12, 2))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][10] */                                               \
    VMOVUPS(MEM(RAX, R12, 2), YMM(T0))                  /* C[0:8][10] <- ymm_T0 */                                                             \
    VEXTRACTF32X8(YMM(T0), ZMM(R3), IMM(1))             /* ymm_T0 <- R3[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R3), YMM(Rbeta), MEM(RCX, R13, 1))  /* ymm_R3 <- ymm_R3 + beta*C[0:8][3] */                                                \
    VMOVUPS(MEM(RCX, R13, 1), YMM(R3))                  /* C[0:8][3] <- ymm_R3 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, R13, 1))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][11] */                                               \
    VMOVUPS(MEM(RAX, R13, 1), YMM(T0))                  /* C[0:8][11] <- ymm_T0 */                                                             \
    VEXTRACTF32X8(YMM(T0), ZMM(R4), IMM(1))             /* ymm_T0 <- R4[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R4), YMM(Rbeta), MEM(RCX, R12, 4))  /* ymm_R4 <- ymm_R4 + beta*C[0:8][4] */                                                \
    VMOVUPS(MEM(RCX, R12, 4), YMM(R4))                  /* C[0:8][4] <- ymm_R4 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, R12, 4))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][12] */                                               \
    VMOVUPS(MEM(RAX, R12, 4), YMM(T0))                  /* C[0:8][12] <- ymm_T0 */                                                             \
    VEXTRACTF32X8(YMM(T0), ZMM(R5), IMM(1))             /* ymm_T0 <- R5[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R5), YMM(Rbeta), MEM(RCX, RDX, 1))  /* ymm_R5 <- ymm_R5 + beta*C[0:8][5] */                                                \
    VMOVUPS(MEM(RCX, RDX, 1), YMM(R5))                  /* C[0:8][5] <- ymm_R5 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, RDX, 1))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][13] */                                               \
    VMOVUPS(MEM(RAX, RDX, 1), YMM(T0))                  /* C[0:8][13] <- ymm_T0 */                                                             \
    VEXTRACTF32X8(YMM(T0), ZMM(R6), IMM(1))             /* ymm_T0 <- R6[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R6), YMM(Rbeta), MEM(RCX, R13, 2))  /* ymm_R6 <- ymm_R6 + beta*C[0:8][6] */                                                \
    VMOVUPS(MEM(RCX, R13, 2), YMM(R6))                  /* C[0:8][6] <- ymm_R6 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, R13, 2))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][14] */                                               \
    VMOVUPS(MEM(RAX, R13, 2), YMM(T0))                  /* C[0:8][14] <- ymm_T0 */                                                             \
    VEXTRACTF32X8(YMM(T0), ZMM(R7), IMM(1))             /* ymm_T0 <- R7[8:15] (upper half) */                                                  \
    VFMADD231PS(YMM(R7), YMM(Rbeta), MEM(RCX, R14, 1))  /* ymm_R7 <- ymm_R7 + beta*C[0:8][7] */                                                \
    VMOVUPS(MEM(RCX, R14, 1), YMM(R7))                  /* C[0:8][7] <- ymm_R7 */                                                              \
    VFMADD231PS(YMM(T0), YMM(Rbeta), MEM(RAX, R14, 1))  /* ymm_T0 <- ymm_T0 + beta*C[0:8][15] */                                               \
    VMOVUPS(MEM(RAX, R14, 1), YMM(T0))                  /* C[0:8][15] <- ymm_T0 */                                                             \
    LEA(RCX, MEM(RAX, R12, 8))                          /* RCX <- RAX + 8*cs_c (advance RCX 16 columns) */

/*
beta == 0 case
The input to this macro is the result of TRANSPOSE_8x16:
      R0 = [ col0  | col8  ]     R4 = [ col4  | col12 ]
      R1 = [ col1  | col9  ]     R5 = [ col5  | col13 ]
      R2 = [ col2  | col10 ]     R6 = [ col6  | col14 ]
      R3 = [ col3  | col11 ]     R7 = [ col7  | col15 ]
*/
#define STORE_COL_16_BZ(R0, R1, R2, R3, R4, R5, R6, R7, T0) \
    LEA(RAX, MEM(RCX, R12, 8))               /* base2 = RCX + 8*cs_c (cols 8..15) */  \
    VEXTRACTF32X8(YMM(T0), ZMM(R0), IMM(1))  /* ymm_T0 <- R0[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX), YMM(R0))               /* C[0:8][0] <- ymm_R0 */                \
    VMOVUPS(MEM(RAX), YMM(T0))               /* C[0:8][8] <- ymm_T0 */                \
    VEXTRACTF32X8(YMM(T0), ZMM(R1), IMM(1))  /* ymm_T0 <- R1[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, R12, 1), YMM(R1))       /* C[0:8][1] <- ymm_R1 */                \
    VMOVUPS(MEM(RAX, R12, 1), YMM(T0))       /* C[0:8][9] <- ymm_T0 */                \
    VEXTRACTF32X8(YMM(T0), ZMM(R2), IMM(1))  /* ymm_T0 <- R2[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, R12, 2), YMM(R2))       /* C[0:8][2] <- ymm_R2 */                \
    VMOVUPS(MEM(RAX, R12, 2), YMM(T0))       /* C[0:8][10] <- ymm_T0 */               \
    VEXTRACTF32X8(YMM(T0), ZMM(R3), IMM(1))  /* ymm_T0 <- R3[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, R13, 1), YMM(R3))       /* C[0:8][3] <- ymm_R3 */                \
    VMOVUPS(MEM(RAX, R13, 1), YMM(T0))       /* C[0:8][11] <- ymm_T0 */               \
    VEXTRACTF32X8(YMM(T0), ZMM(R4), IMM(1))  /* ymm_T0 <- R4[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, R12, 4), YMM(R4))       /* C[0:8][4] <- ymm_R4 */                \
    VMOVUPS(MEM(RAX, R12, 4), YMM(T0))       /* C[0:8][12] <- ymm_T0 */               \
    VEXTRACTF32X8(YMM(T0), ZMM(R5), IMM(1))  /* ymm_T0 <- R5[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, RDX, 1), YMM(R5))       /* C[0:8][5] <- ymm_R5 */                \
    VMOVUPS(MEM(RAX, RDX, 1), YMM(T0))       /* C[0:8][13] <- ymm_T0 */               \
    VEXTRACTF32X8(YMM(T0), ZMM(R6), IMM(1))  /* ymm_T0 <- R6[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, R13, 2), YMM(R6))       /* C[0:8][6] <- ymm_R6 */                \
    VMOVUPS(MEM(RAX, R13, 2), YMM(T0))       /* C[0:8][14] <- ymm_T0 */               \
    VEXTRACTF32X8(YMM(T0), ZMM(R7), IMM(1))  /* ymm_T0 <- R7[8:15] (upper half) */    \
    VMOVUPS(MEM(RCX, R14, 1), YMM(R7))       /* C[0:8][7] <- ymm_R7 */                \
    VMOVUPS(MEM(RAX, R14, 1), YMM(T0))       /* C[0:8][15] <- ymm_T0 */               \
    LEA(RCX, MEM(RAX, R12, 8))               /* RCX <- RAX + 8*cs_c (advance RCX 16 columns) */


// Update C when C is general stored
#define UPDATE_C_SCATTERED(R1,R2,R3) \
\
 /* In x86, the gather operations reset the mask to 0, so we have to set to 1 everytime */ \
    KXNORW(K(1), K(0), K(0))                        /* k1 <- 1 (re-arm mask) */               \
    KXNORW(K(2), K(0), K(0))                        /* k2 <- 1 (re-arm mask) */               \
    KXNORW(K(3), K(0), K(0))                        /* k3 <- 1 (re-arm mask) */               \
    VGATHERDPS(ZMM(0) MASK_K(1), MEM(RCX,ZMM(3),1)) /* zmm0 <- C[0:15] */                     \
    /* scale by beta */ \
    VFMADD231PS(ZMM(R1), ZMM(0), ZMM(1))            /* zmmR1 <- zmmR1 + zmm0*zmm1 (beta) */   \
    VGATHERDPS(ZMM(0) MASK_K(2), MEM(RCX,ZMM(4),1)) /* zmm0 <- C[16:31] */                    \
    VFMADD231PS(ZMM(R2), ZMM(0), ZMM(1))            /* zmmR2 <- zmmR2 + zmm0*zmm1 (beta) */   \
    VGATHERDPS(ZMM(0) MASK_K(3), MEM(RCX,ZMM(5),1)) /* zmm0 <- C[32:47] */                    \
    VFMADD231PS(ZMM(R3), ZMM(0), ZMM(1))            /* zmmR3 <- zmmR3 + zmm0*zmm1 (beta) */   \
    /* mask registers are reset to 1 (re-armed) before the scatter instructions */ \
    KXNORW(K(1), K(0), K(0))                        /* k1 <- 1 (re-arm mask) */               \
    KXNORW(K(2), K(0), K(0))                        /* k2 <- 1 (re-arm mask) */               \
    KXNORW(K(3), K(0), K(0))                        /* k3 <- 1 (re-arm mask) */               \
    /* store c */ \
    VSCATTERDPS(MEM(RCX,ZMM(3),1) MASK_K(1), ZMM(R1)) /* C[0:15]  <- zmmR1 */ \
    VSCATTERDPS(MEM(RCX,ZMM(4),1) MASK_K(2), ZMM(R2)) /* C[16:31] <- zmmR2 */ \
    VSCATTERDPS(MEM(RCX,ZMM(5),1) MASK_K(3), ZMM(R3)) /* C[32:47] <- zmmR3 */ \
    LEA(RCX, MEM(RCX,R10,1))

// Update C when C is general/column stored and beta = 0
#define UPDATE_C_SCATTERED_BZ(R1,R2,R3) \
\
    KXNORW(K(1), K(0), K(0))                           /* k1 <- 1 (re-arm mask) */ \
    KXNORW(K(2), K(0), K(0))                           /* k2 <- 1 (re-arm mask) */ \
    KXNORW(K(3), K(0), K(0))                           /* k3 <- 1 (re-arm mask) */ \
    VSCATTERDPS(MEM(RCX,ZMM(3),1) MASK_K(1), ZMM(R1))  /* C[0:15]  <- zmmR1 */     \
    VSCATTERDPS(MEM(RCX,ZMM(4),1) MASK_K(2), ZMM(R2))  /* C[16:31] <- zmmR2 */     \
    VSCATTERDPS(MEM(RCX,ZMM(5),1) MASK_K(3), ZMM(R3))  /* C[32:47] <- zmmR3 */     \
    LEA(RCX, MEM(RCX,R10,1))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[48] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 ,28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

void bli_sgemm_zen5_asm_8x48(
    dim_t k_,
    float *restrict alpha,
    float *restrict a,
    float *restrict b,
    float *restrict beta,
    float *restrict c, inc_t rs_c_, inc_t cs_c_,
    auxinfo_t *data,
    cntx_t *restrict cntx)
{
    (void)data;
    (void)cntx;
    (void)cs_c_;

    const int32_t* offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_ * 4; // convert strides to bytes
    const int64_t cs_c = cs_c_ * 4; // convert strides to bytes

    BEGIN_ASM()

    ZERO_REGISTERS()

    MOV(RAX, VAR(a))    // RAX <- a
    MOV(RBX, VAR(b))    // RBX <- b
    MOV(RCX, VAR(c))    // RCX <- c
    MOV(R10, VAR(rs_c)) // R10 <- rs_c

    LEA(R12, MEM(RCX, 63)) // R12 <- RCX + 63 (prefetch pointer)

    K_LOOP()

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSS(ZMM(0), MEM(RAX)) // zmm0 <- broadcast(alpha)
    VBROADCASTSS(ZMM(1), MEM(RBX)) // zmm1 <- broadcast(beta)

    LEA(R13, MEM(R10, R10, 2)) // R13 <- rs_c*3
    LEA(RDX, MEM(R10, R10, 4)) // RDX <- rs_c*5
    LEA(R14, MEM(R10, R13, 2)) // R14 <- rs_c*7

    VXORPS(ZMM(2), ZMM(2), ZMM(2)) // zmm2 <- 0 (used for beta comparison)

#ifdef ENABLE_COL_GEN_STORE
    MOV(R12, VAR(cs_c)) // R12 <- cs_c
    CMP(R10, IMM(4))
    JE(COLUPDATE)        // jump to COLUPDATE if rs_c(R10) == 1

    CMP(R12, IMM(4))     // R12 = cs_c
    JNE(SCATTERUPDATE)   // if cs_c(R12) != 1 jump to scatterupdate
#endif


#ifdef BETA_OPTIMIZATION       // if beta = 0 and beta = 1 are handled
    MOV(RAX, IMM(1))
    CVTSI2SS(XMM(3), RAX)      // xmm3 <- (float)1 

    VCOMISS(XMM(1), XMM(2))    // if beta == 0
    JZ(BETA_ZERO)              // jump to BETA_ZERO if beta == 0

    VCOMISS(XMM(1), XMM(3))
    JNZ(BETA_NZ_N1)            // jump to BETA_NZ_N1 if beta != 1

    // row1
    VFMADD213PS(ZMM( 8), ZMM(0), MEM(RCX))     // zmm8 <- zmm8*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM( 9), ZMM(0), MEM(RCX,64))  // zmm9 <- zmm9*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(10), ZMM(0), MEM(RCX,128)) // zmm10 <- zmm10*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX    ), ZMM( 8))         // C[row] <- zmm8
    VMOVUPS(MEM(RCX, 64), ZMM( 9))         // C[row] <- zmm9
    VMOVUPS(MEM(RCX,128), ZMM(10))         // C[row] <- zmm10
	
    // row2
    VFMADD213PS(ZMM(11), ZMM(0), MEM(RCX, R10, 1     )) // zmm11 <- zmm11*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(12), ZMM(0), MEM(RCX, R10, 1, 64 )) // zmm12 <- zmm12*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(13), ZMM(0), MEM(RCX, R10, 1, 128)) // zmm13 <- zmm13*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, R10, 1     ), ZMM(11))         // C[row] <- zmm11
    VMOVUPS(MEM(RCX, R10, 1, 64 ), ZMM(12))         // C[row] <- zmm12
    VMOVUPS(MEM(RCX, R10, 1, 128), ZMM(13))         // C[row] <- zmm13
	
    // row3
    VFMADD213PS(ZMM(14), ZMM(0), MEM(RCX, R10, 2     )) // zmm14 <- zmm14*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(15), ZMM(0), MEM(RCX, R10, 2, 64 )) // zmm15 <- zmm15*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(16), ZMM(0), MEM(RCX, R10, 2, 128)) // zmm16 <- zmm16*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, R10, 2     ), ZMM(14))         // C[row] <- zmm14
    VMOVUPS(MEM(RCX, R10, 2, 64 ), ZMM(15))         // C[row] <- zmm15
    VMOVUPS(MEM(RCX, R10, 2, 128), ZMM(16))         // C[row] <- zmm16
	
    // row4
    VFMADD213PS(ZMM(17), ZMM(0), MEM(RCX, R13, 1     )) // zmm17 <- zmm17*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(18), ZMM(0), MEM(RCX, R13, 1, 64 )) // zmm18 <- zmm18*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(19), ZMM(0), MEM(RCX, R13, 1, 128)) // zmm19 <- zmm19*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, R13, 1     ), ZMM(17))         // C[row] <- zmm17
    VMOVUPS(MEM(RCX, R13, 1, 64 ), ZMM(18))         // C[row] <- zmm18
    VMOVUPS(MEM(RCX, R13, 1, 128), ZMM(19))         // C[row] <- zmm19
	
    // row5
    VFMADD213PS(ZMM(20), ZMM(0), MEM(RCX, R10, 4     )) // zmm20 <- zmm20*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(21), ZMM(0), MEM(RCX, R10, 4, 64 )) // zmm21 <- zmm21*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(22), ZMM(0), MEM(RCX, R10, 4, 128)) // zmm22 <- zmm22*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, R10, 4     ), ZMM(20))         // C[row] <- zmm20
    VMOVUPS(MEM(RCX, R10, 4, 64 ), ZMM(21))         // C[row] <- zmm21
    VMOVUPS(MEM(RCX, R10, 4, 128), ZMM(22))         // C[row] <- zmm22
	
    // row6
    VFMADD213PS(ZMM(23), ZMM(0), MEM(RCX, RDX, 1     )) // zmm23 <- zmm23*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(24), ZMM(0), MEM(RCX, RDX, 1, 64 )) // zmm24 <- zmm24*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(25), ZMM(0), MEM(RCX, RDX, 1, 128)) // zmm25 <- zmm25*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, RDX, 1     ), ZMM(23))         // C[row] <- zmm23
    VMOVUPS(MEM(RCX, RDX, 1, 64 ), ZMM(24))         // C[row] <- zmm24
    VMOVUPS(MEM(RCX, RDX, 1, 128), ZMM(25))         // C[row] <- zmm25
	
    // row7
    VFMADD213PS(ZMM(26), ZMM(0), MEM(RCX, R13, 2     )) // zmm26 <- zmm26*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(27), ZMM(0), MEM(RCX, R13, 2, 64 )) // zmm27 <- zmm27*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(28), ZMM(0), MEM(RCX, R13, 2, 128)) // zmm28 <- zmm28*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, R13, 2     ), ZMM(26))         // C[row] <- zmm26
    VMOVUPS(MEM(RCX, R13, 2, 64 ), ZMM(27))         // C[row] <- zmm27
    VMOVUPS(MEM(RCX, R13, 2, 128), ZMM(28))         // C[row] <- zmm28
	
    // row8
    VFMADD213PS(ZMM(29), ZMM(0), MEM(RCX, R14, 1     )) // zmm29 <- zmm29*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(30), ZMM(0), MEM(RCX, R14, 1, 64 )) // zmm30 <- zmm30*alpha + C[row] (beta==1)
    VFMADD213PS(ZMM(31), ZMM(0), MEM(RCX, R14, 1, 128)) // zmm31 <- zmm31*alpha + C[row] (beta==1)
    VMOVUPS(MEM(RCX, R14, 1     ), ZMM(29))         // C[row] <- zmm29
    VMOVUPS(MEM(RCX, R14, 1, 64 ), ZMM(30))         // C[row] <- zmm30
    VMOVUPS(MEM(RCX, R14, 1, 128), ZMM(31))         // C[row] <- zmm31
    JMP(END)

    LABEL(BETA_ZERO)
    // row1
    VMULPS(ZMM( 8), ZMM( 8), ZMM(0)) // zmm8 <- zmm8*alpha
    VMULPS(ZMM( 9), ZMM( 9), ZMM(0)) // zmm9 <- zmm9*alpha
    VMULPS(ZMM(10), ZMM(10), ZMM(0)) // zmm10 <- zmm10*alpha
    VMOVUPS(MEM(RCX    ), ZMM( 8))   // C[row] <- zmm8
    VMOVUPS(MEM(RCX, 64), ZMM( 9))   // C[row] <- zmm9
    VMOVUPS(MEM(RCX,128), ZMM(10))   // C[row] <- zmm10

    // row2
    VMULPS(ZMM(11), ZMM(11), ZMM(0))        // zmm11 <- zmm11*alpha
    VMULPS(ZMM(12), ZMM(12), ZMM(0))        // zmm12 <- zmm12*alpha
    VMULPS(ZMM(13), ZMM(13), ZMM(0))        // zmm13 <- zmm13*alpha
    VMOVUPS(MEM(RCX, R10, 1     ), ZMM(11)) // C[row] <- zmm11
    VMOVUPS(MEM(RCX, R10, 1, 64 ), ZMM(12)) // C[row] <- zmm12
    VMOVUPS(MEM(RCX, R10, 1, 128), ZMM(13)) // C[row] <- zmm13

    // row3
    VMULPS(ZMM(14), ZMM(14), ZMM(0))        // zmm14 <- zmm14*alpha
    VMULPS(ZMM(15), ZMM(15), ZMM(0))        // zmm15 <- zmm15*alpha
    VMULPS(ZMM(16), ZMM(16), ZMM(0))        // zmm16 <- zmm16*alpha
    VMOVUPS(MEM(RCX, R10, 2     ), ZMM(14)) // C[row] <- zmm14
    VMOVUPS(MEM(RCX, R10, 2, 64 ), ZMM(15)) // C[row] <- zmm15
    VMOVUPS(MEM(RCX, R10, 2, 128), ZMM(16)) // C[row] <- zmm16

    // row4
    VMULPS(ZMM(17), ZMM(17), ZMM(0))        // zmm17 <- zmm17*alpha
    VMULPS(ZMM(18), ZMM(18), ZMM(0))        // zmm18 <- zmm18*alpha
    VMULPS(ZMM(19), ZMM(19), ZMM(0))        // zmm19 <- zmm19*alpha
    VMOVUPS(MEM(RCX, R13, 1     ), ZMM(17)) // C[row] <- zmm17
    VMOVUPS(MEM(RCX, R13, 1, 64 ), ZMM(18)) // C[row] <- zmm18
    VMOVUPS(MEM(RCX, R13, 1, 128), ZMM(19)) // C[row] <- zmm19

    // row5
    VMULPS(ZMM(20), ZMM(20), ZMM(0))        // zmm20 <- zmm20*alpha
    VMULPS(ZMM(21), ZMM(21), ZMM(0))        // zmm21 <- zmm21*alpha
    VMULPS(ZMM(22), ZMM(22), ZMM(0))        // zmm22 <- zmm22*alpha
    VMOVUPS(MEM(RCX, R10, 4     ), ZMM(20)) // C[row] <- zmm20
    VMOVUPS(MEM(RCX, R10, 4, 64 ), ZMM(21)) // C[row] <- zmm21
    VMOVUPS(MEM(RCX, R10, 4, 128), ZMM(22)) // C[row] <- zmm22

    // row6
    VMULPS(ZMM(23), ZMM(23), ZMM(0))        // zmm23 <- zmm23*alpha
    VMULPS(ZMM(24), ZMM(24), ZMM(0))        // zmm24 <- zmm24*alpha
    VMULPS(ZMM(25), ZMM(25), ZMM(0))        // zmm25 <- zmm25*alpha
    VMOVUPS(MEM(RCX, RDX, 1     ), ZMM(23)) // C[row] <- zmm23
    VMOVUPS(MEM(RCX, RDX, 1, 64 ), ZMM(24)) // C[row] <- zmm24
    VMOVUPS(MEM(RCX, RDX, 1, 128), ZMM(25)) // C[row] <- zmm25

    // row7
    VMULPS(ZMM(26), ZMM(26), ZMM(0))        // zmm26 <- zmm26*alpha
    VMULPS(ZMM(27), ZMM(27), ZMM(0))        // zmm27 <- zmm27*alpha
    VMULPS(ZMM(28), ZMM(28), ZMM(0))        // zmm28 <- zmm28*alpha
    VMOVUPS(MEM(RCX, R13, 2     ), ZMM(26)) // C[row] <- zmm26
    VMOVUPS(MEM(RCX, R13, 2, 64 ), ZMM(27)) // C[row] <- zmm27
    VMOVUPS(MEM(RCX, R13, 2, 128), ZMM(28)) // C[row] <- zmm28

    // row8
    VMULPS(ZMM(29), ZMM(29), ZMM(0))        // zmm29 <- zmm29*alpha
    VMULPS(ZMM(30), ZMM(30), ZMM(0))        // zmm30 <- zmm30*alpha
    VMULPS(ZMM(31), ZMM(31), ZMM(0))        // zmm31 <- zmm31*alpha
    VMOVUPS(MEM(RCX, R14, 1     ), ZMM(29)) // C[row] <- zmm29
    VMOVUPS(MEM(RCX, R14, 1, 64 ), ZMM(30)) // C[row] <- zmm30
    VMOVUPS(MEM(RCX, R14, 1, 128), ZMM(31)) // C[row] <- zmm31

    JMP(END)


    LABEL(BETA_NZ_N1) // beta not zero or not 1
#endif //BETA_OPTIMIZATION
    // row1
    VMULPS(ZMM( 8), ZMM( 8), ZMM(0))           // zmm8 <- zmm8*alpha
    VMULPS(ZMM( 9), ZMM( 9), ZMM(0))           // zmm9 <- zmm9*alpha
    VMULPS(ZMM(10), ZMM(10), ZMM(0))           // zmm10 <- zmm10*alpha
    VFMADD231PS(ZMM( 8), ZMM(1), MEM(RCX))     // zmm8 <- zmm8 + beta*C[row]
    VFMADD231PS(ZMM( 9), ZMM(1), MEM(RCX,64))  // zmm9 <- zmm9 + beta*C[row]
    VFMADD231PS(ZMM(10), ZMM(1), MEM(RCX,128)) // zmm10 <- zmm10 + beta*C[row]
    VMOVUPS(MEM(RCX    ), ZMM( 8))             // C[row] <- zmm8
    VMOVUPS(MEM(RCX, 64), ZMM( 9))             // C[row] <- zmm9
    VMOVUPS(MEM(RCX,128), ZMM(10))             // C[row] <- zmm10

    // row2
    VMULPS(ZMM(11), ZMM(11), ZMM(0))                    // zmm11 <- zmm11*alpha
    VMULPS(ZMM(12), ZMM(12), ZMM(0))                    // zmm12 <- zmm12*alpha
    VMULPS(ZMM(13), ZMM(13), ZMM(0))                    // zmm13 <- zmm13*alpha
    VFMADD231PS(ZMM(11), ZMM(1), MEM(RCX, R10, 1     )) // zmm11 <- zmm11 + beta*C[row]
    VFMADD231PS(ZMM(12), ZMM(1), MEM(RCX, R10, 1, 64 )) // zmm12 <- zmm12 + beta*C[row]
    VFMADD231PS(ZMM(13), ZMM(1), MEM(RCX, R10, 1, 128)) // zmm13 <- zmm13 + beta*C[row]
    VMOVUPS(MEM(RCX, R10, 1     ), ZMM(11))             // C[row] <- zmm11
    VMOVUPS(MEM(RCX, R10, 1, 64 ), ZMM(12))             // C[row] <- zmm12
    VMOVUPS(MEM(RCX, R10, 1, 128), ZMM(13))             // C[row] <- zmm13

    // row3
    VMULPS(ZMM(14), ZMM(14), ZMM(0))                    // zmm14 <- zmm14*alpha
    VMULPS(ZMM(15), ZMM(15), ZMM(0))                    // zmm15 <- zmm15*alpha
    VMULPS(ZMM(16), ZMM(16), ZMM(0))                    // zmm16 <- zmm16*alpha
    VFMADD231PS(ZMM(14), ZMM(1), MEM(RCX, R10, 2     )) // zmm14 <- zmm14 + beta*C[row]
    VFMADD231PS(ZMM(15), ZMM(1), MEM(RCX, R10, 2, 64 )) // zmm15 <- zmm15 + beta*C[row]
    VFMADD231PS(ZMM(16), ZMM(1), MEM(RCX, R10, 2, 128)) // zmm16 <- zmm16 + beta*C[row]
    VMOVUPS(MEM(RCX, R10, 2     ), ZMM(14))             // C[row] <- zmm14
    VMOVUPS(MEM(RCX, R10, 2, 64 ), ZMM(15))             // C[row] <- zmm15
    VMOVUPS(MEM(RCX, R10, 2, 128), ZMM(16))             // C[row] <- zmm16

    // row4
    VMULPS(ZMM(17), ZMM(17), ZMM(0))                    // zmm17 <- zmm17*alpha
    VMULPS(ZMM(18), ZMM(18), ZMM(0))                    // zmm18 <- zmm18*alpha
    VMULPS(ZMM(19), ZMM(19), ZMM(0))                    // zmm19 <- zmm19*alpha
    VFMADD231PS(ZMM(17), ZMM(1), MEM(RCX, R13, 1     )) // zmm17 <- zmm17 + beta*C[row]
    VFMADD231PS(ZMM(18), ZMM(1), MEM(RCX, R13, 1, 64 )) // zmm18 <- zmm18 + beta*C[row]
    VFMADD231PS(ZMM(19), ZMM(1), MEM(RCX, R13, 1, 128)) // zmm19 <- zmm19 + beta*C[row]
    VMOVUPS(MEM(RCX, R13, 1     ), ZMM(17))             // C[row] <- zmm17
    VMOVUPS(MEM(RCX, R13, 1, 64 ), ZMM(18))             // C[row] <- zmm18
    VMOVUPS(MEM(RCX, R13, 1, 128), ZMM(19))             // C[row] <- zmm19

    // row5
    VMULPS(ZMM(20), ZMM(20), ZMM(0))                    // zmm20 <- zmm20*alpha
    VMULPS(ZMM(21), ZMM(21), ZMM(0))                    // zmm21 <- zmm21*alpha
    VMULPS(ZMM(22), ZMM(22), ZMM(0))                    // zmm22 <- zmm22*alpha
    VFMADD231PS(ZMM(20), ZMM(1), MEM(RCX, R10, 4     )) // zmm20 <- zmm20 + beta*C[row]
    VFMADD231PS(ZMM(21), ZMM(1), MEM(RCX, R10, 4, 64 )) // zmm21 <- zmm21 + beta*C[row]
    VFMADD231PS(ZMM(22), ZMM(1), MEM(RCX, R10, 4, 128)) // zmm22 <- zmm22 + beta*C[row]
    VMOVUPS(MEM(RCX, R10, 4     ), ZMM(20))             // C[row] <- zmm20
    VMOVUPS(MEM(RCX, R10, 4, 64 ), ZMM(21))             // C[row] <- zmm21
    VMOVUPS(MEM(RCX, R10, 4, 128), ZMM(22))             // C[row] <- zmm22

    // row6
    VMULPS(ZMM(23), ZMM(23), ZMM(0))                    // zmm23 <- zmm23*alpha
    VMULPS(ZMM(24), ZMM(24), ZMM(0))                    // zmm24 <- zmm24*alpha
    VMULPS(ZMM(25), ZMM(25), ZMM(0))                    // zmm25 <- zmm25*alpha
    VFMADD231PS(ZMM(23), ZMM(1), MEM(RCX, RDX, 1     )) // zmm23 <- zmm23 + beta*C[row]
    VFMADD231PS(ZMM(24), ZMM(1), MEM(RCX, RDX, 1, 64 )) // zmm24 <- zmm24 + beta*C[row]
    VFMADD231PS(ZMM(25), ZMM(1), MEM(RCX, RDX, 1, 128)) // zmm25 <- zmm25 + beta*C[row]
    VMOVUPS(MEM(RCX, RDX, 1     ), ZMM(23))             // C[row] <- zmm23
    VMOVUPS(MEM(RCX, RDX, 1, 64 ), ZMM(24))             // C[row] <- zmm24
    VMOVUPS(MEM(RCX, RDX, 1, 128), ZMM(25))             // C[row] <- zmm25

    // row7
    VMULPS(ZMM(26), ZMM(26), ZMM(0))                    // zmm26 <- zmm26*alpha
    VMULPS(ZMM(27), ZMM(27), ZMM(0))                    // zmm27 <- zmm27*alpha
    VMULPS(ZMM(28), ZMM(28), ZMM(0))                    // zmm28 <- zmm28*alpha
    VFMADD231PS(ZMM(26), ZMM(1), MEM(RCX, R13, 2     )) // zmm26 <- zmm26 + beta*C[row]
    VFMADD231PS(ZMM(27), ZMM(1), MEM(RCX, R13, 2, 64 )) // zmm27 <- zmm27 + beta*C[row]
    VFMADD231PS(ZMM(28), ZMM(1), MEM(RCX, R13, 2, 128)) // zmm28 <- zmm28 + beta*C[row]
    VMOVUPS(MEM(RCX, R13, 2     ), ZMM(26))             // C[row] <- zmm26
    VMOVUPS(MEM(RCX, R13, 2, 64 ), ZMM(27))             // C[row] <- zmm27
    VMOVUPS(MEM(RCX, R13, 2, 128), ZMM(28))             // C[row] <- zmm28

    // row8
    VMULPS(ZMM(29), ZMM(29), ZMM(0))                    // zmm29 <- zmm29*alpha
    VMULPS(ZMM(30), ZMM(30), ZMM(0))                    // zmm30 <- zmm30*alpha
    VMULPS(ZMM(31), ZMM(31), ZMM(0))                    // zmm31 <- zmm31*alpha
    VFMADD231PS(ZMM(29), ZMM(1), MEM(RCX, R14, 1     )) // zmm29 <- zmm29 + beta*C[row]
    VFMADD231PS(ZMM(30), ZMM(1), MEM(RCX, R14, 1, 64 )) // zmm30 <- zmm30 + beta*C[row]
    VFMADD231PS(ZMM(31), ZMM(1), MEM(RCX, R14, 1, 128)) // zmm31 <- zmm31 + beta*C[row]
    VMOVUPS(MEM(RCX, R14, 1     ), ZMM(29))             // C[row] <- zmm29
    VMOVUPS(MEM(RCX, R14, 1, 64 ), ZMM(30))             // C[row] <- zmm30
    VMOVUPS(MEM(RCX, R14, 1, 128), ZMM(31))             // C[row] <- zmm31

#ifdef ENABLE_COL_GEN_STORE
    JMP(END)

    LABEL(COLUPDATE)
    // Column-major C (unit row stride rs_c == 1): use an in-register 8x16
    // transpose per 16-wide group (TRANSPOSE_8x16) followed by contiguous
    // column stores (STORE_COL_16 / _BZ), avoiding the slow gather/scatter.
    //
    // R12 = cs_c (already loaded).  R13/RDX/R14 currently hold rs_c*{3,5,7}
    // from the row-major setup; recompute them as cs_c*{3,5,7} for columns.
    LEA(R13, MEM(R12, R12, 2)) // R13 <- cs_c*3
    LEA(RDX, MEM(R12, R12, 4)) // RDX <- cs_c*5
    LEA(R14, MEM(R12, R13, 2)) // R14 <- cs_c*7

    // Scale all 24 accumulators by alpha (zmm0) up front.  This makes the
    // subsequent transpose (which reuses zmm0..7 as scratch) and the column
    // stores independent of alpha, and frees zmm0 as a scratch register.
    VMULPS(ZMM( 8), ZMM( 8), ZMM(0)) // zmm8 <- zmm8*alpha
    VMULPS(ZMM( 9), ZMM( 9), ZMM(0)) // zmm9 <- zmm9*alpha
    VMULPS(ZMM(10), ZMM(10), ZMM(0)) // zmm10 <- zmm10*alpha
    VMULPS(ZMM(11), ZMM(11), ZMM(0)) // zmm11 <- zmm11*alpha
    VMULPS(ZMM(12), ZMM(12), ZMM(0)) // zmm12 <- zmm12*alpha
    VMULPS(ZMM(13), ZMM(13), ZMM(0)) // zmm13 <- zmm13*alpha
    VMULPS(ZMM(14), ZMM(14), ZMM(0)) // zmm14 <- zmm14*alpha
    VMULPS(ZMM(15), ZMM(15), ZMM(0)) // zmm15 <- zmm15*alpha
    VMULPS(ZMM(16), ZMM(16), ZMM(0)) // zmm16 <- zmm16*alpha
    VMULPS(ZMM(17), ZMM(17), ZMM(0)) // zmm17 <- zmm17*alpha
    VMULPS(ZMM(18), ZMM(18), ZMM(0)) // zmm18 <- zmm18*alpha
    VMULPS(ZMM(19), ZMM(19), ZMM(0)) // zmm19 <- zmm19*alpha
    VMULPS(ZMM(20), ZMM(20), ZMM(0)) // zmm20 <- zmm20*alpha
    VMULPS(ZMM(21), ZMM(21), ZMM(0)) // zmm21 <- zmm21*alpha
    VMULPS(ZMM(22), ZMM(22), ZMM(0)) // zmm22 <- zmm22*alpha
    VMULPS(ZMM(23), ZMM(23), ZMM(0)) // zmm23 <- zmm23*alpha
    VMULPS(ZMM(24), ZMM(24), ZMM(0)) // zmm24 <- zmm24*alpha
    VMULPS(ZMM(25), ZMM(25), ZMM(0)) // zmm25 <- zmm25*alpha
    VMULPS(ZMM(26), ZMM(26), ZMM(0)) // zmm26 <- zmm26*alpha
    VMULPS(ZMM(27), ZMM(27), ZMM(0)) // zmm27 <- zmm27*alpha
    VMULPS(ZMM(28), ZMM(28), ZMM(0)) // zmm28 <- zmm28*alpha
    VMULPS(ZMM(29), ZMM(29), ZMM(0)) // zmm29 <- zmm29*alpha
    VMULPS(ZMM(30), ZMM(30), ZMM(0)) // zmm30 <- zmm30*alpha
    VMULPS(ZMM(31), ZMM(31), ZMM(0)) // zmm31 <- zmm31*alpha

    VCOMISS(XMM(1), XMM(2))
    JE(COLSTORBZ)                  // beta == 0 -> pure-store path
    // beta != 0: transpose each 16-wide group, then FMA-store columns.
    // Each transpose reuses zmm0..7 as scratch (clobbering beta in zmm1),
    // so we re-broadcast beta afterwards for STORE_COL_16.
    TRANSPOSE_8x16( 8, 11, 14, 17, 20, 23, 26, 29,  0, 1, 2, 3, 4, 5, 6, 7)
    TRANSPOSE_8x16( 9, 12, 15, 18, 21, 24, 27, 30,  0, 1, 2, 3, 4, 5, 6, 7)
    TRANSPOSE_8x16(10, 13, 16, 19, 22, 25, 28, 31,  0, 1, 2, 3, 4, 5, 6, 7)
    VBROADCASTSS(ZMM(1), MEM(RBX))                     // zmm1 <- broadcast(beta) (re-broadcast, clobbered by transpose)
    STORE_COL_16( 8, 11, 14, 17, 20, 23, 26, 29, 0, 1) // cols  0:15
    STORE_COL_16( 9, 12, 15, 18, 21, 24, 27, 30, 0, 1) // cols 16:31
    STORE_COL_16(10, 13, 16, 19, 22, 25, 28, 31, 0, 1) // cols 32:47
    JMP(END)

    LABEL(COLSTORBZ)
    // beta == 0: transpose each group, then plain column stores.
    TRANSPOSE_8x16( 8, 11, 14, 17, 20, 23, 26, 29,  0, 1, 2, 3, 4, 5, 6, 7)
    TRANSPOSE_8x16( 9, 12, 15, 18, 21, 24, 27, 30,  0, 1, 2, 3, 4, 5, 6, 7)
    TRANSPOSE_8x16(10, 13, 16, 19, 22, 25, 28, 31,  0, 1, 2, 3, 4, 5, 6, 7)
    STORE_COL_16_BZ( 8, 11, 14, 17, 20, 23, 26, 29, 1) // cols  0:15
    STORE_COL_16_BZ( 9, 12, 15, 18, 21, 24, 27, 30, 1) // cols 16:31
    STORE_COL_16_BZ(10, 13, 16, 19, 22, 25, 28, 31, 1) // cols 32:47
    JMP(END)

    LABEL(SCATTERUPDATE)
    // if C is general stride
    VMULPS(ZMM( 8), ZMM( 8), ZMM(0)) // zmm8 <- zmm8*alpha (all accumulators scaled the same way)
    VMULPS(ZMM( 9), ZMM( 9), ZMM(0)) // zmm9 <- zmm9*alpha
    VMULPS(ZMM(10), ZMM(10), ZMM(0)) // zmm10 <- zmm10*alpha
    VMULPS(ZMM(11), ZMM(11), ZMM(0)) // zmm11 <- zmm11*alpha
    VMULPS(ZMM(12), ZMM(12), ZMM(0)) // zmm12 <- zmm12*alpha
    VMULPS(ZMM(13), ZMM(13), ZMM(0)) // zmm13 <- zmm13*alpha
    VMULPS(ZMM(14), ZMM(14), ZMM(0)) // zmm14 <- zmm14*alpha
    VMULPS(ZMM(15), ZMM(15), ZMM(0)) // zmm15 <- zmm15*alpha
    VMULPS(ZMM(16), ZMM(16), ZMM(0)) // zmm16 <- zmm16*alpha
    VMULPS(ZMM(17), ZMM(17), ZMM(0)) // zmm17 <- zmm17*alpha
    VMULPS(ZMM(18), ZMM(18), ZMM(0)) // zmm18 <- zmm18*alpha
    VMULPS(ZMM(19), ZMM(19), ZMM(0)) // zmm19 <- zmm19*alpha
    VMULPS(ZMM(20), ZMM(20), ZMM(0)) // zmm20 <- zmm20*alpha
    VMULPS(ZMM(21), ZMM(21), ZMM(0)) // zmm21 <- zmm21*alpha
    VMULPS(ZMM(22), ZMM(22), ZMM(0)) // zmm22 <- zmm22*alpha
    VMULPS(ZMM(23), ZMM(23), ZMM(0)) // zmm23 <- zmm23*alpha
    VMULPS(ZMM(24), ZMM(24), ZMM(0)) // zmm24 <- zmm24*alpha
    VMULPS(ZMM(25), ZMM(25), ZMM(0)) // zmm25 <- zmm25*alpha
    VMULPS(ZMM(26), ZMM(26), ZMM(0)) // zmm26 <- zmm26*alpha
    VMULPS(ZMM(27), ZMM(27), ZMM(0)) // zmm27 <- zmm27*alpha
    VMULPS(ZMM(28), ZMM(28), ZMM(0)) // zmm28 <- zmm28*alpha
    VMULPS(ZMM(29), ZMM(29), ZMM(0)) // zmm29 <- zmm29*alpha
    VMULPS(ZMM(30), ZMM(30), ZMM(0)) // zmm30 <- zmm30*alpha
    VMULPS(ZMM(31), ZMM(31), ZMM(0)) // zmm31 <- zmm31*alpha

    MOV(R13, VAR(offsetPtr))

    VPBROADCASTD(ZMM(0), R12D)              // zmm0 <- broadcast(cs_c) (16 copies)
    VPMULLD(ZMM(3), ZMM(0), MEM(R13))       // zmm3 <- [0:15]*cs_c
    VPMULLD(ZMM(4), ZMM(0), MEM(R13, 16*4)) // zmm4 <- [16:31]*cs_c
    VPMULLD(ZMM(5), ZMM(0), MEM(R13,32*4))  // zmm5 <- [32:47]*cs_c

    VCOMISS(XMM(1), XMM(2))
    JE(GENSTORBZ)                          // if beta == 0 jump
    UPDATE_C_SCATTERED( 8,  9, 10)         // scale by beta and store
    UPDATE_C_SCATTERED(11, 12, 13)
    UPDATE_C_SCATTERED(14, 15, 16)
    UPDATE_C_SCATTERED(17, 18, 19)
    UPDATE_C_SCATTERED(20, 21, 22)
    UPDATE_C_SCATTERED(23, 24, 25)
    UPDATE_C_SCATTERED(26, 27, 28)
    UPDATE_C_SCATTERED(29, 30, 31)
    JMP(END)
	
    LABEL(GENSTORBZ)
    UPDATE_C_SCATTERED_BZ( 8,  9, 10)
    UPDATE_C_SCATTERED_BZ(11, 12, 13)
    UPDATE_C_SCATTERED_BZ(14, 15, 16)
    UPDATE_C_SCATTERED_BZ(17, 18, 19)
    UPDATE_C_SCATTERED_BZ(20, 21, 22)
    UPDATE_C_SCATTERED_BZ(23, 24, 25)
    UPDATE_C_SCATTERED_BZ(26, 27, 28)
    UPDATE_C_SCATTERED_BZ(29, 30, 31)
#endif

    LABEL(END)


        END_ASM( : // output operands
                 : // input operands
                 [k] "m"(k),
                 [a] "m"(a),
                 [b] "m"(b),
                 [alpha] "m"(alpha),
                 [beta] "m"(beta),
                 [c] "m"(c),
                 [rs_c] "m"(rs_c),
                 [cs_c] "m"(cs_c),
                 [offsetPtr] "m"(offsetPtr) : // register clobber list
                 "rax", "rbx", "rcx", "rdx", "rdi", "r10", "r12", "r13", "r14",
                 "k0", "k1", "k2", "k3", "xmm1", "xmm2", "xmm3",
                 "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6",
                 "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
                 "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
                 "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25",
                 "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory")
}


/* C = A*B (beta is 0) */
#define UPDATE_C_BETA_0(R1, R2, R3) \
    VMOVUPS(MEM(RCX     ), ZMM(R1)) \
    VXORPS(ZMM(R1), ZMM(R1), ZMM(R1)) \
    VMOVUPS(MEM(RCX,  64), ZMM(R2)) \
    VXORPS(ZMM(R2), ZMM(R2), ZMM(R2)) \
    VMOVUPS(MEM(RCX, 128), ZMM(R3)) \
    VXORPS(ZMM(R3), ZMM(R3), ZMM(R3)) \
    LEA(RCX, MEM(RCX, R10, 1)) \

/* C += A*B */
#define UPDATE_C_BETA_1(R1, R2, R3) \
    VADDPS(ZMM(R1), ZMM(R1), MEM(RCX)) /* C += A*B */ \
    VADDPS(ZMM(R2), ZMM(R2), MEM(RCX, 64)) \
    VADDPS(ZMM(R3), ZMM(R3), MEM(RCX, 128)) \
    VMOVUPS(MEM(RCX     ), ZMM(R1)) \
    VXORPS(ZMM(R1), ZMM(R1), ZMM(R1)) \
    VMOVUPS(MEM(RCX,  64), ZMM(R2)) \
    VXORPS(ZMM(R2), ZMM(R2), ZMM(R2)) \
    VMOVUPS(MEM(RCX, 128), ZMM(R3)) \
    VXORPS(ZMM(R3), ZMM(R3), ZMM(R3)) \
    LEA(RCX, MEM(RCX, R10, 1)) \

/* C = A*B - C */
#define UPDATE_C_BETA_M1(R1, R2, R3) \
    VSUBPS(ZMM(R1), ZMM(R1), MEM(RCX)) \
    VSUBPS(ZMM(R2), ZMM(R2), MEM(RCX, 64)) \
    VSUBPS(ZMM(R3), ZMM(R3), MEM(RCX, 128)) \
    VMOVUPS(MEM(RCX     ), ZMM(R1)) \
    VXORPS(ZMM(R1), ZMM(R1), ZMM(R1)) \
    VMOVUPS(MEM(RCX,  64), ZMM(R2)) \
    VXORPS(ZMM(R2), ZMM(R2), ZMM(R2)) \
    VMOVUPS(MEM(RCX, 128), ZMM(R3)) \
    VXORPS(ZMM(R3), ZMM(R3), ZMM(R3)) \
    LEA(RCX, MEM(RCX, R10, 1)) \

/* C = (beta*c) + (A*B) */
#define UPDATE_C_BETA_N(R1, R2, R3) \
    VFMADD231PS(ZMM(R1), ZMM(1), MEM(RCX))  \
    VFMADD231PS(ZMM(R2), ZMM(1), MEM(RCX,64)) \
    VFMADD231PS(ZMM(R3), ZMM(1), MEM(RCX,128)) \
    \
    VMOVUPS(MEM(RCX    ), ZMM(R1)) \
    VXORPS(ZMM(R1), ZMM(R1), ZMM(R1)) \
    VMOVUPS(MEM(RCX, 64), ZMM(R2)) \
    VXORPS(ZMM(R2), ZMM(R2), ZMM(R2)) \
    VMOVUPS(MEM(RCX,128), ZMM(R3)) \
    VXORPS(ZMM(R3), ZMM(R3), ZMM(R3)) \
    LEA(RCX, MEM(RCX, R10, 1)) \


#define PRE_K_LOOP() \
        const int64_t n   = n0;                                       \
        const int64_t m   = m0;                                       \
        const int64_t k   = k0;                                       \
        const int64_t rs_c= ldc0 * 4;                                 \
        const int64_t cs_c= 4;                                        \
        BEGIN_ASM()                                                   \
                                                                      \
        MOV(RDI, VAR(n))    /* load N into RDI */                     \
        MOV(RSI, VAR(m))    /* load M into RSI */                     \
        MOV(RDX, VAR(k))    /* load K into RDX */                     \
        MOV(RCX, VAR(c))    /* load C macro panel pointer into RCX*/  \
        MOV(R8 , VAR(a))    /* load A macro panel pointer into R8 */  \
        MOV(R9 , VAR(b))    /* load B macro panel pointer into R9 */  \
        MOV(R10, VAR(rs_c)) /* load ldc into R10*/                    \
                                                                      \
        SAR(RSI, IMM(3))    /* m_iter = M/8 */                        \
                                                                      \
        ZERO_REGISTERS()    /* zero accumulation registers */         \
                                                                      \
        MOV(VAR(m), RSI)    /* backup m_iter into stack */            \
        MOV(R15, R8)        /* backup A macro panel pointer to R15 */ \
        MOV(R11, RCX)       /* backup C macro panel pointer to R11 */ \
                                                                      \
        CMP(RDI, IMM(0))    /* check if n is zero */                  \
        JLE(ENDJR)          /* JMP to endjr if n <= 0*/               \
                                                                      \
    LOOP_ALIGN                  \
    LABEL(LOOPJR) /* JR loop */ \
                                \
        MOV(VAR(n), RDI)                                                \
        MOV(R8, R15)     /* restore A macro panel pointer */            \
        MOV(RSI, VAR(m)) /* copy m_iter to RSI */                       \
        MOV(RCX, R11)    /* restore pointer to C macro panel pointer */ \
        TEST(RSI, RSI)                                                  \
                                                                        \
        JZ(ENDIR)        /* Jump to ENDIR if m_iter(RSI) == 0*/         \
        LOOP_ALIGN                                                      \
        LABEL(LOOPIR)                                                   \
            MOV(RAX, R8)           /* Move A micro panel pointer to RAX */ \
            MOV(RBX, R9)           /* Move B micro panel pointer to RBX */ \
            LEA(R12, MEM(RCX, 63)) /* calculate c_prefetch pointer */


#define POST_K_LOOP() \
            LABEL(END_MICRO_KER)                                       \
                                                                       \
            MOV(R13, VAR(k))         /* move k_iter into R13 */        \
            IMUL(R13, IMM(8))        /* k_iter *= 8 */                 \
            LEA(R8, MEM(R8, R13, 4)) /* a_next_upanel = A + (k*8*4) */ \
                                                                       \
            DEC(RSI)                 /* decrement m_iter */            \
            JNZ(LOOPIR)                                                \
                                                                       \
        LABEL(ENDIR)                                                \
                                                                    \
        MOV(R14, VAR(k))         /* move k_iter into R14 */         \
        IMUL(R14, IMM(48))       /* k_iter *= 48 */                 \
        LEA(R9, MEM(R9, R14, 4)) /* b_next_upanel = B + (k*48*4) */ \
        LEA(R11, MEM(R11, 48*4)) /* c_next_upanel = C + (48*4) */   \
        MOV(RDI, VAR(n))                                            \
        SUB(RDI, IMM(48))        /* subtract NR(48) from N */       \
        JNZ(LOOPJR)                                                 \
                                                                    \
    LABEL(ENDJR) \
                 \
        END_ASM \
        (       \
          : /* output operands */ \
          : /* input operands */  \
            [n]       "m" (n),    \
            [m]       "m" (m),    \
            [k]       "m" (k),    \
            [c]       "m" (c),    \
            [a]       "m" (a),    \
            [b]       "m" (b),    \
            [beta]    "m" (beta), \
            [rs_c]    "m" (rs_c), \
            [cs_c]    "m" (cs_c)  \
          : /* register clobber list */ \
            "rax", "rbx", "rcx", "rdi", "rdx", "rsi", "r8", "r9",          \
            "r10", "r11", "r12", "r13", "r14", "r15", "xmm1", "xmm2",      \
            "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6",        \
            "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",    \
            "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",          \
            "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25",          \
            "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory" \
        )

/*
    Macro kernel for C = A*B (beta = 0)
    Only Row major stored C is supported.
*/
BLIS_INLINE void bli_sgemm_zen5_asm_8x48_macro_kernel_b0
(
    dim_t   n0,
    dim_t   m0,
    dim_t   k0,
    float*  c,
    float*  a,
    float*  b,
    dim_t   ldc0,
    float*  beta
)
{
    PRE_K_LOOP()
    K_LOOP()
    UPDATE_C_BETA_0( 8,  9, 10)
    UPDATE_C_BETA_0(11, 12, 13)
    UPDATE_C_BETA_0(14, 15, 16)
    UPDATE_C_BETA_0(17, 18, 19)
    UPDATE_C_BETA_0(20, 21, 22)
    UPDATE_C_BETA_0(23, 24, 25)
    UPDATE_C_BETA_0(26, 27, 28)
    UPDATE_C_BETA_0(29, 30, 31)
    POST_K_LOOP()
}


/*
    Macro kernel for C = C + (A*B) (beta = 1)
    Only Row major stored C is supported.
*/
BLIS_INLINE void bli_sgemm_zen5_asm_8x48_macro_kernel_b1
(
    dim_t   n0,
    dim_t   m0,
    dim_t   k0,
    float* c,
    float* a,
    float* b,
    dim_t   ldc0,
    float* beta
)
{
    PRE_K_LOOP()
    K_LOOP()
    UPDATE_C_BETA_1( 8,  9, 10)
    UPDATE_C_BETA_1(11, 12, 13)
    UPDATE_C_BETA_1(14, 15, 16)
    UPDATE_C_BETA_1(17, 18, 19)
    UPDATE_C_BETA_1(20, 21, 22)
    UPDATE_C_BETA_1(23, 24, 25)
    UPDATE_C_BETA_1(26, 27, 28)
    UPDATE_C_BETA_1(29, 30, 31)
    POST_K_LOOP()
}


/*
    Macro kernel for C = (A*B) - C (beta = 1)
    Only Row major stored C is supported.
*/
BLIS_INLINE void bli_sgemm_zen5_asm_8x48_macro_kernel_bm1
(
    dim_t   n0,
    dim_t   m0,
    dim_t   k0,
    float* c,
    float* a,
    float* b,
    dim_t   ldc0,
    float* beta
)
{
    PRE_K_LOOP()
    K_LOOP()
    MOV(RBX, VAR(beta))
    VBROADCASTSS(ZMM(1), MEM(RBX))
    UPDATE_C_BETA_M1( 8,  9, 10)
    UPDATE_C_BETA_M1(11, 12, 13)
    UPDATE_C_BETA_M1(14, 15, 16)
    UPDATE_C_BETA_M1(17, 18, 19)
    UPDATE_C_BETA_M1(20, 21, 22)
    UPDATE_C_BETA_M1(23, 24, 25)
    UPDATE_C_BETA_M1(26, 27, 28)
    UPDATE_C_BETA_M1(29, 30, 31)
    POST_K_LOOP()

}

/*
    Macro kernel for C = (beta*C) + (A*B)
    Only Row major stored C is supported.
*/
BLIS_INLINE void bli_sgemm_zen5_asm_8x48_macro_kernel_bn
(
    dim_t   n0,
    dim_t   m0,
    dim_t   k0,
    float* c,
    float* a,
    float* b,
    dim_t   ldc0,
    float* beta
)
{
    PRE_K_LOOP()
    K_LOOP()
    MOV(RBX, VAR(beta))
    VBROADCASTSS(ZMM(1), MEM(RBX))
    UPDATE_C_BETA_N( 8,  9, 10)
    UPDATE_C_BETA_N(11, 12, 13)
    UPDATE_C_BETA_N(14, 15, 16)
    UPDATE_C_BETA_N(17, 18, 19)
    UPDATE_C_BETA_N(20, 21, 22)
    UPDATE_C_BETA_N(23, 24, 25)
    UPDATE_C_BETA_N(26, 27, 28)
    UPDATE_C_BETA_N(29, 30, 31)
    POST_K_LOOP()

}

/*
    SGEMM 8x48 Macro kernel for fringe cases.
    MR = 8, NR = 48
    Only row major stored C is supported by this kernel.
    Alpha scaling is done at packing.
*/
void bli_sgemm_zen5_asm_8x48_macro_kernel_fringe
(
    dim_t   n0,
    dim_t   m0,
    dim_t   k0,
    float* c,
    float* a,
    float* b,
    dim_t   ldc0,
    float* beta
)
{
    const int64_t n = n0;
    const int64_t m = m0;
    const int64_t k = k0;
    const int64_t ldc = ldc0;
    // Create temporary buffer for C
    float ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( float ) ]
                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

    dim_t ldct = 48;
    float alpha = 1;           // only alpha=1 is supported by macro kernel
    float zero = 0;

    dim_t m_left = m % 8;      // M % MR (computed by this kernel)
    dim_t m_main = m - m_left; // already computed by main kernel (multiple of MR)

    dim_t n_left = n % 48;     // N % NR (computed by this kernel)
    dim_t n_main = n - n_left; // already computed by main kernel (multiple of NR)
    float *a_temp = a;
    float *b_temp = b;
    float *c_actual = c;


    if ( m_left )
    {
        // loop along N dimension
        // initial m_main rows of 'C' are aready computed,
        // to compute remaining m_left rows, pointer 'C'
        // matrix shoule be moved forward by m_main rows,
        // and pointer 'A' should point to  (m_main / MR)th
        // micropanel.
        // To move 'A' pointer to (m_main / MR)th micropanel.
        //    A += (ps_a)    * (m_main / MR)
        // => A += (k * MR)  * (m_main / MR)
        // => A += k * m_main
        //
        // To Move 'C' pointer ahead by m_main rows,
        //    C += (ldc * m_main)
        a_temp = a + ( k * m_main );
        c_actual = c + ( ldc * m_main );
        for(dim_t j = 0; j < n_main; j += 48 )
        {
            bli_sgemm_zen5_asm_8x48
            (
                k,
                &alpha,
                a_temp,
                // move B pointer to next micropanel of packB (( j / NR)th micropanel)
                //     B += ( j / NR) * ps_b;
                // =>  B += ( j / NR ) * ( k * NR );
                // =>  B += j * k
                b + ( j * k ),
                &zero,
                ct,
                ldct,
                1,
                NULL,
                NULL
            );

            // copy GEMM result from 'ct' into 'c'.
            // 'n' will always be NR(48) for region 1, fringe case when
            // both M and N are less than MR and NR respectively
            // is handled in n_left region.
            PASTEMAC(s,xpbys_mxn)( m_left, 48,
            ct,  ldct, 1,
            beta,
            // move 'C' pointer ahead by j columns.
            c_actual + ( j ), ldc,  1 );
        }
    }
    // #ENDREGEION m_left

    // #REGEION n_left
    if ( (n % 48) )
    {
        // loop along M dimension
        // initial n_main rows of 'C' are aready computed,
        // to compute remaining n_left rows, pointer 'C'
        // matrix shoule be moved forward by n_main columns,
        // and pointer 'B' should point to  (n_main / NR)th
        // micropanel.
        // To move 'B' pointer to (n_main / NR)th micropanel.
        //    B += (ps_b)    * (n_main / NR)
        // => B += (k * NR)  * (n_main / NR)
        // => B += k * n_main
        //
        // To Move 'C' pointer ahead by n_main columns,
        //    C += (n_main)
        b_temp = b + ( k * n_main );
        c_actual = c + ( n_main );
        for (dim_t i = 0; i < m; i += 8 )
        {
            bli_sgemm_zen5_asm_8x48
            (
                k,
                &alpha,
                // move A pointer to next micropanel of packA (( i / MR)th micropanel)
                //     A += ( i / MR) * ps_a;
                // =>  A += ( i / MR ) * ( k * MR );
                // =>  A += i * k
                a + ( i * k),
                b_temp,
                &zero,
                ct,
                ldct,
                1,
                NULL,
                NULL
            );
            // remaning compute along M dimension = m - i
            dim_t m_curr = m - i;
            // if M remainder compute > 8, then only MR is
            // is solved in current iteration.
            if (m_curr > 8) m_curr = 8;

            // copy GEMM result from 'ct' into 'c'.
            PASTEMAC(s,xpbys_mxn)( m_curr, n_left,
            ct,  ldct, 1,
            beta,
            // move 'C' pointer ahead by i rows.
            c_actual + ( ldc * i ), ldc,  1 );
        }
    }
    // #ENDREGEION n_left

}

/*
    SGEMM 8x48 Macro kernel.
    MR = 8, NR = 48
    Only row major stored C is supported by this kernel.
    Alpha scaling is not supported.
*/
void bli_sgemm_zen5_asm_8x48_macro_kernel
(
    dim_t   n,
    dim_t   m,
    dim_t   k,
    float* c,
    float* a,
    float* b,
    dim_t   ldc,
    float* beta
)
{
    if(*beta == 1)
    {
        bli_sgemm_zen5_asm_8x48_macro_kernel_b1
        (
            n - (n % 48), // remaining N will be handled by fringe kernel.
            m - (m % 8),  // remaining M will be handled by fringe kernel.
            k,
            c,
            a,
            b,
            ldc,
            beta
        );
    }
    else if(*beta == -1)
    {
        bli_sgemm_zen5_asm_8x48_macro_kernel_bm1
        (
            n - (n % 48),
            m - (m % 8),
            k,
            c,
            a,
            b,
            ldc,
            beta
        );
    }
    else if (*beta == 0)
    {
        bli_sgemm_zen5_asm_8x48_macro_kernel_b0
        (
            n - (n % 48),
            m - (m % 8),
            k,
            c,
            a,
            b,
            ldc,
            beta
        );
    }
    else
    {
        bli_sgemm_zen5_asm_8x48_macro_kernel_bn
        (
            n - (n % 48),
            m - (m % 8),
            k,
            c,
            a,
            b,
            ldc,
            beta
        );
    }

    if ( n % 48 || m % 8)
    {
        bli_sgemm_zen5_asm_8x48_macro_kernel_fringe
        (
            n,
            m,
            k,
            c,
            a,
            b,
            ldc,
            beta
        );
    }
}
