/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/*
   rrc:
     --------        ------        | | | | | | | |
     --------        ------        | | | | | | | |
     --------   +=   ------ ...    | | | | | | | |
     --------        ------        | | | | | | | |
     --------        ------              :
     --------        ------              :

   Assumptions:
   - C is row-stored and B is column-stored;
   - A is row-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential microkernel is well-suited for
   a dot-product-based accumulation that performs vector loads from
   both A and B.

   NOTE: These kernels implicitly support column-oriented IO, implemented
   via an a high-level transposition of the entire operation. A and B will
   effectively remain row- and column-stored, respectively, but C will then
   effectively appear column-stored. Thus, this kernel may be used for both
   rrc and crc cases.
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


void bli_dgemmsup_rd_haswell_asm_6x8m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t n_left = n0 % 8;

	// First check whether this is a edge case in the n dimension. If so,
	// dispatch other 6x?m kernels, as needed.
	if ( n_left )
	{
		double* restrict cij = c;
		double* restrict bj  = b;
		double* restrict ai  = a;

		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_dgemmsup_rd_haswell_asm_6x4m
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_6x2m
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 0
			const dim_t nr_cur = 1;

			bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;
	uint64_t m_left = m0 % 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	//mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rdx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	//mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// r14 = rax = a
	// rdx = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(imm(0), r15)                   // jj = 0;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;



	mov(var(m_iter), r9)               // ii = m_iter;

	label(.DLOOP3X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif


	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
#endif
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	

	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 2
	
#if 1
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
#if 1
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	
	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)
	
	                                   // ymm4  ymm7  ymm10 ymm13  
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15
	
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)


	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	lea(mem(r12), rcx)                 // rcx = c_iijj;
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	


	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;
	jne(.DLOOP3X4I)                    // iterate again if ii != 0.




	add(imm(4), r15)                   // jj += 4;
	cmp(imm(8), r15)                   // compare jj to 8
	jl(.DLOOP3X4J)                     // if jj < 8, jump to beginning
	                                   // of jj loop; otherwise, loop ends.



	label(.DRETURN)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 8;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict bj  = b;
		double* restrict ai  = a + i_edge*rs_a;

		if ( 2 == m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_2x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			//cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_1x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*
24x24 block

                         1 1 1 1 1 1  1 1 1 1 2 2 2 2
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5  6 7 8 9 0 1 2 3
     |- - - - - - - -|- - - - - - - -| - - - - - - - -|
0    |               |               |                |
1    | m_off_24 = 0  |               |                |
2    | n_off_24 = 0  |               |                |
3    |    m_idx = 0  |               |                |
4    |    n_idx = 0  |               |                |
5    |- - - - - - - -|- - - - - - - -|- - - - - - - - |
6    |               |               |                |
7    | m_off_24 = 6  | m_off_24 = 6  |                |
8    | n_off_24 = 0  | n_off_24 = 8  |                |
9    |    m_idx = 1  |    m_idx = 1  |                |
10   |    n_idx = 0  |    n_idx = 1  |                |
11   |- - - - - - - -|- - - - - - - -|- - - - - - - - |
12   |               |               |                |
13   |               | m_off_24 = 12 | m_off_24 = 12  |
14   |               | n_off_24 = 8  | n_off_24 = 16  |
15   |               |    m_idx = 2  |    m_idx = 2   |
16   |               |    n_idx = 1  |    n_idx = 2   |
17   |- - - - - - - -|- - - - - - - -|- - - - - - - - |
18   |               |               |                |
19   |               |               | m_off_24 = 18  |
20   |               |               | n_off_24 = 16  |
21   |               |               |    m_idx = 3   |
22   |               |               |    n_idx = 2   |
23   |- - - - - - - -|- - - - - - - -|- - - - - - - - |
*/


#define SUBITER_K4_3x4(a, b) \
\
	vmovupd(mem(a       ), ymm0) \
	vmovupd(mem(a, r8, 1), ymm1) \
	vmovupd(mem(a, r8, 2), ymm2) \
	add(imm(4*8), a) \
\
	vmovupd(mem(b        ), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm4) \
	vfmadd231pd(ymm1, ymm3, ymm5) \
	vfmadd231pd(ymm2, ymm3, ymm6) \
\
	vmovupd(mem(b, r11, 1), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm7) \
	vfmadd231pd(ymm1, ymm3, ymm8) \
	vfmadd231pd(ymm2, ymm3, ymm9) \
\
	vmovupd(mem(b, r11, 2), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm10) \
	vfmadd231pd(ymm1, ymm3, ymm11) \
	vfmadd231pd(ymm2, ymm3, ymm12) \
\
	vmovupd(mem(b, r13, 1), ymm3) \
	add(imm(4*8), b) \
	vfmadd231pd(ymm0, ymm3, ymm13) \
	vfmadd231pd(ymm1, ymm3, ymm14) \
	vfmadd231pd(ymm2, ymm3, ymm15) \

#define SUBITER_K1_3x4(a, b) \
\
	vmovsd(mem(a       ), xmm0) \
	vmovsd(mem(a, r8, 1), xmm1) \
	vmovsd(mem(a, r8, 2), xmm2) \
	add(imm(1*8), a) \
\
	vmovsd(mem(b        ), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm4) \
	vfmadd231pd(ymm1, ymm3, ymm5) \
	vfmadd231pd(ymm2, ymm3, ymm6) \
\
	vmovsd(mem(b, r11, 1), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm7) \
	vfmadd231pd(ymm1, ymm3, ymm8) \
	vfmadd231pd(ymm2, ymm3, ymm9) \
\
	vmovsd(mem(b, r11, 2), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm10) \
	vfmadd231pd(ymm1, ymm3, ymm11) \
	vfmadd231pd(ymm2, ymm3, ymm12) \
\
	vmovsd(mem(b, r13, 1), xmm3) \
	add(imm(1*8), b) \
	vfmadd231pd(ymm0, ymm3, ymm13) \
	vfmadd231pd(ymm1, ymm3, ymm14) \
	vfmadd231pd(ymm2, ymm3, ymm15) \

#define SUBITER_K4_2x4(a, b) \
\
	vmovupd(mem(a       ), ymm0) \
	vmovupd(mem(a, r8, 1), ymm1) \
	add(imm(4*8), a) \
\
	vmovupd(mem(b        ), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm4) \
	vfmadd231pd(ymm1, ymm3, ymm5) \
\
	vmovupd(mem(b, r11, 1), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm7) \
	vfmadd231pd(ymm1, ymm3, ymm8) \
\
	vmovupd(mem(b, r11, 2), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm10) \
	vfmadd231pd(ymm1, ymm3, ymm11) \
\
	vmovupd(mem(b, r13, 1), ymm3) \
	add(imm(4*8), b) \
	vfmadd231pd(ymm0, ymm3, ymm13) \
	vfmadd231pd(ymm1, ymm3, ymm14) \

#define SUBITER_K1_2x4(a, b) \
\
	vmovsd(mem(a       ), xmm0) \
	vmovsd(mem(a, r8, 1), xmm1) \
	add(imm(1*8), a) \
\
	vmovsd(mem(b        ), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm4) \
	vfmadd231pd(ymm1, ymm3, ymm5) \
\
	vmovsd(mem(b, r11, 1), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm7) \
	vfmadd231pd(ymm1, ymm3, ymm8) \
\
	vmovsd(mem(b, r11, 2), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm10) \
	vfmadd231pd(ymm1, ymm3, ymm11) \
\
	vmovsd(mem(b, r13, 1), xmm3) \
	add(imm(1*8), b) \
	vfmadd231pd(ymm0, ymm3, ymm13) \
	vfmadd231pd(ymm1, ymm3, ymm14) \

#define SUBITER_K4_1x4(a, b) \
\
	vmovupd(mem(a       ), ymm0) \
	add(imm(4*8), a) \
	vmovupd(mem(b        ), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm4) \
	vmovupd(mem(b, r11, 1), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm7) \
	vmovupd(mem(b, r11, 2), ymm3) \
	vfmadd231pd(ymm0, ymm3, ymm10) \
	vmovupd(mem(b, r13, 1), ymm3) \
	add(imm(4*8), b) \
	vfmadd231pd(ymm0, ymm3, ymm13) \

#define SUBITER_K1_1x4(a, b) \
\
	vmovsd(mem(a       ), xmm0) \
	add(imm(1*8), a) \
	vmovsd(mem(b        ), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm4) \
	vmovsd(mem(b, r11, 1), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm7) \
	vmovsd(mem(b, r11, 2), xmm3) \
	vfmadd231pd(ymm0, ymm3, ymm10) \
	vmovsd(mem(b, r13, 1), xmm3) \
	add(imm(1*8), b) \
	vfmadd231pd(ymm0, ymm3, ymm13) \

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 0 and n_offset is 0(0x0)
(0x0)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		0    x x x x x x x x
|		1    - x x x x x x x
m		2    - - x x x x x x
off		3    - - - x x x x x
24		4    - - - - x x x x
|		5    - - - - - x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_0x0_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.
	label(.DPOSTACCUM_BLOCK1)

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	                                   // now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm5, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx ,2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	jmp(.DDONE_BLOCK1)                 // jump to end.
	label(.DBETAZERO_BLOCK1)

	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm5, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx ,2*8))
	add(rdi, rcx)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	label(.DDONE_BLOCK1)
	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a
	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	SUBITER_K4_1x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_1x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8))
	SUBITER_K4_1x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_1x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	SUBITER_K4_1x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_1x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm1, mem(rcx ,3*8))
	jmp(.DDONE_BLOCK2)                 // jump to end.
	label(.DBETAZERO_BLOCK2)

	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm1, mem(rcx ,3*8))
	label(.DDONE_BLOCK2)
	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a
	dec(r9)                            // ii -= 1;
	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.
	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.
	label(.DPOSTACCUM_BLOCK3)
	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)

	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm6, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm6, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 0(6x0)
(6x0)_U


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		6    - - - - - - x x
|		7    - - - - - - - x
m		8    - - - - - - - -
off		9    - - - - - - - -
24		10   - - - - - - - -
|		11   - - - - - - - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_6x0_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(4), r15)                   // jj = 4;
// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

// ----------------------- Block 3
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_2x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_2x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_2x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_2x4(rax, rbx)
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_2x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_2x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))
	add(rdi, rcx)

	// vfmadd231pd(mem(rcx), ymm3, ymm6)
	// vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))
	add(rdi, rcx)


	label(.DDONE_BLOCK3)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 0, n_offset is 0(0x0) and m_offset is 6, n_offset is 0 (6x0)
(0x0)+(6x0)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		0    x x x x x x x x
|		1    - x x x x x x x
m		2    - - x x x x x x
off		3    - - - x x x x x
24		4    - - - - x x x x
|		5    - - - - - x x x
↓
↑		6    - - - - - - x x
|		7    - - - - - - - x
m		8    - - - - - - - -
off		9    - - - - - - - -
24		10   - - - - - - - -
|		11   - - - - - - - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_0x0_combined_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm5, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx ,2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm5, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx ,2*8))

	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))



	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	SUBITER_K4_1x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_1x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	SUBITER_K4_1x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_1x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	SUBITER_K4_1x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_1x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm1, mem(rcx ,3*8))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm1, mem(rcx ,3*8))


	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm6, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm6, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	a += 6 * rs_a;
	c += 6 * rs_c;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))
	add(rdi, rcx)

	// vfmadd231pd(mem(rcx), ymm3, ymm6)
	// vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))
	add(rdi, rcx)


	label(.DDONE_BLOCK3)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 8(6x8)
(6x8)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		6    x x x x x x x x
|		7    x x x x x x x x
m		8    x x x x x x x x
off		9    - x x x x x x x
24		10   - - x x x x x x
|		11   - - - x x x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_6x8_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm4, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm4, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))


	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1

	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3

	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1

	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3

	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 8(12x8)
(12x8)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		12   - - - - x x x x
|		13   - - - - - x x x
m		14   - - - - - - x x
off		15   - - - - - - - x
24		16   - - - - - - - -
|		17   - - - - - - - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_12x8_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(4), r15)                   // jj = 0;

// ----------------------- Block 3

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm5, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx ,2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm5, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx ,2*8))

	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))



	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_1x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_1x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_1x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_1x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_1x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_1x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm1, mem(rcx ,3*8))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm1, mem(rcx ,3*8))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 16(12x16)
(12x16)_U


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		12   x x x x x x x x
|		13   x x x x x x x x
m		14   x x x x x x x x
off		15   x x x x x x x x
24		16   x x x x x x x x
|		17   - x x x x x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_12x16_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm6, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm6, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))


	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 8(12x8)
(12x8)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		12   - - - - x x x x
|		13   - - - - - x x x
m		14   - - - - - - x x
off		15   - - - - - - - x
24		16   - - - - - - - -
|		17   - - - - - - - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_18x16_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_2x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_2x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_2x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_2x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_2x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_2x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))

	// vfmadd231pd(mem(rcx), ymm3, ymm6)
	// vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))


	label(.DDONE_BLOCK1)
// ----------------------- Block 2
	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm4, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovhpd(xmm4, mem(rcx, 1*8))
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovupd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovhpd(xmm1, mem(rcx, 3*8))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 0 and n_offset is 0(0x0)
(0x0)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		0    x - - - - - - -
|		1    x x - - - - - -
m		2    x x x - - - - -
off		3    x x x x - - - -
24		4    x x x x x - - -
|		5    x x x x x x - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_0x0_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovlpd(xmm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(xmm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(xmm6, mem(rcx))
	vextractf128(imm(1), ymm6, xmm1 )
	vmovlpd(xmm1, mem(rcx, 2*8))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vmovlpd(xmm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm6, mem(rcx))
	vextractf128(imm(1), ymm6, xmm1 )
	vmovlpd(xmm1, mem(rcx, 2*8))


	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovlpd(xmm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(xmm6, mem(rcx))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	add(rdi, rcx)

	vmovlpd(xmm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm6, mem(rcx))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 0(6x0)
(6x0)_L


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		6    x x x x x x x -
|		7    x x x x x x x x
m		8    x x x x x x x x
off		9    x x x x x x x x
24		10   x x x x x x x x
|		11   x x x x x x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_6x0_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm4, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm4, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 8(6x8)
(6x8)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		6    - - - - - - - -
|		7    - - - - - - - -
m		8    x - - - - - - -
off		9    x x - - - - - -
24		10   x x x - - - - -
|		11   x x x x - - - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_6x8_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	add(rdi, rcx)
	add(rdi, rcx)
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovlpd(xmm6, mem(rcx))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)

	add(rdi, rcx)
	add(rdi, rcx)
	vmovlpd(xmm6, mem(rcx))



	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vextractf128(imm(1), ymm5, xmm1 )
	vmovupd(xmm5, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm5, xmm1 )
	vmovupd(xmm5, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK2)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 8(12x8)
(12x8)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		12   x x x x x - - -
|		13   x x x x x x - -
m		14   x x x x x x x -
off		15   x x x x x x x x
24		16   x x x x x x x x
|		17   x x x x x x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_12x8_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovlpd(xmm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(xmm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm6, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))

	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	vmovlpd(xmm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm5, mem(rcx))
	add(rdi, rcx)

	vextractf128(imm(1), ymm6, xmm1 )
	vmovupd(xmm6, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))

	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 1
	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	SUBITER_K4_3x4(rax, rbx)
	// ---------------------------------- iteration 3
	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	SUBITER_K4_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	SUBITER_K1_3x4(rax, rbx)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))


	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 16(12x16)
(12x16)_L


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		12   - - - - - - - -
|		13   - - - - - - - -
m		14   - - - - - - - -
off		15   - - - - - - - -
24		16   x - - - - - - -
|		17   x x - - - - - -
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_12x16_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovlpd(xmm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(xmm6, mem(rcx))


	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	add(rdi, rcx)

	vmovlpd(xmm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm6, mem(rcx))

	label(.DDONE_BLOCK2)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 12, n_offset is 16(12x16) and m_offset is 18, n_offset is 16 (18x16)
(16x12)+(16x18)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		12   - - - - - - - -
|		13   - - - - - - - -
m		14   - - - - - - - -
off		15   - - - - - - - -
24		16   x - - - - - - -
|		17   x x - - - - - -
↓
↑		18   x x x - - - - -
|		19   x x x x - - - -
m		20   x x x x x - - -
off		21   x x x x x x - -
24		22   x x x x x x x -
|		23   x x x x x x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_16x12_combined_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovlpd(xmm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(xmm6, mem(rcx))

	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	add(rdi, rcx)

	vmovlpd(xmm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm6, mem(rcx))

	label(.DDONE_BLOCK2)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	a += 6 * rs_a;
	c += 6 * rs_c;
	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm4, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm4, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm6, ymm6)

	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	add(rdi, rcx)
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovlpd(xmm6, mem(rcx))

	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	add(rdi, rcx)
	add(rdi, rcx)
	vmovlpd(xmm6, mem(rcx))

	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(xmm5, mem(rcx))
	vextractf128(imm(1), ymm5, xmm1 )
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm5, mem(rcx))
	vextractf128(imm(1), ymm5, xmm1 )
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 18 and n_offset is 16(18x16)
(18x16)_L


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		18   x x x - - - - -
|		19   x x x x - - - -
m		20   x x x x x - - -
off		21   x x x x x x - -
24		22   x x x x x x x -
|		23   x x x x x x x x
↓


*/
void bli_dgemmsup_rd_haswell_asm_6x8m_18x16_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

	mov(var(rs_a), r8)                 // load rs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(imm(0), r15)                   // jj = 0;

// ----------------------- Block 1

	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK1)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK1)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK1)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK1)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK1)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK1)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK1)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK1)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK1)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK1)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK1)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK1)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK1)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm4, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK1)                 // jump to end.

	label(.DBETAZERO_BLOCK1)


	vextractf128(imm(1), ymm4, xmm1 )
	vmovupd(xmm4, mem(rcx))
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK1)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 2
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK2)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK2)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK2)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK2)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK2)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK2)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK2)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK2)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK2)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK2)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK2)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK2)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)


	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK2)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK2)                 // jump to end.

	label(.DBETAZERO_BLOCK2)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))

	label(.DDONE_BLOCK2)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

	add(imm(4), r15)                   // jj += 4;

// ----------------------- Block 3
	mov(var(a), r14)                   // load address of a
	mov(var(b), rdx)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c*sizeof(double) = 1*8
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;

	mov(var(m_iter), r9)               // ii = m_iter;

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;


	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK3)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK3)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK3)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK3)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK3)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK3)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK3)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK3)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK3)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK3)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK3)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK3)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm6, ymm6)


	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK3)              // if ZF = 1, jump to beta == 0 case


	add(rdi, rcx)
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovlpd(xmm6, mem(rcx))

	jmp(.DDONE_BLOCK3)                 // jump to end.

	label(.DBETAZERO_BLOCK3)


	add(rdi, rcx)
	add(rdi, rcx)
	vmovlpd(xmm6, mem(rcx))

	label(.DDONE_BLOCK3)

	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;

// ----------------------- Block 4
 	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b_jj;

	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a

	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4_BLOCK4)          // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.

	label(.DLOOPKITER16_BLOCK4)        // MAIN LOOP

	// ---------------------------------- iteration 0

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 2

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16_BLOCK4)          // iterate again if i != 0.

	label(.DCONSIDKITER4_BLOCK4)

	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1_BLOCK4)          // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.

	label(.DLOOPKITER4_BLOCK4)         // EDGE LOOP (ymm)

	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4_BLOCK4)           // iterate again if i != 0.

	label(.DCONSIDKLEFT1_BLOCK4)

	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM_BLOCK4)             // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.

	label(.DLOOPKLEFT1_BLOCK4)         // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.

	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;

	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1_BLOCK4)           // iterate again if i != 0.

	label(.DPOSTACCUM_BLOCK4)

	                                   // ymm4  ymm7  ymm10 ymm13
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15

	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // xmm4[0] = sum(ymm4);  xmm4[1] = sum(ymm7)
	                                   // xmm4[2] = sum(ymm10); xmm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // xmm5[0] = sum(ymm5);  xmm5[1] = sum(ymm8)
	                                   // xmm5[2] = sum(ymm11); xmm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // xmm6[0] = sum(ymm6);  xmm6[1] = sum(ymm9)
	                                   // xmm6[2] = sum(ymm12); xmm6[3] = sum(ymm15)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	lea(mem(r12), rcx)                 // rcx = c_iijj;
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO_BLOCK4)              // if ZF = 1, jump to beta == 0 case


	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(xmm5, mem(rcx))
	vextractf128(imm(1), ymm5, xmm1 )
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	jmp(.DDONE_BLOCK4)                 // jump to end.

	label(.DBETAZERO_BLOCK4)


	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm5, mem(rcx))
	vextractf128(imm(1), ymm5, xmm1 )
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))


	label(.DDONE_BLOCK4)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rd_haswell_asm_6x4m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 3;
	uint64_t m_left = m0 % 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rdx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// r14 = rax = a
	// rdx = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(var(m_iter), r9)               // ii = m_iter;

	label(.DLOOP3X4I)                  // LOOP OVER ii = [ m_iter .. 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif


	lea(mem(r12), rcx)                 // rcx = c + 3*ii*rs_c;
	lea(mem(r14), rax)                 // rax = a + 3*ii*rs_a;
	lea(mem(rdx), rbx)                 // rbx = b;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
#endif
	lea(mem(r8,  r8,  4), rcx)         // rcx = 5*rs_a
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 2
	
#if 1
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 3

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)

#if 1
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rcx, 1, 0*8)) // prefetch rax + 5*rs_a
#endif
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	
	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)
	
	                                   // ymm4  ymm7  ymm10 ymm13  
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15
	
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // ymm4[0] = sum(ymm4);  ymm4[1] = sum(ymm7)
	                                   // ymm4[2] = sum(ymm10); ymm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // ymm5[0] = sum(ymm5);  ymm5[1] = sum(ymm8)
	                                   // ymm5[2] = sum(ymm11); ymm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // ymm6[0] = sum(ymm6);  ymm6[1] = sum(ymm9)
	                                   // ymm6[2] = sum(ymm12); ymm6[3] = sum(ymm15)



	
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	lea(mem(r12), rcx)                 // rcx = c + 3*ii*rs_c;
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	


	lea(mem(r12, rdi, 2), r12)         //
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)         //
	lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

	dec(r9)                            // ii -= 1;
	jne(.DLOOP3X4I)                    // iterate again if ii != 0.




	label(.DRETURN)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 4;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict bj  = b;
		double* restrict ai  = a + i_edge*rs_a;

		if ( 2 == m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_2x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			//cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_1x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
	}
	
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rd_haswell_asm_6x2m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()
	
	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rdx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	//lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	//lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// r14 = rax = a
	// rdx = rbx = b
	// r9  = m dim index ii

	mov(var(m_iter), r9)               // ii = m_iter;

	label(.DLOOP3X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif


	lea(mem(r12), rcx)                 // rcx = c_ii;
	lea(mem(r14), rax)                 // rax = a_ii;
	lea(mem(rdx), rbx)                 // rbx = b;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	lea(mem(rcx, rdi, 2), r10)         //
	lea(mem(r10, rdi, 1), r10)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(r10, 1*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(r10, rdi, 1, 1*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(r10, rdi, 2, 1*8)) // prefetch c + 5*rs_c
#endif
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rbp, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	
#if 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rbp, 1, 0*8)) // prefetch rax + 5*rs_a
#endif

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)

#if 0
	prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*rs_a
	prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*rs_a
	prefetch(0, mem(rax, rbp, 1, 0*8)) // prefetch rax + 5*rs_a
#endif
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rbx        ), xmm0)
	vmovsd(mem(rbx, r11, 1), xmm1)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;

	vmovsd(mem(rax        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovsd(mem(rax, r8,  1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovsd(mem(rax, r8,  2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovsd(mem(rax, r13, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovsd(mem(rax, r8,  4), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovsd(mem(rax, r15, 1), xmm3)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)

	                                   // ymm4  ymm5
	                                   // ymm6  ymm7
	                                   // ymm8  ymm9
	                                   // ymm10 ymm11
	                                   // ymm12 ymm13
	                                   // ymm14 ymm15
	
	vhaddpd( ymm5, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm4 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm5)

	vhaddpd( ymm7, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm6 )

	vhaddpd( ymm9, ymm8, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm8 )

	vhaddpd( ymm11, ymm10, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm10 )

	vhaddpd( ymm13, ymm12, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm12 )

	vhaddpd( ymm15, ymm14, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm14 )

	                                   // xmm4[0:1]  = sum(ymm4)  sum(ymm5)
	                                   // xmm6[0:1]  = sum(ymm6)  sum(ymm7)
	                                   // xmm8[0:1]  = sum(ymm8)  sum(ymm9)
	                                   // xmm10[0:1] = sum(ymm10) sum(ymm11)
	                                   // xmm12[0:1] = sum(ymm12) sum(ymm13)
	                                   // xmm14[0:1] = sum(ymm14) sum(ymm15)


	
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4,  xmm4)          // scale by alpha
	vmulpd(xmm0, xmm6,  xmm6)
	vmulpd(xmm0, xmm8,  xmm8)
	vmulpd(xmm0, xmm10, xmm10)
	vmulpd(xmm0, xmm12, xmm12)
	vmulpd(xmm0, xmm14, xmm14)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm8)
	vmovupd(xmm8, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm10)
	vmovupd(xmm10, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm12)
	vmovupd(xmm12, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm14)
	vmovupd(xmm14, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm8, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm10, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm12, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm14, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	



	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	lea(mem(r14, r8,  4), r14)         //
	lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a

	dec(r9)                            // ii -= 1;
	jne(.DLOOP3X4I)                    // iterate again if ii != 0.




	label(.DRETURN)

	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 2;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict bj  = b;
		double* restrict ai  = a + i_edge*rs_a;

		if ( 3 <= m_left )
		{
			const dim_t mr_cur = 3;

			bli_dgemmsup_rd_haswell_asm_3x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 2 <= m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_2x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_1x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

