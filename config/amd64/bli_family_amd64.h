/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_FAMILY_AMD64_H
#define BLIS_FAMILY_AMD64_H

//To enable framework optimizations for EPYC family processors.
//With this macro defined, we can call kernels directly from
//BLAS interfaces for levels 1 & 2.
//This macro needs to be defined for all EPYC configurations.
#define BLIS_CONFIG_EPYC


// For zen3 architecture we dynamically change block sizes
// based on number of threads. These values were determined
// by running benchmarks on zen3 platform.

#ifdef BLIS_ENABLE_MULTITHREADING

#define BLIS_GEMM_DYNAMIC_BLOCK_SIZE_UPDATE(cntx, rntm,  c) {           \
                                                                        \
    if (bli_is_double(bli_obj_dt(&c))) {                                \
        const dim_t nt = rntm->num_threads;                             \
        const dim_t m = bli_obj_length(&c);                             \
        const dim_t n = bli_obj_width(&c);                              \
                                                                        \
        blksz_t blkszs[BLIS_NUM_BLKSZS];                                \
        if (nt >= 32 && (m > 7800 || n > 7800)) {                       \
            bli_blksz_init_easy(&blkszs[BLIS_MC],   144,    72,   144,    72 ); \
            bli_blksz_init_easy(&blkszs[BLIS_KC],   256,   512,   256,   256 ); \
            bli_blksz_init_easy(&blkszs[BLIS_NC],  4080,  4080,  4080,  4080 ); \
                                                                        \
            bli_cntx_set_blkszs(                                        \
                BLIS_NAT, 3,                                            \
                BLIS_NC, &blkszs[BLIS_NC], BLIS_NR,                     \
                BLIS_KC, &blkszs[BLIS_KC], BLIS_KR,                     \
                BLIS_MC, &blkszs[BLIS_MC], BLIS_MR,                     \
                cntx);                                                  \
        } else {                                                        \
            bli_blksz_init_easy(&blkszs[BLIS_MC],   144,    72,   144,    72 ); \
            bli_blksz_init_easy(&blkszs[BLIS_KC],   256,   256,   256,   256 ); \
            bli_blksz_init_easy(&blkszs[BLIS_NC],  4080,  4080,  4080,  4080 ); \
                                                                        \
            bli_cntx_set_blkszs(                                        \
                BLIS_NAT, 3,                                            \
                BLIS_NC, &blkszs[BLIS_NC], BLIS_NR,                     \
                BLIS_KC, &blkszs[BLIS_KC], BLIS_KR,                     \
                BLIS_MC, &blkszs[BLIS_MC], BLIS_MR,                     \
                cntx);                                                  \
        }                                                               \
    }                                                                   \
}
#else
#define BLIS_GEMM_DYNAMIC_BLOCK_SIZE_UPDATE(cntx, rntm, c) {}
#endif

// Place holder for bundle configuration.

#endif

