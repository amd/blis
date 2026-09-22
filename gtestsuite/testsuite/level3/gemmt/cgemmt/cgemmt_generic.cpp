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

#include <gtest/gtest.h>
#include "level3/gemmt/test_gemmt.h"

// Disable tests for BLIS_TYPED case due to compiler errors.
#ifndef TEST_BLIS_TYPED

// ----------------------------- alpha = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_alpha0_path,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0, 1, 2, 79),                              // k
            ::testing::Values(scomplex{0.0, 0.0}),                       // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

// ----------------------------- k = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k0_path,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0),                                        // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

//----------------------------- n = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_n1_path,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1),                                        // n
            ::testing::Values(1, 2, 103),                                // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

//----------------------------- k = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k1_path,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 79),                                 // n
            ::testing::Values(1),                                        // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

//-------------------- No GEMMT tiny code path yet but use this name ---------------------
INSTANTIATE_TEST_SUITE_P(
        expect_tiny_path,
        cgemmtGeneric3,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(2, 43),                                    // n
            ::testing::Values(2, 19),                                    // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 7),                                     // increment to the leading dim of a
            ::testing::Values(0, 4),                                     // increment to the leading dim of b
            ::testing::Values(0, 11),                                    // increment to the leading dim of c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric3Print<scomplex>()
    );

// ----------------------------- SUP implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sup_path,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(3, 32, 253),                               // n
            ::testing::Values(5, 17, 244),                               // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

// ----------------------------- Native implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_native_path,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(263, 577),                                 // n
            ::testing::Values(3, 47),                                    // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

// ----------------------------- Extreme K value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_K,
        cgemmtGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('u','l'),                                  // uplo u:upper, l:lower
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 6),                                     // n
            ::testing::Values(9689, 33444),                              // k
            ::testing::Values(                     scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // alpha
            ::testing::Values(scomplex{0.0, 0.0},  scomplex{1.0, 0.0},
                              scomplex{-1.0, 0.0}, scomplex{0.0, 0.7},
                              scomplex{1.1, 0.59}),                      // beta
            ::testing::Values(0, 3),                                     // increment to the leading dim of a, b and c
            ::testing::Values(true, false)                               // is memory test
        ),
        ::gemmtMemGeneric1Print<scomplex>()
    );

#endif // ifndef TEST_BLIS_TYPED
