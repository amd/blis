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
#include "level3/gemm/test_gemm.h"

// ----------------------------- alpha = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_alpha0_path,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 103),                                // m
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0, 1, 2, 79),                              // k
            ::testing::Values(0.0),                                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

// ----------------------------- k = 0 --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k0_path,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 103),                                // m
            ::testing::Values(1, 2, 114),                                // n
            ::testing::Values(0),                                        // k
            ::testing::Values(     1.0, -1.0, 2.3),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

//----------------------------- m = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_m1_path,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1),                                        // m
            ::testing::Values(1, 2, 79),                                 // n
            ::testing::Values(1, 2, 103),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

//----------------------------- n = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_n1_path,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 79),                                 // m
            ::testing::Values(1),                                        // n
            ::testing::Values(1, 2, 103),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

//----------------------------- k = 1 ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_k1_path,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 2, 103),                                // m
            ::testing::Values(1, 2, 79),                                 // n
            ::testing::Values(1),                                        // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

//----------------------------- bli_dgemm_tiny kernel ------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_tiny_path,
        dgemmGeneric3,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(3, 81, 138),                               // m
            ::testing::Values(2, 35, 100),                               // n
            ::testing::Values(5, 12, 24),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 7),                                     // increment to the leading dim of a
            ::testing::Values(0, 4),                                     // increment to the leading dim of b
            ::testing::Values(0, 11)                                     // increment to the leading dim of c
        ),
        ::gemmGeneric3Print<double>()
    );

//----------------------------- dgemm_small kernel -----------------------------------
// Tests both bli_dgemm_small and bli_dgemm_small_At
INSTANTIATE_TEST_SUITE_P(
        expect_small_path,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            // Covers all possible combinations of storage schemes
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(5, 19, 32, 44),                            // m
            ::testing::Values(25, 27, 32),                               // n
            ::testing::Values(5, 17, 24),                                // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

// ----------------------------- SUP implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_sup_path,
        dgemmGeneric1,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(877, 1002),                                // m
            ::testing::Values(453, 567),                                 // n
            ::testing::Values(105, 124),                                 // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

// ----------------------------- Extreme M value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_M,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(9689, 33444),                              // m
            ::testing::Values(1, 6),                                     // n
            ::testing::Values(1, 34),                                    // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

// ----------------------------- Extreme N value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_N,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 6),                                     // m
            ::testing::Values(9689, 33444),                              // n
            ::testing::Values(1, 34),                                    // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

// ----------------------------- Extreme K value --------------------------------------------
INSTANTIATE_TEST_SUITE_P(
        extreme_K,
        dgemmGeneric1,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS_LIKE
                             ,'r'
#endif
            ),                                                           // storage format
            ::testing::Values('n', 'c', 't'),                            // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1, 34),                                    // m
            ::testing::Values(1, 6),                                     // n
            ::testing::Values(9689, 33444),                              // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );
