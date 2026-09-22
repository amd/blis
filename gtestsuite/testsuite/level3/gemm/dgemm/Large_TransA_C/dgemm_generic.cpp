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

#include <gtest/gtest.h>
#include "level3/gemm/test_gemm.h"

// ----------------------------- Native implementation --------------------------------------
INSTANTIATE_TEST_SUITE_P(
        expect_native_path,
        dgemmGeneric1,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('c'),                                      // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(1103),                                     // m
            ::testing::Values(1107),                                     // n
            ::testing::Values(2313),                                     // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );

INSTANTIATE_TEST_SUITE_P(
        expect_native_path_Large,
        dgemmGeneric1,
        ::testing::Combine(
            // Storage of A and B is handled by packing
            ::testing::Values('c'),                                      // storage format
            ::testing::Values('c'),                                      // transa
            ::testing::Values('n', 'c', 't'),                            // transb
            ::testing::Values(2203),                                     // m
            ::testing::Values(5417),                                     // n
            ::testing::Values(2313),                                     // k
            ::testing::Values(     1.0, -1.0, 1.7),                      // alpha
            ::testing::Values(0.0, 1.0, -1.0, 2.3),                      // beta
            ::testing::Values(0, 3)                                      // increment to the leading dim of a, b and c
        ),
        ::gemmGeneric1Print<double>()
    );
