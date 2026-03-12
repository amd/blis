/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include "common/data_generators.h"
// This function returns a reference to a thread-local RandomDataPool instance.
// It is required to ensure that the pool has been generated before use in tests and that the constructor
// is called correctly and exactly once.
// Then the same numbers will be reused in each case, eliminating the start-up time for generating random numbers.
// Each unique combination of template parameters will create a separate pool instance.
// So, if in the same program, get_pool<float, -1, 1>() and get_pool<double, -10, 10>() are called,
// two separate pools will be created and maintained.
template<typename T, gtint_t min = -1, gtint_t max = 1, gtint_t pool_size = 12346> // pool size needs to be a multiple of 2 for complex types
testinghelpers::datagenerators::RandomDataPool<T>& get_pool() {
    using real_type = typename testinghelpers::type_info<T>::real_type;
    static thread_local testinghelpers::datagenerators::RandomDataPool<T> pool(
      static_cast<real_type>(min),
      static_cast<real_type>(max),
      pool_size);
    return pool;
}

// Pool with smaller values to preserve numerical stability in certain tests
template<typename T, gtint_t min = -1, gtint_t max = 1, gtint_t pool_size = 12346> // pool size needs to be a multiple of 2 for complex types
testinghelpers::datagenerators::RandomDataPool<T>& get_tiny_pool() {  
    using real_type = typename testinghelpers::type_info<T>::real_type;
    static thread_local testinghelpers::datagenerators::RandomDataPool<T> pool(
      static_cast<real_type>(min)*0.1,
      static_cast<real_type>(max)*0.1,
      pool_size);
    return pool;
}