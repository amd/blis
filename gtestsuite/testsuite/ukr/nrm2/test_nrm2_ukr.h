/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "util/nrm2/nrm2.h"
#include <limits>
#include "util/ref_nrm2.h"
#include "inc/check_error.h"

// Defining the function pointer type for ?norm2fv vectorized kernels
// It is based on two template parameters :
// T : datatype of input vector x
// RT : datatype of output norm
template<typename T, typename RT>
using nrm2_ker_ft = void (*)
                    (
                      dim_t    n,
                      T*   x, inc_t incx,
                      RT* norm,
                      cntx_t*  cntx
                    );

// Function to test the ?norm2fv micro-kernels
// The function is templatized based on the datatype of the input and output operands.
// The first parameter(function pointer) uses these template parameters to take the appropriate type.
template<typename T, typename RT>
static void test_nrm2_ukr( nrm2_ker_ft<T, RT> ukr_fp, gtint_t n, gtint_t incx, double thresh,
                        bool is_memory_test = false)
{
    // Pointers to obtain the required memory.
    T *x, *x_copy;
    gtint_t size_x = testinghelpers::buff_dim( n, incx ) * sizeof( T );

    // Create the objects for the input and output operands
    // The kernel does not expect the memory to be aligned
    testinghelpers::ProtectedBuffer x_buffer( size_x, false, is_memory_test );

    // Creating x_copy, to save the contents of x
    testinghelpers::ProtectedBuffer x_copy_buffer( size_x, false, false );

    // Acquire the first greenzone for x
    x = ( T* )x_buffer.greenzone_1;
    x_copy = ( T* )x_copy_buffer.greenzone_1; // For x_copy, there is no greenzone_2

    // Initiaize the memory with random data
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );

    // Copying the contents of x to x_copy
    memcpy( x_copy, x, size_x );

    RT norm = 0.0;
    // Add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Call the ukr function.
        // This call is made irrespective of is_memory_test.
        // This will check for out of bounds access with first redzone(if memory test is true)
        // Else, it will just call the ukr function.
        ukr_fp( n, x, incx, &norm, NULL );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone
            x = ( T* )x_buffer.greenzone_2;

            // Copy the data for x from x_copy accordingly
            memcpy( x, x_copy, size_x );

            norm = 0.0;
            ukr_fp( n, x, incx, &norm, NULL );
        }
    }
    catch(const std::exception& e)
    {
        // Reset to default signal handler
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // Show failure in case seg fault was detected
        FAIL() << "Memory Test Failed";
    }
    // Reset to default signal handler
    testinghelpers::ProtectedBuffer::stop_signal_handler();

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    RT norm_ref = testinghelpers::ref_nrm2<T>( n, x, incx );

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<RT>( norm, norm_ref, thresh );

}