/*

   BLISPP
   C++ test driver for BLIS CPP rotg routine and reference blis rotg routine.

   Copyright (C) 2019 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include <cmath>
#include <complex>
#include <iostream>
#include "blis.hh"
#include "test.hh"

using namespace blis;
using namespace std;
//#define PRINT

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T>
void test_rotg(T a_input, T b_input)
{

    T a = a_input;
    T b = b_input;
    T c, s;
    T a_ref, b_ref, c_ref, s_ref;

      if (b_input == T{0})
      {
         a_ref = a_input;
         b_ref = T{0};
         c_ref = T{1};
         s_ref = T{0};
      }
      else
      {
         const T scale = std::abs(a_input) + std::abs(b_input);
         const T roe = std::abs(a_input) > std::abs(b_input) ? a_input : b_input;
         const T r = std::copysign(
            scale * std::sqrt((a_input / scale) * (a_input / scale) +
                              (b_input / scale) * (b_input / scale)),
            roe);
         a_ref = r;
         c_ref = a_input / r;
         s_ref = b_input / r;
         b_ref = T{1};
         if (std::abs(a_input) > std::abs(b_input))
            b_ref = s_ref;
         else if (c_ref != T{0})
            b_ref = T{1} / c_ref;
      }

#ifdef PRINT
        cout<< "Before blis::rotg \na Value : " << a << "\n" ;
        cout<< "b Value : " << b << "\n" ;
#endif
    blis::rotg<T>( 
            &a,
            &b,
            &c,
            &s
            );

#ifdef PRINT
        cout<< "After blis::rotg \na Value : " << a << "\n" ;
        cout<< "b Value : " << b << "\n" ;
        cout<< "c Value : " << c << "\n" ;
        cout<< "s Value : " << s << "\n" ;
#endif

#ifdef PRINT
        cout<< "Expected Output\na Value : " << a_ref << "\n" ;
        cout<< "b Value : " << b_ref << "\n" ;
        cout<< "c Value : " << c_ref << "\n" ;
        cout<< "s Value : " << s_ref << "\n" ;
#endif
   const double threshold = 5.0 * test_epsilon<T>();
    if( test_error(a, a_ref) > threshold ||
       test_error(b, b_ref) > threshold ||
       test_error(c, c_ref) > threshold ||
       test_error(s, s_ref) > threshold )
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
     else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);

}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
   test_rotg<float>(1.0f, 1.0f);
   test_rotg<float>(-2.0f, 1.0f);
   test_rotg<float>(1.0f, -2.0f);
   test_rotg<double>(1.0, 0.0);
   test_rotg<double>(-2.0, 1.0);
   test_rotg<double>(1.0, -2.0);
    return 0;

}
