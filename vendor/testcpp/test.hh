/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * test.hh
 *
 *
 * Purpose:
 * this header file contains all function prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#ifndef TEST_HH
#define TEST_HH

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <math.h>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

// Allocate memory and initialise memory with random values
void allocate_init_buffer(int *aIn, int m, int n)
{
  aIn =  new int [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aIn[ i ] = ((int) rand() / ((int) RAND_MAX / 2.0)) - 1.0;
  }
}

void allocate_init_buffer(float *&aIn, int m, int n)
{
  aIn =  new float [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aIn[ i ] = ((float) rand() / ((float) RAND_MAX / 2.0)) - 1.0;
  }
}

void allocate_init_buffer(double *&aIn, int m, int n)
{
  aIn =  new double [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aIn[ i ] = ((double) rand() / ((double) RAND_MAX / 2.0)) - 1.0;
  }
}
void allocate_init_buffer(complex<float> *&aIn, int m, int n)
{
  aIn =  new complex<float> [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     float real = ((float) rand() / ((float) RAND_MAX / 2.0)) - 1.0;
     float imag = ((float) rand() / ((float) RAND_MAX / 2.0)) - 1.0;
     aIn[i] = {real,imag};
  }
}
void allocate_init_buffer(complex<double> *&aIn, int m, int n)
{
  aIn =  new complex<double> [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     double real = ((double) rand() / ((double) RAND_MAX / 2.0)) - 1.0;
     double imag = ((double) rand() / ((double) RAND_MAX / 2.0)) - 1.0;
     aIn[i] = {real,imag};
  }
}

template< typename T >
void copy_buffer(T *aSrc, T *&aDest, int m, int n)
{
  aDest =  new T [m*n];
  for ( int i = 0; i < m*n; i ++ ) {
     aDest[i] = aSrc[i];
  }
}

template< typename T >
struct test_real_type
{
  using type = T;
};

template< typename T >
struct test_real_type<complex<T>>
{
  using type = T;
};

inline int test_logical_index(int i, int n, int increment)
{
  return increment < 0 ? (n - 1 - i) * (-increment) : i * increment;
}

inline int test_lower_packed_index(int n, int row, int column)
{
  return column * n - column * (column - 1) / 2 + row - column;
}

template< typename T >
void reference_spr(int n, T alpha, const T *x, int incx, T *a)
{
  for (int column = 0; column < n; ++column)
    for (int row = column; row < n; ++row)
      a[test_lower_packed_index(n, row, column)] +=
        alpha * x[test_logical_index(row, n, incx)] *
                x[test_logical_index(column, n, incx)];
}

template< typename T >
void reference_spr2(int n, T alpha, const T *x, int incx,
                    const T *y, int incy, T *a)
{
  for (int column = 0; column < n; ++column)
    for (int row = column; row < n; ++row)
      a[test_lower_packed_index(n, row, column)] += alpha *
        (x[test_logical_index(row, n, incx)] *
         y[test_logical_index(column, n, incy)] +
         y[test_logical_index(row, n, incy)] *
         x[test_logical_index(column, n, incx)]);
}

template< typename T >
void reference_hpr(int n, typename test_real_type<T>::type alpha,
                   const T *x, int incx, T *a)
{
  for (int column = 0; column < n; ++column)
  {
    for (int row = column; row < n; ++row)
      a[test_lower_packed_index(n, row, column)] += alpha *
        x[test_logical_index(row, n, incx)] *
        std::conj(x[test_logical_index(column, n, incx)]);

    const int diagonal = test_lower_packed_index(n, column, column);
    a[diagonal] = T{a[diagonal].real(), 0};
  }
}

template< typename T >
void reference_hpr2(int n, T alpha, const T *x, int incx,
                    const T *y, int incy, T *a)
{
  for (int column = 0; column < n; ++column)
  {
    for (int row = column; row < n; ++row)
      a[test_lower_packed_index(n, row, column)] +=
        alpha * x[test_logical_index(row, n, incx)] *
                std::conj(y[test_logical_index(column, n, incy)]) +
        std::conj(alpha) * y[test_logical_index(row, n, incy)] *
                           std::conj(x[test_logical_index(column, n, incx)]);

    const int diagonal = test_lower_packed_index(n, column, column);
    a[diagonal] = T{a[diagonal].real(), 0};
  }
}

template< typename T >
void reference_tbmv(int n, int k, const T *a, int lda, T *x, int incx)
{
  std::vector<T> result(n);
  for (int row = 0; row < n; ++row)
  {
    T value = T{};
    const int first_column = std::max(0, row - k);
    for (int column = first_column; column <= row; ++column)
      value += a[(row - column) + column * lda] *
               x[test_logical_index(column, n, incx)];
    result[row] = value;
  }

  for (int row = 0; row < n; ++row)
    x[test_logical_index(row, n, incx)] = result[row];
}

template< typename T >
void reference_tpmv(int n, const T *a, T *x, int incx)
{
  std::vector<T> result(n);
  for (int row = 0; row < n; ++row)
  {
    T value = T{};
    for (int column = 0; column <= row; ++column)
      value += a[test_lower_packed_index(n, row, column)] *
               x[test_logical_index(column, n, incx)];
    result[row] = value;
  }

  for (int row = 0; row < n; ++row)
    x[test_logical_index(row, n, incx)] = result[row];
}

template< typename T >
double test_epsilon()
{
  using real_type = typename test_real_type<T>::type;
  return std::numeric_limits<real_type>::epsilon();
}

template< typename T >
double test_type_factor(double, double, double, double);

template<>
inline double test_type_factor<float>(double s, double, double, double)
{
  return s;
}

template<>
inline double test_type_factor<double>(double, double d, double, double)
{
  return d;
}

template<>
inline double test_type_factor<complex<float>>(double, double, double c, double)
{
  return c;
}

template<>
inline double test_type_factor<complex<double>>(double, double, double, double z)
{
  return z;
}

template< typename T >
double test_threshold(double operation_count,
                      double s_factor,
                      double d_factor,
                      double c_factor,
                      double z_factor)
{
  return operation_count *
         test_type_factor<T>(s_factor, d_factor, c_factor, z_factor) *
         test_epsilon<T>();
}

template< typename T >
double test_threshold(double operation_count,
                      double s_factor,
                      double d_factor,
                      double c_factor,
                      double z_factor,
                      double int_s_factor,
                      double int_d_factor,
                      double int_c_factor,
                      double int_z_factor)
{
#ifdef BLIS_INT_ELEMENT_TYPE
  return test_threshold<T>(operation_count,
                           int_s_factor,
                           int_d_factor,
                           int_c_factor,
                           int_z_factor);
#else
  return test_threshold<T>(operation_count,
                           s_factor,
                           d_factor,
                           c_factor,
                           z_factor);
#endif
}

template< typename T >
double test_error(const T& actual, const T& reference)
{
  const double actual_magnitude = std::abs(actual);
  const double reference_magnitude = std::abs(reference);
  if (!std::isfinite(actual_magnitude) || !std::isfinite(reference_magnitude))
    return std::numeric_limits<double>::infinity();

  if (actual == reference)
    return 0.0;

  const double absolute_error = std::abs(actual - reference);
  return reference_magnitude > 1.0
    ? absolute_error / reference_magnitude
    : absolute_error;
}

template< typename T >
int computeExactV(int n, const T *actual, const T *reference)
{
  for (int i = 0; i < n; ++i)
    if (actual[i] != reference[i])
      return 1;

  return 0;
}

template< typename TA, typename TB >
double test_scal_threshold()
{
  const bool adjusted_complex_float =
    std::is_same<TA, complex<float>>::value &&
    std::is_same<TB, complex<float>>::value;
  return (adjusted_complex_float ? 1.01 : 1.0) * test_epsilon<TB>();
}

template< typename T >
int computeErrorM(
        int    lda,
        int    lda_ref,
        int    m,
        int    n,
        const T *actual,
        const T *reference,
        double threshold
        )
{

    int ret = 0;
    for ( int i = 0; i < m; i ++ ) {
      for ( int j = 0; j < n; j ++ ) {
        const T& actual_value = actual[j * lda + i];
        const T& reference_value = reference[j * lda_ref + i];
        const double error = test_error(actual_value, reference_value);
        if ( error > threshold )  {
                ret = 1;
            }
        }
    }
    return ret;

}



 template< typename T >
 int computeErrorV(
         int    n,
   const T *actual,
   const T *reference,
         double threshold
         )
 {
     int ret = 0;
         for ( int i = 0; i < n; i ++ ) {
           const double error = test_error(actual[i], reference[i]);
           if ( error > threshold)  {
                 ret = 1;
             }
         }

       return ret;

 }

/*
 *printing matrix and vector
 *
 */

template <typename T>
void printmatrix(
        T *A,
        int    lda,
        int    m,
        int    n,
        char *func_str
        )
{
    int    i, j;
    cout << func_str <<"\n";
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
		cout<< A[j * lda + i]<<" ";
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
void printvector(
         T *X,
         int    m,
        char *func_str
         )
 {
     int    i;
    cout << func_str <<"\n";
     for ( i = 0; i < m; i ++ ) {
                  cout<< X[i]<<" ";
                  cout<<"\n";
          }
          printf("\n");

 } 


#endif
