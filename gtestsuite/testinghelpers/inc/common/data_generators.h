/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include <random>
#include <type_traits>
#include "common/testing_helpers.h"

namespace testinghelpers {
namespace datagenerators {

// Setting an enum class to make random data generation more robust.
enum class ElementType {FP, INT};
// Define a static variable to be used as the default argument in
// the generators, depending on CMake configuration.
#ifdef BLIS_INT_ELEMENT_TYPE
// Integer random values will be used in testing.
static const ElementType GenericET = ElementType::INT;
#else
// Floating-point random values will be used in testing.
static const ElementType GenericET = ElementType::FP;
#endif

/***************************************************
 *             High-Performance Data Pool
****************************************************/
/**
 * @brief High-performance random data generator using pre-generated pools
 * 
 * This class generates a large pool of random numbers once and then cycles
 * through them for maximum performance. Each pool contains 100,000 random
 * numbers with a fixed seed (94) for reproducible testing.
 * 
 * @tparam T Data type (float, double, scomplex, dcomplex)
 */
template<typename T>
class RandomDataPool {
private:
    using real_T = typename testinghelpers::type_info<T>::real_type;
    static constexpr size_t DEFAULT_POOL_SIZE = 12346;
    
    // Self-contained generator with fixed seed 94
    std::mt19937 pool_generator_;
    std::vector<real_T> pool_;
    std::uniform_real_distribution<real_T> dist_fp_;
    std::uniform_int_distribution<gtint_t> dist_int_;
    size_t pool_size_ = DEFAULT_POOL_SIZE;
    size_t index_;
    ElementType internal_datatype_ = GenericET;
    real_T from_, to_;
    
    void fill_pool();

public:
    explicit RandomDataPool(real_T from, real_T to, ElementType datatype = GenericET);
    explicit RandomDataPool(real_T from, real_T to, size_t pool_size, ElementType datatype = GenericET);
    
    // Fast random access - cycles through the same pool repeatedly
    inline __attribute__((__always_inline__)) real_T next();
    // Return the size of the pool
    size_t size() const;
    // Set starting index for a specific test
    // This is set so that different tests use different parts of the pool
    // but in a way that it is truly reproducible when we run the same test
    // as part of a full suite or in isolation.
    // We have two versions, one for vectors and one for matrices.
    void set_index(gtint_t m, gtint_t n);
    void set_index(gtint_t m, gtint_t n, gtint_t k);
    // Reset the range of generated random numbers
    // This will re-generate the pool with the appropriate range
    void reset(real_T from, real_T to);
    // Optimized vector generation
    std::vector<T> get_random_vector(gtint_t n, gtint_t inc);

    /**
    * @brief Returns a random fp type (float, double, scomplex, dcomplex)
    *        that lies in the range [from, to].
    *
    * @param[in, out] alpha the random fp
    */
    void getfp(T* alpha);
    /**
    * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
    *        with elements that follow a uniform distribution in the range [from, to].
    * @param[in] n length of vector x
    * @param[in] incx increments of vector x
    * @param[in, out] x the random fp vector
    */
    void getfp(gtint_t n, gtint_t incx, T* x);
    /**
    * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
    *        with elements that follow a uniform distribution in the range [from, to].
    * @param[in] storage storage type of matrix A, row or column major
    * @param[in] m, n dimensions of matrix A
    * @param[in, out] a the random fp matrix A
    * @param[in] lda leading dimension of matrix A
    * @param[in] stridea stride between two "continuous" elements in matrix A
    */
    void getfp(char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, gtint_t stridea = 1);
    /**
    * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
    *        with elements that follow a uniform distribution in the range [from, to].
    * @param[in] storage storage type of matrix A, row or column major
    * @param[in] m, n dimensions of matrix A
    * @param[in, out] a the random fp matrix A
    * @param[in] trans transposition of matrix A
    * @param[in] lda leading dimension of matrix A
    * @param[in] stridea stride between two "continuous" elements in matrix A
    */
    void getfp(char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, gtint_t stridea = 1);

    void randomgenerators(gtint_t n, gtint_t incx, T* x);

    void randomgenerators(char storage, gtint_t m, gtint_t n,
        T* a, gtint_t lda, gtint_t stridea = 1);

    void randomgenerators(char storage, gtint_t m, gtint_t n,
        T* a, char transa, gtint_t lda, gtint_t stridea = 1);

    void randomgenerators(char storage, char uplo, gtint_t k, T* a, gtint_t lda);

    std::vector<T> get_random_matrix(char storage, char trans, gtint_t m, gtint_t n,
                        gtint_t lda, gtint_t stridea = 1);

    std::vector<T> get_random_matrix(char storage, char uplo, gtint_t k, gtint_t lda);
};

/***************************************************
 *             Floating Point Generators
****************************************************/
/**
 * @brief Returns a random fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, T1* alpha);
/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, gtint_t n, gtint_t incx, T1* x);

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimensions of matrix A
 * @param[in, out] a the random fp matrix A
 * @param[in] lda leading dimension of matrix A
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, gtint_t lda, gtint_t stridea = 1 );

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimensions of matrix A
 * @param[in, out] a the random fp matrix A
 * @param[in] trans transposition of matrix A
 * @param[in] lda leading dimension of matrix A
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, char transa, gtint_t lda, gtint_t stridea = 1 );

/***************************************************
 *              Integer Generators
****************************************************/
/**
 * @brief Returns a random integer converted to an fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T>
void getint(int from, int to, T* alpha);

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T>
void getint(int from, int to, gtint_t n, gtint_t incx, T* x);

/**
 * @brief Returns a random fp matrix (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimensions of matrix A
 * @param[in, out] a the random fp matrix A
 * @param[in] lda leading dimension of matrix A
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, gtint_t stridea = 1 );

/**
 * @brief Returns a random fp matrix (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimensions of matrix A
 * @param[in, out] a the random fp matrix A
 * @param[in] trans transposition of matrix A
 * @param[in] lda leading dimension of matrix A
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, gtint_t stridea = 1 );

template<typename T1, typename T2, typename T3>
void randomgenerators(T2 from, T3 to, gtint_t n, gtint_t incx, T1* x, ElementType datatype = GenericET);

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, gtint_t lda, gtint_t stridea = 1, ElementType datatype = GenericET );

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, char transa, gtint_t lda, gtint_t stridea = 1, ElementType datatype = GenericET );

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, char uplo, gtint_t k,
                    T1* a, gtint_t lda, ElementType datatype = GenericET );

} //end of namespace datagenerators

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_matrix(T2 from, T3 to, char storage, char trans, gtint_t m, gtint_t n,
                    gtint_t lda, gtint_t stridea = 1, datagenerators::ElementType datatype = datagenerators::GenericET);

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_matrix(T2 from, T3 to, char storage, char uplo, gtint_t k, gtint_t lda, datagenerators::ElementType datatype = datagenerators::GenericET );

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_vector(T2 from, T3 to, gtint_t n, gtint_t incx, datagenerators::ElementType datatype = datagenerators::GenericET);

template<typename T>
void set_vector( gtint_t n, gtint_t incx, T* x, T value );

template<typename T>
void set_matrix( char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, T value );

template<typename T>
void set_matrix( char storage, gtint_t n, T* a, char uplo, gtint_t lda, T value );

template<typename T>
std::vector<T> get_vector( gtint_t n, gtint_t incx, T value );

template<typename T>
std::vector<T> get_matrix( char storage, char trans, gtint_t m, gtint_t n, gtint_t lda, T value );

template<typename T>
void set_ev_mat( char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T exval, T* m );

/*
    Function to set few values of a matrix to values relative to DBL_MAX/DBL_MIN
    These values are used to create overflow and underflow scenarios
*/
template<typename T>
void set_overflow_underflow_mat(char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T* a, gtint_t mode, gtint_t input_range);

} //end of namespace testinghelpers
