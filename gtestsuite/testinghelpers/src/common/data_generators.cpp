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

#include <random>
#include <type_traits>
#include "common/testing_helpers.h"
#include "common/data_generators.h"

namespace testinghelpers {
namespace datagenerators {

constexpr int RANDOM_POOL_SEED = 94;

/***************************************************
 *             Data Pool Generators
****************************************************/
template<typename T>
RandomDataPool<T>::RandomDataPool(real_T from, real_T to, ElementType datatype)
    : pool_generator_(RANDOM_POOL_SEED),           // Initialize with seed
        pool_(DEFAULT_POOL_SIZE),      // Initialize with size
        index_(0),
        internal_datatype_(datatype),
        from_(from),
        to_(to){
    if (datatype == ElementType::FP) {
        dist_fp_ = std::uniform_real_distribution<real_T>(from, to);
    } else {
        dist_int_ = std::uniform_int_distribution<gtint_t>(static_cast<gtint_t>(from), static_cast<gtint_t>(to));
    }
    fill_pool();  // Generate random values ONCE during construction
}
    
template<typename T>
RandomDataPool<T>::RandomDataPool(real_T from, real_T to, size_t pool_size, ElementType datatype)
    : pool_generator_(RANDOM_POOL_SEED),           // Initialize with seed
        pool_(pool_size),      // Initialize with pool_size
        pool_size_(pool_size),
        index_(0),
        internal_datatype_(datatype),
        from_(from),
        to_(to){
    if (pool_size_ % 2 != 0) {
        throw std::runtime_error("RandomDataPool: pool_size must be a multiple of 2 for complex number support");
    }
    if (datatype == ElementType::FP) {
        dist_fp_ = std::uniform_real_distribution<real_T>(from, to);
    } else {
        dist_int_ = std::uniform_int_distribution<gtint_t>(static_cast<gtint_t>(from), static_cast<gtint_t>(to));
    }
    fill_pool();  // Generate random values ONCE during construction
}

template<typename T>
void RandomDataPool<T>::fill_pool() {
    // Generate exactly pool_size_ random numbers ONCE
    if (internal_datatype_ == ElementType::FP) {
        for (size_t i = 0; i < pool_size_; ++i) {
                pool_[i] = dist_fp_(pool_generator_);
        }
    } else {
        for (size_t i = 0; i < pool_size_; ++i) {
                pool_[i] = dist_int_(pool_generator_);
        }
    }
}

template<typename T>
void RandomDataPool<T>::set_index(gtint_t m, gtint_t n) {
    gtint_t start_index = m * 5 + (n + 4) * 3;
    // Use modulo operation to ensure we stay within pool size
    index_ = start_index % pool_size_;
    // Ensure index is even for complex number optimization
    if (index_ % 2 != 0) index_ += 1;
}

template<typename T>
void RandomDataPool<T>::set_index(gtint_t m, gtint_t n, gtint_t k) { 
    gtint_t start_index = m * 6 + (n + 5) * 4 + (k + 3) * 2;
    // Use modulo operation to ensure we stay within pool size
    index_ = start_index % pool_size_;
    // Ensure index is even for complex number optimization
    if (index_ % 2 != 0) index_ += 1;
}

// Fast random access - cycles through the same pool repeatedly
template<typename T>
inline __attribute__((__always_inline__)) typename RandomDataPool<T>::real_T RandomDataPool<T>::next() {
    typename RandomDataPool<T>::real_T value = pool_[index_];  // Copy current element
    ++index_;
    
    // When we reach the end, wrap around to the beginning (reuse same numbers)
    if (index_ >= pool_size_) {
        index_ = 0;  // Reset to first element, reuse existing values
    }
    return value;
} 

template<typename T>
void RandomDataPool<T>::reset(real_T from, real_T to) {
    from_ = from;
    to_ = to;
    if (internal_datatype_ == ElementType::FP) {
        dist_fp_.param(typename std::uniform_real_distribution<real_T>::param_type(from, to));
    } else {
        dist_int_.param(typename std::uniform_int_distribution<gtint_t>::param_type(static_cast<gtint_t>(from), static_cast<gtint_t>(to)));
    }
    fill_pool();  // Only regenerate when range changes
    index_ = 0;   // Reset position
}

template<typename T>
size_t RandomDataPool<T>::size() const {
    return pool_size_;
}

/**
 * @brief Returns a random fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T>
void RandomDataPool<T>::getfp(T* alpha) {
    if constexpr (testinghelpers::type_info<T>::is_real)
        *alpha = next();
    else
        *alpha = {next(), next()};
}

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T>
void RandomDataPool<T>::getfp(gtint_t n, gtint_t incx, T* x)
{
    if (incx == 1) {
        real_T* pool_data = pool_.data();
        real_T* x_as_real;
        gtint_t n_real;
        
        if constexpr (testinghelpers::type_info<T>::is_real) {
            x_as_real = reinterpret_cast<real_T*>(x);
            n_real = n;
        } else {
            // For complex: treat n complex numbers as 2*n real numbers
            // Since set_index ensures index is always a multiple of 2,
            // we can safely reinterpret complex array as real array
            x_as_real = reinterpret_cast<real_T*>(x);
            n_real = n * 2;
        }
        
        // Common fast path: direct memory copy for contiguous data
        gtint_t remaining = n_real;
        gtint_t dest_idx = 0;
        
        while (remaining > 0) {
            gtint_t chunk_size = (std::min)(remaining, static_cast<gtint_t>(pool_size_ - index_));
            
            // Direct memory copy - much faster than individual assignments
            memcpy(&x_as_real[dest_idx], &pool_data[index_], chunk_size * sizeof(real_T));
            
            dest_idx += chunk_size;
            remaining -= chunk_size;
            index_ += chunk_size;
            
            if (index_ >= pool_size_) {
                index_ = 0;
            }
        }
    } else {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        // Use vectorized fill instead of element-by-element loop
        std::fill_n(x, testinghelpers::buff_dim(n, incx), T{-1.2345e38});
        // General case with stride
        const gtint_t abs_incx = std::abs(incx);
        if constexpr (testinghelpers::type_info<T>::is_real) {
            for (gtint_t i = 0; i < n; ++i) {
                x[i * abs_incx] = next();  // Cycles through pool
            }
        } else {
            for (gtint_t i = 0; i < n; ++i) {
                x[i * abs_incx] = {next(), next()};  // Cycles through pool
            }
        }
    }
    
}

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimensions of matrix A
 * @param[in, out] a the random fp matrix A
 * @param[in] lda leading dimension of matrix A
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T>
void RandomDataPool<T>::getfp(char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, gtint_t stridea )
{
    if((storage == 'c') || (storage == 'C')) {
        if (m > 0)
        {
            // Fill each column by calling the vector version
            for(gtint_t j=0; j<n; j++)
            {
                getfp(m, stridea, &a[j * lda]);
                // Fill trailing elements with sentinel value
                gtint_t last_filled = (m - 1) * stridea + 1;
                if (last_filled < lda) {
                    std::fill_n(&a[last_filled + j*lda], lda - last_filled, T{-1.2345e38});
                }
            }
        }
        else
        {
            std::fill_n(a, static_cast<size_t>(n * lda), T{-1.2345e38});
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        if (n > 0)
        {
            for(gtint_t i=0; i<m; i++)
            {
                getfp(n, stridea, &a[i * lda]);
                // Fill trailing elements with sentinel value
                gtint_t last_filled = (n - 1) * stridea + 1;
                if (last_filled < lda) {
                    std::fill_n(&a[last_filled + i*lda], lda - last_filled, T{-1.2345e38});
                }
            }
        }
        else
        {
            std::fill_n(a, static_cast<size_t>(m * lda), T{-1.2345e38});
        }
    }
}

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
template<typename T>
void RandomDataPool<T>::getfp(char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, gtint_t stridea ) {
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }
    this->getfp(storage, m, n, a, lda, stridea );
}

// Optimized vector generation
template<typename T>
std::vector<T> RandomDataPool<T>::get_random_vector(gtint_t n, gtint_t inc) {
    // Create vector for the given sizes.
    std::vector<T> x( testinghelpers::buff_dim(n, inc) );
    this->randomgenerators( n, inc, x.data());
    return x;
}

// Optimized vrandom generation for matrices
template<typename T>
void RandomDataPool<T>::randomgenerators(gtint_t n, gtint_t incx, T* x) {
    this->getfp(n, incx, x );
}

template<typename T>
void RandomDataPool<T>::randomgenerators(char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, gtint_t stridea) {
    this->getfp(storage, m, n, a, lda, stridea);
}

// This is used in gemv and needs to be optimized
template<typename T>
void RandomDataPool<T>::randomgenerators(char storage, gtint_t m, gtint_t n,
    T* a, char transa, gtint_t lda, gtint_t stridea){
    this->getfp(storage, m, n, a, transa, lda, stridea);
}

template<typename T>
void RandomDataPool<T>::randomgenerators(char storage, char uplo, gtint_t k, T* a, gtint_t lda){
    this->randomgenerators(storage, k, k, a, lda, 1);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<k; j++)
        {
            for(gtint_t i=0; i<k; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
    else
    {
        for(gtint_t i=0; i<k; i++)
        {
            for(gtint_t j=0; j<k; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[j+i*lda] = T{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}
    
template<typename T>
std::vector<T> RandomDataPool<T>::get_random_matrix(char storage, char trans, gtint_t m, gtint_t n,
                        gtint_t lda, gtint_t stridea) {
    std::vector<T> a(matsize(storage, trans, m, n, lda));
    this->randomgenerators(storage, m, n, a.data(), trans, lda, stridea);
    return a;
}

template<typename T>
std::vector<T> RandomDataPool<T>::get_random_matrix(char storage, char uplo, gtint_t k, gtint_t lda) {
    // Create matrix for the given sizes.
    std::vector<T> a( testinghelpers::matsize( storage, 'n', k, k, lda ) );
    this->randomgenerators(storage, uplo, k, a.data(), lda);
    return a;
}


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
void getfp(T2 from, T3 to, T1* alpha)
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    std::mt19937                              generator(RANDOM_POOL_SEED);
    std::uniform_real_distribution<real_T>    distr(from, to);
    if constexpr (testinghelpers::type_info<T1>::is_real)
        *alpha = distr(generator);
    else
        *alpha = {distr(generator), distr(generator)};
}

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, gtint_t n, gtint_t incx, T1* x)
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    std::mt19937                              generator(RANDOM_POOL_SEED);
    // Generate the values from the uniform distribution that
    // the BLAS routine should read and/or modify.
    std::uniform_real_distribution<real_T>    distr(from, to);

    // Optimize based on stride pattern
    if (incx == 1) {
        // Fast path for contiguous memory
        if constexpr (testinghelpers::type_info<T1>::is_real) {
            for (gtint_t i = 0; i < n; ++i) {
                x[i] = distr(generator);
            }
        } else {
            for (gtint_t i = 0; i < n; ++i) {
                x[i] = {distr(generator), distr(generator)};
            }
        }
    } else {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        // Use vectorized fill instead of element-by-element loop
        std::fill_n(x, testinghelpers::buff_dim(n, incx), T1{-1.2345e38});
        // General case with stride
        const gtint_t abs_incx = std::abs(incx);
        if constexpr (testinghelpers::type_info<T1>::is_real) {
            for (gtint_t i = 0; i < n; ++i) {
                x[i * abs_incx] = distr(generator);
            }
        } else {
            for (gtint_t i = 0; i < n; ++i) {
                x[i * abs_incx] = {distr(generator), distr(generator)};
            }
        }
    }
}

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
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, gtint_t lda, gtint_t stridea )
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    std::mt19937                              generator(RANDOM_POOL_SEED);
    std::uniform_real_distribution<real_T>    distr(from, to);

    if((storage == 'c') || (storage == 'C'))
    {
        if (m > 0)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if constexpr (testinghelpers::type_info<T1>::is_real)
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T1{-1.2345e38};

                        a[i*stridea+j*lda] = real_T(distr(generator));
                    }
                    a[(m-1)*stridea+j*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T1{-1.2345e38};

                        a[i*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(m-1)*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t i=(m-1)*stridea+1; i<lda; i++)
                {
                    a[i+j*lda] = T1{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t j=0; j<n; j++)
            {
                for(gtint_t i=0; i<lda; i++)
                {
                    a[i+j*lda] = T1{-1.2345e38};
                }
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        if (n > 0)
        {
            for(gtint_t i=0; i<m; i++)
            {
                if constexpr (testinghelpers::type_info<T1>::is_real)
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T1{-1.2345e38};

                        a[j*stridea+i*lda] = real_T(distr(generator));
                    }
                    a[(n-1)*stridea+i*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T1{-1.2345e38};

                        a[j*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(n-1)*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t j=(n-1)*stridea+1; j<lda; j++)
                {
                    a[j+i*lda] = T1{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t i=0; i<m; i++)
            {
                for(gtint_t j=0; j<lda; j++)
                {
                    a[j+i*lda] = T1{-1.2345e38};
                }
            }
        }
    }
}
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
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, char transa, gtint_t lda, gtint_t stridea )
{
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }
    getfp<T1>( from, to, storage, m, n, a, lda, stridea );
}

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
void getint(int from, int to, T* alpha)
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                              generator(RANDOM_POOL_SEED);
    std::uniform_int_distribution<int>    distr(from, to);
    if constexpr (testinghelpers::type_info<T>::is_real)
        *alpha = real_T(distr(generator));
    else
        *alpha = {real_T(distr(generator)), real_T(distr(generator))};
}
/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T>
void getint(int from, int to, gtint_t n, gtint_t incx, T* x)
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                              generator(RANDOM_POOL_SEED);
    // Generate the values from the uniform distribution that
    // the BLAS routine should read and/or modify.
    std::uniform_int_distribution<int>    distr(from, to);

    if (incx == 1) {
        if constexpr (testinghelpers::type_info<T>::is_real) {
            for (gtint_t i = 0; i < n; ++i) {
                x[i] = real_T(distr(generator));
            }
        } else {
            for (gtint_t i = 0; i < n; ++i) {
                x[i] = {real_T(distr(generator)), real_T(distr(generator))};
            }
        }
    } else {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        std::fill_n(x, testinghelpers::buff_dim(n, incx), T{-1.2345e38});

        const gtint_t abs_incx = std::abs(incx);
        if constexpr (testinghelpers::type_info<T>::is_real) {
            for (gtint_t i = 0; i < n; ++i) {
                x[i * abs_incx] = real_T(distr(generator));
            }
        } else {
            for (gtint_t i = 0; i < n; ++i) {
                x[i * abs_incx] = {real_T(distr(generator)), real_T(distr(generator))};
            }
        }
    }
}

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
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, gtint_t stridea )
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                              generator(RANDOM_POOL_SEED);
    std::uniform_int_distribution<int>    distr(from, to);

    if((storage == 'c') || (storage == 'C'))
    {
        if (m > 0)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T{-1.2345e38};

                        a[i*stridea+j*lda] = real_T(distr(generator));
                    }
                    a[(m-1)*stridea+j*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T{-1.2345e38};

                        a[i*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(m-1)*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t i=(m-1)*stridea+1; i<lda; i++)
                {
                    a[i+j*lda] = T{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t j=0; j<n; j++)
            {
                for(gtint_t i=0; i<lda; i++)
                {
                    a[i+j*lda] = T{-1.2345e38};
                }
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        if (n > 0)
        {
            for(gtint_t i=0; i<m; i++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T{-1.2345e38};

                        a[j*stridea+i*lda] = real_T(distr(generator));
                    }
                    a[(n-1)*stridea+i*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T{-1.2345e38};

                        a[j*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(n-1)*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t j=(n-1)*stridea+1; j<lda; j++)
                {
                    a[j+i*lda] = T{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t i=0; i<m; i++)
            {
                for(gtint_t j=0; j<lda; j++)
                {
                    a[j+i*lda] = T{-1.2345e38};
                }
            }
        }
    }
}

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
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, gtint_t stridea )
{
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }
    getint<T>( from, to, storage, m, n, a, lda, stridea );
}

template<typename T1, typename T2, typename T3>
void randomgenerators(T2 from, T3 to, gtint_t n, gtint_t incx, T1* x, ElementType datatype) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, n, incx, x );
    else
        getfp<T1>( from, to, n, incx, x );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, gtint_t lda, gtint_t stridea, ElementType datatype ) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, storage, m, n, a, lda, stridea );
    else
        getfp<T1>( from, to, storage, m, n, a, lda, stridea );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, char transa, gtint_t lda, gtint_t stridea, ElementType datatype ) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, storage, m, n, a, transa, lda, stridea );
    else
        getfp<T1>( from, to, storage, m, n, a, transa, lda, stridea );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, char uplo, gtint_t k,
                    T1* a, gtint_t lda, ElementType datatype ) {
    testinghelpers::datagenerators::randomgenerators<T1>(from, to, storage, k, k, a, lda, 1, datatype);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<k; j++)
        {
            for(gtint_t i=0; i<k; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T1{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T1{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
    else
    {
        for(gtint_t i=0; i<k; i++)
        {
            for(gtint_t j=0; j<k; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[j+i*lda] = T1{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T1{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}

} //end of namespace datagenerators

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_matrix(T2 from, T3 to, char storage, char trans, gtint_t m, gtint_t n,
                    gtint_t lda, gtint_t stridea, datagenerators::ElementType datatype )
{
    std::vector<T1> a(matsize(storage, trans, m, n, lda));
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, storage, m, n, a.data(), trans, lda, stridea, datatype );
    return a;
}

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_matrix(T2 from, T3 to, char storage, char uplo, gtint_t k, gtint_t lda, datagenerators::ElementType datatype )
{
    // Create matrix for the given sizes.
    std::vector<T1> a( testinghelpers::matsize( storage, 'n', k, k, lda ) );
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, storage, uplo, k, a.data(), lda, datatype );
    return a;
}

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_vector(T2 from, T3 to, gtint_t n, gtint_t incx, datagenerators::ElementType datatype )
{
    // Create vector for the given sizes.
    std::vector<T1> x( testinghelpers::buff_dim(n, incx) );
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, n, incx, x.data(), datatype );
    return x;
}

template<typename T>
void set_vector( gtint_t n, gtint_t incx, T* x, T value )
{
    if (incx == 1) {
        std::fill_n(x, n, value);
    } else {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        std::fill_n(x, testinghelpers::buff_dim(n, incx), T{-1.2345e38});
        
        const gtint_t abs_incx = std::abs(incx);
        for (gtint_t i = 0; i < n; ++i) {
            x[i * abs_incx] = value;
        }
    }
}

template<typename T>
void set_matrix( char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, T value )
{
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }

    if((storage == 'c') || (storage == 'C'))
    {
        for( gtint_t j = 0 ; j < n ; j++ )
        {
            for( gtint_t i = 0 ; i < m ; i++ )
            {
                a[i+j*lda] = value ;
            }
            for(gtint_t i=m; i<lda; i++)
            {
                a[i+j*lda] = T{-1.2345e38};
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        for( gtint_t i = 0 ; i < m ; i++ )
        {
            for( gtint_t j = 0 ; j < n ; j++ )
            {
                a[j+i*lda] = value ;
            }
            for(gtint_t j=n; j<lda; j++)
            {
                a[j+i*lda] = T{-1.2345e38};
            }
        }
    }
}

template<typename T>
void set_matrix( char storage, gtint_t n, T* a, char uplo, gtint_t lda, T value )
{
    testinghelpers::set_matrix<T>(storage, n, n, a, 'n', lda, value );
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<n; j++)
        {
            for(gtint_t i=0; i<n; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
    else
    {
        for(gtint_t i=0; i<n; i++)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[j+i*lda] = T{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}

template<typename T>
std::vector<T> get_vector( gtint_t n, gtint_t incx, T value )
{
    // Create vector for the given sizes.
    std::vector<T> x( testinghelpers::buff_dim(n, incx) );
    testinghelpers::set_vector( n, incx, x.data(), value );
    return x;
}

template<typename T>
std::vector<T> get_matrix( char storage, char trans, gtint_t m, gtint_t n, gtint_t lda, T value )
{
    std::vector<T> a( matsize( storage, trans, m, n, lda ) );
    testinghelpers::set_matrix<T>( storage, m, n, a.data(), trans, lda, value );
    return a;
}

template<typename T>
void set_ev_mat( char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T exval, T* m )
{
    // Setting the exception values on the indices passed as arguments
    if ( storage == 'c' || storage == 'C' )
    {
      if ( trns == 'n' || trns == 'N' )
        m[i + j*ld] = exval;
      else
        m[j + i*ld] = exval;
    }
    else
    {
      if ( trns == 'n' || trns == 'N' )
        m[i*ld + j] = exval;
      else
        m[j*ld + i] = exval;
    }
}

/*
    Function to set few values of a matrix to values relative to DBL_MAX/DBL_MIN
    These values are used to create overflow and underflow scenarios
*/
template<typename T>
void set_overflow_underflow_mat(char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T* a, gtint_t mode, gtint_t input_range)
{
    /* Calculate index where overflow/underflow values need to be inserted */
    gtint_t indexA = 0;

    if ( storage == 'c' || storage == 'C' )
    {
      if ( trns == 'n' || trns == 'N' )
      {
        indexA = i + j*ld;
      }
      else
      {
        indexA = j + i*ld;
      }
    }
    else
    {
      if ( trns == 'n' || trns == 'N' )
      {
        indexA = i*ld + j;
      }
      else
      {
        indexA = j*ld + i;
      }
    }

    using RT = typename testinghelpers::type_info<T>::real_type;
    std::vector<gtint_t> exponent(12);

    if (std::is_same<RT, double>::value)
    {
      exponent = {23, 203, 18, 180, 123, 130, 185, 178, 108, 158, 185, 220};
    }
    else if (std::is_same<RT, float>::value)
    {
      exponent = {3, 20, 8, 2, 30, 28, 8, 10, 33, 24, 8, 22};
    }

    RT limits_val;

    // Helper lambda to compute power with correct type
    auto compute_power = [](RT base, gtint_t exp) -> RT {
        return std::pow(base, static_cast<RT>(exp));
    };


    /* When mode is set to 0, values relative to DBL_MAX are inserted into the input matrices */
    if(mode == 0)
    {
        limits_val = (std::numeric_limits<RT>::max)();
        switch(input_range)
        {
            case -1:
                if constexpr (testinghelpers::type_info<T>::is_real) {
                    a[0] = limits_val/ compute_power(RT(10), exponent[0]);
                    a[indexA] = limits_val/ compute_power(RT(10), exponent[1]);
                } else {
                    a[0] = {limits_val / compute_power(RT(10), exponent[0]), RT(0)};  // Complex with real part, zero imaginary
                    a[indexA] = {limits_val / compute_power(RT(10), exponent[1]), RT(0)};  // Complex with real part, zero imaginary
                }
                break;

            case 0:
                if constexpr (testinghelpers::type_info<T>::is_real) {
                    a[0] = -(limits_val/ compute_power(RT(10), exponent[4]));
                    a[indexA] = -(limits_val/ compute_power(RT(10), exponent[5]));
                } else {
                    a[0] = {-limits_val / compute_power(RT(10), exponent[4]), RT(0)};  // Complex with real part, zero imaginary
                    a[indexA] = {-limits_val / compute_power(RT(10), exponent[5]), RT(0)};  // Complex with real part, zero imaginary
                }
                break;

            case 1:
                if constexpr (testinghelpers::type_info<T>::is_real) {
                    a[0] = limits_val/ compute_power(RT(10), exponent[8]);
                    a[indexA] = limits_val/ compute_power(RT(10), exponent[9]);
                } else {
                    a[0] = {limits_val / compute_power(RT(10), exponent[8]), RT(0)};  // Complex with real part, zero imaginary
                    a[indexA] = {limits_val / compute_power(RT(10), exponent[9]), RT(0)};  // Complex with real part, zero imaginary
                }
                break;
        }
    }
    /* When mode is set to 1, values relative to DBL_MIN are inserted into the input matrices*/
    else
    {
        limits_val = (std::numeric_limits<RT>::min)();
        switch(input_range)
        {
            case -1:
                if constexpr (testinghelpers::type_info<T>::is_real) {
                     a[0] = limits_val * compute_power(RT(10), exponent[0]);
                     a[indexA] = limits_val * compute_power(RT(10), exponent[1]);
                } else {
                     a[0] = {limits_val * compute_power(RT(10), exponent[0]), RT(0)};  // Complex with real part, zero imaginary
                     a[indexA] = {limits_val * compute_power(RT(10), exponent[1]), RT(0)};  // Complex with real part, zero imaginary
                }
                break;

            case 0:
                if constexpr (testinghelpers::type_info<T>::is_real) {
                     a[0] = -(limits_val * compute_power(RT(10), exponent[4]));
                     a[indexA] = -(limits_val * compute_power(RT(10), exponent[5]));
                } else {
                     a[0] = {-limits_val * compute_power(RT(10), exponent[4]), RT(0)};  // Complex with real part, zero imaginary
                     a[indexA] = {-limits_val * compute_power(RT(10), exponent[5]), RT(0)};  // Complex with real part, zero imaginary
                }
                break;

            case 1:
                if constexpr (testinghelpers::type_info<T>::is_real) {
                     a[0] = limits_val * compute_power(RT(10), exponent[8]);
                     a[indexA] = limits_val * compute_power(RT(10), exponent[9]);
                } else {
                     a[0] = {limits_val * compute_power(RT(10), exponent[8]), RT(0)};  // Complex with real part, zero imaginary
                     a[indexA] = {limits_val * compute_power(RT(10), exponent[9]), RT(0)};  // Complex with real part, zero imaginary
                }
                break;
        }

    }
}

} //end of namespace testinghelpers

// Add these explicit template instantiations at the end of data_generators.cpp

/***************************************************
 *          Explicit Template Instantiations
****************************************************/

// getfp scalar instantiations
template void testinghelpers::datagenerators::getfp<float, int, int>(int, int, float*);
template void testinghelpers::datagenerators::getfp<double, int, int>(int, int, double*);
template void testinghelpers::datagenerators::getfp<scomplex, int, int>(int, int, scomplex*);
template void testinghelpers::datagenerators::getfp<dcomplex, int, int>(int, int, dcomplex*);

template void testinghelpers::datagenerators::getfp<float, float, float>(float, float, float*);
template void testinghelpers::datagenerators::getfp<double, double, double>(double, double, double*);
template void testinghelpers::datagenerators::getfp<scomplex, float, float>(float, float, scomplex*);
template void testinghelpers::datagenerators::getfp<dcomplex, double, double>(double, double, dcomplex*);

// getfp vector instantiations
template void testinghelpers::datagenerators::getfp<float, int, int>(int, int, gtint_t, gtint_t, float*);
template void testinghelpers::datagenerators::getfp<double, int, int>(int, int, gtint_t, gtint_t, double*);
template void testinghelpers::datagenerators::getfp<scomplex, int, int>(int, int, gtint_t, gtint_t, scomplex*);
template void testinghelpers::datagenerators::getfp<dcomplex, int, int>(int, int, gtint_t, gtint_t, dcomplex*);

template void testinghelpers::datagenerators::getfp<float, float, float>(float, float, gtint_t, gtint_t, float*);
template void testinghelpers::datagenerators::getfp<double, double, double>(double, double, gtint_t, gtint_t, double*);
template void testinghelpers::datagenerators::getfp<scomplex, float, float>(float, float, gtint_t, gtint_t, scomplex*);
template void testinghelpers::datagenerators::getfp<dcomplex, double, double>(double, double, gtint_t, gtint_t, dcomplex*);

// getfp matrix instantiations
template void testinghelpers::datagenerators::getfp<float, int, int>(int, int, char, gtint_t, gtint_t, float*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<double, int, int>(int, int, char, gtint_t, gtint_t, double*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<scomplex, int, int>(int, int, char, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<dcomplex, int, int>(int, int, char, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t);

template void testinghelpers::datagenerators::getfp<float, float, float>(float, float, char, gtint_t, gtint_t, float*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<double, double, double>(double, double, char, gtint_t, gtint_t, double*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<scomplex, float, float>(float, float, char, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<dcomplex, double, double>(double, double, char, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t);

// getfp matrix with transpose instantiations
template void testinghelpers::datagenerators::getfp<float, int, int>(int, int, char, gtint_t, gtint_t, float*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<double, int, int>(int, int, char, gtint_t, gtint_t, double*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<scomplex, int, int>(int, int, char, gtint_t, gtint_t, scomplex*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<dcomplex, int, int>(int, int, char, gtint_t, gtint_t, dcomplex*, char, gtint_t, gtint_t);

template void testinghelpers::datagenerators::getfp<float, float, float>(float, float, char, gtint_t, gtint_t, float*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<double, double, double>(double, double, char, gtint_t, gtint_t, double*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<scomplex, float, float>(float, float, char, gtint_t, gtint_t, scomplex*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getfp<dcomplex, double, double>(double, double, char, gtint_t, gtint_t, dcomplex*, char, gtint_t, gtint_t);

// getint scalar instantiations
template void testinghelpers::datagenerators::getint<float>(int, int, float*);
template void testinghelpers::datagenerators::getint<double>(int, int, double*);
template void testinghelpers::datagenerators::getint<scomplex>(int, int, scomplex*);
template void testinghelpers::datagenerators::getint<dcomplex>(int, int, dcomplex*);

// getint vector instantiations
template void testinghelpers::datagenerators::getint<float>(int, int, gtint_t, gtint_t, float*);
template void testinghelpers::datagenerators::getint<double>(int, int, gtint_t, gtint_t, double*);
template void testinghelpers::datagenerators::getint<scomplex>(int, int, gtint_t, gtint_t, scomplex*);
template void testinghelpers::datagenerators::getint<dcomplex>(int, int, gtint_t, gtint_t, dcomplex*);

// getint matrix instantiations
template void testinghelpers::datagenerators::getint<float>(int, int, char, gtint_t, gtint_t, float*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getint<double>(int, int, char, gtint_t, gtint_t, double*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getint<scomplex>(int, int, char, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getint<dcomplex>(int, int, char, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t);

// getint matrix with transpose instantiations
template void testinghelpers::datagenerators::getint<float>(int, int, char, gtint_t, gtint_t, float*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getint<double>(int, int, char, gtint_t, gtint_t, double*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getint<scomplex>(int, int, char, gtint_t, gtint_t, scomplex*, char, gtint_t, gtint_t);
template void testinghelpers::datagenerators::getint<dcomplex>(int, int, char, gtint_t, gtint_t, dcomplex*, char, gtint_t, gtint_t);

// randomgenerators instantiations
template void testinghelpers::datagenerators::randomgenerators<float, int, int>(int, int, gtint_t, gtint_t, float*, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, int, int>(int, int, gtint_t, gtint_t, double*, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, int, int>(int, int, gtint_t, gtint_t, scomplex*, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, int, int>(int, int, gtint_t, gtint_t, dcomplex*, testinghelpers::datagenerators::ElementType);

template void testinghelpers::datagenerators::randomgenerators<float, float, float>(float, float, gtint_t, gtint_t, float*, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, double, double>(double, double, gtint_t, gtint_t, double*, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, float, float>(float, float, gtint_t, gtint_t, scomplex*, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, double, double>(double, double, gtint_t, gtint_t, dcomplex*, testinghelpers::datagenerators::ElementType);

// randomgenerators matrix instantiations
template void testinghelpers::datagenerators::randomgenerators<float, int, int>(int, int, char, gtint_t, gtint_t, float*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, int, int>(int, int, char, gtint_t, gtint_t, double*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, int, int>(int, int, char, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, int, int>(int, int, char, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

template void testinghelpers::datagenerators::randomgenerators<float, float, float>(float, float, char, gtint_t, gtint_t, float*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, double, double>(double, double, char, gtint_t, gtint_t, double*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, float, float>(float, float, char, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, double, double>(double, double, char, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

// randomgenerators matrix with transpose instantiations
template void testinghelpers::datagenerators::randomgenerators<float, int, int>(int, int, char, gtint_t, gtint_t, float*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, int, int>(int, int, char, gtint_t, gtint_t, double*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, int, int>(int, int, char, gtint_t, gtint_t, scomplex*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, int, int>(int, int, char, gtint_t, gtint_t, dcomplex*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

template void testinghelpers::datagenerators::randomgenerators<float, float, float>(float, float, char, gtint_t, gtint_t, float*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, double, double>(double, double, char, gtint_t, gtint_t, double*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, float, float>(float, float, char, gtint_t, gtint_t, scomplex*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, double, double>(double, double, char, gtint_t, gtint_t, dcomplex*, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

// randomgenerators triangular matrix instantiations
template void testinghelpers::datagenerators::randomgenerators<float, int, int>(int, int, char, char, gtint_t, float*, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, int, int>(int, int, char, char, gtint_t, double*, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, int, int>(int, int, char, char, gtint_t, scomplex*, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, int, int>(int, int, char, char, gtint_t, dcomplex*, gtint_t, testinghelpers::datagenerators::ElementType);

template void testinghelpers::datagenerators::randomgenerators<float, float, float>(float, float, char, char, gtint_t, float*, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<double, double, double>(double, double, char, char, gtint_t, double*, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<scomplex, float, float>(float, float, char, char, gtint_t, scomplex*, gtint_t, testinghelpers::datagenerators::ElementType);
template void testinghelpers::datagenerators::randomgenerators<dcomplex, double, double>(double, double, char, char, gtint_t, dcomplex*, gtint_t, testinghelpers::datagenerators::ElementType);

// get_random_vector instantiations (the ones causing your linking error)
template std::vector<float> testinghelpers::get_random_vector<float, int, int>(int, int, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<double> testinghelpers::get_random_vector<double, int, int>(int, int, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<scomplex> testinghelpers::get_random_vector<scomplex, int, int>(int, int, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<dcomplex> testinghelpers::get_random_vector<dcomplex, int, int>(int, int, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

template std::vector<float> testinghelpers::get_random_vector<float, float, float>(float, float, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<double> testinghelpers::get_random_vector<double, double, double>(double, double, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<scomplex> testinghelpers::get_random_vector<scomplex, float, float>(float, float, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<dcomplex> testinghelpers::get_random_vector<dcomplex, double, double>(double, double, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

// get_random_matrix instantiations
template std::vector<float> testinghelpers::get_random_matrix<float, int, int>(int, int, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<double> testinghelpers::get_random_matrix<double, int, int>(int, int, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<scomplex> testinghelpers::get_random_matrix<scomplex, int, int>(int, int, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<dcomplex> testinghelpers::get_random_matrix<dcomplex, int, int>(int, int, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

template std::vector<float> testinghelpers::get_random_matrix<float, float, float>(float, float, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<double> testinghelpers::get_random_matrix<double, double, double>(double, double, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<scomplex> testinghelpers::get_random_matrix<scomplex, float, float>(float, float, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<dcomplex> testinghelpers::get_random_matrix<dcomplex, double, double>(double, double, char, char, gtint_t, gtint_t, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

// get_random_matrix triangular instantiations
template std::vector<float> testinghelpers::get_random_matrix<float, int, int>(int, int, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<double> testinghelpers::get_random_matrix<double, int, int>(int, int, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<scomplex> testinghelpers::get_random_matrix<scomplex, int, int>(int, int, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<dcomplex> testinghelpers::get_random_matrix<dcomplex, int, int>(int, int, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

template std::vector<float> testinghelpers::get_random_matrix<float, float, float>(float, float, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<double> testinghelpers::get_random_matrix<double, double, double>(double, double, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<scomplex> testinghelpers::get_random_matrix<scomplex, float, float>(float, float, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);
template std::vector<dcomplex> testinghelpers::get_random_matrix<dcomplex, double, double>(double, double, char, char, gtint_t, gtint_t, testinghelpers::datagenerators::ElementType);

// set_vector instantiations
template void testinghelpers::set_vector<float>(gtint_t, gtint_t, float*, float);
template void testinghelpers::set_vector<double>(gtint_t, gtint_t, double*, double);
template void testinghelpers::set_vector<scomplex>(gtint_t, gtint_t, scomplex*, scomplex);
template void testinghelpers::set_vector<dcomplex>(gtint_t, gtint_t, dcomplex*, dcomplex);

// set_matrix instantiations
template void testinghelpers::set_matrix<float>(char, gtint_t, gtint_t, float*, char, gtint_t, float);
template void testinghelpers::set_matrix<double>(char, gtint_t, gtint_t, double*, char, gtint_t, double);
template void testinghelpers::set_matrix<scomplex>(char, gtint_t, gtint_t, scomplex*, char, gtint_t, scomplex);
template void testinghelpers::set_matrix<dcomplex>(char, gtint_t, gtint_t, dcomplex*, char, gtint_t, dcomplex);

// set_matrix triangular instantiations
template void testinghelpers::set_matrix<float>(char, gtint_t, float*, char, gtint_t, float);
template void testinghelpers::set_matrix<double>(char, gtint_t, double*, char, gtint_t, double);
template void testinghelpers::set_matrix<scomplex>(char, gtint_t, scomplex*, char, gtint_t, scomplex);
template void testinghelpers::set_matrix<dcomplex>(char, gtint_t, dcomplex*, char, gtint_t, dcomplex);

// get_vector instantiations
template std::vector<float> testinghelpers::get_vector<float>(gtint_t, gtint_t, float);
template std::vector<double> testinghelpers::get_vector<double>(gtint_t, gtint_t, double);
template std::vector<scomplex> testinghelpers::get_vector<scomplex>(gtint_t, gtint_t, scomplex);
template std::vector<dcomplex> testinghelpers::get_vector<dcomplex>(gtint_t, gtint_t, dcomplex);

// get_matrix instantiations
template std::vector<float> testinghelpers::get_matrix<float>(char, char, gtint_t, gtint_t, gtint_t, float);
template std::vector<double> testinghelpers::get_matrix<double>(char, char, gtint_t, gtint_t, gtint_t, double);
template std::vector<scomplex> testinghelpers::get_matrix<scomplex>(char, char, gtint_t, gtint_t, gtint_t, scomplex);
template std::vector<dcomplex> testinghelpers::get_matrix<dcomplex>(char, char, gtint_t, gtint_t, gtint_t, dcomplex);

// set_ev_mat instantiations
template void testinghelpers::set_ev_mat<float>(char, char, gtint_t, gtint_t, gtint_t, float, float*);
template void testinghelpers::set_ev_mat<double>(char, char, gtint_t, gtint_t, gtint_t, double, double*);
template void testinghelpers::set_ev_mat<scomplex>(char, char, gtint_t, gtint_t, gtint_t, scomplex, scomplex*);
template void testinghelpers::set_ev_mat<dcomplex>(char, char, gtint_t, gtint_t, gtint_t, dcomplex, dcomplex*);

// set_overflow_underflow_mat instantiations
template void testinghelpers::set_overflow_underflow_mat<float>(char, char, gtint_t, gtint_t, gtint_t, float*, gtint_t, gtint_t);
template void testinghelpers::set_overflow_underflow_mat<double>(char, char, gtint_t, gtint_t, gtint_t, double*, gtint_t, gtint_t);
template void testinghelpers::set_overflow_underflow_mat<scomplex>(char, char, gtint_t, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t);
template void testinghelpers::set_overflow_underflow_mat<dcomplex>(char, char, gtint_t, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t);

// RandomDataPool explicit instantiations
template class testinghelpers::datagenerators::RandomDataPool<float>;
template class testinghelpers::datagenerators::RandomDataPool<double>;
template class testinghelpers::datagenerators::RandomDataPool<scomplex>;
template class testinghelpers::datagenerators::RandomDataPool<dcomplex>;
