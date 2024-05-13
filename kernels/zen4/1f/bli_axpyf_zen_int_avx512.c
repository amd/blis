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

#include "immintrin.h"
#include "blis.h"

#if defined __clang__
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define UNROLL_LOOP_FULL() _Pragma("GCC unroll 32")
#else
    #define UNROLL_LOOP_FULL()
#endif

#define GENTFUNC_AXPYF(FUSE_FACTOR) \
    void PASTEMAC2(daxpyf_zen_int, FUSE_FACTOR, _avx512) \
      ( \
       conj_t           conja, \
       conj_t           conjx, \
       dim_t            m, \
       dim_t            b_n, \
       double* restrict alpha, \
       double* restrict a, inc_t inca, inc_t lda, \
       double* restrict x, inc_t incx, \
       double* restrict y0, inc_t incy, \
       cntx_t* restrict cntx \
     ) \
{ \
    const dim_t         fuse_fac       = FUSE_FACTOR; \
    const dim_t         n_elem_per_reg = 8; \
    dim_t               i = 0; \
    \
    __m512d chi[fuse_fac]; \
    __m512d av[1]; \
    __m512d yv[1]; \
    double* as[fuse_fac] __attribute__((aligned(64))); \
    double* y = y0; \
    \
    /* If either dimension is zero, or if alpha is zero, return early.*/ \
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return; \
    \
    /* If b_n is not equal to the fusing factor, then perform the entire
       operation as a loop over axpyv. */ \
    if ( b_n != fuse_fac ) \
    { \
        daxpyv_ker_ft f = bli_daxpyv_zen_int_avx512; \
        \
        for ( i = 0; i < b_n; ++i ) \
        { \
            double* a1   = a + (0  )*inca + (i  )*lda; \
            double* chi1 = x + (i  )*incx; \
            double* y1   = y + (0  )*incy; \
            double  alphavchi1; \
            \
            bli_dcopycjs( conjx, *chi1, alphavchi1 ); \
            bli_dscals( *alpha, alphavchi1 ); \
            \
            f \
            ( \
              conja, \
              m, \
              &alphavchi1, \
              a1, inca, \
              y1, incy, \
              cntx \
            ); \
        } \
        return; \
    } \
    \
    /* At this point, we know that b_n is exactly equal to the fusing factor.*/ \
    UNROLL_LOOP_FULL() \
    for (dim_t ii = 0; ii < fuse_fac; ++ii) \
    { \
        as[ii] = a + (ii * lda); \
        chi[ii] = _mm512_set1_pd( (*alpha) * (*(x + ii * incx)) ); \
    } \
    /* If there are vectorized iterations, perform them with vector
     instructions.*/ \
    if ( inca == 1 && incy == 1 ) \
    { \
        __mmask8 m_mask; \
        m_mask = (1 << 8) - 1; \
        for ( ; i < m; i += 8) \
        { \
            if ( (m - i) < 8) m_mask = (1 << (m - i)) - 1; \
            yv[0] = _mm512_mask_loadu_pd( chi[0], m_mask, y ); \
             \
            UNROLL_LOOP_FULL() \
            for(int ii = 0; ii < fuse_fac; ++ii) \
            { \
                av[0] = _mm512_maskz_loadu_pd( m_mask, as[ii] ); \
                as[ii] += n_elem_per_reg; \
                yv[0] = _mm512_fmadd_pd( av[0], chi[ii], yv[0]); \
            } \
            _mm512_mask_storeu_pd( (double *)(y ), m_mask, yv[0] ); \
            \
            y += n_elem_per_reg; \
        } \
    } \
    else \
    { \
        double       yc = *y; \
        double       chi_s[fuse_fac]; \
         \
        UNROLL_LOOP_FULL() \
        for (dim_t ii = 0; ii < fuse_fac; ++ii) \
        { \
            chi_s[ii] = *(x + ii * incx) * *alpha; \
        } \
        for ( i = 0; (i + 0) < m ; ++i ) \
        { \
            yc = *y; \
            UNROLL_LOOP_FULL() \
            for (dim_t ii = 0 ; ii < fuse_fac; ++ii) \
            { \
                yc += chi_s[ii] * (*as[ii]); \
                as[ii] += inca; \
            } \
            *y = yc; \
            y += incy;  \
        } \
    } \
} \

// Generate two axpyf kernels with fuse_factor = 8 and 32
GENTFUNC_AXPYF(8)
GENTFUNC_AXPYF(32)

#ifdef BLIS_ENABLE_OPENMP
/*
* Multihreaded AVX512 DAXPYF kernel with fuse factor 32
*/
void bli_daxpyf_zen_int32_avx512_mt
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    /*
      Initializing the number of thread to one
      to avoid compiler warnings
    */
    dim_t nt = 1;
    /*
      For the given problem size and architecture, the function
      returns the optimum number of threads with AOCL dynamic enabled
      else it returns the number of threads requested by the user.
    */
    bli_nthreads_l1f
    (
        BLIS_AXPYF_KER,
        BLIS_DOUBLE,
        BLIS_DOUBLE,
        bli_arch_query_id(),
        m,
        &nt
    );

    _Pragma("omp parallel num_threads(nt)")
    {
        const dim_t tid = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        // if num threads requested and num thread available
        // is not same then use single thread
        if( nt_real != nt )
        {
            if( tid == 0 )
            {
                bli_daxpyf_zen_int32_avx512
                (
                    conja,
                    conjx,
                    m,
                    b_n,
                    alpha,
                    a,
                    inca,
                    lda,
                    x,
                    incx,
                    y,
                    incy,
                    cntx
                );
            }
        }
        else
        {
            dim_t job_per_thread, offset;

            // Obtain the job-size and region for compute
            // Calculate y_start and a_start for current thread
            bli_normfv_thread_partition( m, nt_real, &offset, &job_per_thread, 32, incy, tid );
            double* restrict y_start = y + offset;
            bli_normfv_thread_partition( m, nt_real, &offset, &job_per_thread, 32, inca, tid );
            double* restrict a_start = a + offset;

            // call axpyf kernel
            bli_daxpyf_zen_int32_avx512
            (
                conja,
                conjx,
                job_per_thread,
                b_n,
                alpha,
                a_start,
                inca,
                lda,
                x,
                incx,
                y_start,
                incy,
                cntx
            );
        }
    }
}
#endif


void bli_zaxpyf_zen_int_2_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    dim_t fuse_fac = 2;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a sequence of calls to zaxpyf kernels, with fuse-factor
    // 4 and 2 and a single call to zaxpyv, based on the need.
    if ( b_n != fuse_fac )
    {
        dcomplex *a1 = a;
        dcomplex *chi1 = x;
        dcomplex *y1 = y;
        dcomplex alpha_chi1;

        // Vectorization of alpha scaling of X
        __m128d x_vec, alpha_real, alpha_imag, temp[2];
        alpha_real = _mm_loaddup_pd((double *)alpha);
        alpha_imag = _mm_loaddup_pd((double *)alpha + 1);

        x_vec = _mm_loadu_pd((double *)chi1);

        if ( bli_is_conj( conjx ) )
        {
            __m128d conj_set;
            conj_set = _mm_set_pd(-0.0, 0.0);

            x_vec = _mm_xor_pd(conj_set, x_vec);
        }

        temp[0] = _mm_mul_pd(x_vec, alpha_real);
        temp[1] = _mm_mul_pd(x_vec, alpha_imag);

        temp[1] = _mm_permute_pd(temp[1], 0b01);

        temp[0] = _mm_addsub_pd(temp[0], temp[1]);

        _mm_storeu_pd((double *)&alpha_chi1, temp[0]);

        bli_zaxpyv_zen_int_avx512
        (
            conja,
            m,
            &alpha_chi1,
            a1, inca,
            y1, incy,
            cntx
        );

        return;
    }

    // Declaring and initializing the iterator and pointers
    dim_t i = 0;

    double *a_ptr[2];
    double *y0 = (double *)y;

    a_ptr[0] = (double *)a;
    a_ptr[1] = (double *)(a + 1 * lda);

    /* Alpha scaling of X can be vectorized
       irrespective of the incx  and should
       be avoided when alpha is 1 */
    __m128d x_vec[2];

    x_vec[0] = _mm_loadu_pd((double *)(x + 0 * incx));
    x_vec[1] = _mm_loadu_pd((double *)(x + 1 * incx));

    if ( bli_is_conj( conjx ) )
    {
        __m128d conj_set;
        conj_set = _mm_set_pd(-0.0, 0.0);

        // The sequence of xor operations flip the sign bit
        // of imaginary components in X vector
        x_vec[0] = _mm_xor_pd(conj_set, x_vec[0]);
        x_vec[1] = _mm_xor_pd(conj_set, x_vec[1]);
    }

    // Special case handling when alpha == -1 + 0i
    if( alpha->real == -1.0 && alpha->imag == 0.0 )
    {
        __m128d zero_reg = _mm_setzero_pd();

        x_vec[0] = _mm_sub_pd(zero_reg, x_vec[0]);
        x_vec[1] = _mm_sub_pd(zero_reg, x_vec[1]);
    }
    // General case of scaling with alpha
    else if (!(bli_zeq1(*alpha)))
    {
        __m128d alpha_real, alpha_imag, temp[2];
        alpha_real = _mm_loaddup_pd((double *)alpha);
        alpha_imag = _mm_loaddup_pd(((double *)alpha) + 1);

        // Scaling with imaginary part of alpha
        temp[0] = _mm_mul_pd(x_vec[0], alpha_imag);
        temp[1] = _mm_mul_pd(x_vec[1], alpha_imag);

        // Scaling with real part of alpha
        x_vec[0] = _mm_mul_pd(x_vec[0], alpha_real);
        x_vec[1] = _mm_mul_pd(x_vec[1], alpha_real);

        // Permuting the registers to get the following pattern
        // t[0] : xI0*alphaI
        //        xR0*alphaI, and so on
        temp[0] = _mm_permute_pd(temp[0], 0x01);
        temp[1] = _mm_permute_pd(temp[1], 0x01);

        // Addsub to complete the complex arithmetic as such:
        // x_vec[0] : xR0*alphaR - xI0*alphaI
        //            xI0*alphaR + xR0*alphaI, and so on
        x_vec[0] = _mm_addsub_pd(x_vec[0], temp[0]);
        x_vec[1] = _mm_addsub_pd(x_vec[1], temp[1]);
    }

    if ( (inca == 1) && (incy == 1) )
    {
        // Temporary registers to store permuted alpha*X values
        __m128d temp[2];

        temp[0] = _mm_shuffle_pd(x_vec[0], x_vec[0], 0x01);
        temp[1] = _mm_shuffle_pd(x_vec[1], x_vec[1], 0x01);

        // Declaring 4 registers, for re-use over the loops
        // alpha_x_real[0] = xR0*alphaR  xR0*alphaR ...
        // alpah_x_imag[0] = xI0*alphaI  xI0*alphaI ...
        __m512d alpha_x_real[2], alpha_x_imag[2];

        alpha_x_real[0] = _mm512_broadcastsd_pd(x_vec[0]);
        alpha_x_real[1] = _mm512_broadcastsd_pd(x_vec[1]);

        alpha_x_imag[0] = _mm512_broadcastsd_pd(temp[0]);
        alpha_x_imag[1] = _mm512_broadcastsd_pd(temp[1]);

        // Registers to load A, accumulate real and imag scaling separately
        __m512d a_vec[2];
        __m512d real_acc, imag_acc, y_vec;
        __m512d zero_reg = _mm512_setzero_pd();

        // Execute the loops is m >= 4(AVX-512 unmasked code-section)
        if( m >= 4 )
        {
            if ( bli_is_noconj(conja) )
            {
                for (; (i + 7) < m; i += 8)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    // Load next 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    // Load next 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0 + 8);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0 + 8, y_vec);

                    y0 += 16;
                    a_ptr[0] += 16;
                    a_ptr[1] += 16;
                }

                for (; (i + 3) < m; i += 4)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    y0 += 8;
                    a_ptr[0] += 8;
                    a_ptr[1] += 8;
                }
            }
            else
            {
                for (; (i + 7) < m; i += 8)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    // Load next 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    // Load next 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0 + 8);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0 + 8, y_vec);

                    y0 += 16;
                    a_ptr[0] += 16;
                    a_ptr[1] += 16;
                }

                for (; (i + 3) < m; i += 4)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    y0 += 8;
                    a_ptr[0] += 8;
                    a_ptr[1] += 8;
                }
            }
        }
        if( i < m )
        {
            __mmask8 m_mask = (1 << 2*(m - i)) - 1;
            if( bli_is_noconj(conja) )
            {
                // Load remaining elements from first 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[0]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[1]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                // Load remaining elements of Y vector
                y_vec = _mm512_maskz_loadu_pd(m_mask, y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                real_acc = _mm512_add_pd(real_acc, imag_acc);

                y_vec = _mm512_add_pd(y_vec, real_acc);

                // Store onto Y vector
                _mm512_mask_storeu_pd(y0, m_mask, y_vec);
            }
            else
            {
                // Load remaining elements from first 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[0]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[1]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                // Load remaining elements of Y vector
                y_vec = _mm512_maskz_loadu_pd(m_mask, y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                real_acc = _mm512_add_pd(real_acc, imag_acc);

                y_vec = _mm512_add_pd(y_vec, real_acc);

                // Store onto Y vector
                _mm512_mask_storeu_pd(y0, m_mask, y_vec);
            }
        }
    }
    else
    {
        // Perform the computation with 128-bit registers,
        // since dcomplex is 128 bits in size
        __m128d a_vec[2], y_vec, real_acc, imag_acc, temp[2];

        // Unpacking and storing real and imaginary components
        // of alpha*X stored in x_vec[0...7]
        temp[0] = _mm_unpackhi_pd(x_vec[0], x_vec[0]);
        temp[1] = _mm_unpackhi_pd(x_vec[1], x_vec[1]);

        x_vec[0] = _mm_unpacklo_pd(x_vec[0], x_vec[0]);
        x_vec[1] = _mm_unpacklo_pd(x_vec[1], x_vec[1]);

        if ( bli_is_noconj(conja) )
        {
            for (; i < m; i++)
            {
                // Load elements from first 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[0]);
                a_vec[1] = _mm_loadu_pd(a_ptr[1]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_mul_pd(a_vec[0], x_vec[0]);
                imag_acc = _mm_mul_pd(a_vec[0], temp[0]);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[1], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[1], imag_acc);

                // Load Y vector
                y_vec = _mm_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm_permute_pd(imag_acc, 0b01);
                real_acc = _mm_addsub_pd(real_acc, imag_acc);

                y_vec = _mm_add_pd(y_vec, real_acc);

                // Store Y vector
                _mm_storeu_pd(y0, y_vec);

                y0 += 2 * incy;
                a_ptr[0] += 2 * inca;
                a_ptr[1] += 2 * inca;
            }
        }
        else
        {
            for (; i < m; i++)
            {
                // Load elements from first 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[0]);
                a_vec[1] = _mm_loadu_pd(a_ptr[1]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_mul_pd(a_vec[0], x_vec[0]);
                imag_acc = _mm_mul_pd(a_vec[0], temp[0]);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[1], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[1], imag_acc);

                // Load Y vector
                y_vec = _mm_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                real_acc = _mm_permute_pd(real_acc, 0b01);
                real_acc = _mm_addsub_pd(imag_acc, real_acc);
                real_acc = _mm_permute_pd(real_acc, 0b01);

                y_vec = _mm_add_pd(y_vec, real_acc);

                // Store Y vector
                _mm_storeu_pd(y0, y_vec);

                y0 += 2 * incy;
                a_ptr[0] += 2 * inca;
                a_ptr[1] += 2 * inca;
            }
        }
    }
}

void bli_zaxpyf_zen_int_4_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    dim_t fuse_fac = 4;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a sequence of calls to zaxpyf kernels, with fuse-factor
    // 2 and a single call to zaxpyv, based on the need.
    if ( b_n != fuse_fac )
    {
        dcomplex *a1 = a;
        dcomplex *chi1 = x;
        dcomplex *y1 = y;
        dcomplex alpha_chi1;

        // Buggy, try to mimic 8 kernel
        if( b_n >= 2 )
        {
            bli_zaxpyf_zen_int_2_avx512
            (
              conja,
              conjx,
              m,
              (dim_t)2,
              alpha,
              a1, inca, lda,
              chi1, incx,
              y1, incy,
              cntx
            );

            a1 += 2*lda;
            chi1 += 2*incx;
            b_n -= 2;
        }

        if( b_n == 1 )
        {
            // Vectorization of alpha scaling of X
            __m128d x_vec, alpha_real, alpha_imag, temp[2];
            alpha_real = _mm_loaddup_pd((double *)alpha);
            alpha_imag = _mm_loaddup_pd((double *)alpha + 1);

            x_vec = _mm_loadu_pd((double *)chi1);

            if ( bli_is_conj( conjx ) )
            {
                __m128d conj_set;
                conj_set = _mm_set_pd(-0.0, 0.0);

                x_vec = _mm_xor_pd(conj_set, x_vec);
            }

            temp[0] = _mm_mul_pd(x_vec, alpha_real);
            temp[1] = _mm_mul_pd(x_vec, alpha_imag);

            temp[1] = _mm_permute_pd(temp[1], 0b01);

            temp[0] = _mm_addsub_pd(temp[0], temp[1]);

            _mm_storeu_pd((double *)&alpha_chi1, temp[0]);

            bli_zaxpyv_zen_int_avx512
            (
                conja,
                m,
                &alpha_chi1,
                a1, inca,
                y1, incy,
                cntx
            );
        }

        return;
    }

    // Declaring and initializing the iterator and pointers
    dim_t i = 0;

    double *a_ptr[4];
    double *y0 = (double *)y;

    a_ptr[0] = (double *)a;
    a_ptr[1] = (double *)(a + 1 * lda);
    a_ptr[2] = (double *)(a + 2 * lda);
    a_ptr[3] = (double *)(a + 3 * lda);

    /* Alpha scaling of X can be vectorized
       irrespective of the incx  and should
       be avoided when alpha is 1 */
    __m128d x_vec[4];

    x_vec[0] = _mm_loadu_pd((double *)(x + 0 * incx));
    x_vec[1] = _mm_loadu_pd((double *)(x + 1 * incx));
    x_vec[2] = _mm_loadu_pd((double *)(x + 2 * incx));
    x_vec[3] = _mm_loadu_pd((double *)(x + 3 * incx));

    if ( bli_is_conj( conjx ) )
    {
        __m128d conj_set;
        conj_set = _mm_set_pd(-0.0, 0.0);

        // The sequence of xor operations flip the sign bit
        // of imaginary components in X vector
        x_vec[0] = _mm_xor_pd(conj_set, x_vec[0]);
        x_vec[1] = _mm_xor_pd(conj_set, x_vec[1]);
        x_vec[2] = _mm_xor_pd(conj_set, x_vec[2]);
        x_vec[3] = _mm_xor_pd(conj_set, x_vec[3]);
    }

    // Special case handling when alpha == -1 + 0i
    if( alpha->real == -1.0 && alpha->imag == 0.0 )
    {
        __m128d zero_reg = _mm_setzero_pd();

        x_vec[0] = _mm_sub_pd(zero_reg, x_vec[0]);
        x_vec[1] = _mm_sub_pd(zero_reg, x_vec[1]);
        x_vec[2] = _mm_sub_pd(zero_reg, x_vec[2]);
        x_vec[3] = _mm_sub_pd(zero_reg, x_vec[3]);
    }
    // General case of scaling with alpha
    else if (!(bli_zeq1(*alpha)))
    {
        __m128d alpha_real, alpha_imag, temp[4];
        alpha_real = _mm_loaddup_pd((double *)alpha);
        alpha_imag = _mm_loaddup_pd(((double *)alpha) + 1);

        // Scaling with imaginary part of alpha
        temp[0] = _mm_mul_pd(x_vec[0], alpha_imag);
        temp[1] = _mm_mul_pd(x_vec[1], alpha_imag);
        temp[2] = _mm_mul_pd(x_vec[2], alpha_imag);
        temp[3] = _mm_mul_pd(x_vec[3], alpha_imag);

        // Scaling with real part of alpha
        x_vec[0] = _mm_mul_pd(x_vec[0], alpha_real);
        x_vec[1] = _mm_mul_pd(x_vec[1], alpha_real);
        x_vec[2] = _mm_mul_pd(x_vec[2], alpha_real);
        x_vec[3] = _mm_mul_pd(x_vec[3], alpha_real);

        // Permuting the registers to get the following pattern
        // t[0] : xI0*alphaI
        //        xR0*alphaI, and so on
        temp[0] = _mm_permute_pd(temp[0], 0x01);
        temp[1] = _mm_permute_pd(temp[1], 0x01);
        temp[2] = _mm_permute_pd(temp[2], 0x01);
        temp[3] = _mm_permute_pd(temp[3], 0x01);

        // Addsub to complete the complex arithmetic as such:
        // x_vec[0] : xR0*alphaR - xI0*alphaI
        //            xI0*alphaR + xR0*alphaI, and so on
        x_vec[0] = _mm_addsub_pd(x_vec[0], temp[0]);
        x_vec[1] = _mm_addsub_pd(x_vec[1], temp[1]);
        x_vec[2] = _mm_addsub_pd(x_vec[2], temp[2]);
        x_vec[3] = _mm_addsub_pd(x_vec[3], temp[3]);
    }

    if ( (inca == 1) && (incy == 1) )
    {
        // Temporary registers to store permuted alpha*X values
        __m128d temp[4];

        temp[0] = _mm_shuffle_pd(x_vec[0], x_vec[0], 0x01);
        temp[1] = _mm_shuffle_pd(x_vec[1], x_vec[1], 0x01);
        temp[2] = _mm_shuffle_pd(x_vec[2], x_vec[2], 0x01);
        temp[3] = _mm_shuffle_pd(x_vec[3], x_vec[3], 0x01);

        // Declaring 8 registers, for re-use over the loops
        // alpha_x_real[0] = xR0*alphaR  xR0*alphaR ...
        // alpah_x_imag[0] = xI0*alphaI  xI0*alphaI ...
        __m512d alpha_x_real[4], alpha_x_imag[4];

        alpha_x_real[0] = _mm512_broadcastsd_pd(x_vec[0]);
        alpha_x_real[1] = _mm512_broadcastsd_pd(x_vec[1]);
        alpha_x_real[2] = _mm512_broadcastsd_pd(x_vec[2]);
        alpha_x_real[3] = _mm512_broadcastsd_pd(x_vec[3]);

        alpha_x_imag[0] = _mm512_broadcastsd_pd(temp[0]);
        alpha_x_imag[1] = _mm512_broadcastsd_pd(temp[1]);
        alpha_x_imag[2] = _mm512_broadcastsd_pd(temp[2]);
        alpha_x_imag[3] = _mm512_broadcastsd_pd(temp[3]);

        // Registers to load A, accumulate real and imag scaling separately
        __m512d a_vec[4];
        __m512d real_acc, imag_acc, y_vec;
        __m512d zero_reg = _mm512_setzero_pd();

        // Execute the loops is m >= 4(AVX-512 unmasked code-section)
        if( m >= 4 )
        {
            if ( bli_is_noconj(conja) )
            {
                for (; (i + 7) < m; i += 8)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    // Load next 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1] + 8);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2] + 8);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load next 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0 + 8);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0 + 8, y_vec);

                    y0 += 16;
                    a_ptr[0] += 16;
                    a_ptr[1] += 16;
                    a_ptr[2] += 16;
                    a_ptr[3] += 16;
                }

                for (; (i + 3) < m; i += 4)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    y0 += 8;
                    a_ptr[0] += 8;
                    a_ptr[1] += 8;
                    a_ptr[2] += 8;
                    a_ptr[3] += 8;
                }
            }
            else
            {
                for (; (i + 7) < m; i += 8)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    // Load next 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1] + 8);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2] + 8);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load next 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0 + 8);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0 + 8, y_vec);

                    y0 += 16;
                    a_ptr[0] += 16;
                    a_ptr[1] += 16;
                    a_ptr[2] += 16;
                    a_ptr[3] += 16;
                }

                for (; (i + 3) < m; i += 4)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    y0 += 8;
                    a_ptr[0] += 8;
                    a_ptr[1] += 8;
                    a_ptr[2] += 8;
                    a_ptr[3] += 8;
                }
            }
        }
        if( i < m )
        {
            __mmask8 m_mask = (1 << 2*(m - i)) - 1;
            if( bli_is_noconj(conja) )
            {
                // Load remaining elements from first 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[0]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[1]);
                a_vec[2] = _mm512_maskz_loadu_pd(m_mask, a_ptr[2]);
                a_vec[3] = _mm512_maskz_loadu_pd(m_mask, a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                // Load remaining elements of Y vector
                y_vec = _mm512_maskz_loadu_pd(m_mask, y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                real_acc = _mm512_add_pd(real_acc, imag_acc);

                y_vec = _mm512_add_pd(y_vec, real_acc);

                // Store onto Y vector
                _mm512_mask_storeu_pd(y0, m_mask, y_vec);
            }
            else
            {
                // Load remaining elements from first 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[0]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[1]);
                a_vec[2] = _mm512_maskz_loadu_pd(m_mask, a_ptr[2]);
                a_vec[3] = _mm512_maskz_loadu_pd(m_mask, a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                // Load remaining elements of Y vector
                y_vec = _mm512_maskz_loadu_pd(m_mask, y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                real_acc = _mm512_add_pd(real_acc, imag_acc);

                y_vec = _mm512_add_pd(y_vec, real_acc);

                // Store onto Y vector
                _mm512_mask_storeu_pd(y0, m_mask, y_vec);
            }
        }
    }
    else
    {
        // Perform the computation with 128-bit registers,
        // since dcomplex is 128 bits in size
        __m128d a_vec[4], y_vec, real_acc, imag_acc, temp[4];

        // Unpacking and storing real and imaginary components
        // of alpha*X stored in x_vec[0...7]
        temp[0] = _mm_unpackhi_pd(x_vec[0], x_vec[0]);
        temp[1] = _mm_unpackhi_pd(x_vec[1], x_vec[1]);
        temp[2] = _mm_unpackhi_pd(x_vec[2], x_vec[2]);
        temp[3] = _mm_unpackhi_pd(x_vec[3], x_vec[3]);

        x_vec[0] = _mm_unpacklo_pd(x_vec[0], x_vec[0]);
        x_vec[1] = _mm_unpacklo_pd(x_vec[1], x_vec[1]);
        x_vec[2] = _mm_unpacklo_pd(x_vec[2], x_vec[2]);
        x_vec[3] = _mm_unpacklo_pd(x_vec[3], x_vec[3]);

        if ( bli_is_noconj(conja) )
        {
            for (; i < m; i++)
            {
                // Load elements from first 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[0]);
                a_vec[1] = _mm_loadu_pd(a_ptr[1]);
                a_vec[2] = _mm_loadu_pd(a_ptr[2]);
                a_vec[3] = _mm_loadu_pd(a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_mul_pd(a_vec[0], x_vec[0]);
                imag_acc = _mm_mul_pd(a_vec[0], temp[0]);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[1], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[1], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[2], x_vec[2], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[2], temp[2], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[3], x_vec[3], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[3], temp[3], imag_acc);

                // Load Y vector
                y_vec = _mm_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm_permute_pd(imag_acc, 0b01);
                real_acc = _mm_addsub_pd(real_acc, imag_acc);

                y_vec = _mm_add_pd(y_vec, real_acc);

                // Store Y vector
                _mm_storeu_pd(y0, y_vec);

                y0 += 2 * incy;
                a_ptr[0] += 2 * inca;
                a_ptr[1] += 2 * inca;
                a_ptr[2] += 2 * inca;
                a_ptr[3] += 2 * inca;
            }
        }
        else
        {
            for (; i < m; i++)
            {
                // Load elements from first 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[0]);
                a_vec[1] = _mm_loadu_pd(a_ptr[1]);
                a_vec[2] = _mm_loadu_pd(a_ptr[2]);
                a_vec[3] = _mm_loadu_pd(a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_mul_pd(a_vec[0], x_vec[0]);
                imag_acc = _mm_mul_pd(a_vec[0], temp[0]);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[1], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[1], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[2], x_vec[2], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[2], temp[2], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[3], x_vec[3], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[3], temp[3], imag_acc);

                // Load Y vector
                y_vec = _mm_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                real_acc = _mm_permute_pd(real_acc, 0b01);
                real_acc = _mm_addsub_pd(imag_acc, real_acc);
                real_acc = _mm_permute_pd(real_acc, 0b01);

                y_vec = _mm_add_pd(y_vec, real_acc);

                // Store Y vector
                _mm_storeu_pd(y0, y_vec);

                y0 += 2 * incy;
                a_ptr[0] += 2 * inca;
                a_ptr[1] += 2 * inca;
                a_ptr[2] += 2 * inca;
                a_ptr[3] += 2 * inca;
            }
        }
    }
}

void bli_zaxpyf_zen_int_8_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    dim_t fuse_fac = 8;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a sequence of calls to zaxpyf kernels, with fuse-factor
    // 4 and 2 and a single call to zaxpyv, based on the need.
    if ( b_n != fuse_fac )
    {
        dcomplex *a1 = a;
        dcomplex *chi1 = x;
        dcomplex *y1 = y;
        dcomplex alpha_chi1;

        if( b_n >= 4 )
        {
            bli_zaxpyf_zen_int_4_avx512
            (
              conja,
              conjx,
              m,
              (dim_t)4,
              alpha,
              a1, inca, lda,
              chi1, incx,
              y1, incy,
              cntx
            );

            a1 += 4*lda;
            chi1 += 4*incx;
            b_n -= 4;
        }

        // Buggy, try to mimic 8 kernel
        if( b_n >= 2 )
        {
            bli_zaxpyf_zen_int_2_avx512
            (
              conja,
              conjx,
              m,
              (dim_t)2,
              alpha,
              a1, inca, lda,
              chi1, incx,
              y1, incy,
              cntx
            );

            a1 += 2*lda;
            chi1 += 2*incx;
            b_n -= 2;
        }

        if( b_n == 1 )
        {
            // Vectorization of alpha scaling of X
            __m128d x_vec, alpha_real, alpha_imag, temp[2];
            alpha_real = _mm_loaddup_pd((double *)alpha);
            alpha_imag = _mm_loaddup_pd((double *)alpha + 1);

            x_vec = _mm_loadu_pd((double *)chi1);

            if ( bli_is_conj( conjx ) )
            {
                __m128d conj_set;
                conj_set = _mm_set_pd(-0.0, 0.0);

                x_vec = _mm_xor_pd(conj_set, x_vec);
            }

            temp[0] = _mm_mul_pd(x_vec, alpha_real);
            temp[1] = _mm_mul_pd(x_vec, alpha_imag);

            temp[1] = _mm_permute_pd(temp[1], 0b01);

            temp[0] = _mm_addsub_pd(temp[0], temp[1]);

            _mm_storeu_pd((double *)&alpha_chi1, temp[0]);

            bli_zaxpyv_zen_int_avx512
            (
                conja,
                m,
                &alpha_chi1,
                a1, inca,
                y1, incy,
                cntx
            );
        }

        return;
    }

    // Declaring and initializing the iterator and pointers
    dim_t i = 0;

    double *a_ptr[8];
    double *y0 = (double *)y;

    a_ptr[0] = (double *)a;
    a_ptr[1] = (double *)(a + 1 * lda);
    a_ptr[2] = (double *)(a + 2 * lda);
    a_ptr[3] = (double *)(a + 3 * lda);

    a_ptr[4] = (double *)(a + 4 * lda);
    a_ptr[5] = (double *)(a + 5 * lda);
    a_ptr[6] = (double *)(a + 6 * lda);
    a_ptr[7] = (double *)(a + 7 * lda);

    /* Alpha scaling of X can be vectorized
       irrespective of the incx  and should
       be avoided when alpha is 1 */
    __m128d x_vec[8];

    x_vec[0] = _mm_loadu_pd((double *)(x + 0 * incx));
    x_vec[1] = _mm_loadu_pd((double *)(x + 1 * incx));
    x_vec[2] = _mm_loadu_pd((double *)(x + 2 * incx));
    x_vec[3] = _mm_loadu_pd((double *)(x + 3 * incx));

    x_vec[4] = _mm_loadu_pd((double *)(x + 4 * incx));
    x_vec[5] = _mm_loadu_pd((double *)(x + 5 * incx));
    x_vec[6] = _mm_loadu_pd((double *)(x + 6 * incx));
    x_vec[7] = _mm_loadu_pd((double *)(x + 7 * incx));

    if ( bli_is_conj( conjx ) )
    {
        __m128d conj_set;
        conj_set = _mm_set_pd(-0.0, 0.0);

        // The sequence of xor operations flip the sign bit
        // of imaginary components in X vector
        x_vec[0] = _mm_xor_pd(conj_set, x_vec[0]);
        x_vec[1] = _mm_xor_pd(conj_set, x_vec[1]);
        x_vec[2] = _mm_xor_pd(conj_set, x_vec[2]);
        x_vec[3] = _mm_xor_pd(conj_set, x_vec[3]);

        x_vec[4] = _mm_xor_pd(conj_set, x_vec[4]);
        x_vec[5] = _mm_xor_pd(conj_set, x_vec[5]);
        x_vec[6] = _mm_xor_pd(conj_set, x_vec[6]);
        x_vec[7] = _mm_xor_pd(conj_set, x_vec[7]);

    }

    // Special case handling when alpha == -1 + 0i
    if( alpha->real == -1.0 && alpha->imag == 0.0 )
    {
        __m128d zero_reg = _mm_setzero_pd();

        x_vec[0] = _mm_sub_pd(zero_reg, x_vec[0]);
        x_vec[1] = _mm_sub_pd(zero_reg, x_vec[1]);
        x_vec[2] = _mm_sub_pd(zero_reg, x_vec[2]);
        x_vec[3] = _mm_sub_pd(zero_reg, x_vec[3]);

        x_vec[4] = _mm_sub_pd(zero_reg, x_vec[4]);
        x_vec[5] = _mm_sub_pd(zero_reg, x_vec[5]);
        x_vec[6] = _mm_sub_pd(zero_reg, x_vec[6]);
        x_vec[7] = _mm_sub_pd(zero_reg, x_vec[7]);
    }
    // General case of scaling with alpha
    else if (!(bli_zeq1(*alpha)))
    {
        __m128d alpha_real, alpha_imag, temp[4];
        alpha_real = _mm_loaddup_pd((double *)alpha);
        alpha_imag = _mm_loaddup_pd(((double *)alpha) + 1);

        // Scaling with imaginary part of alpha
        temp[0] = _mm_mul_pd(x_vec[0], alpha_imag);
        temp[1] = _mm_mul_pd(x_vec[1], alpha_imag);
        temp[2] = _mm_mul_pd(x_vec[2], alpha_imag);
        temp[3] = _mm_mul_pd(x_vec[3], alpha_imag);

        // Scaling with real part of alpha
        x_vec[0] = _mm_mul_pd(x_vec[0], alpha_real);
        x_vec[1] = _mm_mul_pd(x_vec[1], alpha_real);
        x_vec[2] = _mm_mul_pd(x_vec[2], alpha_real);
        x_vec[3] = _mm_mul_pd(x_vec[3], alpha_real);

        // Permuting the registers to get the following pattern
        // t[0] : xI0*alphaI
        //        xR0*alphaI, and so on
        temp[0] = _mm_permute_pd(temp[0], 0x01);
        temp[1] = _mm_permute_pd(temp[1], 0x01);
        temp[2] = _mm_permute_pd(temp[2], 0x01);
        temp[3] = _mm_permute_pd(temp[3], 0x01);

        // Addsub to complete the complex arithmetic as such:
        // x_vec[0] : xR0*alphaR - xI0*alphaI
        //            xI0*alphaR + xR0*alphaI, and so on
        x_vec[0] = _mm_addsub_pd(x_vec[0], temp[0]);
        x_vec[1] = _mm_addsub_pd(x_vec[1], temp[1]);
        x_vec[2] = _mm_addsub_pd(x_vec[2], temp[2]);
        x_vec[3] = _mm_addsub_pd(x_vec[3], temp[3]);

        // Scaling with imaginary part of alpha
        temp[0] = _mm_mul_pd(x_vec[4], alpha_imag);
        temp[1] = _mm_mul_pd(x_vec[5], alpha_imag);
        temp[2] = _mm_mul_pd(x_vec[6], alpha_imag);
        temp[3] = _mm_mul_pd(x_vec[7], alpha_imag);

        // Scaling with real part of alpha
        x_vec[4] = _mm_mul_pd(x_vec[4], alpha_real);
        x_vec[5] = _mm_mul_pd(x_vec[5], alpha_real);
        x_vec[6] = _mm_mul_pd(x_vec[6], alpha_real);
        x_vec[7] = _mm_mul_pd(x_vec[7], alpha_real);

        // Permuting the registers to get the following pattern
        // t[0] : xI0*alphaI  xR0*alphaI
        temp[0] = _mm_permute_pd(temp[0], 0x01);
        temp[1] = _mm_permute_pd(temp[1], 0x01);
        temp[2] = _mm_permute_pd(temp[2], 0x01);
        temp[3] = _mm_permute_pd(temp[3], 0x01);

        // Addsub to complete the complex arithmetic as such:
        // x_vec[0] : ( xR0*alphaR - xI0*alphaI )  ( xI0*alphaR + xR0*alphaI )
        x_vec[4] = _mm_addsub_pd(x_vec[4], temp[0]);
        x_vec[5] = _mm_addsub_pd(x_vec[5], temp[1]);
        x_vec[6] = _mm_addsub_pd(x_vec[6], temp[2]);
        x_vec[7] = _mm_addsub_pd(x_vec[7], temp[3]);
    }

    if ( (inca == 1) && (incy == 1) )
    {
        // Temporary registers to store permuted alpha*X values
        __m128d temp[8];

        temp[0] = _mm_shuffle_pd(x_vec[0], x_vec[0], 0x01);
        temp[1] = _mm_shuffle_pd(x_vec[1], x_vec[1], 0x01);
        temp[2] = _mm_shuffle_pd(x_vec[2], x_vec[2], 0x01);
        temp[3] = _mm_shuffle_pd(x_vec[3], x_vec[3], 0x01);

        temp[4] = _mm_shuffle_pd(x_vec[4], x_vec[4], 0x01);
        temp[5] = _mm_shuffle_pd(x_vec[5], x_vec[5], 0x01);
        temp[6] = _mm_shuffle_pd(x_vec[6], x_vec[6], 0x01);
        temp[7] = _mm_shuffle_pd(x_vec[7], x_vec[7], 0x01);

        // Declaring 16 registers, for re-use over the loops
        // alpha_x_real[0] = xR0*alphaR  xR0*alphaR ...
        // alpah_x_imag[0] = xI0*alphaI  xI0*alphaI ...
        __m512d alpha_x_real[8], alpha_x_imag[8];

        alpha_x_real[0] = _mm512_broadcastsd_pd(x_vec[0]);
        alpha_x_real[1] = _mm512_broadcastsd_pd(x_vec[1]);
        alpha_x_real[2] = _mm512_broadcastsd_pd(x_vec[2]);
        alpha_x_real[3] = _mm512_broadcastsd_pd(x_vec[3]);
        alpha_x_real[4] = _mm512_broadcastsd_pd(x_vec[4]);
        alpha_x_real[5] = _mm512_broadcastsd_pd(x_vec[5]);
        alpha_x_real[6] = _mm512_broadcastsd_pd(x_vec[6]);
        alpha_x_real[7] = _mm512_broadcastsd_pd(x_vec[7]);

        alpha_x_imag[0] = _mm512_broadcastsd_pd(temp[0]);
        alpha_x_imag[1] = _mm512_broadcastsd_pd(temp[1]);
        alpha_x_imag[2] = _mm512_broadcastsd_pd(temp[2]);
        alpha_x_imag[3] = _mm512_broadcastsd_pd(temp[3]);
        alpha_x_imag[4] = _mm512_broadcastsd_pd(temp[4]);
        alpha_x_imag[5] = _mm512_broadcastsd_pd(temp[5]);
        alpha_x_imag[6] = _mm512_broadcastsd_pd(temp[6]);
        alpha_x_imag[7] = _mm512_broadcastsd_pd(temp[7]);

        // Registers to load A, accumulate real and imag scaling separately
        __m512d a_vec[4];
        __m512d real_acc, imag_acc, y_vec;
        __m512d zero_reg = _mm512_setzero_pd();

        // Execute the loops is m >= 4(AVX-512 unmasked code-section)
        if( m >= 4 )
        {
            if ( bli_is_noconj(conja) )
            {
                for (; (i + 7) < m; i += 8)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements from next 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[4]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[5]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[6]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[7]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    // Load next 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1] + 8);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2] + 8);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load next 4 elements from next 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[4] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[5] + 8);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[6] + 8);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[7] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                    // Load next 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0 + 8);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0 + 8, y_vec);

                    y0 += 16;
                    a_ptr[0] += 16;
                    a_ptr[1] += 16;
                    a_ptr[2] += 16;
                    a_ptr[3] += 16;
                    a_ptr[4] += 16;
                    a_ptr[5] += 16;
                    a_ptr[6] += 16;
                    a_ptr[7] += 16;
                }

                for (; (i + 3) < m; i += 4)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements from next 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[4]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[5]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[6]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[7]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    y0 += 8;
                    a_ptr[0] += 8;
                    a_ptr[1] += 8;
                    a_ptr[2] += 8;
                    a_ptr[3] += 8;
                    a_ptr[4] += 8;
                    a_ptr[5] += 8;
                    a_ptr[6] += 8;
                    a_ptr[7] += 8;
                }
            }
            else
            {
                for (; (i + 7) < m; i += 8)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements from next 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[4]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[5]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[6]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[7]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    // Load next 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1] + 8);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2] + 8);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load next 4 elements from next 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[4] + 8);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[5] + 8);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[6] + 8);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[7] + 8);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                    // Load next 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0 + 8);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0 + 8, y_vec);

                    y0 += 16;
                    a_ptr[0] += 16;
                    a_ptr[1] += 16;
                    a_ptr[2] += 16;
                    a_ptr[3] += 16;
                    a_ptr[4] += 16;
                    a_ptr[5] += 16;
                    a_ptr[6] += 16;
                    a_ptr[7] += 16;
                }

                for (; (i + 3) < m; i += 4)
                {
                    // Load first 4 elements from first 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[0]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[1]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[2]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[3]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                    imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                    // Load first 4 elements from next 4 columns of A
                    a_vec[0] = _mm512_loadu_pd(a_ptr[4]);
                    a_vec[1] = _mm512_loadu_pd(a_ptr[5]);
                    a_vec[2] = _mm512_loadu_pd(a_ptr[6]);
                    a_vec[3] = _mm512_loadu_pd(a_ptr[7]);

                    // Multiply the loaded columns of A by alpha*X(real and imag)
                    real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                    real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                    imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                    // Load first 4 elements of Y vector
                    y_vec = _mm512_loadu_pd(y0);

                    // Permute and reduce the complex and real parts
                    imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                    real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                    real_acc = _mm512_add_pd(real_acc, imag_acc);

                    y_vec = _mm512_add_pd(y_vec, real_acc);

                    // Store onto Y vector
                    _mm512_storeu_pd(y0, y_vec);

                    y0 += 8;
                    a_ptr[0] += 8;
                    a_ptr[1] += 8;
                    a_ptr[2] += 8;
                    a_ptr[3] += 8;
                    a_ptr[4] += 8;
                    a_ptr[5] += 8;
                    a_ptr[6] += 8;
                    a_ptr[7] += 8;
                }
            }
        }
        if( i < m )
        {
            __mmask8 m_mask = (1 << 2*(m - i)) - 1;
            if( bli_is_noconj(conja) )
            {
                // Load remaining elements from first 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[0]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[1]);
                a_vec[2] = _mm512_maskz_loadu_pd(m_mask, a_ptr[2]);
                a_vec[3] = _mm512_maskz_loadu_pd(m_mask, a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                // Load remaining elements from next 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[4]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[5]);
                a_vec[2] = _mm512_maskz_loadu_pd(m_mask, a_ptr[6]);
                a_vec[3] = _mm512_maskz_loadu_pd(m_mask, a_ptr[7]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                // Load remaining elements of Y vector
                y_vec = _mm512_maskz_loadu_pd(m_mask, y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                imag_acc = _mm512_fmaddsub_pd(zero_reg, zero_reg, imag_acc);
                real_acc = _mm512_add_pd(real_acc, imag_acc);

                y_vec = _mm512_add_pd(y_vec, real_acc);

                // Store onto Y vector
                _mm512_mask_storeu_pd(y0, m_mask, y_vec);
            }
            else
            {
                // Load remaining elements from first 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[0]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[1]);
                a_vec[2] = _mm512_maskz_loadu_pd(m_mask, a_ptr[2]);
                a_vec[3] = _mm512_maskz_loadu_pd(m_mask, a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_mul_pd(a_vec[0], alpha_x_real[0]);
                imag_acc = _mm512_mul_pd(a_vec[0], alpha_x_imag[0]);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[1], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[1], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[2], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[2], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[3], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[3], imag_acc);

                // Load remaining elements from next 4 columns of A
                a_vec[0] = _mm512_maskz_loadu_pd(m_mask, a_ptr[4]);
                a_vec[1] = _mm512_maskz_loadu_pd(m_mask, a_ptr[5]);
                a_vec[2] = _mm512_maskz_loadu_pd(m_mask, a_ptr[6]);
                a_vec[3] = _mm512_maskz_loadu_pd(m_mask, a_ptr[7]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_real[4], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[0], alpha_x_imag[4], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_real[5], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[1], alpha_x_imag[5], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_real[6], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[2], alpha_x_imag[6], imag_acc);

                real_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_real[7], real_acc);
                imag_acc = _mm512_fmadd_pd(a_vec[3], alpha_x_imag[7], imag_acc);

                // Load remaining elements of Y vector
                y_vec = _mm512_maskz_loadu_pd(m_mask, y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm512_permute_pd(imag_acc, 0x55);
                real_acc = _mm512_fmsubadd_pd(zero_reg, zero_reg, real_acc);
                real_acc = _mm512_add_pd(real_acc, imag_acc);

                y_vec = _mm512_add_pd(y_vec, real_acc);

                // Store onto Y vector
                _mm512_mask_storeu_pd(y0, m_mask, y_vec);
            }
        }
    }
    else
    {
        // Perform the computation with 128-bit registers,
        // since dcomplex is 128 bits in size
        __m128d a_vec[4], y_vec, real_acc, imag_acc, temp[8];

        // Unpacking and storing real and imaginary components
        // of alpha*X stored in x_vec[0...7]
        temp[0] = _mm_unpackhi_pd(x_vec[0], x_vec[0]);
        temp[1] = _mm_unpackhi_pd(x_vec[1], x_vec[1]);
        temp[2] = _mm_unpackhi_pd(x_vec[2], x_vec[2]);
        temp[3] = _mm_unpackhi_pd(x_vec[3], x_vec[3]);
        temp[4] = _mm_unpackhi_pd(x_vec[4], x_vec[4]);
        temp[5] = _mm_unpackhi_pd(x_vec[5], x_vec[5]);
        temp[6] = _mm_unpackhi_pd(x_vec[6], x_vec[6]);
        temp[7] = _mm_unpackhi_pd(x_vec[7], x_vec[7]);

        x_vec[0] = _mm_unpacklo_pd(x_vec[0], x_vec[0]);
        x_vec[1] = _mm_unpacklo_pd(x_vec[1], x_vec[1]);
        x_vec[2] = _mm_unpacklo_pd(x_vec[2], x_vec[2]);
        x_vec[3] = _mm_unpacklo_pd(x_vec[3], x_vec[3]);
        x_vec[4] = _mm_unpacklo_pd(x_vec[4], x_vec[4]);
        x_vec[5] = _mm_unpacklo_pd(x_vec[5], x_vec[5]);
        x_vec[6] = _mm_unpacklo_pd(x_vec[6], x_vec[6]);
        x_vec[7] = _mm_unpacklo_pd(x_vec[7], x_vec[7]);

        if ( bli_is_noconj(conja) )
        {
            for (; i < m; i++)
            {
                // Load elements from first 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[0]);
                a_vec[1] = _mm_loadu_pd(a_ptr[1]);
                a_vec[2] = _mm_loadu_pd(a_ptr[2]);
                a_vec[3] = _mm_loadu_pd(a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_mul_pd(a_vec[0], x_vec[0]);
                imag_acc = _mm_mul_pd(a_vec[0], temp[0]);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[1], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[1], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[2], x_vec[2], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[2], temp[2], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[3], x_vec[3], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[3], temp[3], imag_acc);

                // Load elements from next 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[4]);
                a_vec[1] = _mm_loadu_pd(a_ptr[5]);
                a_vec[2] = _mm_loadu_pd(a_ptr[6]);
                a_vec[3] = _mm_loadu_pd(a_ptr[7]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_fmadd_pd(a_vec[0], x_vec[4], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[0], temp[4], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[5], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[5], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[2], x_vec[6], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[2], temp[6], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[3], x_vec[7], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[3], temp[7], imag_acc);

                // Load Y vector
                y_vec = _mm_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                imag_acc = _mm_permute_pd(imag_acc, 0b01);
                real_acc = _mm_addsub_pd(real_acc, imag_acc);

                y_vec = _mm_add_pd(y_vec, real_acc);

                // Store Y vector
                _mm_storeu_pd(y0, y_vec);

                y0 += 2 * incy;
                a_ptr[0] += 2 * inca;
                a_ptr[1] += 2 * inca;
                a_ptr[2] += 2 * inca;
                a_ptr[3] += 2 * inca;
                a_ptr[4] += 2 * inca;
                a_ptr[5] += 2 * inca;
                a_ptr[6] += 2 * inca;
                a_ptr[7] += 2 * inca;
            }
        }
        else
        {
            for (; i < m; i++)
            {
                // Load elements from first 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[0]);
                a_vec[1] = _mm_loadu_pd(a_ptr[1]);
                a_vec[2] = _mm_loadu_pd(a_ptr[2]);
                a_vec[3] = _mm_loadu_pd(a_ptr[3]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_mul_pd(a_vec[0], x_vec[0]);
                imag_acc = _mm_mul_pd(a_vec[0], temp[0]);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[1], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[1], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[2], x_vec[2], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[2], temp[2], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[3], x_vec[3], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[3], temp[3], imag_acc);

                // Load elements from next 4 columns of A
                a_vec[0] = _mm_loadu_pd(a_ptr[4]);
                a_vec[1] = _mm_loadu_pd(a_ptr[5]);
                a_vec[2] = _mm_loadu_pd(a_ptr[6]);
                a_vec[3] = _mm_loadu_pd(a_ptr[7]);

                // Multiply the loaded columns of A by alpha*X(real and imag)
                real_acc = _mm_fmadd_pd(a_vec[0], x_vec[4], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[0], temp[4], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[1], x_vec[5], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[1], temp[5], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[2], x_vec[6], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[2], temp[6], imag_acc);

                real_acc = _mm_fmadd_pd(a_vec[3], x_vec[7], real_acc);
                imag_acc = _mm_fmadd_pd(a_vec[3], temp[7], imag_acc);

                // Load Y vector
                y_vec = _mm_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                real_acc = _mm_permute_pd(real_acc, 0b01);
                real_acc = _mm_addsub_pd(imag_acc, real_acc);
                real_acc = _mm_permute_pd(real_acc, 0b01);

                y_vec = _mm_add_pd(y_vec, real_acc);

                // Store Y vector
                _mm_storeu_pd(y0, y_vec);

                y0 += 2 * incy;
                a_ptr[0] += 2 * inca;
                a_ptr[1] += 2 * inca;
                a_ptr[2] += 2 * inca;
                a_ptr[3] += 2 * inca;
                a_ptr[4] += 2 * inca;
                a_ptr[5] += 2 * inca;
                a_ptr[6] += 2 * inca;
                a_ptr[7] += 2 * inca;
            }
        }
    }
}
