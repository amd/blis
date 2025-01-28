/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"
#include "blis_int_type.h"

// Benchmark application to process aocl logs generated by BLIS library.
#ifndef DT
#define DT BLIS_DOUBLE
#endif

//#define PRINT

#define AOCL_MATRIX_INITIALISATION


/* For BLIS since logs are collected at BLAS interfaces
 * we disable cblas interfaces for this benchmark application
 */

#ifdef BLIS_ENABLE_CBLAS
/* #define CBLAS */
#endif

// C = alpha * op(A) * op(AT) + beta * C
// where op(X) is one of op(X) = X, or op(X) = XT, or op(X) = XH,
// alpha and beta are scalars,
// A and C are matrices:
// op(A) is an n-by-k matrix,
// op(AT) is a k-by-n matrix,
// C is an n-by-n symmetric matrix.

int main( int argc, char** argv )
{
    obj_t a, c;
    obj_t c_save;
    obj_t alpha, beta;
    dim_t n, k;
    dim_t p_inc = 0; // to keep track of number of inputs
    num_t dt;
    //ind_t ind;
    char dt_ch;
    int   r, n_repeats;
    trans_t  transa;
    uplo_t uploc;

    double dtime;
    double dtime_save;
    double gflops;

    FILE* fin  = NULL;
    FILE* fout = NULL;

    n_repeats = N_REPEAT; //This macto will get from Makefile.

    dt = DT;

    if (argc < 3)
      {
        printf("Usage: ./test_syrk_XX.x input.csv output.csv\n");
        exit(1);
      }
    fin = fopen(argv[1], "r");
    if (fin == NULL)
      {
        printf("Error opening the file %s\n", argv[1]);
        exit(1);
      }
    fout = fopen(argv[2], "w");
    if (fout == NULL)
      {
        printf("Error opening output file %s\n", argv[2]);
        exit(1);
      }
    fprintf(fout, "Func Dt uploc transa n k alphaR alphaI lda betaR betaI ldc gflops\n");


    inc_t lda;
    inc_t ldc;

    char stor_scheme, transA_c, uplo_c;
    double alpha_r, beta_r, alpha_i, beta_i;
    dim_t m_trans, n_trans;
    char tmp[256]; // to store function name, line no present in logs.

    stor_scheme = 'C'; // since logs are collected at BLAS APIs

    // {S,D,C,Z}{ uploc, transa, n, k, alpha_real, alpha_imag, lda, beta_real, beta_imag, ldc}
    while (fscanf(fin, "%s %c %c %c " INT_FS INT_FS " %lf %lf " UINT_FS " %lf %lf " UINT_FS "\n",\
			tmp, &dt_ch, &uplo_c, &transA_c, &n, &k, &alpha_r,\
			&alpha_i, &lda, &beta_r, &beta_i, &ldc) == 12)
    {
        if (dt_ch == 'D' || dt_ch == 'd') dt = BLIS_DOUBLE;
        else if (dt_ch == 'Z' || dt_ch == 'z') dt = BLIS_DCOMPLEX;
        else if (dt_ch == 'S' || dt_ch == 's') dt = BLIS_FLOAT;
        else if (dt_ch == 'C' || dt_ch == 'c') dt = BLIS_SCOMPLEX;
        else
            {
              printf("Invalid data type %c\n", dt_ch);
              continue;
            }

        if(uplo_c == 'U' || uplo_c == 'u') uploc = BLIS_UPPER;
        else if (uplo_c == 'L' || uplo_c == 'l') uploc = BLIS_LOWER;
        else
        {
            printf("Invalid option for uplo\n");
            continue;
        }

        if( transA_c == 'n' || transA_c == 'N') transa = BLIS_NO_TRANSPOSE;
        else if (transA_c == 't' || transA_c == 'T') transa = BLIS_TRANSPOSE;
        else if ( transA_c == 'c' || transA_c == 'C') transa = BLIS_CONJ_TRANSPOSE;
        else
        {
            printf("Invalid option for transA \n");
            continue;
        }


        bli_obj_create( dt, 1, 1, 0, 0, &alpha);
        bli_obj_create( dt, 1, 1, 0, 0, &beta );

        if(stor_scheme == 'c' || stor_scheme == 'C')
        {
          // Column storage
          // leading dimension should be greater than number of rows
	  if( n > ldc ) continue;

          bli_set_dims_with_trans( transa, n, k, &m_trans, &n_trans);
	  if( m_trans > lda ) continue;
          bli_obj_create( dt, m_trans, n_trans, 1, lda, &a );

          bli_obj_create( dt, n, n, 1, ldc, &c );
          bli_obj_create( dt, n, n, 1, ldc, &c_save );
        }
        else if (stor_scheme == 'r' || stor_scheme == 'R')
        {
          bli_set_dims_with_trans( transa, n, k, &m_trans, &n_trans);

          // row storage
          // leading dimension shold be greater than number of columns

          if((n_trans > lda) || (n > ldc)) continue;

          bli_obj_create( dt, m_trans, n_trans, lda, 1, &a );

          bli_obj_create( dt, n, n, ldc, 1, &c );
          bli_obj_create( dt, n, n, ldc, 1, &c_save );
        }
        else
        {
            printf("Invalid storage schemes\n");
            continue;
        }
#ifndef CBLAS
        if( ( stor_scheme == 'R' ) || ( stor_scheme == 'r' ) )
        {
            printf("BLAS APIs doesn't support row-storage\n");
            continue;
        }
#endif
        bli_obj_set_struc( BLIS_SYMMETRIC, &c );
        bli_obj_set_uplo( uploc, &c );

#ifdef AOCL_MATRIX_INITIALISATION
        bli_randm( &a );
        bli_randm( &c );
#endif
        bli_mksymm( &c );
        bli_mktrim( &c );

        bli_obj_set_conjtrans( transa, &a);

        bli_setsc( alpha_r,alpha_i, &alpha );
        bli_setsc( beta_r, beta_i,  &beta  );

        bli_obj_set_struc( BLIS_SYMMETRIC, &c_save );
        bli_obj_set_uplo( uploc, &c_save);
        bli_copym( &c, &c_save );

        dtime_save = DBL_MAX;

        for ( r = 0; r < n_repeats; ++r )
        {
            bli_copym( &c_save, &c );

#ifdef PRINT
            bli_printm( "a", &a, "%4.1f", "," );
            bli_printm( "c", &c, "%4.1f", "," );
#endif

            dtime = bli_clock();

#ifdef BLIS
            bli_syrk( &alpha,
                      &a,
                      &beta,
                      &c );

#else

#ifdef CBLAS
            enum CBLAS_ORDER cblas_order;
            enum CBLAS_UPLO  cblas_uplo;
            enum CBLAS_TRANSPOSE cblas_transa;

            if ( ( stor_scheme == 'C' ) || ( stor_scheme == 'c' ) )
                cblas_order = CblasColMajor;
            else
                cblas_order = CblasRowMajor;
            if( bli_is_upper( uploc ) )
                cblas_uplo = CblasUpper;
            else
                cblas_uplo = CblasLower;

            if( bli_is_trans( transa ) )
                cblas_transa = CblasTrans;
            else if( bli_is_conjtrans( transa ) )
                cblas_transa = CblasConjTrans;
            else
                cblas_transa = CblasNoTrans;

#else

            f77_char f77_transa;
            f77_char f77_uploc;

            bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
            bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );
#endif

            if ( bli_is_float( dt ) )
            {
                f77_int  kk     = k;
                f77_int  nn     = n;
                float*   alphap = bli_obj_buffer( &alpha );
                float*   ap     = bli_obj_buffer( &a );
                float*   betap  = bli_obj_buffer( &beta );
                float*   cp     = bli_obj_buffer( &c );
#ifdef CBLAS
                cblas_ssyrk( cblas_order,
                              cblas_uplo,
                              cblas_transa,
                              nn,
                              kk,
                              *alphap,
                              ap, lda,
                              *betap,
                              cp, ldc );

#else
                ssyrk_( &f77_uploc,
                         &f77_transa,
                         &nn,
                         &kk,
                         alphap,
                         ap, (f77_int*)&lda,
                         betap,
                         cp, (f77_int*)&ldc );

#endif
            }
            else if ( bli_is_double( dt ) )
            {
                f77_int  kk      = k;
                f77_int  nn      = n;
                double*  alphap = bli_obj_buffer( &alpha );
                double*  ap     = bli_obj_buffer( &a );
                double*  betap  = bli_obj_buffer( &beta );
                double*  cp     = bli_obj_buffer( &c );
#ifdef CBLAS
                cblas_dsyrk( cblas_order,
                              cblas_uplo,
                              cblas_transa,
                              nn,
                              kk,
                              *alphap,
                              ap,lda,
                              *betap,
                              cp, ldc
                            );
#else
                dsyrk_( &f77_uploc,
                         &f77_transa,
                         &nn,
                         &kk,
                         alphap,
                         ap, (f77_int*)&lda,
                         betap,
                         cp, (f77_int*)&ldc
                       );
#endif
            }
            else if ( bli_is_scomplex( dt ) )
            {
                f77_int  kk     = k;
                f77_int  nn     = n;
                scomplex*  alphap = bli_obj_buffer( &alpha );
                scomplex*  ap     = bli_obj_buffer( &a );
                scomplex*  betap  = bli_obj_buffer( &beta );
                scomplex*  cp     = bli_obj_buffer( &c );
#ifdef CBLAS
                cblas_csyrk( cblas_order,
                              cblas_uplo,
                              cblas_transa,
                              nn,
                              kk,
                              alphap,
                              ap, lda,
                              betap,
                              cp, ldc );
#else
                csyrk_( &f77_uploc,
                         &f77_transa,
                         &nn,
                         &kk,
                         alphap,
                         ap, (f77_int*)&lda,
                         betap,
                         cp, (f77_int*)&ldc
                       );

#endif
            }
            else if ( bli_is_dcomplex( dt ) )
            {
                f77_int  kk     = k;
                f77_int  nn     = n;
                dcomplex*  alphap = bli_obj_buffer( &alpha );
                dcomplex*  ap     = bli_obj_buffer( &a );
                dcomplex*  betap  = bli_obj_buffer( &beta );
                dcomplex*  cp     = bli_obj_buffer( &c );
#ifdef CBLAS
                cblas_zsyrk( cblas_order,
                              cblas_uplo,
                              cblas_transa,
                              nn,
                              kk,
                              alphap,
                              ap, lda,
                              betap,
                              cp, ldc );

#else
                zsyrk_( &f77_uploc,
                         &f77_transa,
                         &nn,
                         &kk,
                         alphap,
                         ap, (f77_int*)&lda,
                         betap,
                         cp, (f77_int*)&ldc );

#endif
            }
#endif

#ifdef PRINT
            bli_printm( "c after", &c, "%4.1f", "" );
            exit(1);
#endif


            dtime_save = bli_clock_min_diff( dtime_save, dtime );
        }

        gflops = ( n * k * n ) / ( dtime_save * 1.0e9 );

        if ( bli_is_complex( dt ) ) gflops *= 4.0;

        printf("data_gemm_%s", BLAS);
        p_inc++;

        printf( "( %2lu, 1:4 ) = [ %4lu %4lu %7.2f ];\n",
                ( unsigned long )p_inc,
                ( unsigned long )n,
                ( unsigned long )k, gflops );

        fprintf(fout, "%s %c %c %c " INT_FS INT_FS " %lf %lf " UINT_FS " %lf %lf " UINT_FS " %6.3f\n", \
                tmp, dt_ch, uplo_c, transA_c, n, k,
                alpha_r, alpha_i,
                lda, beta_r, beta_i,
                ldc, gflops
                );

        fflush(fout);

        bli_obj_free( &alpha );
        bli_obj_free( &beta );

        bli_obj_free( &a );
        bli_obj_free( &c );
        bli_obj_free( &c_save );
    }

    fclose(fin);
    fclose(fout);

    return 0;
}
