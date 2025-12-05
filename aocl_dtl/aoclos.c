/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

/*===================================================================
 * File Name :  aoclos.c
 *
 * Description : Abstraction for os services used by DTL.
 *
 *==================================================================*/
#include "blis.h"
#include "aocltpdef.h"
#include "aocldtl.h"
#include "aoclfal.h"
#include "aocldtlcf.h"

#if defined(__linux__)

#include <time.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef BLIS_ENABLE_OPENMP
#include <omp.h>
#endif

#endif

#if BLIS_OS_WINDOWS
#include <process.h>
#endif

// BLIS TODO: This is workaround to check if BLIS is built with
//            openmp support. Ideally we don't want any library
//            specific code in dtl.
#include <blis.h>

#if defined(__linux__)

/*
    Disable instrumentation for these functions as they will also be
    called from compiler generated instrumentation code to trace
    function execution.

    It needs to be part of declaration in the C file so can't be
    moved to header file.

*/

AOCL_TID AOCL_gettid(void) __attribute__((no_instrument_function));
pid_t    AOCL_getpid(void) __attribute__((no_instrument_function));
uint64   AOCL_getTimestamp(void) __attribute__((no_instrument_function));

AOCL_TID AOCL_gettid(void)
{

#ifdef BLIS_ENABLE_OPENMP
  return omp_get_thread_num();
#else
#ifdef BLIS_ENABLE_PTHREADS
  // pthread_self is not suitable for this purpose and may be replaced
  // in a later release with something else. It returns a value of type
  // pthread_t, which on linux is an unsigned long int.
  return (AOCL_TID) pthread_self();
#else
  return 0;
#endif
#endif

}

pid_t  AOCL_getpid(void)
{
    return getpid();
}

uint64 AOCL_getTimestamp(void)
{
    struct timespec tms;

    /* The C11 way */
    if (clock_gettime(CLOCK_REALTIME, &tms))
    {
        return -1;
    }

    /* seconds, multiplied with 1 million */
    uint64 micros = tms.tv_sec * 1000000;
    /* Add full microseconds */
    micros += tms.tv_nsec / 1000;
    /* round up if necessary */
    if (tms.tv_nsec % 1000 >= 500)
    {
        ++micros;
    }
    return micros;
}


#elif BLIS_OS_WINDOWS

AOCL_TID AOCL_gettid(void)
{
#ifdef BLIS_ENABLE_OPENMP
  return omp_get_thread_num();
#else
#ifdef BLIS_ENABLE_PTHREADS
  // pthread_self is not suitable for this purpose and may be replaced
  // in a later release with something else. It returns a value of type
  // pthread_t, whose type may depend upon the operating system. On
  // freeBSD it is a pointer to an empty struct.
  return (AOCL_TID) pthread_self();
#else
  return 0;
#endif
#endif
}

pid_t  AOCL_getpid(void)
{
    return (pid_t) _getpid();
}

uint64 AOCL_getTimestamp(void)
{
    /* stub for other os's */
    return 0;
}

#else  /* Non linux, Non Windows support */

AOCL_TID AOCL_gettid(void)
{
#ifdef BLIS_ENABLE_OPENMP
  return omp_get_thread_num();
#else
#ifdef BLIS_ENABLE_PTHREADS
  // pthread_self is not suitable for this purpose and may be replaced
  // in a later release with something else. It returns a value of type
  // pthread_t, whose type may depend upon the operating system. On
  // freeBSD it is a pointer to an empty struct.
  return (AOCL_TID) pthread_self();
#else
  return 0;
#endif
#endif
}

pid_t  AOCL_getpid(void)
{
    /* stub for other os's */
    return 0;
}

uint64 AOCL_getTimestamp(void)
{
    /* stub for other os's */
    return 0;
}
#endif
