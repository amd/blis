/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
 * File Name :  aocldtlcf.h
 *
 * Description : This is configuration file for debug and trace
 *               libaray, all debug features (except auto trace)
 *               can be enabled/disabled in this file.
 *
 *==================================================================*/

#ifndef _AOCLDTLCF_H_
#define _AOCLDTLCF_H_

/* DTL_DumpData functionality (AOCL_DTL_DUMP_ENABLE) is not used at present */
/* Macro for dumping the log If the user wants to enable dumping he has to
   enable this macro by making it to 1 else 0 */

/* AOCL_DTL_LOG_ENABLE and AOCL_DTL_TRACE_ENABLE now defined via configure
   and cmake options */

/* Select the trace level till which you want to log the data */
/* Default set in configure and CMakeLists.txt is AOCL_DTL_TRACE_LEVEL_NUMBER=5 */
#define AOCL_DTL_TRACE_LEVEL         AOCL_DTL_TRACE_LEVEL_NUMBER

/* user has to explicitly use the below macros to identify
   criticality of the logged message */
#define AOCL_DTL_LEVEL_ALL          (15)
#define AOCL_DTL_LEVEL_TRACE_10     (15)
#define AOCL_DTL_LEVEL_TRACE_9      (14)
#define AOCL_DTL_LEVEL_TRACE_8      (13)
#define AOCL_DTL_LEVEL_TRACE_7      (12)      /* Kernels */
#define AOCL_DTL_LEVEL_TRACE_6      (11)
#define AOCL_DTL_LEVEL_TRACE_5      (10)
#define AOCL_DTL_LEVEL_TRACE_4      (9)
#define AOCL_DTL_LEVEL_TRACE_3      (8)
#define AOCL_DTL_LEVEL_TRACE_2      (7)
#define AOCL_DTL_LEVEL_TRACE_1      (6)       /* BLIS/BLAS API */
#define AOCL_DTL_LEVEL_VERBOSE      (5)
#define AOCL_DTL_LEVEL_INFO         (4)
#define AOCL_DTL_LEVEL_MINOR        (3)
#define AOCL_DTL_LEVEL_MAJOR        (2)
#define AOCL_DTL_LEVEL_CRITICAL     (1)


#define AOCL_DTL_TRACE_FILE         "aocldtl_trace.txt"
#define AOCL_DTL_AUTO_TRACE_FILE    "aocldtl_auto_trace.rawfile"
#define AOCL_DTL_LOG_FILE           "aocldtl_log.txt"

/* The use can use below three macros for different data type while dumping data
 * or specify the size of data type in bytes macro for character data type */
#define AOCL_CHAR_DATA_TYPE         (1)

/* macro for short data type */
#define AOCL_UINT16_DATA_TYPE       (2)

/* macro for String data type */
#define AOCL_STRING_DATA_TYPE       (3)

/* macro for uint32 data type */
#define AOCL_UINT32_DATA_TYPE       (4)

/* macro for printing Hex values */
#define AOCL_LOG_HEX_VALUE          ('x')

/* macro for printing Decimal values */
#define AOCL_LOG_DECIMAL_VALUE      ('d')



#endif /* _AOCLDTLCF_H_ */

/* --------------- End of aocldtlcf.h ----------------- */
