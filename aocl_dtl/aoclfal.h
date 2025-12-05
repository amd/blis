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
 * File Name :  aoclfal.h
 * 
 * Description : Interfaces for platform/os independent file 
 *               handling API's
 * 
 *==================================================================*/

#ifndef _AOCL_FAL_H_
#define _AOCL_FAL_H_

/* The possible error values of FAL */
#define AOCL_FAL_SUCCESS             0
#define AOCL_FAL_CLOSE_ERROR        -1
#define AOCL_FAL_READ_ERROR         -2
#define AOCL_FAL_WRITE_ERROR        -3
#define AOCL_FAL_EOF_ERROR          -6
#define AOCL_FAL_FERROR             -7

/* The type definition for FILE */
#define AOCL_FAL_FILE FILE

/* The FAL function declaration */
int32 AOCL_FAL_Close(
    AOCL_FAL_FILE *fpFilePointer);

int32 AOCL_FAL_Error(
    AOCL_FAL_FILE *fpFilePointer);

AOCL_FAL_FILE *AOCL_FAL_Open(
    const int8 *pchFileName,
    const int8 *pchMode);

int32 AOCL_FAL_Read(
    void *pvBuffer,
    int32 i32Size,
    int32 i32Count,
    AOCL_FAL_FILE *fpFilePointer);

int32 AOCL_FAL_Write(
    const void *pvBuffer,
    int32 i32Size,
    int32 iCount,
    AOCL_FAL_FILE *fpFilePointer);

#endif /* _AOCL_FAL_H_ */

/* --------------- End of aoclfal.h ----------------- */
