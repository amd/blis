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
 * File Name :  aoclflist.h
 *
 * Description : Linked list of open files associated with
 *               each thread. This is used to log the deta
 *               to correct file as per the current thread id.
 *
 *==================================================================*/

#ifndef _AOCL_FLIST_H_
#define _AOCL_FLIST_H_

#include "blis.h"
#include "aocltpdef.h"
#include "aoclfal.h"

typedef struct AOCL_FLIST_Node_t
{
    AOCL_TID tid;
    AOCL_FAL_FILE *fp;
    uint64 u64SavedTimeStamp;
    struct AOCL_FLIST_Node_t *pNext;
} AOCL_FLIST_Node;

bool AOCL_FLIST_IsEmpty(
    AOCL_FLIST_Node *plist);

AOCL_FLIST_Node * AOCL_FLIST_GetNode(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid);

AOCL_FAL_FILE *AOCL_FLIST_GetFile(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid);

AOCL_FAL_FILE *AOCL_FLIST_AddFile(
    const int8 *pchFilePrefix,
    AOCL_FLIST_Node **plist,
    AOCL_TID tid);

void AOCL_FLIST_CloseFile(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid);

void AOCL_FLIST_CloseAll(
    AOCL_FLIST_Node *plist);

#endif /* _AOCL_FLIST_H_ */

/* --------------- End of aoclfist.h ----------------- */
