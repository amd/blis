/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include "bios.h"
#include "common/io/io.h"

#include <stdlib.h>

static bool hostValueSet(const FFstrbuf* value)
{
    const char* str = value->chars;
    const size_t length = value->length;

    return (length > 0) &&
           !(
               ffStrbufStartsWithIgnCaseS(str, length, "To be filled") ||
               ffStrbufStartsWithIgnCaseS(str, length, "To be set") ||
               ffStrbufStartsWithIgnCaseS(str, length, "OEM") ||
               ffStrbufStartsWithIgnCaseS(str, length, "O.E.M.") ||
               ffStrbufIgnCaseCompS(str, length, "None") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "System Product") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "System Product Name") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "System Product Version") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "System Name") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "System Version") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "Default string") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "Undefined") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "Not Specified") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "Not Applicable") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "INVALID") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "Type1ProductConfigId") == 0 ||
               ffStrbufIgnCaseCompS(str, length, "All Series") == 0
           );
}

void ffDetectBios(FFBiosResult* bios)
{
    ffStrbufInit(&bios->error);
    ffStrbufInit(&bios->biosDate);
    ffStrbufInit(&bios->biosRelease);
    ffStrbufInit(&bios->biosVendor);
    ffStrbufInit(&bios->biosVersion);

    ffReadFileBuffer("/sys/devices/virtual/dmi/id/bios_date", &bios->biosDate);
    if (hostValueSet(&bios->biosDate))
        return;

    ffReadFileBuffer("/sys/class/dmi/id/bios_date", &bios->biosDate);
    if (hostValueSet(&bios->biosDate))
        return;

    ffStrbufClear(&bios->biosDate);

    ffReadFileBuffer("/sys/devices/virtual/dmi/id/bios_release", &bios->biosRelease);
    if (hostValueSet(&bios->biosRelease))
        return;

    ffReadFileBuffer("/sys/class/dmi/id/bios_release", &bios->biosRelease);
    if (hostValueSet(&bios->biosRelease))
        return;

    ffStrbufClear(&bios->biosRelease);

    ffReadFileBuffer("/sys/devices/virtual/dmi/id/bios_vendor", &bios->biosVendor);
    if (hostValueSet(&bios->biosVendor))
        return;

    ffReadFileBuffer("/sys/class/dmi/id/bios_vendor", &bios->biosVendor);
    if (hostValueSet(&bios->biosVendor))
        return;

    ffStrbufClear(&bios->biosVendor);

    ffReadFileBuffer("/sys/devices/virtual/dmi/id/bios_version", &bios->biosVersion);
    if (hostValueSet(&bios->biosVersion))
        return;

    ffReadFileBuffer("/sys/class/dmi/id/bios_version", &bios->biosVersion);
    if (hostValueSet(&bios->biosVersion))
        return;

    ffStrbufClear(&bios->biosVersion);
}
