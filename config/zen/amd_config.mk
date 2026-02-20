#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2021 - 2026, Advanced Micro Devices, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# All the common flags for AMD architectures will be added here

# NOTE: The build system will append these variables with various
# general-purpose/configuration-agnostic flags in common.mk. You
# may specify additional flags here as needed.

CPPROCFLAGS    :=
CMISCFLAGS     :=
CPICFLAGS      :=
CWARNFLAGS     :=

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
  COPTFLAGS      := -O0
else
  COPTFLAGS      := -O3
endif

# Flags specific to optimized kernels.
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
CKOPTFLAGS     := $(COPTFLAGS) -fomit-frame-pointer
# Additional flag which is required for lpgemm kernels
CKLPOPTFLAGS     :=

ifeq ($(CC_VENDOR),gcc)
  CKVECFLAGS     := -mavx2 -mfpmath=sse -mfma
else ifeq ($(CC_VENDOR),clang)
  CKVECFLAGS     := -mavx2 -mfpmath=sse -mfma -mno-fma4 -mno-tbm -mno-xop -mno-lwp
  ifeq ($(strip $(shell $(CC) -v |&head -1 |grep -c 'AOCC.LLVM')),1)
    CKVECFLAGS += -mllvm -disable-licm-vrp
  endif
else
  $(error gcc or clang are required for this configuration.)
endif

ifeq ($(CC_VENDOR),clang)
  # But also set these in case we are using upstream LLVM clang
  VENDOR_STRING := $(strip $(shell ${CC_VENDOR} --version | egrep -o '[0-9]+\.[0-9]+\.?[0-9]*'))
  CC_MAJOR := $(shell (echo ${VENDOR_STRING} | cut -d. -f1))
  # Detect whether this is AOCC from the compiler version string
  CLANG_VERSION_STRING := $(strip $(shell $(CC) --version 2>&1 | head -1))
  # Extract if the compiler is AOCC
  IS_AOCC := $(findstring AOCC,$(CLANG_VERSION_STRING))
  AOCC_VERSION_STRING :=
  ifneq ($(IS_AOCC),)
    # AOCC detected - extract version string
    # Try to match AOCC_x.y.z format first, then remove AOCC_ prefix if found
    AOCC_VERSION_STRING := $(strip $(shell $(CC) --version 2>&1 | grep -oE 'AOCC_[0-9]+\.[0-9]+\.[0-9]+' | head -1 | sed 's/AOCC_//'))
    # If AOCC_x.y.z not found, try AOCC_x_y_z format
    ifeq ($(AOCC_VERSION_STRING),)
      AOCC_VERSION_STRING := $(strip $(shell $(CC) --version 2>&1 | grep -oE 'AOCC_[0-9]+_[0-9]+_[0-9]+' | head -1 | sed 's/AOCC_//'))
      # Replace underscores with dots in the version string
      AOCC_VERSION_STRING := $(shell echo $(AOCC_VERSION_STRING) | sed 's/_/./g')
    endif
    # If AOCC_x.y.z or AOCC_x_y_z not found, try AOCC.LLVM.x.y.z format
    ifeq ($(AOCC_VERSION_STRING),)
      AOCC_VERSION_STRING := $(strip $(shell $(CC) --version 2>&1 | grep -oE 'AOCC\.LLVM\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 | sed 's/AOCC\.LLVM\.//'))
    endif
    # If AOCC detected but no version extracted, hard fail
    ifeq ($(AOCC_VERSION_STRING),)
      $(error Could not extract AOCC version from clang version string: $(CLANG_VERSION_STRING))
    endif
    
    # Extract major version number from AOCC version
    AOCC_MAJOR := $(shell (echo ${AOCC_VERSION_STRING} | cut -d. -f1))
  endif
  
  #$(error Detected AOCC compiler version $(AOCC_VERSION_STRING), major version $(AOCC_MAJOR), clang version $(VENDOR_STRING), CC_MAJOR $(CC_MAJOR))
endif # clang

# Flags specific to reference kernels.
CROPTFLAGS     := $(CKOPTFLAGS)
ifeq ($(CC_VENDOR),gcc)
  CRVECFLAGS     := $(CKVECFLAGS) -funsafe-math-optimizations -ffp-contract=fast
else ifeq ($(CC_VENDOR),clang)
  CRVECFLAGS     := $(CKVECFLAGS) -funsafe-math-optimizations -ffp-contract=fast
else
  CRVECFLAGS     := $(CKVECFLAGS)
endif

