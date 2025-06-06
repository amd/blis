#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2019 - 2024, Advanced Micro Devices, Inc. All rights reserved.
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

# FLAGS that are specific to the 'zen3' architecture are added here.
# FLAGS that are common for all the AMD architectures are present in
# config/zen/amd_config.mk.

# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := zen3
#CONFIGS_INCL   += $(THIS_CONFIG)

# Include file containing common flags for all AMD architectures
AMD_CONFIG_FILE := amd_config.mk
AMD_CONFIG_PATH := $(BASE_SHARE_PATH)/config/zen
-include $(AMD_CONFIG_PATH)/$(AMD_CONFIG_FILE)

#
# --- Determine the C compiler and related flags ---
#

# NOTE: The build system will append these variables with various
# general-purpose/configuration-agnostic flags in common.mk. You
# may specify additional flags here as needed.

CPPROCFLAGS    :=
CMISCFLAGS     :=
CPICFLAGS      :=
CWARNFLAGS     :=

ifneq ($(DEBUG_TYPE),off)
  CDBGFLAGS    := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
  COPTFLAGS    := -O0
else
  COPTFLAGS    := -O3
endif

# Flags specific to optimized kernels.
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
CKOPTFLAGS     := $(COPTFLAGS) -fomit-frame-pointer
# Additional flag which is required for lpgemm kernels
CKLPOPTFLAGS   :=

# gcc or clang version must be at least 4.0
ifeq ($(CC_VENDOR),gcc)
  GCC_VERSION := $(strip $(shell $(CC) -dumpversion | cut -d. -f1))

  ifeq ($(shell test $(GCC_VERSION) -ge 11; echo $$?),0)
    # gcc 11.0 or later
    CKVECFLAGS += -march=znver3
    # Update CKOPTFLAGS for gcc to use O3 optimization without
    # -ftree-pre and -ftree-partial-pre flag. These flag results
    # in suboptimal code generation for instrinsic based kernels.
    # The -ftree-loop-vectorize results in inefficient code gen
    # for amd optimized l1 kernels based on instrinsics.
    CKLPOPTFLAGS += -fno-tree-partial-pre -fno-tree-pre -fno-tree-loop-vectorize -fno-gcse
  else ifeq ($(shell test $(GCC_VERSION) -ge 9; echo $$?),0)
    # gcc 9.0 or later
    CKVECFLAGS += -march=znver2
    CKLPOPTFLAGS += -fno-tree-partial-pre -fno-tree-pre -fno-tree-loop-vectorize -fno-gcse
  else
    # If gcc is older than 9.1.0 but at least 6.1.0, then we can use -march=znver1
    # as the fallback option.
    CKVECFLAGS += -march=znver1 -mno-avx256-split-unaligned-store
    CRVECFLAGS += -march=znver1 -mno-avx256-split-unaligned-store
  endif
endif # gcc

ifeq ($(CC_VENDOR),clang)
  # AOCC clang has various formats for the version line

  # AOCC.LLVM.2.0.0.B191.2019_07_19 clang version 8.0.0 (CLANG: Jenkins AOCC_2_0_0-Build#191) (based on LLVM AOCC.LLVM.2.0.0.B191.2019_07_19)
  # AOCC.LLVM.2.1.0.B1030.2019_11_12 clang version 9.0.0 (CLANG: Build#1030) (based on LLVM AOCC.LLVM.2.1.0.B1030.2019_11_12)
  # AMD clang version 10.0.0 (CLANG: AOCC_2.2.0-Build#93 2020_06_25) (based on LLVM Mirror.Version.10.0.0)
  # AMD clang version 11.0.0 (CLANG: AOCC_2.3.0-Build#85 2020_11_10) (based on LLVM Mirror.Version.11.0.0)
  # AMD clang version 12.0.0 (CLANG: AOCC_3.0.0-Build#2 2020_11_05) (based on LLVM Mirror.Version.12.0.0)
  # AMD clang version 14.0.0 (CLANG: AOCC_4.0.0-Build#98 2022_06_15) (based on LLVM Mirror.Version.14.0.0)

  # For our purpose we just want to know if it version 2x or 3x or 4x

  # But also set these in case we are using upstream LLVM clang
  VENDOR_STRING := $(strip $(shell ${CC_VENDOR} --version | egrep -o '[0-9]+\.[0-9]+\.?[0-9]*'))
  CC_MAJOR := $(shell (echo ${VENDOR_STRING} | cut -d. -f1))

  ifeq ($(strip $(shell $(CC) -v |&head -1 |grep -c 'AOCC_4')),1)
    # AOCC version 4x we will enable znver3
    CKVECFLAGS += -march=znver3
  else ifeq ($(strip $(shell $(CC) -v |&head -1 |grep -c 'AOCC_3')),1)
    # AOCC version 3x we will enable znver3
    CKVECFLAGS += -march=znver3
  else ifeq ($(strip $(shell $(CC) -v |&head -1 |grep -c 'AOCC.LLVM.2\|AOCC_2')),1)
    # AOCC version 2x we will enable znver2
    CKVECFLAGS += -march=znver2
  else ifeq ($(shell test $(CC_MAJOR) -ge 9; echo $$?),0)
    # LLVM clang 9.0 or later
    CKVECFLAGS += -march=znver2
  else
    CKVECFLAGS += -march=znver1
  endif
endif # clang

# Flags specific to reference kernels.
CROPTFLAGS     := $(CKOPTFLAGS)
CRVECFLAGS     := $(CKVECFLAGS)

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))

