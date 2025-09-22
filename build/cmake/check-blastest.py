#!/usr/bin/env python3
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.
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

# Import modules
import os
import sys

def check_blastest():
    results_file_path = sys.argv[1]
    results_directory = os.listdir(results_file_path)
    has_failure = False
    is_empty = False
    for fname in results_directory:
        if os.path.isfile(results_file_path + os.sep + fname) and "out" in fname:
            file = open(results_file_path + os.sep + fname, 'r')
            # read all content of a file
            content = file.read()
            if content == "":
                is_empty = True
            # check if string present in a file
            if "*****" in content:
                has_failure = True
    if has_failure:
        print("\033[0;31m At least one BLAS test failed. :( \033[0m")
        print("\033[0;31m Please see the corresponding out.* for details. \033[0m")
        exit(1)
    elif is_empty:
        print("\033[0;31m At least one BLAS test resulted without a PASS. :( \033[0m")
        print("\033[0;31m Please ensure that the corresponding out.* was generated correctly. \033[0m")
        exit(1)
    else:
        print("\033[0;32m All BLAS tests passed! \033[0m")
        exit(0)

check_blastest()
