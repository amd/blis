##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsv_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsv_unb_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsv_unf_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsv_unf_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsv_var_oapi.c
    )

