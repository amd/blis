##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_d6x8m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_d6x8n.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_s6x16m.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_s6x16n.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_d6x8m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_d6x8n.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_s6x16m.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_s6x16n.c
    )
add_subdirectory(d6x8)
#add_subdirectory(s6x16)

