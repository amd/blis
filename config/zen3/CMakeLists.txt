##Copyright (C) 2020, Advanced Micro Devices, Inc ##

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cntx_init_zen3.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_family_zen3.h
   )
