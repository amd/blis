##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bla_amax.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_amin.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_asum.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_axpy.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_copy.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_dot.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemm.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemm3m.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemmt.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_ger.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_hemm.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_hemv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_her.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_her2.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_her2k.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_herk.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_nrm2.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_scal.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_swap.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_symm.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_symv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_syr.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_syr2.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_syr2k.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_syrk.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_trmm.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_trmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_trsm.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_trsv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemm_batch.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_axpby.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_omatcopy.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_imatcopy.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_omatcopy2.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_omatadd.c
	)

#Add all subdirectories
#	add_subdirectory(attic)
#   add_subdirectory(blis)
    add_subdirectory(cblas)
    add_subdirectory(check)
    add_subdirectory(f2c)
