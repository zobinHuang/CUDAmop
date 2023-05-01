# cmake version
cmake_minimum_required(VERSION 3.16.3)

# project info
project(CUDAop LANGUAGES CXX CUDA)

# set executable output path
set(PATH_EXECUTABLE bin)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_EXECUTABLE})
SET(EXECUTABLE_OUTPUT_PATH ../${PATH_EXECUTABLE})

# set library output path
set(PATH_LIBRARY lib)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_LIBRARY})
SET(LIBRARY_OUTPUT_PATH ../${PATH_LIBRARY})

# find CUDA Toolkit Extension library
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
)

# ====================== OPERATOR LIBRARY ======================
file(GLOB_RECURSE SRC_OPERATORS "src/**/*.cu" "src/**/*.cpp")
add_library(_operators ${SRC_OPERATORS})
target_compile_features(_operators PUBLIC cxx_std_11)
target_compile_options(
  _operators PUBLIC 
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
)

# Don't do following operation!
# set_target_properties( 
#   _operators                   
#   PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# )

target_include_directories(_operators PUBLIC include)

# ====================== PROFILING PROGRAM ======================
# >>> vector addition
SET(SRC_VECTOR_ADDITION profiling/vector_addition)
file(GLOB_RECURSE SRC_VECTOR_ADDITION_CPP ${SRC_VECTOR_ADDITION}/*.cpp)
add_executable(basic_vector_addition ${SRC_VECTOR_ADDITION}/basic.cu ${SRC_VECTOR_ADDITION_CPP})
add_executable(unified_vector_addition ${SRC_VECTOR_ADDITION}/unified.cu ${SRC_VECTOR_ADDITION_CPP})
add_executable(unified_prefetch_vector_addition ${SRC_VECTOR_ADDITION}/unified_prefetch.cu ${SRC_VECTOR_ADDITION_CPP})

# >>> matrix multiplication
SET(SRC_MATRIX_MULTIPLICATION profiling/matrix_multiplication)
file(GLOB_RECURSE SRC_MATRIX_MULTIPLICATION_CPP ${SRC_MATRIX_MULTIPLICATION}/*.cpp)
add_executable(basic_matrix_multiplication ${SRC_MATRIX_MULTIPLICATION}/basic.cu ${SRC_MATRIX_MULTIPLICATION_CPP})
add_executable(tiled_matrix_multiplication ${SRC_MATRIX_MULTIPLICATION}/tiled.cu ${SRC_MATRIX_MULTIPLICATION_CPP})
add_executable(aligned_matrix_multiplication ${SRC_MATRIX_MULTIPLICATION}/aligned.cu ${SRC_MATRIX_MULTIPLICATION_CPP})

# >>> sum reduction
SET(SRC_SUM_REDUCTION profiling/sum_reduction)
file(GLOB_RECURSE SRC_SUM_REDUCTION_CPP ${SRC_SUM_REDUCTION}/*.cpp)
add_executable(basic_sum_reduction ${SRC_SUM_REDUCTION}/basic.cu ${SRC_SUM_REDUCTION_CPP})
add_executable(non_divergence_sum_reduction ${SRC_SUM_REDUCTION}/non_divergence.cu ${SRC_SUM_REDUCTION_CPP})

# >>> cuBLAS
SET(SRC_CUBLAS profiling/cublas)
file(GLOB_RECURSE SRC_CUBLAS_CPP ${SRC_CUBLAS}/*.cpp)
add_executable(cublas_vector_addition ${SRC_CUBLAS}/vector_add.cu ${SRC_CUBLAS_CPP})
add_executable(cublas_matrix_multiplication ${SRC_CUBLAS}/matrix_multiplication.cu ${SRC_CUBLAS_CPP})

# >>> SpMV
SET(SRC_SPMV profiling/spmv)
file(GLOB_RECURSE SRC_SPMV_CPP ${SRC_SPMV}/*.cpp)
add_executable(spmv_csr_scalar ${SRC_SPMV}/scalar.cu ${SRC_SPMV_CPP})
add_executable(spmv_csr_vector ${SRC_SPMV}/vector.cu ${SRC_SPMV_CPP})

# >>> global configuration
set(PROFILING_TARGETS 
  basic_vector_addition
  unified_vector_addition
  unified_prefetch_vector_addition
  basic_matrix_multiplication
  tiled_matrix_multiplication
  aligned_matrix_multiplication
  basic_sum_reduction
  non_divergence_sum_reduction
  cublas_vector_addition
  cublas_matrix_multiplication
  spmv_csr_scalar
  spmv_csr_vector
)
foreach( profiling_target ${PROFILING_TARGETS} )
  target_link_libraries(${profiling_target} _operators nvToolsExt -lcublas -lcurand)
  target_compile_features(${profiling_target} PUBLIC cxx_std_11)
  target_include_directories(${profiling_target} PUBLIC include)
  target_compile_options(
    ${profiling_target} PRIVATE 
    # 编译器前后端: arch -> IR, code -> 后端 binary (针对硬件的二进制，小版本号兼容 sm_80 -> sm_86)
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
  )
endforeach( profiling_target ${PROFILING_TARGETS} )