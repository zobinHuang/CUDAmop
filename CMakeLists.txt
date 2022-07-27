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
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
)

# ====================== OPERATOR LIBRARY ======================
file(GLOB_RECURSE SRC_OPERATORS "src/*.cu" "src/*.cpp")
add_library(_operators ${SRC_OPERATORS})
target_compile_features(_operators PUBLIC cxx_std_11)
target_compile_options(
  _operators PUBLIC 
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>
)

# Don't do following operation!
# set_target_properties( 
#   _operators                   
#   PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# )

target_include_directories(_operators PUBLIC include)

# ====================== PROFILING PROGRAM ======================
# >>> basic vector addition
file(GLOB_RECURSE SRC_BASIC_VECTOR_ADDITION "profiling/basic_vector_addition/*.cu")
add_executable(basic_vector_addition ${SRC_BASIC_VECTOR_ADDITION})

# >>> global configuration
set(PROFILING_TARGETS 
  basic_vector_addition
)
target_link_libraries(${PROFILING_TARGETS} _operators nvToolsExt)
target_compile_features(${PROFILING_TARGETS} PUBLIC cxx_std_11)
target_include_directories(${PROFILING_TARGETS} PUBLIC include)
target_compile_options(
  ${PROFILING_TARGETS} PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>
)