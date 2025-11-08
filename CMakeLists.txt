cmake_minimum_required(VERSION 3.18)
project(librediffusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE 1)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
# 75: 20xx
# 86: 30xx
# 87: jetson orin
# 89: 40xx
# 120: 50xx
set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 90a-real 100-real 103-real 110-real 120)
set(CMAKE_CUDA_FLAGS "-use_fast_math ")#-arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 -arch=sm_90 -arch=sm_90a -arch=sm_100 -arch=sm_103 -arch=sm_110 -arch=sm_120 ")
if(NOT MSVC)
  set(CMAKE_C_FLAGS -Wno-deprecated )
  set(CMAKE_CXX_FLAGS "-Wno-deprecated -Wno-deprecated-declarations -flto -O3 -ffast-math")
  set(CMAKE_EXE_LINKER_FLAGS "-flto -O3 -ffast-math")
else()
  set(CMAKE_CXX_FLAGS "/EHsc")
  set(CMAKE_CXX_FLAGS "/O2 /GL /fp:fast /arch:AVX2 /LTCG /EHsc")
  set(CMAKE_EXE_LINKER_FLAGS "/O2 /GL /fp:fast /arch:AVX2 /LTCG")
endif()

# Fetch Corrosion for Rust integration and Catch2 for testing
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

# Fetch Catch2 for unit testing
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v2.13.10
)
FetchContent_MakeAvailable(Catch2)

# Import the CLIP tokenizer Rust library
corrosion_import_crate(
    MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/clip-tokenizer-c/Cargo.toml"
    PROFILE release
    CRATE_TYPES staticlib
)

# Find required packages
find_package(CUDAToolkit REQUIRED)

option(TENSORRT_RTX "Use trt rtx" OFF)
if(WIN32)
if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TENSORRT_INCLUDE_DIR must be set")
endif()
if(NOT TENSORRT_LIB_DIR)
    message(FATAL_ERROR "TENSORRT_LIB_DIR must be set")
endif()
else()
find_path(TENSORRT_INCLUDE_DIR "NvInferRuntime.h")
endif()

# TensorRT paths
if(TENSORRT_RTX)
  set(TENSORRT_LIBRARIES 
      ${TENSORRT_LIB_DIR}/libtensorrt_rtx.so
      ${TENSORRT_LIB_DIR}/libtensorrt_onnxparser_rtx.so
  )
else()
  if(WIN32)
    set(TENSORRT_LIBRARIES
        "${TENSORRT_LIB_DIR}/nvinfer_10.lib"
    )
  else()
    find_library(TENSORRT_LIBRARIES NAMES nvinfer)
  endif()
endif()

# !! export POLYGRAPHY_USE_TENSORRT_RTX=1
# export USE_TRT_RTX=1
# etc.

# Add executable
add_library(librediffusion_cuda STATIC 
    src/kernels.cu
    src/nchw.cu
)
target_compile_options(librediffusion_cuda PRIVATE  -allow-unsupported-compiler)
target_include_directories(librediffusion_cuda PUBLIC
    src
    src/clip-tokenizer-c
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
#    "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/cccl/"
    3rdparty/cutlass/include
    3rdparty/cutlass/tools/util/include
)
add_library(librediffusion SHARED
    src/librediffusion.cpp
    src/librediffusion.embeddings.cpp
    src/librediffusion.images.cpp
    src/librediffusion.init.cpp
    src/librediffusion.unet.cpp
    src/librediffusion.v2v.cpp
    src/librediffusion.vae.cpp
    src/librediffusion.workflows.cpp
    src/librediffusion.hpp
    src/tensorrt_wrappers.cpp
    src/tensorrt_wrappers.hpp
    src/cuda_tensor.hpp 
    src/cuda_tensor.cpp
    src/pcg.hpp

    src/librediffusion_c.cpp
    src/librediffusion_c.h
) 
set_target_properties(librediffusion PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_compile_definitions(librediffusion PRIVATE $<$<BOOL:${WIN32}>:SD_BUILD_DLL>)
target_include_directories(librediffusion PUBLIC
    src
    src/clip-tokenizer-c
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/cccl/"
    3rdparty/cutlass/include
    3rdparty/cutlass/tools/util/include
)

target_link_libraries(librediffusion PUBLIC
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    CUDA::nppig
    CUDA::curand
    clip_tokenizer_c
    librediffusion_cuda
)
