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
# Overridable via -DCMAKE_CUDA_ARCHITECTURES=... (e.g. CI builds a single arch for speed); the cache
# default ships every supported arch.
set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;90a-real;100-real;103-real;110-real;120"
    CACHE STRING "CUDA architectures to build")
set(CMAKE_CUDA_FLAGS "-use_fast_math ")#-arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 -arch=sm_90 -arch=sm_90a -arch=sm_100 -arch=sm_103 -arch=sm_110 -arch=sm_120 ")
if(MSVC)
  # CUDA 13's CCCL headers reject MSVC's traditional preprocessor (fatal error C1189). Use the
  # conforming preprocessor for both the host C++ and the nvcc host pass.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
endif()

# Boost is header-only here (model_cache.hpp uses boost::concurrent_flat_map, needs Boost >= 1.83).
# Normally found as a system include; CI passes its location explicitly (portable across gcc/MSVC,
# unlike CPLUS_INCLUDE_PATH which MSVC ignores).
if(BOOST_INCLUDE_DIR)
  include_directories(${BOOST_INCLUDE_DIR})
endif()
option(LRD_COVERAGE "Build host C++ with gcov coverage instrumentation" OFF)
if(NOT MSVC)
  set(CMAKE_C_FLAGS -Wno-deprecated )
  if(LRD_COVERAGE)
    # Coverage build: no LTO (breaks gcno mapping), no fast-math, -O0 for exact branch coverage.
    # nvcc-compiled .cu files use CMAKE_CUDA_FLAGS so they stay un-instrumented (gcov can't read device code anyway).
    set(CMAKE_CXX_FLAGS "-Wno-deprecated -Wno-deprecated-declarations -O0 -g --coverage -fprofile-update=atomic -fno-inline")
    set(CMAKE_EXE_LINKER_FLAGS "--coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "--coverage")
  else()
    set(CMAKE_CXX_FLAGS "-Wno-deprecated -Wno-deprecated-declarations -flto -O3 -ffast-math")
    set(CMAKE_EXE_LINKER_FLAGS "-flto -O3 -ffast-math")
  endif()
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

# NOTE: the Qwen tokenizer (FLUX.2-klein-4B) lives INSIDE clip-tokenizer-c so there is a
# single Rust staticlib (two Rust staticlibs would duplicate the Rust std runtime at link time).
# The qwen_tokenizer_* C symbols are exported by clip_tokenizer_c.

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
  # Standard TensorRT (the project's runtime; TENSORRT_RTX is abandoned). Locate libnvinfer in
  # TENSORRT_LIB_DIR: libnvinfer.so on Linux, nvinfer_<major>.lib on Windows. (Pass TENSORRT_LIBRARIES
  # directly to override.)
  find_library(TENSORRT_LIBRARIES NAMES nvinfer nvinfer_11 nvinfer_10
               HINTS ${TENSORRT_LIB_DIR} ${TENSORRT_LIB_DIR}/lib)
endif()

# !! export POLYGRAPHY_USE_TENSORRT_RTX=1
# export USE_TRT_RTX=1
# etc.

# Add executable
add_library(librediffusion_cuda STATIC
    src/kernels.cu
    src/kernels_klein.cu
    src/kernels_rife.cu
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
    src/librediffusion.flux2.cpp
    src/librediffusion.flux2.hpp
    src/librediffusion.hpp
    src/tensorrt_wrappers.cpp
    src/tensorrt_wrappers.hpp
    src/tensorrt_wrappers_klein.cpp
    src/tensorrt_wrappers_klein.hpp
    src/tensorrt_wrappers_rife.cpp
    src/tensorrt_wrappers_rife.hpp
    src/librediffusion.rife.cpp
    src/librediffusion.rife.hpp
    src/librediffusion_c.rife.cpp
    src/librediffusion_c.flux2.cpp
    src/cuda_tensor.hpp 
    src/cuda_tensor.cpp
    src/pcg.hpp

    src/librediffusion_c.cpp
    src/librediffusion_c.h
) 
set_target_properties(librediffusion PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_compile_definitions(librediffusion PRIVATE $<$<BOOL:${WIN32}>:LIBREDIFFUSION_BUILD_DLL>)
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

if(WIN32)
  # System libs the Rust staticlib (clip_tokenizer_c) pulls in but doesn't propagate as import deps:
  # BCryptGenRandom (bcrypt), plus the usual Rust-std-on-Windows set.
  target_link_libraries(librediffusion PUBLIC bcrypt ntdll userenv ws2_32 advapi32)
endif()
