cmake_minimum_required(VERSION 3.18)
project(streamdiffusion_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)
# Import the CLIP tokenizer Rust library
corrosion_import_crate(
    MANIFEST_PATH "clip-tokenizer-c/Cargo.toml"
    PROFILE release
    CRATE_TYPES staticlib
)

# Find packages
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# PyBind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_CMAKE_DIR})

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090

# TensorRT-RTX paths
set(TENSORRT_RTX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../TensorRT-RTX-1.1.1.26")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${TENSORRT_RTX_ROOT}/include
)

# Link to existing CUDA kernels library and TensorRT-RTX
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${TENSORRT_RTX_ROOT}/lib
)

# Source files
set(SOURCES
    stream_diffusion.cpp
    tensorrt_wrappers.cpp
    bindings.cpp
    kernels.cu
)

# Create PyBind11 module
pybind11_add_module(streamdiffusion_cpp ${SOURCES})

# Link libraries
target_link_libraries(streamdiffusion_cpp PRIVATE
    streamdiffusion_cuda  # Our existing CUDA kernels
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    tensorrt_rtx
)

# Set output directory
set_target_properties(streamdiffusion_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Compiler flags
target_compile_options(streamdiffusion_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)
