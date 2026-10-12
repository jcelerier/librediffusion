# Standalone build for the C++ validation harness.
# It links against the librediffusion shared library + the C API header. The library
# itself is built by the parent module (or pointed at via LIBREDIFFUSION_BUILD_DIR).
cmake_minimum_required(VERSION 3.16)
project(librediffusion_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where liblibrediffusion.so lives (defaults to the TRT11 build dir set up earlier).
if(NOT LIBREDIFFUSION_BUILD_DIR)
  set(LIBREDIFFUSION_BUILD_DIR "/media/data1/trt-install/lrd-build")
endif()

# C API header lives in ../src
set(LRD_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_executable(harness harness.cpp)
target_include_directories(harness PRIVATE
  ${LRD_SRC_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library(LRD_LIB
  NAMES librediffusion
  PATHS ${LIBREDIFFUSION_BUILD_DIR}
  NO_DEFAULT_PATH
)
if(NOT LRD_LIB)
  message(FATAL_ERROR "liblibrediffusion.so not found in ${LIBREDIFFUSION_BUILD_DIR}; build the module first or pass -DLIBREDIFFUSION_BUILD_DIR=")
endif()

# Transitive deps of liblibrediffusion.so: TensorRT 11 + CUDA runtime. Default to the
# same locations used to build the module (see memory cpp-harness-design / validation-and-export-scope).
if(NOT TENSORRT_LIB_DIR)
  set(TENSORRT_LIB_DIR "/media/data1/trt-install/trt11-lib")
endif()
if(NOT CUDA_LIB_DIR)
  set(CUDA_LIB_DIR "/usr/local/cuda-13.3/targets/x86_64-linux/lib")
endif()

find_package(CUDAToolkit QUIET)

target_link_libraries(harness PRIVATE ${LRD_LIB})
target_link_directories(harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(harness PRIVATE nvinfer cudart)

# rpath so the harness finds the .so + its TRT/CUDA deps at runtime. The TRT libs are
# the unversioned symlinks in TENSORRT_LIB_DIR pointing at the venv's libnvinfer.so.11.
set(_rpaths
  "${LIBREDIFFUSION_BUILD_DIR}"
  "${TENSORRT_LIB_DIR}"
  "${CUDA_LIB_DIR}"
)
set_target_properties(harness PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# Bit-exactness probe for the shared PCG32 randn (compared against the Python replica).
add_executable(randn_dump randn_dump.cpp)
target_include_directories(randn_dump PRIVATE ${LRD_SRC_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
# launch_randn_fp16 lives in the CUDA static lib (liblibrediffusion_cuda.a), not the .so.
find_library(LRD_CUDA_LIB NAMES librediffusion_cuda PATHS ${LIBREDIFFUSION_BUILD_DIR} NO_DEFAULT_PATH)
target_link_directories(randn_dump PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(randn_dump PRIVATE ${LRD_CUDA_LIB} cudart)
set_target_properties(randn_dump PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# On-device IP-Adapter image-encoder harness (CLIPImageEncoderWrapper vs golden tokens).
add_executable(ipadapter_image_harness ipadapter_image_harness.cpp)
if(NOT TENSORRT_INCLUDE_DIR)
  set(TENSORRT_INCLUDE_DIR "/media/data1/trt-install/TensorRT-oss-11.0/include")
endif()
target_include_directories(ipadapter_image_harness PRIVATE
  ${LRD_SRC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  ${TENSORRT_INCLUDE_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
target_link_directories(ipadapter_image_harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
# The C++ wrapper symbols are hidden in the .so (C-API only is exported), so link the librediffusion
# object files + the CUDA static lib directly. Glob all built .o from the module build.
file(GLOB LRD_OBJS "${LIBREDIFFUSION_BUILD_DIR}/CMakeFiles/librediffusion.dir/src/*.o")
find_library(LRD_CUDA_LIB2 NAMES librediffusion_cuda PATHS ${LIBREDIFFUSION_BUILD_DIR} NO_DEFAULT_PATH)
find_library(LRD_TOK_LIB NAMES clip_tokenizer_c PATHS ${LIBREDIFFUSION_BUILD_DIR} NO_DEFAULT_PATH)
target_link_libraries(ipadapter_image_harness PRIVATE
  ${LRD_OBJS} ${LRD_CUDA_LIB2} ${LRD_TOK_LIB} nvinfer cudart)
set_target_properties(ipadapter_image_harness PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# FLUX.2-klein-4B Phase 2 validation harness (drives librediffusion_flux2_* C-API vs Phase-0 golden).
add_executable(klein_harness klein_harness.cpp)
target_include_directories(klein_harness PRIVATE
  ${LRD_SRC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
target_link_directories(klein_harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(klein_harness PRIVATE ${LRD_LIB} cudart)
set_target_properties(klein_harness PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# FLUX.2-klein-4B Phase 4 streaming validation harness (chat-template ids + cached encoder + stream).
add_executable(klein_stream_harness klein_stream_harness.cpp)
target_include_directories(klein_stream_harness PRIVATE
  ${LRD_SRC_DIR} ${LRD_SRC_DIR}/clip-tokenizer-c ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
target_link_directories(klein_stream_harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(klein_stream_harness PRIVATE ${LRD_LIB} cudart)
set_target_properties(klein_stream_harness PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# Prompt probe: does the prompt actually change klein output? (MAD between prompt outputs)
add_executable(klein_prompt_probe klein_prompt_probe.cpp)
target_include_directories(klein_prompt_probe PRIVATE
  ${LRD_SRC_DIR} ${LRD_SRC_DIR}/clip-tokenizer-c ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
target_link_directories(klein_prompt_probe PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(klein_prompt_probe PRIVATE ${LRD_LIB} cudart)
set_target_properties(klein_prompt_probe PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# FPS harness: reproduces the node's sync / sync-interp / async klein paths headless (no Qt/score)
add_executable(klein_fps_harness klein_fps_harness.cpp)
target_include_directories(klein_fps_harness PRIVATE
  ${LRD_SRC_DIR} ${LRD_SRC_DIR}/clip-tokenizer-c ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
target_link_directories(klein_fps_harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(klein_fps_harness PRIVATE ${LRD_LIB} cudart)
set_target_properties(klein_fps_harness PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# SD/SDXL streaming-FPS harness (the SD counterpart to klein_fps_harness): times the
# img2img per-frame loop over a wall-clock window and reports avg FPS + pacing percentiles.
add_executable(sd_fps_harness sd_fps_harness.cpp)
target_include_directories(sd_fps_harness PRIVATE
  ${LRD_SRC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda-13.3/targets/x86_64-linux/include)
target_link_directories(sd_fps_harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(sd_fps_harness PRIVATE ${LRD_LIB} nvinfer cudart)
set_target_properties(sd_fps_harness PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

# Video img2img throughput from any SD / SD-Turbo / SDXL bundle, configured the way the node does it
# (bundle.json + the scheduler tables); no golden cell needed. Driven by bench/run_video_bench.py.
add_executable(sd_video_harness sd_video_harness.cpp)
target_include_directories(sd_video_harness PRIVATE ${LRD_SRC_DIR})
target_link_libraries(sd_video_harness PRIVATE ${LRD_LIB})
set_target_properties(sd_video_harness PROPERTIES BUILD_RPATH "${_rpaths}" INSTALL_RPATH "${_rpaths}")

# img2img-turbo skip-VAE validation driver
add_executable(img2img_turbo_harness img2img_turbo_harness.cpp)
target_include_directories(img2img_turbo_harness PRIVATE ${LRD_SRC_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_LIB_DIR}/../include)
target_link_libraries(img2img_turbo_harness PRIVATE ${LRD_LIB})
target_link_directories(img2img_turbo_harness PRIVATE ${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})
target_link_libraries(img2img_turbo_harness PRIVATE nvinfer cudart)
set_target_properties(img2img_turbo_harness PROPERTIES BUILD_RPATH "${_rpaths}" INSTALL_RPATH "${_rpaths}")
