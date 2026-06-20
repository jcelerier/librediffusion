#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_nhwc_to_nchw_rgb(
    int n, int h, int w, const float* in_rgb, float* out_rgb, void* stream);
void launch_nhwc_to_nchw_rgb(
    int n, int h, int w, const __half* in_rgb, __half* out_rgb, void* stream);
void launch_nchw_to_nhwc_rgb(
    int n, int h, int w, const float* in_rgb, float* out_rgb, void* stream);
void launch_nchw_to_nhwc_rgb(
    int n, int h, int w, const __half* in_rgb, __half* out_rgb, void* stream);
