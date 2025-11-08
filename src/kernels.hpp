#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_fp16_to_fp32(const void* input, void* output, int N, void* stream_ptr);

void launch_add_noise_fp16(
    const void* original_samples, const void* noise, void* noisy_samples, float alpha,
    float beta, int N, void* stream_ptr);

void launch_add_noise_direct_fp16(
    void* original_samples, const void* noise, float alpha, float beta, int N,
    void* stream_ptr);

void launch_scheduler_step_fp16(
    const void* model_pred, const void* x_t_latent, void* output, float alpha,
    float beta, float c_skip, float c_out, int batch, int channels, int height,
    int width, void* stream_ptr);

void launch_apply_cfg_fp16(
    const void* noise_pred_uncond, const void* noise_pred_text, void* output,
    float guidance_scale, int N, void* stream_ptr);

void launch_concat(
    void** input_ptrs, int num_inputs, size_t* input_byte_sizes, void* output,
    void* stream_ptr);

void launch_ones_like_fp16(void* output, int N, void* stream_ptr);

void launch_randn_fp16(void* output, unsigned long long seed, int N, void* stream_ptr);

void launch_scalar_mul_inplace_fp16(void* data, float scalar, int N, void* stream_ptr);

void launch_scalar_div_fp16(
    const void* data, void* output, float scalar, int N, void* stream_ptr);

void launch_tensor_clone(const void* src, void* dst, size_t num_bytes, void* stream_ptr);

void launch_fp16_to_fp32(const void* input, void* output, int N, void* stream_ptr);

void launch_fp32_to_fp16(const void* input, void* output, int N, void* stream_ptr);

void launch_tensor_sub_fp16(
    const void* a, const void* b, void* output, int N, void* stream_ptr);

// StreamV2V Feature Injection Kernels
void launch_cosine_similarity(
    const void* x,
    const void* y,
    void* similarities,
    int batch, int seq_len, int cached_seq_len, int hidden_dim,
    void* stream_ptr);

void launch_nearest_neighbor(
    const void* current_features,
    const void* cached_features,
    const void* similarities,
    void* output,
    float threshold,
    int batch, int seq_len, int cached_seq_len, int hidden_dim,
    void* stream_ptr);

void launch_blend_features(
    const void* current,
    const void* neighbor,
    void* output,
    float blend_strength,
    int N,
    void* stream_ptr);

// RGBA to RGB normalization kernels
void launch_rgba_to_rgb_normalized_fp32(
    const void* rgba_in,
    void* rgb_out,
    int n, int h, int w,
    void* stream_ptr);

void launch_rgba_to_rgb_normalized_fp16(
    const void* rgba_in,
    void* rgb_out,
    int n, int h, int w,
    void* stream_ptr);

// Weighted accumulate for embedding blending: acc += weight * src
void launch_weighted_accumulate_fp16(
    void* accumulator,        // In/out: accumulated result
    const void* source,       // Input to add
    float weight,             // Scalar weight
    int N,
    void* stream_ptr);

// Zero-fill a buffer with FP16 zeros
void launch_zero_fill_fp16(void* output, int N, void* stream_ptr);

// RGB to RGBA denormalization kernels
void launch_rgb_to_rgba_denormalized_fp32(
    const void* rgb_in,
    void* rgba_out,
    int n, int h, int w,
    void* stream_ptr);

void launch_rgb_to_rgba_denormalized_fp16(
    const void* rgb_in,
    void* rgba_out,
    int n, int h, int w,
    void* stream_ptr);

// Simple implementation of launch_concat using cudaMemcpy
inline void launch_concat(
    void** input_ptrs, int num_inputs, size_t* input_byte_sizes, void* output,
    void* stream_ptr)
{
  cudaStream_t stream = (cudaStream_t)stream_ptr;
  size_t offset = 0;

  for(int i = 0; i < num_inputs; i++)
  {
    cudaMemcpyAsync(
        (char*)output + offset, input_ptrs[i], input_byte_sizes[i],
        cudaMemcpyDeviceToDevice, stream);
    offset += input_byte_sizes[i];
  }
}
