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

// StreamV2V ToMe: bipartite soft matching (deterministic even/odd split) + scatter-mean merge.
// cat_{k,v,o}: [B, N, h] (N even) = concat(bank, new); dst_{k,v,o}: [B, N/2, h] = compacted bank.
void launch_tome_merge(
    const void* cat_k, const void* cat_v, const void* cat_o,
    void* dst_k, void* dst_v, void* dst_o,
    int B, int N, int h, void* stream_ptr);

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

// ControlNet control image: RGBA uint8 (NHWC [N,H,W,4]) -> RGB fp16 NCHW [N,3,H,W] in [0,1]
// (ToTensor semantics: /255, no [-1,1] shift, channels-first). Alpha ignored.
void launch_rgba_to_rgb_chw_01_fp16(
    const void* rgba_in,
    void* rgb_out,
    int n, int h, int w,
    void* stream_ptr);

// IP-Adapter CLIP image preprocess: RGBA uint8 (NHWC [H,W,4]) -> separable Lanczos-3 resize to
// 224x224 -> CLIP normalize (mean/std) -> RGB fp16 NCHW [1,3,224,224]. Matches PIL LANCZOS + the HF
// CLIPImageProcessor recipe used by diffusers_ipadapter (cos>=0.99 vs golden). Alpha ignored.
void launch_clip_image_preprocess_fp16(
    const void* rgba_in,
    void* chw_out,
    int in_h, int in_w,
    void* stream_ptr);

// Add a control residual into an accumulator (multi-ControlNet sum): acc[i] += src[i] (fp16).
void launch_tensor_add_fp16(
    void* acc, const void* src, int n, void* stream_ptr);

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

// img2img-turbo host<->device image conversions (fp32, channels-first):
// RGBA8 NHWC [H,W,4] -> RGB fp32 CHW [3,H,W] in [0,1]; and CHW [-1,1] -> RGBA8 (alpha=255).
void launch_rgba_to_chw01_f32(const void* rgba, void* chw, int H, int W, void* stream_ptr);
void launch_chw_m1p1_to_rgba_f32(const void* chw, void* rgba, int H, int W, void* stream_ptr);

// ===== FLUX.2-klein-4B kernels (bf16) =====
void launch_klein_stack3_7680(
    const void* l0, const void* l1, const void* l2, void* out, int B, int Lt, int D, void* stream_ptr);
void launch_klein_patchify(const void* in, void* out, int B, int C, int Lh, int Lw, void* stream_ptr);
void launch_klein_unpatchify(const void* in, void* out, int B, int C, int Th, int Tw, void* stream_ptr);
void launch_klein_pack(const void* in, void* out, int B, int C, int Th, int Tw, void* stream_ptr);
void launch_klein_unpack(const void* in, void* out, int B, int C, int Th, int Tw, void* stream_ptr);
void launch_klein_bn(const void* in, void* out, const void* mean, const void* std_, int B, int C,
                     int HW, int denorm, void* stream_ptr);
void launch_klein_img_ids(void* out, int Th, int Tw, void* stream_ptr);
void launch_klein_txt_ids(void* out, int Lt, void* stream_ptr);
void launch_klein_euler_axpy(void* x, const void* v, float dt, long N, void* stream_ptr);
// out = (1-t)*a + t*b (bf16) — flow-match img2img start blend
void launch_klein_lerp(void* out, const void* a, const void* b, float t, long N, void* stream_ptr);
// inpaint per-step blend: x = mask*x + (1-mask)*(sigma_next*noise + (1-sigma_next)*ref); mask per-token [Lp]
void launch_klein_inpaint_blend(void* x, const void* ref, const void* noise, const float* mask,
                                float sigma_next, long Lp, int hidden, void* stream_ptr);
void launch_klein_randn_bf16(void* out, unsigned long long seed, long N, void* stream_ptr);
void launch_klein_bf16_to_fp32(const void* in, void* out, long N, void* stream_ptr);
void launch_klein_fp32_to_bf16(const void* in, void* out, long N, void* stream_ptr);
void launch_klein_chw_to_rgba_u8(const void* in, void* out, int H, int W, void* stream_ptr);
// Streaming / reference image:
void launch_klein_rgba_u8_to_chw_m1p1(const void* in, void* out, int H, int W, void* stream_ptr);
void launch_klein_ref_ids(void* out, int Th, int Tw, float t_offset, void* stream_ptr);
void launch_klein_concat_seq(const void* a, const void* b, void* out, long Na, long Nb, void* stream_ptr);

// ===== RIFE (IFNet) frame-interpolation helpers (fp16, model-agnostic) =====
// RGBA uint8 NHWC [H,W,4] <-> RGB fp16 CHW [3,H,W] in [0,1] (RIFE's I/O domain).
void launch_rife_rgba_to_chw01(const void* rgba, void* chw, int H, int W, void* stream_ptr);
void launch_rife_chw01_to_rgba(const void* chw, void* rgba, int H, int W, void* stream_ptr);
// Pack B (img0,img1) pairs into engine input [B,6,H,W] from prevs/nexts [B,3,H,W].
void launch_rife_pack_pairs(
    const void* prevs, const void* nexts, void* out, int B, int H, int W, void* stream_ptr);
// Interleave subdivision level: dst[2B-1] = {src[k] at even, mids[k] at odd}.
void launch_rife_interleave(
    const void* src, const void* mids, void* dst, int B, int H, int W, void* stream_ptr);

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

// img2img-turbo: closed-form 1-step DDPM x0 = (latent - sqrt(1-acp)*model_pred)/sqrt(acp). All DEVICE fp32.
void launch_ddpm_1step_x0(
    const float* latent, const float* model_pred, float acp, long N, float* out, void* stream_ptr);
