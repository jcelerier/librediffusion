/**
 * FLUX.2-klein-4B TensorRT wrappers (bf16 I/O).
 *
 * Engines (bf16 + fp8 transformer):
 *  - transformer: hidden_states[B,Lp,128] bf16, encoder_hidden_states[B,512,7680] bf16,
 *                 timestep[B] fp32, img_ids[B,Lp,4] fp32, txt_ids[B,512,4] fp32 -> velocity[B,Lp,128] bf16
 *  - qwen3_encoder: input_ids[B,512] int64, attention_mask[B,512] int64 -> encoder_hidden_states[B,512,7680] bf16
 *  - vae_encoder: image[B,3,H,W] bf16 -> latent[B,32,H/8,W/8] bf16
 *  - vae_decoder: latent[B,32,h,w] bf16 -> image[B,3,8h,8w] bf16
 *
 * Callers pass DEVICE pointers; the wrappers bind them directly (no internal staging copies).
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

#include "NvInfer.h"

namespace librediffusion
{
class CachedTensorRTEngine;

class Flux2TransformerWrapper
{
public:
  explicit Flux2TransformerWrapper(const std::string& engine_path);
  ~Flux2TransformerWrapper();
  Flux2TransformerWrapper(const Flux2TransformerWrapper&) = delete;
  Flux2TransformerWrapper& operator=(const Flux2TransformerWrapper&) = delete;

  // All pointers DEVICE. Lp = packed image tokens, Lt = text tokens (512), D = 7680.
  void forward(
      const __nv_bfloat16* hidden_states, const __nv_bfloat16* encoder_hidden_states,
      const float* timestep, const float* img_ids, const float* txt_ids,
      __nv_bfloat16* velocity, int batch, int Lp, int Lt, int D, cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void loadEngine(const std::string& engine_path);
};

// FLUX.3-image-klein transformer (flow map, one sample):
//   img[1,Li,128] cond[1,Lc,128] txt[1,Lt,20480] bf16, rope_{img,cond,txt}[1,L,64,4] fp32, t[1] r[1] fp32
//   -> velocity[1,Li,128] bf16. Lc may be 0 (text-to-image).
class Flux3TransformerWrapper
{
public:
  explicit Flux3TransformerWrapper(const std::string& engine_path);
  ~Flux3TransformerWrapper();
  Flux3TransformerWrapper(const Flux3TransformerWrapper&) = delete;
  Flux3TransformerWrapper& operator=(const Flux3TransformerWrapper&) = delete;

  void forward(
      const __nv_bfloat16* img, const __nv_bfloat16* cond, const __nv_bfloat16* txt,
      const float* rope_img, const float* rope_cond, const float* rope_txt, const float* t,
      const float* r, __nv_bfloat16* velocity, int Li, int Lc, int Lt, cudaStream_t stream);

  // profile-0 max of the reference ("cond") sequence
  int max_cond_tokens() const;

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

class Qwen3EncoderWrapper
{
public:
  explicit Qwen3EncoderWrapper(const std::string& engine_path);
  ~Qwen3EncoderWrapper();
  Qwen3EncoderWrapper(const Qwen3EncoderWrapper&) = delete;
  Qwen3EncoderWrapper& operator=(const Qwen3EncoderWrapper&) = delete;

  // input_ids/attention_mask DEVICE int64 [B,Lt]; out encoder_hidden_states DEVICE bf16 [B,Lt,7680]
  void forward(
      const int64_t* input_ids, const int64_t* attention_mask, __nv_bfloat16* encoder_hidden_states,
      int batch, int Lt, int D, cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void loadEngine(const std::string& engine_path);
};

class KleinVAEEncoderWrapper
{
public:
  explicit KleinVAEEncoderWrapper(const std::string& engine_path);
  ~KleinVAEEncoderWrapper();
  KleinVAEEncoderWrapper(const KleinVAEEncoderWrapper&) = delete;
  KleinVAEEncoderWrapper& operator=(const KleinVAEEncoderWrapper&) = delete;

  // image DEVICE bf16 [B,3,H,W] -> latent DEVICE bf16 [B,32,H/8,W/8]
  void encode(const __nv_bfloat16* image, __nv_bfloat16* latent, int batch, int H, int W,
              cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void loadEngine(const std::string& engine_path);
};

class KleinVAEDecoderWrapper
{
public:
  explicit KleinVAEDecoderWrapper(const std::string& engine_path);
  ~KleinVAEDecoderWrapper();
  KleinVAEDecoderWrapper(const KleinVAEDecoderWrapper&) = delete;
  KleinVAEDecoderWrapper& operator=(const KleinVAEDecoderWrapper&) = delete;

  // latent DEVICE bf16 [B,32,h,w] -> image DEVICE bf16 [B,3,8h,8w]
  void decode(const __nv_bfloat16* latent, __nv_bfloat16* image, int batch, int h, int w,
              cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void loadEngine(const std::string& engine_path);
};

} // namespace librediffusion
