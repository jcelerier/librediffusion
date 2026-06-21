/**
 * "img2img-turbo" here means the specific project github.com/GaParmar/img2img-turbo
 * (pix2pix-turbo / CycleGAN-turbo, "One-Step Image Translation with Text-to-Image Models").
 * It is NOT a generic SD-turbo img2img workflow — that is the ordinary img2img path
 * (librediffusion_img2img). This is a distinct architecture: one-step, no CFG, no scheduler,
 * with 4 VAE encoder->decoder skip connections baked into the VAE engines.
 *
 * C++ pipeline — SD-turbo one-step img2img with 4 VAE encoder->decoder skip connections.
 *
 * Engines (fp32 I/O; the skip 1x1 convs, /scaling_factor and gamma are baked inside the decoder):
 *   vae_encoder : image[1,3,H,W] -> latent[1,4,h,w] (=mode*sf), enc0[1,128,H,W], enc1[1,128,H/2,W/2],
 *                                   enc2[1,256,H/4,W/4], enc3[1,512,h,w]
 *   unet        : latent[1,4,h,w], ehs[1,77,1024] (t=999 baked) -> model_pred[1,4,h,w]
 *   vae_decoder : latent_scaled[1,4,h,w], s0=enc3, s1=enc2, s2=enc1, s3=enc0 -> image[1,3,H,W] in [-1,1]
 *
 * Flow: encode -> unet -> closed-form 1-step x0 -> decode(x0, skips REVERSED). No CFG, no multi-step,
 * no noise add. Callers pass DEVICE fp32 pointers.
 */
#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "NvInfer.h"
#include "cuda_tensor.hpp"

namespace librediffusion
{
class CachedTensorRTEngine;

struct Img2ImgTurboEngines
{
  std::string unet;
  std::string vae_encoder;
  std::string vae_decoder;
};

class Img2ImgTurboPipeline
{
public:
  explicit Img2ImgTurboPipeline(const Img2ImgTurboEngines& paths);
  ~Img2ImgTurboPipeline();
  Img2ImgTurboPipeline(const Img2ImgTurboPipeline&) = delete;
  Img2ImgTurboPipeline& operator=(const Img2ImgTurboPipeline&) = delete;

  // DEVICE fp32: image[1,3,H,W], ehs[1,77,1024] -> out[1,3,H,W] in [-1,1].
  void forward(const float* image, const float* ehs, float* out, cudaStream_t stream);

  // HOST entry point (mirrors flux2_stream_frame): RGBA8 NHWC [H,W,4] in + host ehs[1,77,1024]
  // -> RGBA8 NHWC [H,W,4] out. Does the host<->device copies + RGBA<->CHW conversion internally so
  // callers (e.g. the score node, which has no CUDA) only deal in plain bytes. Synchronizes `stream`.
  void forward_rgba(
      const unsigned char* in_rgba, const float* ehs_host, unsigned char* out_rgba,
      cudaStream_t stream);

  // Like forward_rgba but the embedding is a DEVICE fp16 [1,77,1024] (e.g. straight from
  // clip_compute_embeddings); converted to the persistent fp32 ehs buffer internally.
  void forward_rgba_dev(
      const unsigned char* in_rgba, const void* ehs_dev_fp16, unsigned char* out_rgba,
      cudaStream_t stream);

  // sd-turbo scheduler alphas_cumprod[999] (the 1-step constant). Override if a model differs.
  // Clamp to (0,1] so the kernel's sqrtf(acp)/sqrtf(1-acp) can never hit a div-by-zero / NaN.
  void set_alpha_cumprod(float a) { acp_ = a < 1e-6f ? 1e-6f : (a > 1.0f ? 1.0f : a); }

private:
  std::shared_ptr<CachedTensorRTEngine> e_enc_, e_unet_, e_dec_;
  std::unique_ptr<nvinfer1::IExecutionContext> c_enc_, c_unet_, c_dec_;
  std::unique_ptr<CUDATensor<float>> latent_, enc0_, enc1_, enc2_, enc3_, model_pred_, x0_;
  // forward_rgba() persistent scratch (no per-frame allocation): device RGBA8 in/out + the fp32
  // CHW image / ehs / out the device forward() consumes.
  std::unique_ptr<CUDATensor<unsigned char>> rgba_in_, rgba_out_;
  std::unique_ptr<CUDATensor<float>> image_, ehs_, out_;

  float acp_ = 0.00466009508818388f; // sd-turbo alphas_cumprod[999]
  int H_ = 512, W_ = 512, lh_ = 64, lw_ = 64;
  void alloc();
  // shared body of forward_rgba / forward_rgba_dev (image upload+convert, forward, out download);
  // assumes ehs_ is already populated.
  void run_rgba(const unsigned char* in_rgba, unsigned char* out_rgba, cudaStream_t stream);
};

} // namespace librediffusion
