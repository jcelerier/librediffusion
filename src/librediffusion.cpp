#include "librediffusion.hpp"

#include "tensorrt_wrappers.hpp"
#include "kernels.hpp"
#include "nchw.hpp"

#include <cmath>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "pcg.hpp"
namespace librediffusion
{

template class CUDATensor<__half>;
template class CUDATensor<float>;
template class CUDATensor<int>;

LibreDiffusionPipeline::LibreDiffusionPipeline(const LibreDiffusionConfig& config)
    : config_(config)
{
  init_cuda();

  init_npp();

  init_engines();

  init_buffers();

  reseed(config_.seed);
}

LibreDiffusionPipeline::~LibreDiffusionPipeline()
{
  if(graph_exec_)
    cudaGraphExecDestroy(graph_exec_);
  if(graph_)
    cudaGraphDestroy(graph_);
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}


void LibreDiffusionPipeline::reseed(int64_t rs)
{
  config_.seed = rs;
  // init_noise: the SHARED counter-based PCG32 Gaussian (launch_randn_fp16) — bit-identical to the
  // Python pcg32_randn (deterministic_noise.py) and the txt2img path, so img2img generated noise
  // matches python<->C++ element-for-element. (Previously this was a host-side std::normal_distribution
  // over a single pcg stream seeded (seed, seed+1), which is a DIFFERENT RNG and did NOT match the
  // shared per-element counter+Box-Muller kernel — img2img noise silently diverged from the goldens.)
  launch_randn_fp16(init_noise_->data(), config_.seed, (int)init_noise_->size(), stream_);

  // stock_noise: zeros (updated during denoising, matches Python torch.zeros_like)
  cudaMemsetAsync(
      stock_noise_->data(), 0, stock_noise_->size() * sizeof(__half), stream_);

  // Initialize temporal state noise buffers if in V2V mode
  if(config_.mode == PipelineMode::TEMPORAL_V2V)
  {
    if(temporal_state_.randn_noise)
    {
      // Copy first slice of init_noise to randn_noise and warp_noise
      int single_latent_size = 1 * 4 * config_.latent_height * config_.latent_width;
      cudaMemcpyAsync(
          temporal_state_.randn_noise->data(), init_noise_->data(),
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream_);
      cudaMemcpyAsync(
          temporal_state_.warp_noise->data(), init_noise_->data(),
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream_);
    }
  }

  cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::set_guidance_scale(float g)
{
  config_.guidance_scale = g;
}

void LibreDiffusionPipeline::set_delta(float g)
{
  config_.delta = g;
}

void LibreDiffusionPipeline::prepare_scheduler(
    std::span<float> timesteps, std::span<float> alpha_prod_t_sqrt,
    std::span<float> beta_prod_t_sqrt, std::span<float> c_skip, std::span<float> c_out)
{
  // Copy to host side
  alpha_prod_t_sqrt_host_.assign(alpha_prod_t_sqrt.begin(), alpha_prod_t_sqrt.end());
  beta_prod_t_sqrt_host_.assign(beta_prod_t_sqrt.begin(), beta_prod_t_sqrt.end());
  c_skip_host_.assign(c_skip.begin(), c_skip.end());
  c_out_host_.assign(c_out.begin(), c_out.end());

  // Allocate and copy scheduler parameters
  alpha_prod_t_sqrt_ = std::make_unique<CUDATensor<float>>(config_.denoising_steps);
  beta_prod_t_sqrt_ = std::make_unique<CUDATensor<float>>(config_.denoising_steps);
  c_skip_ = std::make_unique<CUDATensor<float>>(config_.denoising_steps);
  c_out_ = std::make_unique<CUDATensor<float>>(config_.denoising_steps);
  sub_timesteps_ = std::make_unique<CUDATensor<float>>(config_.denoising_steps);

  // Copy scheduler parameters to device

  cudaMemcpy(
      alpha_prod_t_sqrt_->data(), alpha_prod_t_sqrt_host_.data(),
      config_.denoising_steps * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(
      beta_prod_t_sqrt_->data(), beta_prod_t_sqrt_host_.data(),
      config_.denoising_steps * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_skip_->data(), c_skip_host_.data(), config_.denoising_steps * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      c_out_->data(), c_out_host_.data(), config_.denoising_steps * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      sub_timesteps_->data(), timesteps.data(), config_.denoising_steps * sizeof(float),
      cudaMemcpyHostToDevice);

  cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::set_init_noise(const __half* noise)
{
  // Copy noise from Python to our internal buffer
  size_t noise_size = init_noise_->size();
  cudaMemcpyAsync(
      init_noise_->data(), noise, noise_size * sizeof(__half), cudaMemcpyDeviceToDevice,
      stream_);
  cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::set_controlnet_cond(int index, const __half* cond, int img_h, int img_w)
{
  // Store ONE [3, img_h, img_w] row (device, [0,1]); tiled to unet_batch_size at inference time.
  if(index < 0 || index >= (int)controlnet_cond_.size())
    throw std::runtime_error("set_controlnet_cond: index out of range");
  size_t n = (size_t)3 * img_h * img_w;
  // Update IN PLACE (allocate only if absent/too small). Reallocating every frame moves the device
  // address, which silently breaks CUDA-graph replay: the captured cond-tiling D2D copy bakes the
  // capture-time address and would read a stale buffer forever (observed as "graph bakes the cond").
  if(!controlnet_cond_[index] || controlnet_cond_[index]->size() < n)
    controlnet_cond_[index] = std::make_unique<CUDATensor<__half>>(n);
  cudaMemcpyAsync(controlnet_cond_[index]->data(), cond, n * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream_);
  cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::set_controlnet_cond_rgba(
    int index, const uint8_t* cpu_rgba, int img_h, int img_w)
{
  // Convenience path: host RGBA uint8 [H,W,4] -> RGB fp16 NCHW [3,H,W] in [0,1] on-device.
  if(index < 0 || index >= (int)controlnet_cond_.size())
    throw std::runtime_error("set_controlnet_cond_rgba: index out of range");
  size_t rgba_n = (size_t)img_h * img_w * 4;
  if(!controlnet_rgba_tmp_ || controlnet_rgba_tmp_->size() < rgba_n)
    controlnet_rgba_tmp_ = std::make_unique<CUDATensor<uint8_t>>(rgba_n);
  cudaMemcpyAsync(controlnet_rgba_tmp_->data(), cpu_rgba, rgba_n * sizeof(uint8_t),
                  cudaMemcpyHostToDevice, stream_);
  size_t n = (size_t)3 * img_h * img_w;
  // In place (see set_controlnet_cond): a stable device address is required for CUDA-graph replay.
  if(!controlnet_cond_[index] || controlnet_cond_[index]->size() < n)
    controlnet_cond_[index] = std::make_unique<CUDATensor<__half>>(n);
  launch_rgba_to_rgb_chw_01_fp16(controlnet_rgba_tmp_->data(), controlnet_cond_[index]->data(),
                                 1, img_h, img_w, stream_);
  cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::set_controlnet_scale(int index, float scale)
{
  if(index < 0 || index >= (int)controlnet_scales_.size())
    throw std::runtime_error("set_controlnet_scale: index out of range");
  controlnet_scales_[index] = scale;
}

void LibreDiffusionPipeline::set_ipadapter_tokens(
    const __half* pos, const __half* neg, int num_tokens, int dim)
{
  size_t n = (size_t)num_tokens * dim;
  ipadapter_num_tokens_ = num_tokens;
  ipadapter_tokens_pos_ = std::make_unique<CUDATensor<__half>>(n);
  cudaMemcpyAsync(ipadapter_tokens_pos_->data(), pos, n * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream_);
  if(neg)
  {
    ipadapter_tokens_neg_ = std::make_unique<CUDATensor<__half>>(n);
    cudaMemcpyAsync(ipadapter_tokens_neg_->data(), neg, n * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream_);
  }
  else
  {
    ipadapter_tokens_neg_.reset();  // cfg-none/self use pos for all rows
  }
  cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::set_ipadapter_image(
    const uint8_t* cpu_rgba, int img_h, int img_w)
{
  if(!ipadapter_image_encoder_)
    throw std::runtime_error(
        "set_ipadapter_image: no on-device image encoder configured (set "
        "ipadapter_image_encoder_path + ipadapter_image_proj_path before init)");

  int N = ipadapter_image_encoder_->numTokens();
  int dim = ipadapter_image_encoder_->tokenDim();
  size_t n = (size_t)N * dim;

  // Encode directly into the persistent pos/neg token buffers (allocate if needed).
  if(!ipadapter_tokens_pos_ || ipadapter_tokens_pos_->size() < n)
    ipadapter_tokens_pos_ = std::make_unique<CUDATensor<__half>>(n);
  if(!ipadapter_tokens_neg_ || ipadapter_tokens_neg_->size() < n)
    ipadapter_tokens_neg_ = std::make_unique<CUDATensor<__half>>(n);

  ipadapter_image_encoder_->encodeImage(
      cpu_rgba, img_h, img_w, ipadapter_tokens_pos_->data(),
      ipadapter_tokens_neg_->data(), stream_);
  ipadapter_num_tokens_ = N;
}

void LibreDiffusionPipeline::set_ipadapter_scale(float scale)
{
  int n = unet_ ? unet_->numIpLayers() : 0;
  if(n <= 0) n = 16;  // SD1.5 base default if not yet known
  ipadapter_scale_vec_.assign(n, scale);
}

void LibreDiffusionPipeline::set_ipadapter_scale_vector(const float* per_layer, int num_ip_layers)
{
  ipadapter_scale_vec_.assign(per_layer, per_layer + num_ip_layers);
}

}
