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
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}


void LibreDiffusionPipeline::reseed(int64_t rs)
{
  config_.seed = rs;
  // Initialize noise buffers on host, then copy to device
  // init_noise: random normal distribution (using Box-Muller transform)
  {
    size_t noise_size = init_noise_->size();
    static thread_local std::vector<__half> host_noise;
    host_noise.resize(noise_size);

    // Initialize PCG with seed
    pcg rng;
    rng.seed(config_.seed, config_.seed + 1);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    // Generate random normal values
    for(size_t i = 0; i < noise_size; ++i)
    {
      host_noise[i] = normal_dist(rng);
    }

    // Copy to device
    cudaMemcpyAsync(
        init_noise_->data(), host_noise.data(), noise_size * sizeof(__half),
        cudaMemcpyHostToDevice, stream_);
  }

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

}
