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
  // Drain the stream first: the buffers and TensorRT contexts below may still be referenced by
  // queued work, and img2img ends with an async D2H into the CALLER's host buffer.
  if(stream_)
    cudaStreamSynchronize(stream_);

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

namespace
{
inline float coeff_at(const std::vector<float>& v, int i, const char* name)
{
  if(i < 0 || (size_t)i >= v.size())
    throw std::out_of_range(
        std::string("scheduler coefficient ") + name + "[" + std::to_string(i)
        + "] out of range (schedule has " + std::to_string(v.size())
        + " entries) — prepare_scheduler was not called with a schedule of this length");
  return v[(size_t)i];
}
} // namespace

float LibreDiffusionPipeline::alpha_at(int i) const
{
  return coeff_at(alpha_prod_t_sqrt_host_, i, "alpha_prod_t_sqrt");
}
float LibreDiffusionPipeline::beta_at(int i) const
{
  return coeff_at(beta_prod_t_sqrt_host_, i, "beta_prod_t_sqrt");
}
float LibreDiffusionPipeline::c_skip_at(int i) const
{
  return coeff_at(c_skip_host_, i, "c_skip");
}
float LibreDiffusionPipeline::c_out_at(int i) const
{
  return coeff_at(c_out_host_, i, "c_out");
}

const char* LibreDiffusionPipeline::inference_readiness() const
{
  // Conditioning: every UNet forward reads prompt_embeds_->data() unconditionally.
  if(!prompt_embeds_)
    return "prompt embeddings not prepared (call librediffusion_prepare_embeds first)";

  // Scheduler: sub_timesteps_ is the device buffer every forward binds; the coefficient vectors are
  // indexed host-side, and an empty vector's data() is nullptr, so [0] is a null read.
  if(!sub_timesteps_ || alpha_prod_t_sqrt_host_.empty() || beta_prod_t_sqrt_host_.empty()
     || c_skip_host_.empty() || c_out_host_.empty())
    return "scheduler not prepared (call librediffusion_prepare_scheduler first)";

  // reinit_buffers can raise batch_size / frame_buffer_size after the schedule was uploaded, and the
  // forwards read sub_timesteps_ by that extent.
  if(sub_timesteps_->size() < (size_t)timestep_extent())
    return "the scheduler was prepared for a smaller batch (call librediffusion_prepare_scheduler "
           "again after changing batch_size or frame_buffer_size)";

  // SDXL additionally binds the pooled embeddings and the time ids on every path.
  if(config_.model_type == ModelType::SDXL_TURBO && (!text_embeds_ || !time_ids_))
    return "SDXL conditioning not prepared (call librediffusion_prepare_sdxl_conditioning first)";

  return nullptr;
}

const char* LibreDiffusionPipeline::encode_readiness() const
{
  if(!vae_encoder_)
    return "no VAE encoder (call librediffusion_pipeline_init_engines, and configure "
           "vae_encoder_path first)";
  return nullptr;
}

const char* LibreDiffusionPipeline::decode_readiness() const
{
  if(!vae_decoder_)
    return "no VAE decoder (call librediffusion_pipeline_init_engines, and configure "
           "vae_decoder_path first)";
  return nullptr;
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

  // Size everything by the arrays the caller actually passed, NOT config_.denoising_steps. The two can
  // disagree on the live-update path (the wrapper's updateScheduler pushes new coefficients without
  // syncing the pipeline's denoising_steps), and using the stale config count would memcpy past the end
  // of the *_host_ vectors (an OOB read). timesteps.size() is authoritative and self-consistent.
  const size_t n = timesteps.size();
  // Contract: all five coefficient arrays describe the same timesteps, so they must be the same length
  // (the C-API builds every span from one num_timesteps; updateScheduler fills them in one loop). Guard
  // it so a future caller mismatch is a clear error, not a silent OOB read in the memcpys below.
  // NOTE: deliberately do NOT touch config_.denoising_steps here — it sizes the batch buffers allocated
  // in init_buffers()/reinit_buffers(); mutating it on this light path (which does not reallocate) would
  // desync the step count from the buffers. A genuine step-count change goes through need_rebuild ->
  // reinit_buffers (which sets denoising_steps AND reallocates).
  if(alpha_prod_t_sqrt.size() != n || beta_prod_t_sqrt.size() != n
     || c_skip.size() != n || c_out.size() != n)
    throw invalid_dimensions_error(
        "prepare_scheduler: coefficient spans must all match timesteps.size()");
  if(n == 0)
    throw invalid_argument_error("prepare_scheduler: an empty schedule is not a schedule");
  // ...and they must match config_.denoising_steps, which is what the denoise loops iterate and what
  // init_buffers()/reinit_buffers() sized every batch buffer from. A step-count change goes through
  // reinit_buffers first, so this never fires on a legitimate live update.
  if((int)n != config_.denoising_steps)
    throw invalid_dimensions_error(
        "prepare_scheduler: schedule has " + std::to_string(n)
        + " timesteps but the pipeline is configured for "
        + std::to_string(config_.denoising_steps)
        + " denoising steps (reinit_buffers with the new step count first)");

  // NaN/inf propagate through the whole latent in one multiply; alpha == 0 is a divisor on the
  // 1-step turbo path. Timestep VALUES stay unconstrained beyond finiteness.
  // NOTE: the library is compiled with -ffast-math, so std::isfinite() and `x == 0.f` are NOT
  // usable here — the compiler may assume no NaN/inf exists. Inspect the IEEE-754 bits instead.
  auto bits = [](float f) {
    unsigned int u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  };
  auto reject_non_finite = [&](std::span<float> v, const char* name) {
    for(size_t i = 0; i < v.size(); i++)
      if((bits(v[i]) & 0x7F800000u) == 0x7F800000u) // exponent all ones => inf or NaN
        throw invalid_argument_error(
            std::string("prepare_scheduler: ") + name + "[" + std::to_string(i)
            + "] is not a finite number");
  };
  reject_non_finite(timesteps, "timesteps");
  reject_non_finite(alpha_prod_t_sqrt, "alpha_prod_t_sqrt");
  reject_non_finite(beta_prod_t_sqrt, "beta_prod_t_sqrt");
  reject_non_finite(c_skip, "c_skip");
  reject_non_finite(c_out, "c_out");
  for(size_t i = 0; i < alpha_prod_t_sqrt.size(); i++)
    if((bits(alpha_prod_t_sqrt[i]) & 0x7FFFFFFFu) == 0u) // +0.0 or -0.0
      throw invalid_argument_error(
          "prepare_scheduler: alpha_prod_t_sqrt[" + std::to_string(i)
          + "] is zero; it is a divisor on the single-step path");
  auto reuse = [&](std::unique_ptr<CUDATensor<float>>& b, size_t want) {
    if(!b || b->size() != want) { b = std::make_unique<CUDATensor<float>>(want); }
  };
  reuse(alpha_prod_t_sqrt_, n);
  reuse(beta_prod_t_sqrt_, n);
  reuse(c_skip_, n);
  reuse(c_out_, n);

  // sub_timesteps_ is the one buffer the UNet paths read by BATCH extent, not by step index: every
  // forward copies `total_batch` (or batch_size) floats out of it, which equals `n` only when
  // batch_size == denoising_steps and frame_buffer_size == 1. Hold the schedule cycled to the extent
  // the forwards actually read, so the first `n` rows stay bit-identical.
  const size_t extent = std::max(n, (size_t)timestep_extent());
  reuse(sub_timesteps_, extent);
  std::vector<float> cycled(extent);
  for(size_t i = 0; i < extent; i++)
    cycled[i] = timesteps[i % n];

  // Copy scheduler parameters to device
  cudaMemcpy(alpha_prod_t_sqrt_->data(), alpha_prod_t_sqrt_host_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(beta_prod_t_sqrt_->data(), beta_prod_t_sqrt_host_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_skip_->data(), c_skip_host_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_out_->data(), c_out_host_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(sub_timesteps_->data(), cycled.data(), extent * sizeof(float), cudaMemcpyHostToDevice);

  cudaStreamSynchronize(stream_);

  // Invalidate any captured 1-step CUDA graph. The graphed single-step body bakes the scheduler
  // coefficients (alpha/beta/c_skip/c_out) as BY-VALUE kernel arguments at capture time (they leave the
  // captured region as host scalars into launch_scheduler_step_fp16, NOT via a device buffer read), so
  // refreshing the host vectors + device buffers above does NOT change what a graph replay computes. A
  // live timestep/coefficient edit must therefore force a recapture, exactly as set_lora_scale does for
  // its baked value. capture_signature() only hashes buffer ADDRESSES (stable here), so it cannot catch a
  // value-only change — hence this explicit invalidation.
  graph_ready_ = false;
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
  // The scale reaches the captured 1-step graph via a skip-when-unchanged host-side H2D inside
  // ControlNetWrapper::forward (staged into scale_buffer_). A graph REPLAY does not call forward, so
  // that re-stage never runs and the engine keeps the capture-time scale. capture_signature() hashes
  // only buffer addresses (this scale is a host scalar), so a value change is invisible to it. Force a
  // recapture — same treatment as set_lora_scale / prepare_scheduler. Change-gated in the wrapper, so
  // one recapture per (rare) edit.
  graph_ready_ = false;
}

void LibreDiffusionPipeline::set_ipadapter_tokens(
    const __half* pos, const __half* neg, int num_tokens, int dim)
{
  size_t n = (size_t)num_tokens * dim;
  ipadapter_num_tokens_ = num_tokens;
  // Grow-only: reuse the existing device buffer when the shape is unchanged so its address stays
  // stable across frames. A moving address would silently desync a captured CUDA graph whose body
  // memcpy's from these buffers (see run_single_step_body IP path + capture_signature). Reallocate
  // only when the buffer is absent or too small.
  if(!ipadapter_tokens_pos_ || ipadapter_tokens_pos_->size() < n)
    ipadapter_tokens_pos_ = std::make_unique<CUDATensor<__half>>(n);
  cudaMemcpyAsync(ipadapter_tokens_pos_->data(), pos, n * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream_);
  if(neg)
  {
    if(!ipadapter_tokens_neg_ || ipadapter_tokens_neg_->size() < n)
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
  // Consumed via a skip-when-unchanged host-side H2D inside forward_ipadapter (staged into
  // ipadapter_scale_buffer_); a graph REPLAY skips forward_ipadapter so the re-stage never runs and the
  // engine keeps the capture-time scale. Force a recapture on a value change (mirrors set_lora_scale).
  graph_ready_ = false;
}

void LibreDiffusionPipeline::set_ipadapter_scale_vector(const float* per_layer, int num_ip_layers)
{
  ipadapter_scale_vec_.assign(per_layer, per_layer + num_ip_layers);
  graph_ready_ = false;  // see set_ipadapter_scale — force recapture so a live scale change reaches the engine
}

int LibreDiffusionPipeline::num_runtime_loras() const
{
  return unet_ ? unet_->numRuntimeLoras() : 0;
}

void LibreDiffusionPipeline::set_lora_scale(int idx, float scale)
{
  // UNetWrapper::setLoraScale bounds-checks and returns silently, so refuse the slot here instead.
  const int slots = num_runtime_loras();
  if(idx < 0 || idx >= slots)
    throw std::out_of_range(
        "set_lora_scale: slot " + std::to_string(idx) + " but the UNet engine declares "
        + std::to_string(slots) + " runtime LoRA slot(s)");
  unet_->setLoraScale(idx, scale);
  // The captured 1-step CUDA graph bakes the lora_scale H2D (contents staged before enqueue). A value
  // change must re-stage -> force a recapture (the buffer address is stable + hashed in capture_signature,
  // so a no-op set leaves the graph intact). Same discipline as a prompt/scheduler change.
  graph_ready_ = false;
}

void LibreDiffusionPipeline::set_lora_scale_vector(const float* scales, int n)
{
  const int slots = num_runtime_loras();
  if(n < 0 || n > slots)
    throw std::out_of_range(
        "set_lora_scale_vector: " + std::to_string(n) + " scales but the UNet engine declares "
        + std::to_string(slots) + " runtime LoRA slot(s)");
  for(int i = 0; i < n; i++)
    unet_->setLoraScale(i, scales[i]);
  graph_ready_ = false;
}

}
