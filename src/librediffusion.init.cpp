#include "librediffusion.hpp"

#include "kernels.hpp"
#include "nchw.hpp"
#include "tensorrt_wrappers.hpp"

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

void LibreDiffusionPipeline::init_cuda()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if(deviceCount == 0)
    throw std::runtime_error("CUDA not available");
  else if(deviceCount == 1)
    this->config_.device = 0;
  else
    this->config_.device = std::clamp(this->config_.device, 0, deviceCount - 1);

  cudaSetDevice(this->config_.device);
  cudaError_t err = cudaStreamCreate(&stream_);
  if(err != cudaSuccess)
  {
    throw std::runtime_error(
        std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
  }
}

void LibreDiffusionPipeline::init_engines()
{
  bool use_v2v = (config_.mode == PipelineMode::TEMPORAL_V2V);
  unet_ = std::make_unique<UNetWrapper>(config_.unet_engine_path, use_v2v);
  vae_encoder_ = std::make_unique<VAEEncoderWrapper>(config_.vae_encoder_path);
  vae_decoder_ = std::make_unique<VAEDecoderWrapper>(config_.vae_decoder_path);

  // ControlNet (optional, multi): create a wrapper per configured net. Enabled only when at least one
  // net is configured AND the UNet engine is control-aware (declares input_control_* inputs).
  // NOTE: init_engines() can be called more than once (ctor + pipeline_init_all); clear first so the
  // wrapper/cond/scale vectors don't ACCUMULATE duplicates across calls.
  controlnets_.clear();
  controlnet_cond_.clear();
  controlnet_scales_.clear();
  controlnet_enabled_ = false;
  if(!config_.controlnets.empty())
  {
    if(!unet_->hasControlInputs())
    {
      std::cout << "Warning: " << config_.controlnets.size() << " controlnet(s) configured but the UNet "
                   "engine has no input_control_* inputs; ControlNet DISABLED (use a control-aware "
                   "unet.engine)" << std::endl;
    }
    else
    {
      for(const auto& spec : config_.controlnets)
      {
        controlnets_.push_back(std::make_unique<ControlNetWrapper>(spec.engine_path));
        controlnet_cond_.push_back(nullptr);  // filled by set_controlnet_cond[_rgba]
        controlnet_scales_.push_back(spec.conditioning_scale);
      }
      controlnet_enabled_ = true;
    }
  }

  // IP-Adapter: auto-enable when the UNet engine is an IP variant (declares ipadapter_scale). The image
  // tokens are fed host-side via set_ipadapter_tokens; default the per-layer scale vector to a uniform
  // config_.ipadapter_scale (length = the engine's num_ip_layers).
  ipadapter_enabled_ = unet_->hasIpAdapter();
  if(ipadapter_enabled_)
  {
    int n = unet_->numIpLayers();
    if(n <= 0) n = 16;
    ipadapter_scale_vec_.assign(n, config_.ipadapter_scale);
    ipadapter_num_tokens_ = config_.ipadapter_num_tokens;
    std::cout << "Note: IP-Adapter enabled (" << n << " layers, " << ipadapter_num_tokens_
              << " image tokens)" << std::endl;

    // Optional on-device image encoder: when both engine paths are set, load the CLIP image encoder +
    // projection so the host can feed a raw style image via set_ipadapter_image (instead of tokens).
    ipadapter_image_encoder_.reset();
    if(!config_.ipadapter_image_encoder_path.empty()
       && !config_.ipadapter_image_proj_path.empty())
    {
      ipadapter_image_encoder_ = std::make_unique<CLIPImageEncoderWrapper>(
          config_.ipadapter_image_encoder_path, config_.ipadapter_image_proj_path);
      std::cout << "Note: IP-Adapter on-device image encoder loaded ("
                << ipadapter_image_encoder_->numTokens() << " tokens x "
                << ipadapter_image_encoder_->tokenDim() << ")" << std::endl;
    }
  }
}

void LibreDiffusionPipeline::init_buffers()
{
  // Calculate sizes
  int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;
  int buffer_size = config_.denoising_steps * config_.frame_buffer_size * 4
                    * config_.latent_height * config_.latent_width;

  vae_encoded_x_t_latent_ = std::make_unique<CUDATensor<__half>>(latent_size);
  unet_output_x_0_pred_ = std::make_unique<CUDATensor<__half>>(latent_size);

  // Case !denoising_batch && config_.denoising_steps == 1
  {
    // See logic in predict_x0_batch_impl_single_step
    predict_x0_batch_x_t_latent = std::make_unique<CUDATensor<__half>>(latent_size);
    {
      int total_batch = config_.batch_size
                        + (config_.denoising_steps - 1) * config_.frame_buffer_size;
      int denoised_size = total_batch * 4 * config_.latent_height * config_.latent_width;
      predict_x0_batch_denoised = std::make_unique<CUDATensor<__half>>(denoised_size);

      int unet_batch_size = config_.batch_size;

      if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
      { // cfg_type="full" doubles the batch
        const int doubled_size = latent_size * 2;
        unet_batch_size = config_.batch_size * 2;
        predict_x0_batch_unet_input_latent
            = std::make_unique<CUDATensor<__half>>(doubled_size);
        predict_x0_batch_unet_input_timestep
            = std::make_unique<CUDATensor<float>>(config_.batch_size * 2);
      }
      else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
      { // cfg_type="initialize" adds 1 element
        int single_latent_size = 4 * config_.latent_height * config_.latent_width;
        int extended_size = latent_size + single_latent_size;
        unet_batch_size = config_.batch_size + 1;
        predict_x0_batch_unet_input_latent
            = std::make_unique<CUDATensor<__half>>(extended_size);
        predict_x0_batch_unet_input_timestep
            = std::make_unique<CUDATensor<float>>(config_.batch_size + 1);
      }
      else
      { // cfg_type="none" or "self" - no change
        predict_x0_batch_unet_input_latent
            = std::make_unique<CUDATensor<__half>>(latent_size);
        predict_x0_batch_unet_input_timestep
            = std::make_unique<CUDATensor<float>>(config_.batch_size);
      }

      int latent_elements_single
          = unet_batch_size * 4 * config_.latent_height * config_.latent_width;
      predict_x0_batch_model_pred
          = std::make_unique<CUDATensor<__half>>(latent_elements_single);
      predict_x0_batch_unet_input_latent_fp32_single
          = std::make_unique<CUDATensor<float>>(latent_elements_single);

      if(unet_batch_size > config_.batch_size)
      {
        int embedding_size = config_.text_seq_len * config_.text_hidden_dim;
        int total_embedding_size = unet_batch_size * embedding_size;
        predict_x0_batch_unet_encoder_hidden_states_single
            = std::make_unique<CUDATensor<__half>>(total_embedding_size);
      }
      else
      {
        int embedding_size
            = config_.batch_size * config_.text_seq_len * config_.text_hidden_dim;
        predict_x0_batch_unet_encoder_hidden_states_single
            = std::make_unique<CUDATensor<__half>>(embedding_size);
      }

      predict_x0_batch_unet_output
          = std::make_unique<CUDATensor<__half>>(latent_elements_single);
    }
  }

  // Allocate persistent buffers
  if(config_.denoising_steps > 1)
  {
    int buffer_batch = (config_.denoising_steps - 1) * config_.frame_buffer_size;
    int buffer_size_elems
        = buffer_batch * 4 * config_.latent_height * config_.latent_width;
    x_t_latent_buffer_ = std::make_unique<CUDATensor<__half>>(buffer_size_elems);

    // Initialize with zeros
    cudaMemset(x_t_latent_buffer_->data(), 0, buffer_size_elems * sizeof(__half));
  }

  // Allocate noise buffers
  // When use_denoising_batch=true, we need noise for each timestep (denoising_steps), not batch_size
  int noise_batch_size
      = config_.use_denoising_batch ? config_.denoising_steps : config_.batch_size;
  init_noise_ = std::make_unique<CUDATensor<__half>>(
      noise_batch_size * 4 * config_.latent_height * config_.latent_width);

  stock_noise_ = std::make_unique<CUDATensor<__half>>(
      noise_batch_size * 4 * config_.latent_height * config_.latent_width);

  // Pre-allocate additional reusable buffers
  // These sizes handle the maximum possible cases to avoid reallocation
  // For batched denoising, the actual batch size is total_batch, not config_.batch_size
  // total_batch = batch_size + (denoising_steps - 1) * frame_buffer_size
  // Additionally, cfg_type="initialize" adds 1 extra element, and cfg_type="full" doubles
  int base_batch = config_.use_denoising_batch
                       ? config_.batch_size
                             + (config_.denoising_steps - 1) * config_.frame_buffer_size
                       : config_.batch_size;
  int max_unet_batch = base_batch;
  if(config_.guidance_scale > 1.0f && config_.cfg_type == 1) {
    max_unet_batch = base_batch * 2;  // cfg_type="full" doubles the batch
  } else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3) {
    max_unet_batch = base_batch + 1;  // cfg_type="initialize" adds 1 element
  } else {
    max_unet_batch = base_batch;  // cfg_type="none" or "self" - no change
  }


  unet_input_latent_ = std::make_unique<CUDATensor<__half>>(
      max_unet_batch * 4 * config_.latent_height * config_.latent_width);

  unet_input_timestep_ = std::make_unique<CUDATensor<float>>(max_unet_batch);

  int max_embedding_size
      = max_unet_batch * config_.text_seq_len * config_.text_hidden_dim;
  unet_encoder_hidden_states_ = std::make_unique<CUDATensor<__half>>(max_embedding_size);

  model_pred_tmp_ = std::make_unique<CUDATensor<__half>>(
      max_unet_batch * 4 * config_.latent_height * config_.latent_width);

  unet_output_buffer_ = std::make_unique<CUDATensor<__half>>(
      max_unet_batch * 4 * config_.latent_height * config_.latent_width);

  // Initialize StreamV2V temporal state (only if mode is TEMPORAL_V2V)
  if(config_.mode == PipelineMode::TEMPORAL_V2V)
  {
    // Allocate randn_noise and warp_noise (same size as one latent)
    int single_latent_size = 1 * 4 * config_.latent_height * config_.latent_width;
    temporal_state_.randn_noise
        = std::make_unique<CUDATensor<__half>>(single_latent_size);
    temporal_state_.warp_noise
        = std::make_unique<CUDATensor<__half>>(single_latent_size);

    temporal_state_.frame_id = 0;
  }
}

void LibreDiffusionPipeline::init_npp()
{
  NppStreamContext npp_stream_;

  int device = config_.device;
  cudaGetDevice(&device);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);

  npp_stream_.nCudaDeviceId = device;
  npp_stream_.nMultiProcessorCount = prop.multiProcessorCount;
  npp_stream_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  npp_stream_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  npp_stream_.nSharedMemPerBlock = prop.sharedMemPerBlock;
  npp_stream_.nCudaDevAttrComputeCapabilityMajor = prop.major;
  npp_stream_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
  npp_stream_.hStream = this->stream_;
}

void LibreDiffusionPipeline::reinit_buffers(const LibreDiffusionConfig& new_config)
{
  // Preserve engine paths and mode (these cannot change without engine reload)
  // But update all other parameters
  config_.width = new_config.width;
  config_.height = new_config.height;
  config_.latent_width = new_config.latent_width;
  config_.latent_height = new_config.latent_height;
  config_.batch_size = new_config.batch_size;
  config_.denoising_steps = new_config.denoising_steps;
  config_.frame_buffer_size = new_config.frame_buffer_size;
  config_.guidance_scale = new_config.guidance_scale;
  config_.delta = new_config.delta;
  config_.do_add_noise = new_config.do_add_noise;
  config_.use_denoising_batch = new_config.use_denoising_batch;
  config_.seed = new_config.seed;
  config_.cfg_type = new_config.cfg_type;
  config_.text_seq_len = new_config.text_seq_len;
  config_.text_hidden_dim = new_config.text_hidden_dim;
  config_.clip_pad_token = new_config.clip_pad_token;
  config_.pooled_embedding_dim = new_config.pooled_embedding_dim;
  config_.time_ids_dim = new_config.time_ids_dim;
  config_.timestep_indices = new_config.timestep_indices;

  // Update temporal coherence parameters (these don't need engine reload)
  config_.use_cached_attn = new_config.use_cached_attn;
  config_.use_feature_injection = new_config.use_feature_injection;
  config_.feature_injection_strength = new_config.feature_injection_strength;
  config_.feature_similarity_threshold = new_config.feature_similarity_threshold;
  config_.cache_interval = new_config.cache_interval;
  config_.cache_maxframes = new_config.cache_maxframes;
  config_.use_tome_cache = new_config.use_tome_cache;
  config_.tome_ratio = new_config.tome_ratio;

  // Reinitialize all buffers with new config
  init_buffers();
}
}
