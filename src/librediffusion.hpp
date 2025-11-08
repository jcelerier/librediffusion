/**
 * LibreDiffusion C++ Implementation
 *
 * High-performance C++/CUDA implementation of the LibreDiffusion inference pipeline.
 * Eliminates Python overhead for maximum performance.
 */

#pragma once

#include "cuda_tensor.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nppi.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace librediffusion
{

// Forward declarations
class UNetWrapper;
class VAEEncoderWrapper;
class VAEDecoderWrapper;

enum class ModelType
{
  SD_15,
  SD_TURBO,
  SDXL_TURBO
};

enum class PipelineMode
{
  SINGLE_FRAME, 
  TEMPORAL_V2V  
};

/**
 * Configuration for LibreDiffusion pipeline
 */
struct LibreDiffusionConfig
{
  int device{0};
  ModelType model_type{ModelType::SD_15};
  int width = 512;
  int height = 512;
  int latent_width = 64;
  int latent_height = 64;
  int batch_size = 1;
  int denoising_steps = 1;
  int frame_buffer_size = 1;
  float guidance_scale = 1.2f;
  float delta = 1.0f; // Scaling factor for stock_noise in self-attention CFG
  bool do_add_noise = true;
  bool use_denoising_batch = true;
  uint64_t seed = 42;

  // CFG type: 0=none, 1=full, 2=self, 3=initialize
  int cfg_type = 2;

  // Text encoder configuration
  int text_seq_len = 77;     // Sequence length for text embeddings
  int text_hidden_dim = 768; // Hidden dimension: 768 for SD1.5, 1024 for SD-Turbo/SD2.1, 2048 for SDXL
  int clip_pad_token = 0;    // Padding token: 0 for SD-Turbo, 49407 for SD1.5

  // SDXL-specific configuration
  int pooled_embedding_dim = 1280; // Pooled text embeddings dimension for SDXL
  int time_ids_dim = 6;            // Time IDs dimension for SDXL

  // Engine paths for TensorRT-RTX
  std::string unet_engine_path = "engines/unet.engine";
  std::string vae_encoder_path = "engines/vae_encoder.engine";
  std::string vae_decoder_path = "engines/vae_decoder.engine";

  std::vector<int> timestep_indices;

  // StreamV2V temporal coherence parameters
  PipelineMode mode = PipelineMode::SINGLE_FRAME;  // Default to single-frame mode
  bool use_cached_attn = true;              // Enable attention caching
  bool use_feature_injection = true;        // Enable feature injection
  float feature_injection_strength = 0.8f;  // Blend ratio (0-1)
  float feature_similarity_threshold = 0.98f; // NN matching threshold
  int cache_interval = 4;                   // Update cache every N frames
  int cache_maxframes = 1;                  // Max frames to cache
  bool use_tome_cache = false;              // Enable token merging (advanced)
  float tome_ratio = 0.5f;                  // Token merging ratio
};

/**
 * StreamV2V temporal coherence state
 * Stores information from previous frames for temporal consistency
 */
struct TemporalState
{
  // Previous frame data
  std::unique_ptr<CUDATensor<__half>> prev_image_tensor;  // [1, 3, H, W]
  std::unique_ptr<CUDATensor<__half>> prev_x_t_latent;    // [1, 4, H/8, W/8]

  // Cached latents for nearest-neighbor matching (maxlen=4)
  std::deque<std::unique_ptr<CUDATensor<__half>>> cached_x_t_latent;

  // Separate noise components
  std::unique_ptr<CUDATensor<__half>> randn_noise;  // Pure random noise
  std::unique_ptr<CUDATensor<__half>> warp_noise;   // Warped noise (future use)

  // Unconditional embeddings (stored separately from conditional)
  std::unique_ptr<CUDATensor<__half>> null_prompt_embeds;  // [1, 77, 768/1024/2048]

  // StreamV2V attention caching (16 layers per frame)
  struct AttentionCache
  {
    std::vector<std::unique_ptr<CUDATensor<__half>>> attention_layers;  // 16 layers
    int frame_id = -1;  // Which frame this cache is from
  };
  std::deque<AttentionCache> cached_attentions;  // maxlen = cache_maxframes

  // Frame counter for cache updates
  int frame_id = 0;
};

/**
 * Main LibreDiffusion pipeline class
 */
class LibreDiffusionPipeline
{
public:
  explicit LibreDiffusionPipeline(const LibreDiffusionConfig& config);
  ~LibreDiffusionPipeline();

  void init_cuda();
  void init_npp();
  void init_engines();
  void init_buffers();

  /**
   * Reinitialize buffers with a new configuration.
   * This allows changing dimensions, denoising_steps, batch_size, etc.
   * without reloading the expensive TensorRT engines.
   *
   * Parameters that can be changed:
   * - width, height, latent_width, latent_height
   * - batch_size, denoising_steps, frame_buffer_size
   * - use_denoising_batch, do_add_noise
   * - guidance_scale, delta, cfg_type
   * - text_seq_len, text_hidden_dim
   * - timestep_indices
   *
   * Parameters that CANNOT be changed (require engine reload):
   * - unet_engine_path, vae_encoder_path, vae_decoder_path
   * - model_type, mode (SINGLE_FRAME vs TEMPORAL_V2V)
   */
  void reinit_buffers(const LibreDiffusionConfig& new_config);

  // Prepare the pipeline with prompt embeddings and scheduler parameters
  void prepare_embeds(
      const __half* prompt_embeds, // [batch_size, seq_len, hidden_dim]
      int seq_len, int hidden_dim);

  // Store null (unconditional) embeddings for StreamV2V
  void prepare_null_embeds(
      const __half* null_embeds, // [1, seq_len, hidden_dim]
      int seq_len, int hidden_dim);

  // Store negative (unconditional) embeddings for CFG guidance
  void prepare_negative_embeds(
      const __half* negative_embeds, // [1, seq_len, hidden_dim]
      int seq_len, int hidden_dim);

  // Blend multiple embeddings with weights for prompt interpolation
  // Creates a weighted sum: result = sum(weights[i] * embeddings[i])
  // Weights are normalized internally if they don't sum to 1.0
  void blend_embeds(
      const __half* const* embeddings,  // Array of embedding device pointers [num_embeddings]
      const float* weights,              // Blend weights (host array) [num_embeddings]
      int num_embeddings,
      int seq_len,
      int hidden_dim);

  // Prepare SDXL-specific conditioning inputs
  void prepare_sdxl_conditioning(
      const __half* text_embeds,  // [batch_size, pooled_dim] - Pooled text embeddings
      const __half* time_ids);    // [batch_size, 6] - Time IDs [h, w, crop_top, crop_left, target_h, target_w]

  void prepare_scheduler(
      std::span<float> timesteps,         // [num_timesteps]
      std::span<float> alpha_prod_t_sqrt, // [num_timesteps] - sqrt(alpha_cumprod)
      std::span<float> beta_prod_t_sqrt,  // [num_timesteps] - sqrt(1 - alpha_cumprod)
      std::span<float> c_skip,            // [num_timesteps] - skip connection scaling
      std::span<float> c_out              // [num_timesteps] - output scaling
  );

  // Set initial noise from Python (for testing/validation)
  void set_init_noise(const __half* noise); // [denoising_steps, 4, latent_h, latent_w]

  void reseed(int64_t seed);

  void set_guidance_scale(float guidance);
  void set_delta(float delta);

  // Resize an image to the correct size
  void rgba_resize(
      Npp8u* device_rgba_input_, int iw, int ih, Npp8u* device_rgba_resized_, int ow,
      int oh);

  // Run inference on input latents
  // Input: x_t_latent [batch, 4, latent_h, latent_w]
  // Output: x_0_pred [batch, 4, latent_h, latent_w]
  void predict_x0_batch(
      const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream = 0);

  // Encode image to latent
  void encode_image(
      const __half* image, // [batch, 3, height, width]
      __half* latent_out,  // [batch, 4, latent_h, latent_w]
      cudaStream_t stream = 0);
  void encode_image(
      const float* image, // [batch, 3, height, width]
      __half* latent_out, // [batch, 4, latent_h, latent_w]
      cudaStream_t stream = 0);

  // Decode latent to image
  void decode_latent(
      const __half* latent, // [batch, 4, latent_h, latent_w]
      __half* image_out,    // [batch, 3, height, width]
      cudaStream_t stream = 0);

  // Full img2img: image -> latent -> denoise -> decode -> image
  void img2img_impl(
      const __half* image_in, // [batch, 3, height, width]
      __half* image_out,      // [batch, 3, height, width]
      cudaStream_t stream = 0);

  void img2img_impl(
      const float* image_in, // [batch, 3, height, width]
      __half* image_out,     // [batch, 3, height, width]
      cudaStream_t stream = 0);

  // GPU-optimized image format conversion methods using Cutlass
  // Convert RGBA NHWC byte image to float/half NCHW tensor on GPU
  void rgba_nhwc_to_nchw_gpu(
      const uint8_t* rgba_nhwc_in,      // [batch, height, width, 4] - RGBA byte format
      float* rgb_nchw_out,               // [batch, 3, height, width] - RGB normalized [-1, 1]
      int width, int height,
      cudaStream_t stream = 0);

  void rgba_nhwc_to_nchw_gpu(
      const uint8_t* rgba_nhwc_in,      // [batch, height, width, 4] - RGBA byte format
      __half* rgb_nchw_out,              // [batch, 3, height, width] - RGB normalized [-1, 1]
      int width, int height,
      cudaStream_t stream = 0);

  // Convert float/half NCHW tensor to RGBA NHWC byte image on GPU
  void nchw_to_rgba_nhwc_gpu(
      const __half* rgb_nchw_in,        // [batch, 3, height, width] - RGB normalized [-1, 1]
      uint8_t* rgba_nhwc_out,            // [batch, height, width, 4] - RGBA byte format
      int width, int height,
      cudaStream_t stream = 0);

  void nchw_to_rgba_nhwc_gpu(
      const float* rgb_nchw_in,         // [batch, 3, height, width] - RGB normalized [-1, 1]
      uint8_t* rgba_nhwc_out,            // [batch, height, width, 4] - RGBA byte format
      int width, int height,
      cudaStream_t stream = 0);

  // Txt2img: random latent -> denoise -> decode -> image
  void txt2img_impl(
      __half* image_out, // [batch, 3, height, width]
      cudaStream_t stream = 0);

  // Txt2img for SD-Turbo: single-step direct prediction without LCM scheduler
  void txt2img_sd_turbo_impl(
      __half* image_out, // [batch, 3, height, width]
      cudaStream_t stream = 0);

  // Simplified high-level inference API - handles all CUDA operations internally
  // Input: CPU RGBA buffer in NHWC format [height, width, 4]
  // Output: CPU RGBA buffer in NHWC format [height, width, 4]
  void img2img(
      const uint8_t* cpu_rgba_input, // CPU buffer: RGBA NHWC format
      uint8_t* cpu_rgba_output,      // CPU buffer: RGBA NHWC format
      int width, int height);

  void txt2img(
      uint8_t* cpu_rgba_output, // CPU buffer: RGBA NHWC format
      int width, int height);

  float* img_preprocess(const uint8_t* device_rgba_input, int width, int height);
  uint8_t* img_postprocess(__half* device_rgba_output, int width, int height);

  // StreamV2V temporal coherence methods
  void enableTemporalCoherence(
      bool use_feature_injection = true,
      float injection_strength = 0.8f,
      float similarity_threshold = 0.98f,
      int cache_interval = 4,
      int max_cached_frames = 1);

  void disableTemporalCoherence();
  void resetTemporalState();
  int getCurrentFrameId() const { return temporal_state_.frame_id; }

  const LibreDiffusionConfig& config() const { return config_; }

  LibreDiffusionConfig config_;

  // CUDA stream for async operations
  cudaStream_t stream_;

  NppStreamContext npp_stream_;

  // Persistent buffers
  std::unique_ptr<CUDATensor<__half>> vae_encoded_x_t_latent_;
  std::unique_ptr<CUDATensor<__half>> unet_output_x_0_pred_;

  std::unique_ptr<CUDATensor<__half>> x_t_latent_buffer_;
  std::unique_ptr<CUDATensor<__half>> init_noise_;
  std::unique_ptr<CUDATensor<__half>> stock_noise_;
  std::unique_ptr<CUDATensor<__half>> prompt_embeds_;
  std::unique_ptr<CUDATensor<__half>> negative_embeds_;  // For CFG negative prompt

  // SDXL-specific buffers
  std::unique_ptr<CUDATensor<__half>> text_embeds_;  // Pooled embeddings [batch, 1280]
  std::unique_ptr<CUDATensor<__half>> time_ids_;     // Time IDs [batch, 6]

  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_x_t_latent;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_model_pred;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_denoised;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_concatenated;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_unet_input_latent;
  std::unique_ptr<CUDATensor<float>> predict_x0_batch_unet_input_timestep;
  std::unique_ptr<CUDATensor<float>> predict_x0_batch_unet_input_latent_fp32;
  std::unique_ptr<CUDATensor<float>> predict_x0_batch_unet_input_latent_fp32_single;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_unet_encoder_hidden_states;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_unet_encoder_hidden_states_single;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_unet_output;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_noise_pred_uncond;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_noise_pred_text;
  std::unique_ptr<CUDATensor<__half>> predict_x0_batch_model_pred_tmp;

  // Additional reusable buffers for various operations
  std::unique_ptr<CUDATensor<__half>> unet_input_latent_;
  std::unique_ptr<CUDATensor<float>> unet_input_timestep_;
  std::unique_ptr<CUDATensor<__half>> unet_encoder_hidden_states_;
  std::unique_ptr<CUDATensor<__half>> model_pred_tmp_;
  std::unique_ptr<CUDATensor<__half>> unet_output_buffer_;

  // Scheduler parameters
  std::unique_ptr<CUDATensor<float>> alpha_prod_t_sqrt_;
  std::unique_ptr<CUDATensor<float>> beta_prod_t_sqrt_;
  std::unique_ptr<CUDATensor<float>> c_skip_;
  std::unique_ptr<CUDATensor<float>> c_out_;

  // Host-side copies of scheduler parameters for CPU access
  std::vector<float> alpha_prod_t_sqrt_host_;
  std::vector<float> beta_prod_t_sqrt_host_;
  std::vector<float> c_skip_host_;
  std::vector<float> c_out_host_;
  std::unique_ptr<CUDATensor<float>> sub_timesteps_;

  // Temporary buffers for format conversion
  std::unique_ptr<CUDATensor<float>> rgb_nhwc_tmp_fp32_;
  std::unique_ptr<CUDATensor<__half>> rgb_nhwc_tmp_fp16_;
  std::unique_ptr<CUDATensor<uint8_t>> device_rgba_input_;
  std::unique_ptr<CUDATensor<uint8_t>> device_rgba_input_vae_size_;
  std::unique_ptr<CUDATensor<uint8_t>> device_rgba_output_vae_size_;
  std::unique_ptr<CUDATensor<uint8_t>> device_rgba_output_;
  std::unique_ptr<CUDATensor<float>> device_nchw_input_;
  std::unique_ptr<CUDATensor<__half>> device_nchw_output_;

  // Model wrappers
  std::unique_ptr<UNetWrapper> unet_;
  std::unique_ptr<VAEEncoderWrapper> vae_encoder_;
  std::unique_ptr<VAEDecoderWrapper> vae_decoder_;

  // StreamV2V temporal state
  TemporalState temporal_state_;

  // Internal operations
  void add_noise(
      const __half* original_samples, const __half* noise, __half* noisy_samples,
      int t_index, int N, cudaStream_t stream);
  void add_noise_direct(
      __half* original_samples, const __half* noise, int t_index, int N,
      cudaStream_t stream);

  void scheduler_step_batch(
      const __half* model_pred, const __half* x_t_latent, __half* denoised_out, int idx,
      int N, cudaStream_t stream);
  void predict_x0_batch_impl_single_step(
      const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream);
  void predict_x0_batch_impl_multi_step_sequential(
      const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream);
  void predict_x0_batch_impl_multi_step_batched(
      const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream);

  void apply_cfg(
      const __half* noise_pred_uncond, const __half* noise_pred_text,
      __half* model_pred_out, int N, cudaStream_t stream);

  void unet_step(
      const __half* x_t_latent, const float* t_list, __half* denoised_out, int idx,
      cudaStream_t stream);
};

} // namespace librediffusion
