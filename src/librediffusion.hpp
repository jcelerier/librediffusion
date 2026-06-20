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
class ControlNetWrapper;
class VAEEncoderWrapper;
class VAEDecoderWrapper;
class CLIPImageEncoderWrapper;

enum class ModelType
{
  SD_15,
  SD_TURBO,
  SDXL_TURBO,
  FLUX2_KLEIN_4B
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

  // ControlNet (multi). Each entry = one ControlNet engine + its conditioning scale. When non-empty
  // AND the UNet engine is control-aware (declares input_control_* inputs), the pipeline runs every
  // ControlNet each UNet step and SUMS their residuals before injection. conditioning_scale is BAKED
  // by each engine. Preprocessing is EXTERNAL: the host feeds each net's control image via the C-API.
  struct ControlNetSpec
  {
    std::string engine_path;
    float conditioning_scale = 1.0f;
  };
  std::vector<ControlNetSpec> controlnets;

  // IP-Adapter (baked into the UNet engine — an IP-variant unet.engine with a longer encoder_hidden_
  // states seq + an ipadapter_scale input). Auto-detected from the engine; no separate path needed.
  // Image tokens are computed HOST-SIDE and fed in via the C-API (external, like controlnet preproc).
  int ipadapter_num_tokens = 4;       // image tokens appended to the 77 text tokens (4 base / 16 plus)
  float ipadapter_scale = 1.0f;       // uniform per-layer scale (overridable per-layer via the C-API)
  // On-device IP-Adapter image encoder (optional). When BOTH paths are set, the pipeline loads a
  // CLIPImageEncoderWrapper so the host can feed a RAW style image (set_ipadapter_image) instead of
  // precomputed tokens. clip_image_encoder = CLIP ViT-H/14 (pixel_values[1,3,224,224]->image_embeds
  // [1,1024]); ip_image_proj = ImageProjModel (image_embeds[1,1024]->ip_tokens[1,N,768]).
  std::string ipadapter_image_encoder_path;  // clip_image_encoder.engine
  std::string ipadapter_image_proj_path;     // ip_image_proj.engine

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

  // CUDA graph: capture the 1-step single_step path as one graph and replay per frame (~+18% on
  // SD-Turbo/CN/IP 1-step; no-op/skip on multi-step, SDXL@1024, klein, V2V). Default OFF — enable per
  // 1-step bundle after validation.
  bool use_cuda_graph = false;
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

  // Set ControlNet `index`'s image-space control input. Two overloads:
  //  - device fp16 [1,3,img_h,img_w] in [0,1] (already preprocessed).
  //  - host RGBA uint8 [img_h,img_w,4] — converted on-device to RGB fp16 NCHW [0,1].
  // Stored as a single row; tiled to unet_batch_size internally each step.
  void set_controlnet_cond(int index, const __half* cond, int img_h, int img_w);
  void set_controlnet_cond_rgba(int index, const uint8_t* cpu_rgba, int img_h, int img_w);
  // Live-adjust a net's conditioning scale (re-baked each step via the engine's scale input).
  void set_controlnet_scale(int index, float scale);

  // IP-Adapter: set the host-computed image tokens (pos + optional neg), shape [num_tokens, dim] each
  // (device fp16). dim must equal the UNet hidden dim. neg may be null (used for the cfg uncond row;
  // base IP-Adapter neg = projection of zeros, supplied by the host).
  void set_ipadapter_tokens(const __half* pos, const __half* neg, int num_tokens, int dim);
  // Uniform scale across all IP layers; or a per-layer vector of length num_ip_layers.
  void set_ipadapter_scale(float scale);
  void set_ipadapter_scale_vector(const float* per_layer, int num_ip_layers);
  // On-device path: take a raw host RGBA style image [img_h,img_w,4], run the CLIP image encoder +
  // projection engines, and set the pos tokens (+ neg tokens = projection of zeros). Requires the
  // image encoder engines to be configured (ipadapter_image_encoder_path / _proj_path). Call once /
  // on change (style is static); re-call only when the style image changes.
  void set_ipadapter_image(const uint8_t* cpu_rgba, int img_h, int img_w);

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

  // Multi-step batched persistent scratch — hoisted from per-call stack CUDATensors so the batched
  // denoise path does NO per-frame cudaMalloc (was the "spurious allocations" FIXME); grow-on-first-use.
  // Prerequisite for capturing the multi-step path into a CUDA graph (a per-call malloc is illegal
  // during capture). Behaviour-neutral: same buffers, just reused across frames.
  std::unique_ptr<CUDATensor<__half>> mp_x_t_latent_;
  std::unique_ptr<CUDATensor<__half>> mp_model_pred_;
  std::unique_ptr<CUDATensor<__half>> mp_denoised_;
  std::unique_ptr<CUDATensor<__half>> mp_concatenated_;
  std::unique_ptr<CUDATensor<float>>  mp_unet_input_latent_fp32_;
  std::unique_ptr<CUDATensor<__half>> mp_tiled_text_embeds_;
  std::unique_ptr<CUDATensor<__half>> mp_tiled_time_ids_;
  std::unique_ptr<CUDATensor<__half>> mp_ext_ehs_;

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

  // Run every configured ControlNet on the current step's inputs and SUM their residuals into
  // controlnet_sum_down_/_mid_. Fills out_down[down_count]/out_mid with the summed residual pointers.
  // SDXL passes text_embeds/time_ids (else nullptr/0). down_count set to 12 (SD1.5) / 9 (SDXL).
  void run_controlnets(
      const __half* sample_fp16, const float* timestep, const __half* ehs,
      const __half* text_embeds, const __half* time_ids,
      int unet_batch_size, int seq_len, int hidden_dim, int pooled_dim,
      const __half** out_down, const __half** out_mid, int& down_count, void* stream);

  // ControlNet (multi, optional). One wrapper + one [1,3,H,W] [0,1] cond buffer + one live scale per
  // net. Each UNet step: run every net, SUM their residuals (controlnet_sum_down_/_mid_), inject once.
  // Preprocessing is EXTERNAL — host feeds each net's cond via set_controlnet_cond[_rgba](index,...).
  std::vector<std::unique_ptr<ControlNetWrapper>> controlnets_;
  std::vector<std::unique_ptr<CUDATensor<__half>>> controlnet_cond_;  // per-net [1,3,H,W] [0,1]
  std::vector<float> controlnet_scales_;                              // per-net (live-adjustable)
  std::unique_ptr<CUDATensor<uint8_t>> controlnet_rgba_tmp_;          // RGBA upload scratch
  // Accumulators for the multi-net residual sum (allocated lazily to unet_batch_size geometry).
  std::vector<std::unique_ptr<CUDATensor<__half>>> controlnet_sum_down_;
  std::unique_ptr<CUDATensor<__half>> controlnet_sum_mid_;
  // Persistent tiled-cond buffer (was a per-frame stack CUDATensor in run_controlnets -> illegal cudaMalloc
  // during CUDA-graph capture). Grown on first use.
  std::unique_ptr<CUDATensor<__half>> controlnet_cond_tiled_;
  bool controlnet_enabled_ = false;

  // IP-Adapter (baked UNet variant). Host-fed image tokens [num_tokens, dim] (pos + neg) + the per-layer
  // scale vector. Enabled when the UNet engine declares ipadapter_scale (unet_->hasIpAdapter()).
  bool ipadapter_enabled_ = false;
  std::unique_ptr<CUDATensor<__half>> ipadapter_tokens_pos_;  // [num_tokens, dim]
  std::unique_ptr<CUDATensor<__half>> ipadapter_tokens_neg_;  // [num_tokens, dim] (zeros-proj for base)
  std::vector<float> ipadapter_scale_vec_;                    // host, length num_ip_layers
  int ipadapter_num_tokens_ = 0;
  // On-device image encoder (optional): turns a raw style image into the pos/neg tokens above.
  std::unique_ptr<CLIPImageEncoderWrapper> ipadapter_image_encoder_;

  // ---- CUDA graph (whole-frame capture of the 1-step single_step path) ----
  // Gated, default OFF (librediffusion_config_set_cuda_graph). Capturable only for denoising_steps==1,
  // mode != TEMPORAL_V2V (multi-step has per-step host branches + changing timestep).
  // ~+18% on SD-Turbo/CN/IP, ~0 on multi-step/SDXL/klein.
  bool graph_enabled_ = false;
  bool graph_ready_ = false;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  uint64_t graph_sig_ = 0;
  // Fixed staging buffer for the single_step input latent: the graph bakes the load_d2d SOURCE pointer,
  // so the volatile caller pointer must be staged into THIS persistent buffer before the captured region.
  std::unique_ptr<CUDATensor<__half>> graph_in_staging_;
  // Hoisted IP-Adapter extended-ehs buffer (was a per-frame stack CUDATensor -> illegal cudaMalloc in capture).
  std::unique_ptr<CUDATensor<__half>> ip_ext_ehs_;
  uint64_t capture_signature() const;            // recapture key: config + persistent-buffer device addresses
  void run_single_step_body(const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream);

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
