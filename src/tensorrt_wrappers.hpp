/**
 * TensorRT-RTX Model Wrappers for LibreDiffusion C++ Implementation
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

// TensorRT-RTX includes
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "model_cache.hpp"

namespace librediffusion
{

// Forward declarations
template <typename T>
class CUDATensor;
class CachedTensorRTEngine;

/**
 * UNet wrapper for TensorRT-RTX
 *
 * Inputs (SD1.5/SD-Turbo):
 *   - sample: [batch, 4, height, width] FP32
 *   - timestep: [batch] FP32
 *   - encoder_hidden_states: [batch, seq_len, hidden_dim] FP16
 *
 * Additional inputs for SDXL:
 *   - text_embeds: [batch, pooled_dim] FP16
 *   - time_ids: [batch, 6] FP16
 *
 * Output:
 *   - latent: [batch, 4, height, width] FP16
 */
class UNetWrapper
{
public:
  UNetWrapper(const std::string& engine_path, bool use_v2v = false);
  ~UNetWrapper();

  // No copy
  UNetWrapper(const UNetWrapper&) = delete;
  UNetWrapper& operator=(const UNetWrapper&) = delete;

  /**
     * Run UNet inference (SD1.5/SD-Turbo)
     *
     * @param sample Latent input [batch, 4, height, width] FP32
     * @param timestep Timestep tensor [batch] FP32
     * @param encoder_hidden_states Prompt embeddings [batch, seq_len, hidden_dim] FP16
     * @param output Output tensor [batch, 4, height, width] FP16
     * @param batch Batch size
     * @param height Latent height
     * @param width Latent width
     * @param seq_len Sequence length
     * @param hidden_dim Hidden dimension
     * @param stream CUDA stream
     */
  void forward(
      const float* sample, const float* timestep, const __half* encoder_hidden_states,
      __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
      cudaStream_t stream);

  /**
     * Run UNet inference (SDXL)
     *
     * @param sample Latent input [batch, 4, height, width] FP32
     * @param timestep Timestep tensor [batch] FP32
     * @param encoder_hidden_states Prompt embeddings [batch, seq_len, hidden_dim] FP16
     * @param text_embeds Pooled text embeddings [batch, pooled_dim] FP16 (SDXL only)
     * @param time_ids Time IDs [batch, 6] FP16 (SDXL only)
     * @param output Output tensor [batch, 4, height, width] FP16
     * @param batch Batch size
     * @param height Latent height
     * @param width Latent width
     * @param seq_len Sequence length
     * @param hidden_dim Hidden dimension
     * @param pooled_dim Pooled embedding dimension
     * @param stream CUDA stream
     */
  void forward_sdxl(
      const float* sample, const float* timestep, const __half* encoder_hidden_states,
      const __half* text_embeds, const __half* time_ids,
      __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
      int pooled_dim, cudaStream_t stream);

  /**
   * Run UNet inference (SD1.5) WITH ControlNet residuals injected.
   *
   * The control-aware UNet engine declares extra inputs input_control_00..input_control_11 +
   * input_control_middle (the 12 down + 1 mid ControlNet residuals); the engine adds them to its
   * own down/mid blocks internally. The residuals are produced by ControlNetWrapper and are ALREADY
   * conditioning_scale-scaled (our SD1.5 controlnet engine bakes scale via its onnx::Cast_4 input),
   * so they are bound here as-is.
   *
   * @param down_residuals 12 device pointers, down_residuals[i] = input_control_{i:02d} [batch, C_i, H_i, W_i] FP16
   * @param mid_residual   device pointer for input_control_middle [batch, 1280, H/8, W/8] FP16
   * (other params as forward())
   */
  void forward_controlnet(
      const float* sample, const float* timestep, const __half* encoder_hidden_states,
      const __half* const* down_residuals, const __half* mid_residual,
      __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
      cudaStream_t stream);

  /**
   * Run UNet inference (SDXL) WITH ControlNet residuals injected. Like forward_sdxl (binds
   * text_embeds/time_ids) PLUS the 9 down + mid control residuals into input_control_00..08 +
   * input_control_middle. Residuals are already scale-baked.
   * @param down_residuals 9 device pointers; @param mid_residual the mid block residual.
   */
  void forward_controlnet_sdxl(
      const float* sample, const float* timestep, const __half* encoder_hidden_states,
      const __half* text_embeds, const __half* time_ids,
      const __half* const* down_residuals, const __half* mid_residual,
      __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
      int pooled_dim, cudaStream_t stream);

  /// True if the loaded engine declares the input_control_* residual inputs (control-aware UNet).
  bool hasControlInputs() const { return has_control_inputs_; }

  /// True if the loaded engine is an IP-Adapter variant (declares the `ipadapter_scale` input).
  bool hasIpAdapter() const { return has_ipadapter_; }
  /// Length of the ipadapter_scale vector (= num cross-attn IP layers), detected at load (0 if none).
  int numIpLayers() const { return num_ip_layers_; }

  /**
   * Run UNet inference (SD1.5) for an IP-Adapter engine. Identical to forward() except:
   *  - encoder_hidden_states is the EXTENDED sequence [batch, 77+num_image_tokens, hidden_dim] (text
   *    tokens ++ image tokens, assembled by the caller); pass the full seq_len (e.g. 81).
   *  - binds the `ipadapter_scale` fp32 vector [num_ip_layers] (per-layer decoupled-attn scale).
   * The IP to_k_ip/to_v_ip + projection are baked into the engine; output is the usual single latent.
   */
  void forward_ipadapter(
      const float* sample, const float* timestep, const __half* encoder_hidden_states,
      const float* ipadapter_scale, int num_ip_layers,
      __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
      cudaStream_t stream);

  /**
   * Run UNet inference (SDXL) for an IP-Adapter engine. Combines forward_sdxl (binds text_embeds +
   * time_ids for SDXL's added conditioning) with forward_ipadapter (extended encoder_hidden_states
   * [batch, 77+num_image_tokens, 2048] ++ the per-layer ipadapter_scale[num_ip_layers] vector). The
   * SDXL IP-variant unet.engine declares: sample, timestep, encoder_hidden_states, text_embeds,
   * time_ids, ipadapter_scale -> latent.
   */
  void forward_ipadapter_sdxl(
      const float* sample, const float* timestep, const __half* encoder_hidden_states,
      const __half* text_embeds, const __half* time_ids,
      const float* ipadapter_scale, int num_ip_layers,
      __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
      int pooled_dim, cudaStream_t stream);

  /**
   * StreamV2V: Access attention output buffers for feature injection
   * Returns the 16 attention layer outputs from the last forward pass
   */
  const std::vector<std::unique_ptr<CUDATensor<__half>>>& getAttentionBuffers() const
  {
    return attention_output_buffers_;
  }

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;  // Shared cached engine
  std::unique_ptr<nvinfer1::IExecutionContext> context_; // Per-wrapper context

  // Cached shapes to avoid reallocation
  struct ShapeCache
  {
    int batch = 0;
    int height = 0;
    int width = 0;
    int seq_len = 0;
    int hidden_dim = 0;
    int pooled_dim = 0;  // SDXL only
    bool initialized = false;
  };
  ShapeCache shape_cache_;

  // Timestep input extent declared by the engine. Most engines (SD1.5/SD-Turbo, and our
  // from-scratch SDXL trace) use a batch-sized timestep [batch]. The PREBUILT SDXL UNet from
  // stabilityai/sdxl-turbo-tensorrt instead has a STATIC timestep shape [1]. We query this at
  // load time so the forward path binds the engine's actual extent rather than assuming [batch]
  // (which would fail allInputDimensionsSpecified for batch != 1 on the prebuilt engine).
  // 0 = dynamic/batch-sized (use the call's batch); >0 = fixed extent the engine requires.
  int timestep_fixed_extent_ = 0;

  // Persistent buffers (like Python's polygraphy implementation)
  // These maintain stable addresses across inference calls
  std::unique_ptr<CUDATensor<float>> sample_buffer_;
  std::unique_ptr<CUDATensor<float>> timestep_buffer_;
  std::unique_ptr<CUDATensor<__half>> encoder_hidden_states_buffer_;
  std::unique_ptr<CUDATensor<__half>> output_buffer_;

  // SDXL-specific buffers
  std::unique_ptr<CUDATensor<__half>> text_embeds_buffer_;
  std::unique_ptr<CUDATensor<__half>> time_ids_buffer_;

  // StreamV2V attention caching
  bool use_v2v_ = false;
  static constexpr int NUM_ATTENTION_OUTPUTS = 16;
  std::vector<std::unique_ptr<CUDATensor<__half>>> attention_output_buffers_;

  // ControlNet residual inputs. Persistent buffers, bound only when the engine declares
  // input_control_* inputs (control-aware UNet). down_residuals copied per-call.
  //
  // The residual COUNT and per-index geometry are NOT hardcoded: they are derived from the loaded
  // UNet engine's own input_control_NN bindings at init (see UNetWrapper ctor). This makes the
  // wrapper model-agnostic — SD1.5 declares 12 down (factors {1,1,1,2,2,2,4,4,4,8,8,8}), the pruned
  // SDXS UNet declares 6 (factors {1,1,2,2,4,4}), and both are read straight from the engine instead
  // of a fixed SD1.5 table. NUM_CONTROL_DOWN remains only as the SD1.5 fallback/MAX bound.
  static constexpr int NUM_CONTROL_DOWN = 12;
  bool has_control_inputs_ = false;
  int num_control_down_ = 0;                      // actual input_control_NN count (6 SDXS / 12 SD1.5)
  std::vector<int> control_down_ch_;              // per-index channel count, from the engine binding
  std::vector<int> control_down_fac_;             // per-index spatial downsample factor (latent_dim / control_dim)
  int control_mid_ch_ = 1280;                     // input_control_middle channels
  int control_mid_fac_ = 8;                       // input_control_middle downsample factor (= last down factor)
  std::vector<std::unique_ptr<CUDATensor<__half>>> control_down_buffers_;
  std::unique_ptr<CUDATensor<__half>> control_mid_buffer_;

  // IP-Adapter: engine declares an `ipadapter_scale` fp32 vector input (+ a longer ehs seq). The IP
  // attention is baked into the engine; we only bind the per-layer scale vector here.
  bool has_ipadapter_ = false;
  int num_ip_layers_ = 0;
  std::unique_ptr<CUDATensor<float>> ipadapter_scale_buffer_;
  // PINNED host staging for the per-layer scale (cudaMemcpyAsync H2D from PAGEABLE host memory is illegal
  // inside CUDA-graph capture; pinned is capturable + faster). Cache last-uploaded values so a steady-state
  // graph replay does NO H2D at all (re-upload only when the scale actually changes).
  float* ipadapter_scale_pinned_ = nullptr;   // cudaMallocHost, length = ipadapter_scale_pinned_cap_
  int ipadapter_scale_pinned_cap_ = 0;
  std::vector<float> ipadapter_scale_last_;    // last host values uploaded (to skip redundant H2D)

  void loadEngine(const std::string& engine_path);
  bool needsReallocation(int batch, int height, int width, int seq_len, int hidden_dim, int pooled_dim = 0);
};

/**
 * ControlNet wrapper for TensorRT (SD1.5).
 *
 * Inputs:
 *   - sample:                [batch, 4, H, W]    FP16  (the x_t latent the UNet sees)
 *   - timestep:              [batch]             FP32
 *   - encoder_hidden_states: [batch, 77, 768]    FP16
 *   - controlnet_cond:       [batch, 3, 8H, 8W]  FP16  (IMAGE-space control, range [0,1])
 *   - onnx::Cast_4:          scalar              FP32  (conditioning_scale; BAKED into the residuals)
 *
 * Outputs (12 down + 1 mid, FP16), channels [320,320,320,320,640,640,640,1280,1280,1280,1280,1280] + mid 1280:
 *   - down_block_00 .. down_block_11, mid_block
 *
 * The residuals come out already scaled by conditioning_scale, so the UNet binds them as-is.
 */
class ControlNetWrapper
{
public:
  static constexpr int MAX_DOWN = 12;  // SD1.5 = 12, SDXL = 9 (numDown() reports the actual count)

  explicit ControlNetWrapper(const std::string& engine_path);
  ~ControlNetWrapper();

  ControlNetWrapper(const ControlNetWrapper&) = delete;
  ControlNetWrapper& operator=(const ControlNetWrapper&) = delete;

  int numDown() const { return num_down_; }   // 12 (SD1.5) or 9 (SDXL)
  bool isSdxl() const { return is_sdxl_; }     // engine has text_embeds/time_ids inputs

  /**
   * Run ControlNet inference. Outputs are written to internal persistent buffers; the returned
   * pointers stay valid until the next forward() call. down_out is filled with numDown() pointers.
   *
   * @param sample            [batch,4,H,W] FP16 (the latent; same x_t fed to the UNet)
   * @param timestep          [batch] FP32
   * @param encoder_hidden_states [batch,77,hidden_dim] FP16
   * @param controlnet_cond   [batch,3,img_h,img_w] FP16, range [0,1]
   * @param conditioning_scale scalar baked into the output residuals
   * @param text_embeds       [batch,1280] FP16 — SDXL only (nullptr for SD1.5)
   * @param time_ids          [batch,6] FP16 — SDXL only (nullptr for SD1.5)
   * @param down_out          OUT: numDown() device ptrs (down_block_00..)
   * @param mid_out           OUT: device ptr (mid_block)
   */
  void forward(
      const __half* sample, const float* timestep, const __half* encoder_hidden_states,
      const __half* controlnet_cond, float conditioning_scale,
      const __half* text_embeds, const __half* time_ids,
      int batch, int height, int width, int img_height, int img_width, int seq_len, int hidden_dim,
      int pooled_dim, const __half** down_out, const __half** mid_out, cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // The scale scalar input name varies by trace: SD1.5 traces it as "onnx::Cast_4"; SDXL declares a
  // proper "conditioning_scale" input. Detected at load (lone scalar FP32 input).
  std::string scale_input_name_;
  int num_down_ = 12;       // actual down_block_* output count (detected at load)
  bool is_sdxl_ = false;    // engine declares text_embeds/time_ids inputs

  struct ShapeCache { int batch=0,height=0,width=0,img_h=0,img_w=0,seq_len=0,hidden_dim=0,pooled_dim=0; bool init=false; };
  ShapeCache shape_cache_;

  std::unique_ptr<CUDATensor<__half>> sample_buffer_;
  std::unique_ptr<CUDATensor<float>> timestep_buffer_;
  std::unique_ptr<CUDATensor<__half>> ehs_buffer_;
  std::unique_ptr<CUDATensor<__half>> cond_buffer_;
  std::unique_ptr<CUDATensor<float>> scale_buffer_;
  // PINNED host staging for the scalar conditioning_scale (was H2D from a stack local -> illegal in CUDA-graph
  // capture). Skip the H2D when unchanged so a captured graph replays H2D-free.
  float* scale_pinned_ = nullptr;
  float scale_last_ = -1e30f;
  std::unique_ptr<CUDATensor<__half>> text_embeds_buffer_;  // SDXL
  std::unique_ptr<CUDATensor<__half>> time_ids_buffer_;     // SDXL
  std::vector<std::unique_ptr<CUDATensor<__half>>> down_buffers_;
  std::unique_ptr<CUDATensor<__half>> mid_buffer_;

  void loadEngine(const std::string& engine_path);
};

/**
 * VAE Encoder wrapper for TensorRT-RTX
 *
 * Input:
 *   - images: [batch, 3, height, width] FP16
 *
 * Output:
 *   - latent: [batch, 4, height/8, width/8] FP16
 */
class VAEEncoderWrapper
{
public:
  VAEEncoderWrapper(const std::string& engine_path);
  ~VAEEncoderWrapper();

  // No copy
  VAEEncoderWrapper(const VAEEncoderWrapper&) = delete;
  VAEEncoderWrapper& operator=(const VAEEncoderWrapper&) = delete;

  /**
     * Encode image to latent
     *
     * @param images Input images [batch, 3, height, width] FP16
     * @param latent Output latent [batch, 4, height/8, width/8] FP16 (engine outputs FP16)
     * @param batch Batch size
     * @param height Image height
     * @param width Image width
     * @param stream CUDA stream
     */
  void encode(
      const __half* images, __half* latent, int batch, int height, int width,
      cudaStream_t stream);

  void encode(
      const float* images, __half* latent, int batch, int height, int width,
      cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;  // Shared cached engine
  std::unique_ptr<nvinfer1::IExecutionContext> context_; // Per-wrapper context

  struct ShapeCache
  {
    int batch = 0;
    int height = 0;
    int width = 0;
    bool initialized = false;
  };
  ShapeCache shape_cache_;

  // Persistent buffer for FP32 input (allocated once, reused for all inferences)
  // This matches Python's behavior of allocating buffers once during initialization
  std::unique_ptr<CUDATensor<float>> images_fp32_buffer_;

  void loadEngine(const std::string& engine_path);
  bool needsReallocation(int batch, int height, int width);
};

/**
 * VAE Decoder wrapper for TensorRT-RTX
 *
 * Input:
 *   - latent: [batch, 4, height, width] FP16
 *
 * Output:
 *   - images: [batch, 3, height*8, width*8] FP16
 */
class VAEDecoderWrapper
{
public:
  VAEDecoderWrapper(const std::string& engine_path);
  ~VAEDecoderWrapper();

  // No copy
  VAEDecoderWrapper(const VAEDecoderWrapper&) = delete;
  VAEDecoderWrapper& operator=(const VAEDecoderWrapper&) = delete;

  /**
     * Decode latent to image
     *
     * @param latent Input latent [batch, 4, height, width] FP32
     * @param images Output images [batch, 3, height*8, width*8] FP16
     * @param batch Batch size
     * @param height Latent height
     * @param width Latent width
     * @param stream CUDA stream
     */
  void decode(
      const __half* latent, __half* images, int batch, int height, int width,
      cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;  // Shared cached engine
  std::unique_ptr<nvinfer1::IExecutionContext> context_; // Per-wrapper context

  struct ShapeCache
  {
    int batch = 0;
    int height = 0;
    int width = 0;
    bool initialized = false;
  };
  ShapeCache shape_cache_;

  // Persistent buffer for FP32 input (allocated once, reused for all inferences)
  // This matches Python's behavior of allocating buffers once during initialization
  std::unique_ptr<CUDATensor<float>> latent_fp32_buffer_;

  void loadEngine(const std::string& engine_path);
  bool needsReallocation(int batch, int height, int width);
};

/**
 * CLIP Text Encoder wrapper for TensorRT-RTX
 *
 * Input:
 *   - input_ids: [batch, 77] INT32
 *
 * Output:
 *   - text_embeddings: [batch, 77, 768] FP16
 */
class CLIPWrapper
{
public:
  CLIPWrapper(const std::string& engine_path);
  ~CLIPWrapper();

  // No copy
  CLIPWrapper(const CLIPWrapper&) = delete;
  CLIPWrapper& operator=(const CLIPWrapper&) = delete;

  /**
     * Encode token IDs to text embeddings
     *
     * @param input_ids Token IDs [batch, 77] INT32
     * @param text_embeddings Output embeddings [batch, 77, hidden_dim] FP16
     *                        hidden_dim is queried from engine (768 for SD1.5, 1024 for SD-Turbo)
     * @param batch Batch size
     * @param stream CUDA stream
     * @param pooler_output Optional output for pooler embeddings (SDXL CLIP2 only) [batch, pooled_dim] FP16
     */
  void encode(
      const int32_t* input_ids, __half* text_embeddings, int batch, cudaStream_t stream,
      __half* pooler_output = nullptr);

  /**
 * Compute prompt embeddings using CLIP engine with dynamic tokenization.
 * Tokenizes the prompt on-the-fly using the Rust tokenizer.
 * Returns device pointer to embeddings (must be freed by caller).
 *
 * @param prompt Text prompt to encode
 * @param stream CUDA stream
 * @param pad_token Token ID for padding (0 for SD-Turbo, 49407 for SD1.5)
 */
  __half*
  computeEmbeddings(const std::string& prompt, cudaStream_t stream, int pad_token = 0);

  /**
 * Compute prompt embeddings with optional pooled output (for SDXL CLIP encoder 2).
 * Tokenizes the prompt on-the-fly using the Rust tokenizer.
 * Returns device pointer to embeddings (must be freed by caller).
 *
 * @param prompt Text prompt to encode
 * @param stream CUDA stream
 * @param pad_token Token ID for padding
 * @param pooled_output Optional output pointer for pooled embeddings [batch, pooled_dim] (can be nullptr)
 * @return Device pointer to text embeddings [batch, 77, hidden_dim]
 */
  __half* computeEmbeddingsWithPooled(
      const std::string& prompt, cudaStream_t stream, int pad_token,
      __half** pooled_output);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;  // Shared cached engine
  std::unique_ptr<nvinfer1::IExecutionContext> context_; // Per-wrapper context

  struct ShapeCache
  {
    int batch = 0;
    bool initialized = false;
  };
  ShapeCache shape_cache_;

  void loadEngine(const std::string& engine_path);
  bool needsReallocation(int batch);
};

/**
 * IP-Adapter CLIP image encoder + projection.
 *
 * Two engines, run back to back to turn a raw style image into IP-Adapter image tokens:
 *   1. clip_image_encoder: pixel_values [B,3,224,224] fp16 -> image_embeds [B,1024] fp16
 *      (CLIP ViT-H/14 CLIPVisionModelWithProjection, from h94/IP-Adapter image_encoder).
 *   2. ip_image_proj:      image_embeds [B,1024] fp16     -> ip_tokens [B,num_tokens,768] fp16
 *      (ImageProjModel = Linear(1024 -> num_tokens*768) + LayerNorm, from the ip-adapter .bin).
 *
 * The pixel_values are produced on-device by launch_clip_image_preprocess_fp16 (Lanczos-3 resize +
 * CLIP normalize). The NEGATIVE tokens (base IP-Adapter) = projection of ZEROS: run the SAME proj
 * engine on a [B,1024] zero buffer (the encoder is NOT re-run). batch fixed to 1 here.
 */
class CLIPImageEncoderWrapper
{
public:
  CLIPImageEncoderWrapper(
      const std::string& encoder_engine_path, const std::string& proj_engine_path);
  ~CLIPImageEncoderWrapper();

  CLIPImageEncoderWrapper(const CLIPImageEncoderWrapper&) = delete;
  CLIPImageEncoderWrapper& operator=(const CLIPImageEncoderWrapper&) = delete;

  /// Number of image tokens the proj engine emits (queried from the ip_tokens output, e.g. 4).
  int numTokens() const { return num_tokens_; }
  /// Cross-attention dim of the image tokens (queried, e.g. 768).
  int tokenDim() const { return token_dim_; }

  /**
   * Encode a host RGBA image into IP-Adapter tokens (pos + neg).
   * @param cpu_rgba host RGBA uint8 [in_h, in_w, 4]
   * @param pos_out  device fp16 [num_tokens, dim] (allocated by caller, sized numTokens()*tokenDim())
   * @param neg_out  device fp16 [num_tokens, dim] (allocated by caller); projection of zeros
   * @param stream   CUDA stream
   */
  void encodeImage(
      const uint8_t* cpu_rgba, int in_h, int in_w, __half* pos_out, __half* neg_out,
      cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> enc_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> enc_context_;
  std::shared_ptr<CachedTensorRTEngine> proj_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> proj_context_;

  int num_tokens_ = 0;
  int token_dim_ = 0;

  // Scratch device buffers (lazily sized).
  std::unique_ptr<CUDATensor<uint8_t>> d_rgba_;       // upload buffer for host RGBA
  std::unique_ptr<CUDATensor<__half>> d_pixel_;       // [1,3,224,224] preprocessed
  std::unique_ptr<CUDATensor<__half>> d_image_embeds_;// [1,1024]
  std::unique_ptr<CUDATensor<__half>> d_zero_embeds_; // [1,1024] zeros (for neg)

  void loadEngines(const std::string& encoder_engine_path, const std::string& proj_engine_path);
};

struct SDXLPromptEmbeddings
{
  __half* embeddings{};
  __half* pooled_embeds{};
  __half* time_ids{};
};

SDXLPromptEmbeddings computeClipEmbeddings_SDXL(
    CLIPWrapper& clip1, CLIPWrapper& clip2, const std::string& prompt, int batch_size,
    int height, int width, cudaStream_t stream);

} // namespace librediffusion
