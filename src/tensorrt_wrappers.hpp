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

  void loadEngine(const std::string& engine_path);
  bool needsReallocation(int batch, int height, int width, int seq_len, int hidden_dim, int pooled_dim = 0);
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
