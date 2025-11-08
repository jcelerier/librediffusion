/**
 * TensorRT-RTX Model Wrappers Implementation
 */

#include "tensorrt_wrappers.hpp"

#include "clip_tokenizer_c.h"
#include "kernels.hpp"
#include "model_cache.hpp"
#include "librediffusion.hpp" // For CUDATensor

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace librediffusion
{

// ============================================================================
// TensorRTLogger Implementation
// ============================================================================

std::string TensorRTLogger::severityToString(Severity severity)
{
  switch(severity)
  {
    case Severity::kVERBOSE:
      return "VERBOSE";
    case Severity::kINFO:
      return "INFO";
    case Severity::kWARNING:
      return "WARNING";
    case Severity::kERROR:
      return "ERROR";
    case Severity::kINTERNAL_ERROR:
      return "INTERNAL_ERROR";
    default:
      return "UNKNOWN";
  }
}

void TensorRTLogger::log(Severity severity, const char* msg) noexcept
{
  // Only log warnings and errors to avoid spam
  if(severity < Severity::kWARNING)
  {
    std::cout << "[TensorRT-RTX " << severityToString(severity) << "] " << msg
              << std::endl;
  }
}

// ============================================================================
// UNetWrapper Implementation
// ============================================================================

UNetWrapper::UNetWrapper(const std::string& engine_path, bool use_v2v)
    : use_v2v_(use_v2v)
{
  loadEngine(engine_path);

  // Initialize attention buffers for V2V mode
  if(use_v2v_)
  {
    attention_output_buffers_.reserve(NUM_ATTENTION_OUTPUTS);
    // Buffers will be allocated on first forward pass when we know the sizes
  }
}

UNetWrapper::~UNetWrapper()
{
  // Unique_ptrs will handle cleanup automatically
}

void UNetWrapper::loadEngine(const std::string& engine_path)
{
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      cached_engine_->createExecutionContext());

  if(!context_)
  {
    throw std::runtime_error("Failed to create execution context");
  }
}

bool UNetWrapper::needsReallocation(
    int batch, int height, int width, int seq_len, int hidden_dim, int pooled_dim)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch
         || shape_cache_.height != height || shape_cache_.width != width
         || shape_cache_.seq_len != seq_len || shape_cache_.hidden_dim != hidden_dim
         || shape_cache_.pooled_dim != pooled_dim;
}

void UNetWrapper::forward(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    cudaStream_t stream)
{
  // Calculate buffer sizes
  int sample_size = batch * 4 * height * width;
  int timestep_size = batch;
  int encoder_size = batch * seq_len * hidden_dim;
  int output_size = batch * 4 * height * width;

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim);

  // Allocate persistent buffers on first call or when shapes change (matches Python)
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_size);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);

    // Update shape cache
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.initialized = true;
  }

  // Copy input data into persistent buffers
  cudaMemcpyAsync(
      sample_buffer_->data(), sample, sample_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      timestep_buffer_->data(), timestep, timestep_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      encoder_hidden_states_buffer_->data(), encoder_hidden_states,
      encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

  // CRITICAL: Synchronize stream after copies to ensure TensorRT reads complete data
  // Even though Python doesn't explicitly sync, tensor.copy_() likely handles this internally
  // Without this, TensorRT may read stale/incomplete data from persistent buffers
  cudaStreamSynchronize(stream);

  // CRITICAL: Set input shapes BEFORE setting tensor addresses (matches Python's exact order)
  // Python calls set_input_shape() in infer(), right before set_tensor_address()
  // This happens even if shapes haven't changed (though Python caches to skip redundant calls)
  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims;
  timestep_dims.nbDims = 1;
  timestep_dims.d[0] = batch;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};

  if(!context_->setInputShape("sample", sample_dims))
  {
    throw std::runtime_error("Failed to set sample shape");
  }
  if(!context_->setInputShape("timestep", timestep_dims))
  {
    throw std::runtime_error("Failed to set timestep shape");
  }
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  }

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses using persistent buffer addresses (stays constant across calls)
  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
  {
    throw std::runtime_error("Failed to set sample tensor address");
  }
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
  {
    throw std::runtime_error("Failed to set timestep tensor address");
  }
  if(!context_->setTensorAddress(
         "encoder_hidden_states", encoder_hidden_states_buffer_->data()))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  }
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // StreamV2V: Set attention output tensor addresses
  if(use_v2v_)
  {
    // Allocate attention buffers on first call
    if(attention_output_buffers_.empty())
    {
      int seq_len_spatial = height * width;  // Spatial sequence length
      for(int i = 0; i < NUM_ATTENTION_OUTPUTS; i++)
      {
        // Get the actual shape from the engine for this attention output
        std::string attn_name = "attention_" + std::to_string(i);
        nvinfer1::Dims attn_dims = context_->getTensorShape(attn_name.c_str());

        // attn_dims should be [batch, seq_len, hidden_dim]
        int attn_hidden_dim = attn_dims.d[2];  // Hidden dimension varies per layer
        int attn_size = batch * seq_len_spatial * attn_hidden_dim;

        attention_output_buffers_.push_back(
            std::make_unique<CUDATensor<__half>>(attn_size));
      }
    }

    // Set tensor addresses for attention outputs
    for(int i = 0; i < NUM_ATTENTION_OUTPUTS; i++)
    {
      std::string attn_name = "attention_" + std::to_string(i);
      if(!context_->setTensorAddress(attn_name.c_str(), attention_output_buffers_[i]->data()))
      {
        throw std::runtime_error("Failed to set " + attn_name + " tensor address");
      }
    }
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }

  // Copy output from persistent buffer back to caller's buffer
  cudaMemcpyAsync(
      output, output_buffer_->data(), output_size * sizeof(__half),
      cudaMemcpyDeviceToDevice, stream);

  // TODO: StreamV2V attention outputs are now in attention_output_buffers_
  // These will be used for feature injection in LibreDiffusionPipeline
}

void UNetWrapper::forward_sdxl(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    const __half* text_embeds, const __half* time_ids,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    int pooled_dim, cudaStream_t stream)
{
  // Calculate buffer sizes
  int sample_size = batch * 4 * height * width;
  int timestep_size = batch;
  int encoder_size = batch * seq_len * hidden_dim;
  int output_size = batch * 4 * height * width;
  int text_embeds_size = batch * pooled_dim;
  int time_ids_size = batch * 6;  // SDXL uses 6 time IDs

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim, pooled_dim);

  // Allocate persistent buffers on first call or when shapes change (matches Python)
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_size);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    text_embeds_buffer_ = std::make_unique<CUDATensor<__half>>(text_embeds_size);
    time_ids_buffer_ = std::make_unique<CUDATensor<__half>>(time_ids_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);

    // Update shape cache
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.pooled_dim = pooled_dim;
    shape_cache_.initialized = true;
  }

  // Copy input data into persistent buffers
  cudaMemcpyAsync(
      sample_buffer_->data(), sample, sample_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      timestep_buffer_->data(), timestep, timestep_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      encoder_hidden_states_buffer_->data(), encoder_hidden_states,
      encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      text_embeds_buffer_->data(), text_embeds,
      text_embeds_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      time_ids_buffer_->data(), time_ids,
      time_ids_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

  // CRITICAL: Synchronize stream after copies to ensure TensorRT reads complete data
  cudaStreamSynchronize(stream);

  // CRITICAL: Set input shapes BEFORE setting tensor addresses (matches Python's exact order)
  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims;
  timestep_dims.nbDims = 1;
  timestep_dims.d[0] = batch;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  nvinfer1::Dims text_embeds_dims;
  text_embeds_dims.nbDims = 2;
  text_embeds_dims.d[0] = batch;
  text_embeds_dims.d[1] = pooled_dim;
  nvinfer1::Dims time_ids_dims;
  time_ids_dims.nbDims = 2;
  time_ids_dims.d[0] = batch;
  time_ids_dims.d[1] = 6;

  if(!context_->setInputShape("sample", sample_dims))
  {
    throw std::runtime_error("Failed to set sample shape");
  }
  if(!context_->setInputShape("timestep", timestep_dims))
  {
    throw std::runtime_error("Failed to set timestep shape");
  }
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  }
  if(!context_->setInputShape("text_embeds", text_embeds_dims))
  {
    throw std::runtime_error("Failed to set text_embeds shape");
  }
  if(!context_->setInputShape("time_ids", time_ids_dims))
  {
    throw std::runtime_error("Failed to set time_ids shape");
  }

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses using persistent buffer addresses (stays constant across calls)
  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
  {
    throw std::runtime_error("Failed to set sample tensor address");
  }
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
  {
    throw std::runtime_error("Failed to set timestep tensor address");
  }
  if(!context_->setTensorAddress(
         "encoder_hidden_states", encoder_hidden_states_buffer_->data()))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  }
  if(!context_->setTensorAddress("text_embeds", text_embeds_buffer_->data()))
  {
    throw std::runtime_error("Failed to set text_embeds tensor address");
  }
  if(!context_->setTensorAddress("time_ids", time_ids_buffer_->data()))
  {
    throw std::runtime_error("Failed to set time_ids tensor address");
  }
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }

  // Copy output from persistent buffer back to caller's buffer
  cudaMemcpyAsync(
      output, output_buffer_->data(), output_size * sizeof(__half),
      cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// VAEEncoderWrapper Implementation
// ============================================================================

VAEEncoderWrapper::VAEEncoderWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

VAEEncoderWrapper::~VAEEncoderWrapper()
{
  // Unique_ptrs will handle cleanup
}

void VAEEncoderWrapper::loadEngine(const std::string& engine_path)
{
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      cached_engine_->createExecutionContext());

  if(!context_)
  {
    throw std::runtime_error("Failed to create execution context");
  }
}

bool VAEEncoderWrapper::needsReallocation(int batch, int height, int width)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch
         || shape_cache_.height != height || shape_cache_.width != width;
}

void VAEEncoderWrapper::encode(
    const __half* images, __half* latent, int batch, int height, int width,
    cudaStream_t stream)
{
  // Allocate persistent FP32 buffer on first call or if shape changed
  // This matches Python's behavior of allocating buffers once
  int images_elements = batch * 3 * height * width;

  if(!images_fp32_buffer_ || needsReallocation(batch, height, width))
  {
    images_fp32_buffer_ = std::make_unique<CUDATensor<float>>(images_elements);

    // Update shape cache
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.initialized = true;
  }

  // ALWAYS set input shape to match Python (don't cache)
  nvinfer1::Dims4 images_dims{batch, 3, height, width};
  if(!context_->setInputShape("images", images_dims))
  {
    throw std::runtime_error("Failed to set images shape");
  }

  // Convert FP16→FP32 into persistent buffer (reuse same buffer each time)
  // This ensures the buffer address stays constant across all calls
  launch_fp16_to_fp32(images, images_fp32_buffer_->data(), images_elements, stream);

  // CRITICAL: Synchronize stream to ensure conversion is complete before TensorRT uses the buffer
  // Without this, TensorRT might read incomplete/wrong data from the FP32 buffer
  cudaStreamSynchronize(stream);

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses (use persistent buffer address)
  if(!context_->setTensorAddress("images", images_fp32_buffer_->data()))
  {
    throw std::runtime_error("Failed to set images tensor address");
  }
  if(!context_->setTensorAddress("latent", latent))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }
}

void VAEEncoderWrapper::encode(
    const float* images, __half* latent, int batch, int height, int width,
    cudaStream_t stream)
{
  // Allocate persistent FP32 buffer on first call or if shape changed
  // This matches Python's behavior of allocating buffers once
  int images_elements = batch * 3 * height * width;

  if(!images_fp32_buffer_ || needsReallocation(batch, height, width))
  {
    images_fp32_buffer_ = std::make_unique<CUDATensor<float>>(images_elements);

    // Update shape cache
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.initialized = true;
  }

  // ALWAYS set input shape to match Python (don't cache)
  nvinfer1::Dims4 images_dims{batch, 3, height, width};
  if(!context_->setInputShape("images", images_dims))
  {
    throw std::runtime_error("Failed to set images shape");
  }

  cudaMemcpy(
      images_fp32_buffer_->data(), (const void*)images, images_elements * sizeof(float),
      cudaMemcpyHostToDevice);

  cudaStreamSynchronize(stream);

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses (use persistent buffer address)
  if(!context_->setTensorAddress("images", images_fp32_buffer_->data()))
  {
    throw std::runtime_error("Failed to set images tensor address");
  }
  if(!context_->setTensorAddress("latent", latent))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }
}

// ============================================================================
// VAEDecoderWrapper Implementation
// ============================================================================

VAEDecoderWrapper::VAEDecoderWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

VAEDecoderWrapper::~VAEDecoderWrapper()
{
  // Unique_ptrs will handle cleanup
}

void VAEDecoderWrapper::loadEngine(const std::string& engine_path)
{
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      cached_engine_->createExecutionContext());

  if(!context_)
  {
    throw std::runtime_error("Failed to create execution context");
  }
}

bool VAEDecoderWrapper::needsReallocation(int batch, int height, int width)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch
         || shape_cache_.height != height || shape_cache_.width != width;
}

void VAEDecoderWrapper::decode(
    const __half* latent, __half* images, int batch, int height, int width,
    cudaStream_t stream)
{
  // Allocate persistent FP32 buffer on first call or if shape changed
  // This matches Python's behavior of allocating buffers once
  int latent_elements = batch * 4 * height * width;
  if(!latent_fp32_buffer_ || needsReallocation(batch, height, width))
  {
    latent_fp32_buffer_ = std::make_unique<CUDATensor<float>>(latent_elements);

    // Set input shape (only needs to be done once or when shape changes)
    nvinfer1::Dims4 latent_dims{batch, 4, height, width};
    if(!context_->setInputShape("latent", latent_dims))
    {
      throw std::runtime_error("Failed to set latent shape");
    }

    // Update shape cache
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.initialized = true;
  }

  // Convert FP16→FP32 into persistent buffer (reuse same buffer each time)
  // This ensures the buffer address stays constant across all calls
  launch_fp16_to_fp32(latent, latent_fp32_buffer_->data(), latent_elements, stream);

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses (use persistent buffer address)
  if(!context_->setTensorAddress("latent", latent_fp32_buffer_->data()))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }
  if(!context_->setTensorAddress("images", images))
  {
    throw std::runtime_error("Failed to set images tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }
}

// ============================================================================
// CLIPWrapper Implementation
// ============================================================================

CLIPWrapper::CLIPWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

CLIPWrapper::~CLIPWrapper()
{
  context_.reset();
}

void CLIPWrapper::loadEngine(const std::string& engine_path)
{
  context_.reset();
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load CLIP engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  context_.reset(cached_engine_->createExecutionContext());
  if(!context_)
  {
    throw std::runtime_error("Failed to create CLIP execution context");
  }
}

bool CLIPWrapper::needsReallocation(int batch)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch;
}

void CLIPWrapper::encode(
    const int32_t* input_ids, __half* text_embeddings, int batch, cudaStream_t stream,
    __half* pooler_output)
{
  assert(cached_engine_ && cached_engine_->isValid());
  // Set input shape [batch, 77]
  nvinfer1::Dims input_dims;
  input_dims.nbDims = 2;
  input_dims.d[0] = batch;
  input_dims.d[1] = 77; // CLIP sequence length

  if(needsReallocation(batch))
  {
    if(!context_->setInputShape("input_ids", input_dims))
    {
      throw std::runtime_error("Failed to set CLIP input_ids shape");
    }

    shape_cache_.batch = batch;
    shape_cache_.initialized = true;
  }

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all CLIP input dimensions specified");
  }

  // Query output dimensions from engine to support both SD1.5 (768) and SD-Turbo (1024)
  nvinfer1::Dims output_dims = context_->getTensorShape("text_embeddings");
  if(output_dims.nbDims != 3)
  {
    throw std::runtime_error("Unexpected CLIP output dimensions");
  }

  int hidden_dim = output_dims.d[2];  // Extract hidden dimension from engine

  // Allocate temporary FP32 buffer for engine output
  // Engine outputs FP32 but we need FP16
  size_t output_size = batch * 77 * hidden_dim;
  float* d_output_fp32;
  cudaMalloc(&d_output_fp32, output_size * sizeof(float));

  // Check if engine has pooler_output (SDXL CLIP encoder 2)
  float* d_pooler_fp32 = nullptr;
  bool has_pooler = false;

  // Safely check if pooler_output exists by iterating I/O tensors
  //int num_io_tensors = engine_->getNbIOTensors();
  //for(int i = 0; i < num_io_tensors; ++i)
  //{
  //  const char* tensor_name = engine_->getIOTensorName(i);
  //  if(std::string(tensor_name) == "pooler_output")
  if(pooler_output)
  {
    has_pooler = true;
    nvinfer1::Dims pooler_dims = context_->getTensorShape("pooler_output");
    size_t pooler_size = batch * pooler_dims.d[1];
    cudaMalloc(&d_pooler_fp32, pooler_size * sizeof(float));
    //    break;
  }
  // }

  // Set tensor addresses
  // Note: TensorRT API requires non-const void*, so we cast away const
  if(!context_->setTensorAddress("input_ids", const_cast<int32_t*>(input_ids)))
  {
    cudaFree(d_output_fp32);
    if(d_pooler_fp32) cudaFree(d_pooler_fp32);
    throw std::runtime_error("Failed to set CLIP input_ids tensor address");
  }
  if(!context_->setTensorAddress("text_embeddings", d_output_fp32))
  {
    cudaFree(d_output_fp32);
    if(d_pooler_fp32) cudaFree(d_pooler_fp32);
    throw std::runtime_error("Failed to set CLIP text_embeddings tensor address");
  }

  // Set pooler_output address if present
  if(has_pooler && !context_->setTensorAddress("pooler_output", d_pooler_fp32))
  {
    cudaFree(d_output_fp32);
    cudaFree(d_pooler_fp32);
    throw std::runtime_error("Failed to set CLIP pooler_output tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    cudaFree(d_output_fp32);
    if(d_pooler_fp32) cudaFree(d_pooler_fp32);
    throw std::runtime_error("Failed to enqueue CLIP inference");
  }

  // Convert FP32 output to FP16
  // Use the existing fp32_to_fp16 kernel
  launch_fp32_to_fp16(d_output_fp32, text_embeddings, output_size, stream);

  // Convert pooler output to FP16 if requested
  if(pooler_output && has_pooler && d_pooler_fp32)
  {
    nvinfer1::Dims pooler_dims = context_->getTensorShape("pooler_output");
    size_t pooler_size = batch * pooler_dims.d[1];
    launch_fp32_to_fp16(d_pooler_fp32, pooler_output, pooler_size, stream);
  }

  // Wait for conversion to complete and cleanup
  cudaStreamSynchronize(stream);
  cudaFree(d_output_fp32);
  if(d_pooler_fp32) cudaFree(d_pooler_fp32);
}

__half* CLIPWrapper::computeEmbeddings(const std::string& prompt, cudaStream_t stream, int pad_token)
{
  // Tokenize the prompt using Rust tokenizer with configurable padding
  std::vector<int32_t> token_ids_host(77);
  int num_tokens = clip_tokenizer_encode_with_padding(prompt.c_str(), token_ids_host.data(), pad_token);

  if(num_tokens < 0)
  {
    throw std::runtime_error("Failed to tokenize prompt: " + prompt);
  }

  // Allocate device memory for token IDs
  int32_t* d_token_ids;
  cudaMalloc(&d_token_ids, num_tokens * sizeof(int32_t));
  cudaMemcpy(
      d_token_ids, token_ids_host.data(), num_tokens * sizeof(int32_t),
      cudaMemcpyHostToDevice);

  // Query output dimensions from engine to support both SD1.5 (768) and SD-Turbo (1024)
  // Need to set input shape first to get valid output shape
  nvinfer1::Dims input_dims;
  input_dims.nbDims = 2;
  input_dims.d[0] = 1;  // batch = 1
  input_dims.d[1] = 77; // sequence length
  context_->setInputShape("input_ids", input_dims);

  nvinfer1::Dims output_dims = context_->getTensorShape("text_embeddings");
  int hidden_dim = output_dims.d[2];  // Extract hidden dimension from engine

  // Allocate device memory for embeddings: [1, 77, hidden_dim]
  __half* d_embeddings;
  size_t embedding_size = 1 * 77 * hidden_dim * sizeof(__half);
  cudaMalloc(&d_embeddings, embedding_size);

  // Run CLIP inference
  this->encode(d_token_ids, d_embeddings, 1, stream);
  cudaStreamSynchronize(stream);

  // Cleanup token IDs
  cudaFree(d_token_ids);

  return d_embeddings;
}

__half* CLIPWrapper::computeEmbeddingsWithPooled(
    const std::string& prompt, cudaStream_t stream, int pad_token,
    __half** pooled_output)
{
  // Tokenize the prompt using Rust tokenizer with configurable padding
  std::vector<int32_t> token_ids_host(77);
  int num_tokens = clip_tokenizer_encode_with_padding(prompt.c_str(), token_ids_host.data(), pad_token);

  if(num_tokens < 0)
  {
    throw std::runtime_error("Failed to tokenize prompt: " + prompt);
  }

  // Allocate device memory for token IDs
  int32_t* d_token_ids;
  cudaMalloc(&d_token_ids, num_tokens * sizeof(int32_t));
  cudaMemcpy(
      d_token_ids, token_ids_host.data(), num_tokens * sizeof(int32_t),
      cudaMemcpyHostToDevice);

  // Set input dimensions first (required before querying output shapes)
  nvinfer1::Dims input_dims;
  input_dims.nbDims = 2;
  input_dims.d[0] = 1;  // batch = 1
  input_dims.d[1] = 77; // sequence length
  if(!context_->setInputShape("input_ids", input_dims))
  {
    cudaFree(d_token_ids);
    throw std::runtime_error("Failed to set input_ids shape");
  }

  if(!context_->allInputDimensionsSpecified())
  {
    cudaFree(d_token_ids);
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Check what kind of CLIP engine this is by checking the text_embeddings output shape
  // CLIP1 (encoder 1): text_embeddings is [batch, 77, 768] - sequence embeddings
  // CLIP2 (encoder 2): has hidden_states [batch, 77, 1280] + text_embeddings [batch, 1280]
  //
  // We detect CLIP2 by checking if text_embeddings is 2D (pooled) instead of 3D (sequence)

  __half* d_sequence_embeddings = nullptr;
  __half* d_pooled_embeddings = nullptr;

  nvinfer1::Dims text_embeddings_dims = context_->getTensorShape("text_embeddings");

  if(text_embeddings_dims.nbDims == 2)
  {
    // CLIP2: text_embeddings is [batch, 1280] (pooled), also has hidden_states [batch, 77, 1280]
    nvinfer1::Dims hidden_states_dims = context_->getTensorShape("hidden_states");

    if(hidden_states_dims.nbDims != 3)
    {
      cudaFree(d_token_ids);
      throw std::runtime_error("CLIP2 hidden_states has unexpected dimensions");
    }

    int hidden_dim = hidden_states_dims.d[2];  // Should be 1280 for CLIP2
    int pooled_dim = text_embeddings_dims.d[1]; // Should be 1280 for CLIP2

    // Allocate output buffers
    size_t sequence_size = 1 * 77 * hidden_dim;
    size_t pooled_size = 1 * pooled_dim;

    // IMPORTANT: Engine outputs mixed precision!
    // - hidden_states: FP32
    // - text_embeddings: FP16
    float* d_hidden_fp32;  // hidden_states output is FP32
    cudaMalloc(&d_hidden_fp32, sequence_size * sizeof(float));
    cudaMalloc(&d_pooled_embeddings, pooled_size * sizeof(__half));  // text_embeddings is FP16
    cudaMalloc(&d_sequence_embeddings, sequence_size * sizeof(__half));  // final FP16 output

    // Set tensor addresses - mixed precision
    if(!context_->setTensorAddress("input_ids", d_token_ids))
    {
      cudaFree(d_token_ids);
      cudaFree(d_hidden_fp32);
      cudaFree(d_sequence_embeddings);
      cudaFree(d_pooled_embeddings);
      throw std::runtime_error("Failed to set CLIP2 input_ids tensor address");
    }
    if(!context_->setTensorAddress("hidden_states", d_hidden_fp32))  // Engine writes FP32 here
    {
      cudaFree(d_token_ids);
      cudaFree(d_hidden_fp32);
      cudaFree(d_sequence_embeddings);
      cudaFree(d_pooled_embeddings);
      throw std::runtime_error("Failed to set CLIP2 hidden_states tensor address");
    }
    if(!context_->setTensorAddress("text_embeddings", d_pooled_embeddings))  // Engine writes FP16 here
    {
      cudaFree(d_token_ids);
      cudaFree(d_hidden_fp32);
      cudaFree(d_sequence_embeddings);
      cudaFree(d_pooled_embeddings);
      throw std::runtime_error("Failed to set CLIP2 text_embeddings tensor address");
    }

    // Run inference
    if(!context_->enqueueV3(stream))
    {
      cudaFree(d_token_ids);
      cudaFree(d_hidden_fp32);
      cudaFree(d_sequence_embeddings);
      cudaFree(d_pooled_embeddings);
      throw std::runtime_error("Failed to enqueue CLIP2 inference");
    }

    // Convert FP32 hidden_states to FP16
    launch_fp32_to_fp16(d_hidden_fp32, d_sequence_embeddings, sequence_size, stream);

    cudaStreamSynchronize(stream);

    // Cleanup
    cudaFree(d_token_ids);
    cudaFree(d_hidden_fp32);  // Free the FP32 intermediate buffer

    // Return both outputs
    if(pooled_output)
    {
      *pooled_output = d_pooled_embeddings;
    }
    else
    {
      cudaFree(d_pooled_embeddings);
    }

    return d_sequence_embeddings;
  }
  else
  {
    // CLIP1: Only has text_embeddings output [batch, 77, 768]
    nvinfer1::Dims output_dims = context_->getTensorShape("text_embeddings");

    if(output_dims.nbDims != 3)
    {
      cudaFree(d_token_ids);
      throw std::runtime_error("CLIP1 text_embeddings has unexpected dimensions");
    }

    int hidden_dim = output_dims.d[2];  // Should be 768 for CLIP1

    // Allocate output buffer
    size_t embedding_size = 1 * 77 * hidden_dim;
    cudaMalloc(&d_sequence_embeddings, embedding_size * sizeof(__half));

    // Run inference using existing encode method
    this->encode(d_token_ids, d_sequence_embeddings, 1, stream, nullptr);

    cudaStreamSynchronize(stream);
    cudaFree(d_token_ids);

    // CLIP1 has no pooled output
    if(pooled_output)
    {
      *pooled_output = nullptr;
    }

    return d_sequence_embeddings;
  }
}

SDXLPromptEmbeddings computeClipEmbeddings_SDXL(
    CLIPWrapper& clip, CLIPWrapper& clip2, const std::string& prompt, int batch_size,
    int height, int width, cudaStream_t clip_stream)
{
  SDXLPromptEmbeddings ret;

  // CLIP encoder 1: 768-dim with EOS padding (49407)
  __half* d_embeds1 = clip.computeEmbeddings(prompt, clip_stream, 49407);

  // CLIP encoder 2: 1280-dim with pooled output, PAD padding (0)
  __half* d_embeds2
      = clip2.computeEmbeddingsWithPooled(prompt, clip_stream, 0, &ret.pooled_embeds);

  // Concatenate embeddings: [batch, 77, 768+1280] = [batch, 77, 2048]
  size_t total_size = batch_size * 77 * 2048;

  cudaMalloc(&ret.embeddings, total_size * sizeof(__half));

  // Copy embeddings sequentially for each batch element and sequence position
  for(int b = 0; b < batch_size; ++b)
  {
    for(int s = 0; s < 77; ++s)
    {
      // Copy 768-dim from encoder 1
      cudaMemcpy(
          ret.embeddings + (b * 77 + s) * 2048, d_embeds1 + (b * 77 + s) * 768,
          768 * sizeof(__half), cudaMemcpyDeviceToDevice);
      // Copy 1280-dim from encoder 2
      cudaMemcpy(
          ret.embeddings + (b * 77 + s) * 2048 + 768, d_embeds2 + (b * 77 + s) * 1280,
          1280 * sizeof(__half), cudaMemcpyDeviceToDevice);
    }
  }

  cudaFree(d_embeds1);
  cudaFree(d_embeds2);

  // Prepare time_ids: [original_height, original_width, crop_top, crop_left, target_height, target_width]
  std::vector<__half> time_ids_host(batch_size * 6);
  for(int i = 0; i < batch_size; ++i)
  {
    time_ids_host[i * 6 + 0] = __half(static_cast<float>(height)); // original_height
    time_ids_host[i * 6 + 1] = __half(static_cast<float>(width));  // original_width
    time_ids_host[i * 6 + 2] = __half(0.0f);                       // crop_top
    time_ids_host[i * 6 + 3] = __half(0.0f);                       // crop_left
    time_ids_host[i * 6 + 4] = __half(static_cast<float>(height)); // target_height
    time_ids_host[i * 6 + 5] = __half(static_cast<float>(width));  // target_width
  }

  cudaMalloc(&ret.time_ids, batch_size * 6 * sizeof(__half));
  cudaMemcpy(
      ret.time_ids, time_ids_host.data(), batch_size * 6 * sizeof(__half),
      cudaMemcpyHostToDevice);

  return ret;
}

} // namespace librediffusion
