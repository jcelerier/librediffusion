/**
 * LibreDiffusion C API Implementation
 *
 * Bridges the C API to the underlying C++ implementation.
 * Compile this file with NVCC or a C++23 compiler that supports CUDA.
 */

#include "librediffusion_c.h"
#include "librediffusion.hpp"
#include "tensorrt_wrappers.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <vector>

#define LIBREDIFFUSION_VERSION_MAJOR 1
#define LIBREDIFFUSION_VERSION_MINOR 0
#define LIBREDIFFUSION_VERSION_PATCH 0
#define LIBREDIFFUSION_VERSION_STRING "1.0.0"

struct librediffusion_config_t
{
  librediffusion::LibreDiffusionConfig cpp_config;
};

struct librediffusion_pipeline_t
{
  std::unique_ptr<librediffusion::LibreDiffusionPipeline> cpp_pipeline;
};

/*===========================================================================*/
/* Thread-Local Error State                                                  */
/*===========================================================================*/

namespace
{
thread_local cudaError_t g_last_cuda_error = cudaSuccess;

void set_cuda_error(cudaError_t err)
{
  g_last_cuda_error = err;
}

librediffusion_error_t check_cuda_error()
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return LIBREDIFFUSION_ERROR_CUDA_ERROR;
  }
  return LIBREDIFFUSION_SUCCESS;
}

template <typename Func>
librediffusion_error_t try_catch_wrapper(Func&& func)
{
  try
  {
    func();
    return check_cuda_error();
  }
  catch (const std::bad_alloc&)
  {
    return LIBREDIFFUSION_ERROR_OUT_OF_MEMORY;
  }
  catch (const std::exception&)
  {
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
  catch (...)
  {
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}
} // anonymous namespace

/*===========================================================================*/
/* Half-Float Conversion Helpers                                             */
/*===========================================================================*/

namespace
{
// Reinterpret librediffusion_half_t (uint16_t) as __half
inline const __half* to_half_ptr(const librediffusion_half_t* ptr)
{
  return reinterpret_cast<const __half*>(ptr);
}

inline __half* to_half_ptr(librediffusion_half_t* ptr)
{
  return reinterpret_cast<__half*>(ptr);
}

inline cudaStream_t to_cuda_stream(librediffusion_stream_t stream)
{
  return static_cast<cudaStream_t>(stream);
}
} // anonymous namespace

/*===========================================================================*/
/* Configuration API Implementation                                          */
/*===========================================================================*/

extern "C" {

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_create(librediffusion_config_handle* config)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() { *config = new librediffusion_config_t{}; });
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_config_destroy(librediffusion_config_handle config)
{
  delete config;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_clone(
    librediffusion_config_handle src, librediffusion_config_handle* dst)
{
  if (!src || !dst)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper(
      [&]() { *dst = new librediffusion_config_t{src->cpp_config}; });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_device(librediffusion_config_handle config, int device)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.device = device;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_model_type(
    librediffusion_config_handle config, librediffusion_model_type_t type)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  switch (type)
  {
    case MODEL_SD_15:
      config->cpp_config.model_type = librediffusion::ModelType::SD_15;
      break;
    case MODEL_SD_TURBO:
      config->cpp_config.model_type = librediffusion::ModelType::SD_TURBO;
      break;
    case MODEL_SDXL_TURBO:
      config->cpp_config.model_type = librediffusion::ModelType::SDXL_TURBO;
      break;
    default:
      return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  }
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_dimensions(
    librediffusion_config_handle config, int width, int height, int latent_width,
    int latent_height)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0 || latent_width <= 0 || latent_height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  config->cpp_config.width = width;
  config->cpp_config.height = height;
  config->cpp_config.latent_width = latent_width;
  config->cpp_config.latent_height = latent_height;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_batch_size(librediffusion_config_handle config, int batch_size)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (batch_size <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  config->cpp_config.batch_size = batch_size;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_denoising_steps(librediffusion_config_handle config, int steps)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (steps <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  config->cpp_config.denoising_steps = steps;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_frame_buffer_size(
    librediffusion_config_handle config, int size)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (size <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  config->cpp_config.frame_buffer_size = size;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_guidance_scale(
    librediffusion_config_handle config, float scale)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.guidance_scale = scale;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_delta(librediffusion_config_handle config, float delta)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.delta = delta;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_add_noise(librediffusion_config_handle config, int enabled)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.do_add_noise = (enabled != 0);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_denoising_batch(
    librediffusion_config_handle config, int enabled)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.use_denoising_batch = (enabled != 0);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_seed(librediffusion_config_handle config, uint64_t seed)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.seed = seed;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_cfg_type(
    librediffusion_config_handle config, librediffusion_cfg_type_t type)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.cfg_type = static_cast<int>(type);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_text_config(
    librediffusion_config_handle config, int seq_len, int hidden_dim, int pad_token)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  config->cpp_config.text_seq_len = seq_len;
  config->cpp_config.text_hidden_dim = hidden_dim;
  config->cpp_config.clip_pad_token = pad_token;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_sdxl_config(
    librediffusion_config_handle config, int pooled_embedding_dim, int time_ids_dim)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (pooled_embedding_dim <= 0 || time_ids_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  config->cpp_config.pooled_embedding_dim = pooled_embedding_dim;
  config->cpp_config.time_ids_dim = time_ids_dim;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_unet_engine(
    librediffusion_config_handle config, const char* path)
{
  if (!config || !path)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    config->cpp_config.unet_engine_path = path;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_vae_encoder(
    librediffusion_config_handle config, const char* path)
{
  if (!config || !path)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    config->cpp_config.vae_encoder_path = path;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_vae_decoder(
    librediffusion_config_handle config, const char* path)
{
  if (!config || !path)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    config->cpp_config.vae_decoder_path = path;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_timestep_indices(
    librediffusion_config_handle config, const int* indices, size_t count)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (count > 0 && !indices)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    config->cpp_config.timestep_indices.assign(indices, indices + count);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_pipeline_mode(
    librediffusion_config_handle config, librediffusion_pipeline_mode_t mode)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  switch (mode)
  {
    case MODE_SINGLE_FRAME:
      config->cpp_config.mode = librediffusion::PipelineMode::SINGLE_FRAME;
      break;
    case MODE_TEMPORAL_V2V:
      config->cpp_config.mode = librediffusion::PipelineMode::TEMPORAL_V2V;
      break;
    default:
      return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  }
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_temporal_params(
    librediffusion_config_handle config, int use_cached_attn, int use_feature_injection,
    float injection_strength, float similarity_threshold, int cache_interval,
    int cache_maxframes)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  config->cpp_config.use_cached_attn = (use_cached_attn != 0);
  config->cpp_config.use_feature_injection = (use_feature_injection != 0);
  config->cpp_config.feature_injection_strength = injection_strength;
  config->cpp_config.feature_similarity_threshold = similarity_threshold;
  config->cpp_config.cache_interval = cache_interval;
  config->cpp_config.cache_maxframes = cache_maxframes;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_tome(
    librediffusion_config_handle config, int enabled, float ratio)
{
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  config->cpp_config.use_tome_cache = (enabled != 0);
  config->cpp_config.tome_ratio = ratio;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_width(librediffusion_config_handle config)
{
  return config ? config->cpp_config.width : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_height(librediffusion_config_handle config)
{
  return config ? config->cpp_config.height : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_latent_width(librediffusion_config_handle config)
{
  return config ? config->cpp_config.latent_width : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_latent_height(librediffusion_config_handle config)
{
  return config ? config->cpp_config.latent_height : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_batch_size(librediffusion_config_handle config)
{
  return config ? config->cpp_config.batch_size : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_denoising_steps(librediffusion_config_handle config)
{
  return config ? config->cpp_config.denoising_steps : 0;
}

/*===========================================================================*/
/* Pipeline API Implementation                                               */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_create(
    librediffusion_config_handle config, librediffusion_pipeline_handle* pipeline)
{
  if (!config || !pipeline)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    auto p = new librediffusion_pipeline_t{};
    p->cpp_pipeline = std::make_unique<librediffusion::LibreDiffusionPipeline>(
        config->cpp_config);
    *pipeline = p;
  });
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_pipeline_destroy(librediffusion_pipeline_handle pipeline)
{
  delete pipeline;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_cuda(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_cuda();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_npp(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_npp();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_engines(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_engines();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_buffers(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_buffers();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_all(librediffusion_pipeline_handle pipeline)
{
  librediffusion_error_t err;

  err = librediffusion_pipeline_init_cuda(pipeline);
  if(err != LIBREDIFFUSION_SUCCESS)
    return err;

  err = librediffusion_pipeline_init_npp(pipeline);
  if(err != LIBREDIFFUSION_SUCCESS)
    return err;

  err = librediffusion_pipeline_init_engines(pipeline);
  if(err != LIBREDIFFUSION_SUCCESS)
    return err;

  err = librediffusion_pipeline_init_buffers(pipeline);
  return err;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_reinit_buffers(
    librediffusion_pipeline_handle pipeline, librediffusion_config_handle config)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!config)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->reinit_buffers(config->cpp_config);
  });
}

/*===========================================================================*/
/* Embedding & Scheduler Preparation                                         */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_embeds(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* prompt_embeds,
    int seq_len, int hidden_dim)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!prompt_embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->prepare_embeds(
        to_half_ptr(prompt_embeds), seq_len, hidden_dim);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_null_embeds(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* null_embeds,
    int seq_len, int hidden_dim)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!null_embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->prepare_null_embeds(
        to_half_ptr(null_embeds), seq_len, hidden_dim);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_negative_embeds(
    librediffusion_pipeline_handle pipeline,
    const librediffusion_half_t* negative_embeds, int seq_len, int hidden_dim)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!negative_embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->prepare_negative_embeds(
        to_half_ptr(negative_embeds), seq_len, hidden_dim);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_blend_embeds(
    librediffusion_pipeline_handle pipeline,
    const librediffusion_half_t* const* embeddings, const float* weights,
    int num_embeddings, int seq_len, int hidden_dim)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!embeddings || !weights)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (num_embeddings <= 0 || seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  return try_catch_wrapper([&]() {
    // Convert librediffusion_half_t* const* to __half* const*
    std::vector<const __half*> embed_ptrs(num_embeddings);
    for(int i = 0; i < num_embeddings; i++)
    {
      embed_ptrs[i] = to_half_ptr(embeddings[i]);
    }
    pipeline->cpp_pipeline->blend_embeds(
        embed_ptrs.data(), weights, num_embeddings, seq_len, hidden_dim);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_sdxl_conditioning(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* text_embeds,
    const librediffusion_half_t* time_ids)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!text_embeds || !time_ids)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->prepare_sdxl_conditioning(
        to_half_ptr(text_embeds), to_half_ptr(time_ids));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_scheduler(
    librediffusion_pipeline_handle pipeline, const float* timesteps,
    const float* alpha_prod_t_sqrt, const float* beta_prod_t_sqrt, const float* c_skip,
    const float* c_out, size_t num_timesteps)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!timesteps || !alpha_prod_t_sqrt || !beta_prod_t_sqrt || !c_skip || !c_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (num_timesteps == 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

  return try_catch_wrapper([&]() {
    std::span<float> ts_span(const_cast<float*>(timesteps), num_timesteps);
    std::span<float> alpha_span(const_cast<float*>(alpha_prod_t_sqrt), num_timesteps);
    std::span<float> beta_span(const_cast<float*>(beta_prod_t_sqrt), num_timesteps);
    std::span<float> skip_span(const_cast<float*>(c_skip), num_timesteps);
    std::span<float> out_span(const_cast<float*>(c_out), num_timesteps);

    pipeline->cpp_pipeline->prepare_scheduler(
        ts_span, alpha_span, beta_span, skip_span, out_span);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_init_noise(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* noise)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!noise)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_init_noise(to_half_ptr(noise));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_reseed(librediffusion_pipeline_handle pipeline, int64_t seed)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->reseed(seed);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_guidance_scale(
    librediffusion_pipeline_handle pipeline, float guidance)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_guidance_scale(guidance);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_delta(librediffusion_pipeline_handle pipeline, float delta)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_delta(delta);
  });
}

/*===========================================================================*/
/* High-Level Inference API                                                  */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_img2img(
    librediffusion_pipeline_handle pipeline, const uint8_t* cpu_rgba_input,
    uint8_t* cpu_rgba_output, int width, int height)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cpu_rgba_input || !cpu_rgba_output)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->img2img(cpu_rgba_input, cpu_rgba_output, width, height);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_txt2img(
    librediffusion_pipeline_handle pipeline, uint8_t* cpu_rgba_output, int width,
    int height)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cpu_rgba_output)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->txt2img(cpu_rgba_output, width, height);
  });
}

/*===========================================================================*/
/* Low-Level GPU Inference API                                               */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_img2img_gpu_half(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* image_in,
    librediffusion_half_t* image_out, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_in || !image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->img2img_impl(
        to_half_ptr(image_in),
        to_half_ptr(image_out),
        to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_img2img_gpu_float(
    librediffusion_pipeline_handle pipeline, const float* image_in,
    librediffusion_half_t* image_out, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_in || !image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->img2img_impl(
        image_in,
        to_half_ptr(image_out),
        to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_txt2img_gpu(
    librediffusion_pipeline_handle pipeline, librediffusion_half_t* image_out,
    librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->txt2img_impl(
        to_half_ptr(image_out),
        to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_txt2img_sd_turbo_gpu(
    librediffusion_pipeline_handle pipeline, librediffusion_half_t* image_out,
    librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->txt2img_sd_turbo_impl(
        to_half_ptr(image_out), to_cuda_stream(stream));
  });
}

/*===========================================================================*/
/* VAE Operations                                                            */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_encode_image_half(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* image,
    librediffusion_half_t* latent_out, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image || !latent_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->encode_image(
        to_half_ptr(image),
        to_half_ptr(latent_out),
        to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_encode_image_float(
    librediffusion_pipeline_handle pipeline, const float* image,
    librediffusion_half_t* latent_out, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image || !latent_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->encode_image(
        image,
        to_half_ptr(latent_out),
        to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_decode_latent(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* latent,
    librediffusion_half_t* image_out, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!latent || !image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->decode_latent(
        to_half_ptr(latent),
        to_half_ptr(image_out),
        to_cuda_stream(stream));
  });
}

/*===========================================================================*/
/* UNet Operations                                                           */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_predict_x0_batch(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* x_t_latent_in,
    librediffusion_half_t* x_0_pred_out, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!x_t_latent_in || !x_0_pred_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->predict_x0_batch(
        to_half_ptr(x_t_latent_in),
        to_half_ptr(x_0_pred_out),
        to_cuda_stream(stream));
  });
}

/*===========================================================================*/
/* Image Format Conversion                                                   */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_rgba_nhwc_to_nchw_float(
    librediffusion_pipeline_handle pipeline, const uint8_t* rgba_nhwc_in,
    float* rgb_nchw_out, int width, int height, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!rgba_nhwc_in || !rgb_nchw_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->rgba_nhwc_to_nchw_gpu(
        rgba_nhwc_in, rgb_nchw_out, width, height, to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_rgba_nhwc_to_nchw_half(
    librediffusion_pipeline_handle pipeline, const uint8_t* rgba_nhwc_in,
    librediffusion_half_t* rgb_nchw_out, int width, int height,
    librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!rgba_nhwc_in || !rgb_nchw_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->rgba_nhwc_to_nchw_gpu(
        rgba_nhwc_in, to_half_ptr(rgb_nchw_out), width, height, to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_nchw_half_to_rgba_nhwc(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* rgb_nchw_in,
    uint8_t* rgba_nhwc_out, int width, int height, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!rgb_nchw_in || !rgba_nhwc_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->nchw_to_rgba_nhwc_gpu(
        to_half_ptr(rgb_nchw_in), rgba_nhwc_out, width, height, to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_nchw_float_to_rgba_nhwc(
    librediffusion_pipeline_handle pipeline, const float* rgb_nchw_in,
    uint8_t* rgba_nhwc_out, int width, int height, librediffusion_stream_t stream)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!rgb_nchw_in || !rgba_nhwc_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->nchw_to_rgba_nhwc_gpu(
        rgb_nchw_in, rgba_nhwc_out, width, height, to_cuda_stream(stream));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_rgba_resize(
    librediffusion_pipeline_handle pipeline, uint8_t* rgba_input, int in_width,
    int in_height, uint8_t* rgba_output, int out_width, int out_height)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!rgba_input || !rgba_output)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (in_width <= 0 || in_height <= 0 || out_width <= 0 || out_height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->rgba_resize(
        rgba_input, in_width, in_height,
        rgba_output, out_width, out_height);
  });
}

/*===========================================================================*/
/* Temporal Coherence                                                        */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_enable_temporal_coherence(
    librediffusion_pipeline_handle pipeline, int use_feature_injection,
    float injection_strength, float similarity_threshold, int cache_interval,
    int max_cached_frames)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->enableTemporalCoherence(
        use_feature_injection != 0,
        injection_strength,
        similarity_threshold,
        cache_interval,
        max_cached_frames);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_disable_temporal_coherence(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->disableTemporalCoherence();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_reset_temporal_state(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->resetTemporalState();
  });
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_get_current_frame_id(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return -1;

  return pipeline->cpp_pipeline->getCurrentFrameId();
}

/*===========================================================================*/
/* CLIP Text Encoder                                                         */
/*===========================================================================*/

struct librediffusion_clip_t
{
  std::unique_ptr<librediffusion::CLIPWrapper> cpp_clip;
};

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_clip_create(const char* engine_path, librediffusion_clip_handle* clip)
{
  if (!engine_path || !clip)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    auto c = new librediffusion_clip_t{};
    c->cpp_clip = std::make_unique<librediffusion::CLIPWrapper>(engine_path);
    *clip = c;
  });
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_clip_destroy(librediffusion_clip_handle clip)
{
  delete clip;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_clip_compute_embeddings(
    librediffusion_clip_handle clip, const char* prompt, int pad_token,
    librediffusion_stream_t stream, librediffusion_half_t** embeddings)
{
  if (!clip || !clip->cpp_clip)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!prompt || !embeddings)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    __half* result = clip->cpp_clip->computeEmbeddings(
        prompt, to_cuda_stream(stream), pad_token);
    *embeddings = reinterpret_cast<librediffusion_half_t*>(result);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_clip_compute_embeddings_sdxl(
    librediffusion_clip_handle clip1, librediffusion_clip_handle clip2,
    const char* prompt, int batch_size, int height, int width,
    librediffusion_stream_t stream, librediffusion_half_t** embeddings,
    librediffusion_half_t** pooled_embeds, librediffusion_half_t** time_ids)
{
  if (!clip1 || !clip1->cpp_clip || !clip2 || !clip2->cpp_clip)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!prompt || !embeddings || !pooled_embeds || !time_ids)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    auto result = librediffusion::computeClipEmbeddings_SDXL(
        *clip1->cpp_clip, *clip2->cpp_clip,
        prompt, batch_size, height, width,
        to_cuda_stream(stream));

    *embeddings = reinterpret_cast<librediffusion_half_t*>(result.embeddings);
    *pooled_embeds = reinterpret_cast<librediffusion_half_t*>(result.pooled_embeds);
    *time_ids = reinterpret_cast<librediffusion_half_t*>(result.time_ids);
  });
}

/*===========================================================================*/
/* Utility Functions                                                         */
/*===========================================================================*/

LIBREDIFFUSION_API const char* LIBREDIFFUSION_CALL
librediffusion_error_string(librediffusion_error_t error)
{
  switch (error)
  {
    case LIBREDIFFUSION_SUCCESS:
      return "Success";
    case LIBREDIFFUSION_ERROR_INVALID_ARGUMENT:
      return "Invalid argument";
    case LIBREDIFFUSION_ERROR_NULL_POINTER:
      return "Null pointer";
    case LIBREDIFFUSION_ERROR_OUT_OF_MEMORY:
      return "Out of memory";
    case LIBREDIFFUSION_ERROR_CUDA_ERROR:
      return "CUDA error";
    case LIBREDIFFUSION_ERROR_ENGINE_LOAD_FAILED:
      return "Failed to load TensorRT engine";
    case LIBREDIFFUSION_ERROR_NOT_INITIALIZED:
      return "Pipeline not initialized";
    case LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS:
      return "Invalid dimensions";
    case LIBREDIFFUSION_ERROR_FILE_NOT_FOUND:
      return "File not found";
    case LIBREDIFFUSION_ERROR_INTERNAL:
    default:
      return "Internal error";
  }
}

LIBREDIFFUSION_API const char* LIBREDIFFUSION_CALL librediffusion_version(void)
{
  return LIBREDIFFUSION_VERSION_STRING;
}

LIBREDIFFUSION_API librediffusion_half_t LIBREDIFFUSION_CALL
librediffusion_float_to_half(float value)
{
  __half h = __float2half(value);
  librediffusion_half_t result;
  std::memcpy(&result, &h, sizeof(result));
  return result;
}

LIBREDIFFUSION_API float LIBREDIFFUSION_CALL
librediffusion_half_to_float(librediffusion_half_t value)
{
  __half h;
  std::memcpy(&h, &value, sizeof(h));
  return __half2float(h);
}

LIBREDIFFUSION_API librediffusion_stream_t LIBREDIFFUSION_CALL
librediffusion_pipeline_get_stream(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return nullptr;

  return static_cast<librediffusion_stream_t>(pipeline->cpp_pipeline->stream_);
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_synchronize(librediffusion_pipeline_handle pipeline)
{
  if (!pipeline || !pipeline->cpp_pipeline)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  cudaError_t err = cudaStreamSynchronize(pipeline->cpp_pipeline->stream_);
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return LIBREDIFFUSION_ERROR_CUDA_ERROR;
  }
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL librediffusion_get_last_cuda_error(void)
{
  return static_cast<int>(g_last_cuda_error);
}

LIBREDIFFUSION_API const char* LIBREDIFFUSION_CALL
librediffusion_get_last_cuda_error_string(void)
{
  return cudaGetErrorString(g_last_cuda_error);
}

/*===========================================================================*/
/* CUDA Memory Management                                                    */
/*===========================================================================*/

LIBREDIFFUSION_API void* LIBREDIFFUSION_CALL librediffusion_cuda_malloc(size_t size)
{
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return nullptr;
  }
  return ptr;
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL librediffusion_cuda_free(void* ptr)
{
  if (ptr)
  {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess)
    {
      set_cuda_error(err);
    }
  }
}

LIBREDIFFUSION_API void* LIBREDIFFUSION_CALL librediffusion_cuda_malloc_host(size_t size)
{
  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return nullptr;
  }
  return ptr;
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL librediffusion_cuda_free_host(void* ptr)
{
  if (ptr)
  {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess)
    {
      set_cuda_error(err);
    }
  }
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_memcpy_h2d(void* dst, const void* src, size_t size)
{
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return LIBREDIFFUSION_ERROR_CUDA_ERROR;
  }
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_memcpy_d2h(void* dst, const void* src, size_t size)
{
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return LIBREDIFFUSION_ERROR_CUDA_ERROR;
  }
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_device_synchronize(void)
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    set_cuda_error(err);
    return LIBREDIFFUSION_ERROR_CUDA_ERROR;
  }
  return LIBREDIFFUSION_SUCCESS;
}

} // extern "C"
