/**
 * LibreDiffusion C API Implementation
 *
 * Bridges the C API to the underlying C++ implementation.
 * Compile this file with NVCC or a C++23 compiler that supports CUDA.
 */

#include "librediffusion_c.h"
#include "librediffusion.hpp"
#include "tensorrt_wrappers.hpp"

#include "cuda_error_state.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
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

/* Handle validation (L-11).
 *
 * A C handle is the one thing a host cannot check for itself, so the library has to. Each handle
 * carries a magic word that is set on construction and CLEARED just before the block is freed, and
 * every entry point checks it. That turns the three mistakes hosts actually make — passing a
 * destroyed handle, destroying twice, and using the out-parameter a failed create never wrote —
 * into ordinary error codes instead of a glibc double-free abort, a SIGSEGV, or (worst of all) a
 * destroyed handle that keeps answering queries with garbage.
 *
 * Reading the magic out of a freed block is not something the standard blesses, but it is the only
 * mitigation available at a C boundary and it converts the overwhelmingly common cases (the block
 * is still mapped and either untouched or reused) into a clean rejection. */
#define LIBREDIFFUSION_CONFIG_MAGIC 0x4C524443u   /* 'LRDC' */
#define LIBREDIFFUSION_PIPELINE_MAGIC 0x4C524450u /* 'LRDP' */

struct librediffusion_config_t
{
  unsigned int magic{LIBREDIFFUSION_CONFIG_MAGIC};
  librediffusion::LibreDiffusionConfig cpp_config;
};

struct librediffusion_pipeline_t
{
  unsigned int magic{LIBREDIFFUSION_PIPELINE_MAGIC};
  std::unique_ptr<librediffusion::LibreDiffusionPipeline> cpp_pipeline;
};

namespace
{
// A live, never-destroyed config handle.
inline bool valid(librediffusion_config_handle c)
{
  return c && c->magic == LIBREDIFFUSION_CONFIG_MAGIC;
}
// A live pipeline handle that also owns a pipeline (i.e. usable for real work).
inline bool valid(librediffusion_pipeline_handle p)
{
  return p && p->magic == LIBREDIFFUSION_PIPELINE_MAGIC && p->cpp_pipeline;
}
// A live pipeline handle, whether or not construction got as far as the pipeline itself. Only
// destroy needs this weaker form.
inline bool valid_handle(librediffusion_pipeline_handle p)
{
  return p && p->magic == LIBREDIFFUSION_PIPELINE_MAGIC;
}
} // anonymous namespace

/*===========================================================================*/
/* Thread-Local Error State                                                  */
/*===========================================================================*/

namespace
{
using librediffusion::cuda_context_lost;
using librediffusion::cuda_error_is_context_fatal;
using librediffusion::drain_cuda_error;
using librediffusion::g_last_cuda_error;
using librediffusion::set_cuda_error;

// For the entry points that hold a cudaError_t of their own: record it, consume whatever the runtime
// still has pending, and answer with the right one of the two codes.
librediffusion_error_t report_cuda_failure(cudaError_t err)
{
  drain_cuda_error();
  set_cuda_error(err);
  if (cuda_error_is_context_fatal(err))
    librediffusion::g_cuda_context_lost.store(true, std::memory_order_relaxed);
  return cuda_error_is_context_fatal(err) ? LIBREDIFFUSION_ERROR_CUDA_CONTEXT_LOST
                                          : LIBREDIFFUSION_ERROR_CUDA_ERROR;
}

librediffusion_error_t check_cuda_error()
{
  cudaError_t err = drain_cuda_error();
  if (err == cudaSuccess)
    return LIBREDIFFUSION_SUCCESS;
  return cuda_error_is_context_fatal(err) ? LIBREDIFFUSION_ERROR_CUDA_CONTEXT_LOST
                                          : LIBREDIFFUSION_ERROR_CUDA_ERROR;
}

template <typename Func>
librediffusion_error_t try_catch_wrapper(Func&& func)
{
  if (cuda_context_lost())
    return LIBREDIFFUSION_ERROR_CUDA_CONTEXT_LOST;
  try
  {
    func();
    return check_cuda_error();
  }
  catch (const std::bad_alloc&)
  {
    drain_cuda_error();
    std::fprintf(stderr, "[librediffusion] OUT_OF_MEMORY\n");
    return LIBREDIFFUSION_ERROR_OUT_OF_MEMORY;
  }
  catch (const std::exception& e)
  {
    // Surface the message so the C-API caller (harness/app) can see WHY an internal error
    // occurred instead of an opaque -99. Without this every throw collapsed to the same code.
    drain_cuda_error();
    std::fprintf(stderr, "[librediffusion] INTERNAL ERROR: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
  catch (...)
  {
    drain_cuda_error();
    std::fprintf(stderr, "[librediffusion] INTERNAL ERROR: unknown (non-std::exception)\n");
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

// Same as try_catch_wrapper, but WITHOUT the trailing check_cuda_error().
//
// L-13: cudaGetLastError() BRINGS UP the CUDA primary context. Wrapping the config entry points —
// which allocate a struct of ints and strings and touch no device — in try_catch_wrapper therefore
// made librediffusion_config_create() cost a full context creation (hundreds of ms, hundreds of MB
// of VRAM) on whatever thread happened to deserialise a preset, pinned it to the DEFAULT device
// before config_set_device was ever read, and left the process unable to fork a working child.
// Measured: a child forked after version() can cudaMalloc; a child forked after config_create()
// cannot (cudaErrorInitializationError).
//
// Use this for any entry point that cannot possibly have produced a CUDA error; keep
// try_catch_wrapper where a CUDA call really may have happened.
template <typename Func>
librediffusion_error_t try_catch_host(Func&& func)
{
  try
  {
    func();
    return LIBREDIFFUSION_SUCCESS;
  }
  catch (const std::bad_alloc&)
  {
    std::fprintf(stderr, "[librediffusion] OUT_OF_MEMORY\n");
    return LIBREDIFFUSION_ERROR_OUT_OF_MEMORY;
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "[librediffusion] INTERNAL ERROR: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
  catch (...)
  {
    std::fprintf(stderr, "[librediffusion] INTERNAL ERROR: unknown (non-std::exception)\n");
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

// Guard every inference entry point against a pipeline whose conditioning / scheduler the host has
// not supplied yet. Construction succeeds long before prepare_embeds()/prepare_scheduler() are
// called, and the denoise path dereferences both unconditionally, so an early frame used to be a
// guaranteed null-deref rather than an error code. Returns SUCCESS when the pipeline is ready.
librediffusion_error_t check_inference_ready(librediffusion_pipeline_handle pipeline)
{
  if (const char* why = pipeline->cpp_pipeline->inference_readiness())
  {
    std::fprintf(stderr, "[librediffusion] NOT_INITIALIZED: %s\n", why);
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  }
  return LIBREDIFFUSION_SUCCESS;
}
// Reject a prepare_* / token call whose DECLARED shape disagrees with the pipeline's configured text
// geometry (L-06). These entry points size the device buffer from the CALL's arguments, but every
// consumer reads config_.text_seq_len * config_.text_hidden_dim back out of it with a raw
// cudaMemcpyAsync — so a smaller declared shape is an out-of-bounds DEVICE read that surfaces later,
// somewhere else, as an opaque cudaErrorInvalidDevice, after the output buffer has already been
// partially written.
librediffusion_error_t check_text_dims(
    librediffusion_pipeline_handle pipeline, int seq_len, int hidden_dim, const char* what)
{
  const auto& cfg = pipeline->cpp_pipeline->config();
  if (seq_len != cfg.text_seq_len || hidden_dim != cfg.text_hidden_dim)
  {
    std::fprintf(
        stderr,
        "[librediffusion] INVALID_DIMENSIONS: %s got [seq=%d, hidden=%d] but the pipeline is "
        "configured for [seq=%d, hidden=%d]\n",
        what, seq_len, hidden_dim, cfg.text_seq_len, cfg.text_hidden_dim);
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  }
  return LIBREDIFFUSION_SUCCESS;
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
  *config = nullptr;

  return try_catch_host([&]() { *config = new librediffusion_config_t{}; });
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_config_destroy(librediffusion_config_handle config)
{
  // NULL is documented as safe; so, now, is a handle that was already destroyed (previously a
  // glibc "double free detected" -> SIGABRT). Clear the magic BEFORE freeing so the block cannot
  // pass validation again even if the allocator hands it straight back out.
  if (!valid(config))
    return;
  config->magic = 0;
  delete config;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_clone(
    librediffusion_config_handle src, librediffusion_config_handle* dst)
{
  if (!valid(src) || !dst)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  *dst = nullptr;

  return try_catch_host([&]() {
    *dst = new librediffusion_config_t{LIBREDIFFUSION_CONFIG_MAGIC, src->cpp_config};
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_device(librediffusion_config_handle config, int device)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.device = device;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_model_type(
    librediffusion_config_handle config, librediffusion_model_type_t type)
{
  if (!valid(config))
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
    case MODEL_FLUX2_KLEIN_4B:
      // klein uses the standalone librediffusion_flux2_* C-API (Flux2Pipeline), not the SD
      // predict_x0 path; the config model_type is recorded for completeness.
      config->cpp_config.model_type = librediffusion::ModelType::FLUX2_KLEIN_4B;
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
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0 || latent_width <= 0 || latent_height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  // Everything that is not <= 0 used to be accepted verbatim, including combinations that cannot
  // describe a real image: 185600x185600 (batch*4*lh*lw overflows the int it is computed in ->
  // negative -> a huge size_t -> a cudaMalloc that fails and whose result is never checked), 513x511
  // (the VAE's 8x downsampling cannot express it), and a latent grid unrelated to the pixel grid
  // (every kernel indexes one and the engine the other). Downstream those were contained only by
  // luck. Refuse them here, where the caller still has a return code to look at.
  //
  // 16384 is well past any diffusion model in existence and keeps every width*height*4 and
  // batch*4*lh*lw product far inside an int.
  constexpr int kMaxDim = 16384;
  if (width > kMaxDim || height > kMaxDim)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  if ((width % 8) != 0 || (height % 8) != 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  if (latent_width != width / 8 || latent_height != height / 8)
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
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (batch_size <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  config->cpp_config.batch_size = batch_size;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_denoising_steps(librediffusion_config_handle config, int steps)
{
  if (!valid(config))
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
  if (!valid(config))
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
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.guidance_scale = scale;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_delta(librediffusion_config_handle config, float delta)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.delta = delta;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_add_noise(librediffusion_config_handle config, int enabled)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.do_add_noise = (enabled != 0);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_denoising_batch(
    librediffusion_config_handle config, int enabled)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.use_denoising_batch = (enabled != 0);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_cuda_graph(librediffusion_config_handle config, int enabled)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.use_cuda_graph = (enabled != 0);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_seed(librediffusion_config_handle config, uint64_t seed)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.seed = seed;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_cfg_type(
    librediffusion_config_handle config, librediffusion_cfg_type_t type)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  config->cpp_config.cfg_type = static_cast<int>(type);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_text_config(
    librediffusion_config_handle config, int seq_len, int hidden_dim, int pad_token)
{
  if (!valid(config))
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
  if (!valid(config))
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
  if (!valid(config) || !path)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_host([&]() {
    config->cpp_config.unet_engine_path = path;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_vae_encoder(
    librediffusion_config_handle config, const char* path)
{
  if (!valid(config) || !path)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_host([&]() {
    config->cpp_config.vae_encoder_path = path;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_vae_decoder(
    librediffusion_config_handle config, const char* path)
{
  if (!valid(config) || !path)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_host([&]() {
    config->cpp_config.vae_decoder_path = path;
  });
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_add_controlnet(
    librediffusion_config_handle config, const char* engine_path, float conditioning_scale)
{
  // Returns the new ControlNet index (>=0), or -1 on error. Preprocessing is EXTERNAL: feed each
  // net's control image per-frame via librediffusion_set_controlnet_cond[_rgba](pipe, index, ...).
  if (!valid(config) || !engine_path)
    return -1;
  try
  {
    config->cpp_config.controlnets.push_back(
        {std::string(engine_path), conditioning_scale});
    return (int)config->cpp_config.controlnets.size() - 1;
  }
  catch (...)
  {
    return -1;
  }
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_timestep_indices(
    librediffusion_config_handle config, const int* indices, size_t count)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (count > 0 && !indices)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_host([&]() {
    config->cpp_config.timestep_indices.assign(indices, indices + count);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_pipeline_mode(
    librediffusion_config_handle config, librediffusion_pipeline_mode_t mode)
{
  if (!valid(config))
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
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  // cache_interval is the divisor of `frame_id % cache_interval` on the img2img path — an integer
  // division, so 0 is a SIGFPE that no try/catch can intercept. cache_maxframes is compared against
  // a size_t, so a negative value becomes SIZE_MAX and the cache deque never drops a frame: one
  // latent of VRAM per frame, forever. Both used to be stored verbatim.
  if (cache_interval < 1 || cache_maxframes < 1)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

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
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  config->cpp_config.use_tome_cache = (enabled != 0);
  config->cpp_config.tome_ratio = ratio;
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_width(librediffusion_config_handle config)
{
  return valid(config) ? config->cpp_config.width : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_height(librediffusion_config_handle config)
{
  return valid(config) ? config->cpp_config.height : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_latent_width(librediffusion_config_handle config)
{
  return valid(config) ? config->cpp_config.latent_width : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_latent_height(librediffusion_config_handle config)
{
  return valid(config) ? config->cpp_config.latent_height : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_batch_size(librediffusion_config_handle config)
{
  return valid(config) ? config->cpp_config.batch_size : 0;
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_denoising_steps(librediffusion_config_handle config)
{
  return valid(config) ? config->cpp_config.denoising_steps : 0;
}

/*===========================================================================*/
/* Pipeline API Implementation                                               */
/*===========================================================================*/

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_create(
    librediffusion_config_handle config, librediffusion_pipeline_handle* pipeline)
{
  if (!valid(config) || !pipeline)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  // ALWAYS write the out-parameter. It used to be left untouched on failure, so a host that checks
  // the handle rather than the return code (or reuses an uninitialised local) carried a wild
  // pointer into every later call — each of which dereferenced it inside its own `!pipeline` guard.
  // The wrapper was leaked on failure too: the throw escaped before `*pipeline = p`, and nothing
  // owned `p`.
  *pipeline = nullptr;

  return try_catch_wrapper([&]() {
    auto p = std::make_unique<librediffusion_pipeline_t>();
    p->cpp_pipeline = std::make_unique<librediffusion::LibreDiffusionPipeline>(
        config->cpp_config);
    *pipeline = p.release();
  });
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_pipeline_destroy(librediffusion_pipeline_handle pipeline)
{
  // NULL and already-destroyed are both no-ops (a second destroy used to re-run the whole dtor
  // chain — CUDA stream, graph, TensorRT contexts — on freed memory, i.e. SIGSEGV). valid_handle()
  // rather than valid(): a pipeline whose construction failed has no cpp_pipeline but still owns
  // the wrapper, and must still be freed.
  if (!valid_handle(pipeline))
    return;
  pipeline->magic = 0;
  delete pipeline;
}

/*===========================================================================*/
/* Engine cache management                                                   */
/*===========================================================================*/
// Host control over the shared TensorRT engine LRU cache. Engines still held by a LIVE pipeline stay
// resident (their VRAM is freed only when that pipeline is destroyed); these only release engines the
// cache alone holds — i.e. unused/previously-loaded models. Use before loading a model through another
// framework to hand back the VRAM our cache is sitting on.

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_engine_cache_clear(void)
{
  try
  {
    auto& engines = librediffusion::GlobalEngineCache::instance().engines();
    int n = (int)engines.size();
    engines.clear();   // drops the cache's refs; unused engines' VRAM is freed now, in-use ones survive
    return n;
  }
  catch (...)
  {
    return -1;
  }
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_engine_cache_count(void)
{
  return (int)librediffusion::GlobalEngineCache::instance().engines().size();
}

LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_engine_cache_set_max_entries(unsigned long long max_entries)
{
  // Caps the LRU; evicts the least-recently-used UNUSED engines down to the cap immediately.
  librediffusion::GlobalEngineCache::instance().engines().set_max_entries((size_t)max_entries);
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_cuda(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_cuda();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_npp(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_npp();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_engines(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->init_engines();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_buffers(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!valid(config))
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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!prompt_embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if (librediffusion_error_t e = check_text_dims(pipeline, seq_len, hidden_dim, "prepare_embeds");
      e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!null_embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if (librediffusion_error_t e = check_text_dims(pipeline, seq_len, hidden_dim, "prepare_null_embeds");
      e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!negative_embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if (librediffusion_error_t e = check_text_dims(pipeline, seq_len, hidden_dim, "prepare_negative_embeds");
      e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!embeddings || !weights)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (num_embeddings <= 0 || seq_len <= 0 || hidden_dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if (librediffusion_error_t e = check_text_dims(pipeline, seq_len, hidden_dim, "blend_embeds");
      e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!text_embeds || !time_ids)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  // No shapes are passed here: both buffers are read using the pipeline's OWN configured dimensions.
  // The least we can do is refuse the call on a pipeline that has no SDXL conditioning to fill, where
  // the dimensions are meaningless and the copy would be sized from stale defaults.
  {
    const auto& cfg = pipeline->cpp_pipeline->config();
    if (cfg.pooled_embedding_dim <= 0 || cfg.time_ids_dim <= 0)
    {
      std::fprintf(
          stderr,
          "[librediffusion] INVALID_DIMENSIONS: prepare_sdxl_conditioning on a pipeline with "
          "pooled_embedding_dim=%d, time_ids_dim=%d (call config_set_sdxl_config first)\n",
          cfg.pooled_embedding_dim, cfg.time_ids_dim);
      return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
    }
  }

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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!noise)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_init_noise(to_half_ptr(noise));
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_controlnet_cond(
    librediffusion_pipeline_handle pipeline, int index, const librediffusion_half_t* cond,
    int img_height, int img_width)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cond)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_controlnet_cond(index, to_half_ptr(cond), img_height, img_width);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_controlnet_cond_rgba(
    librediffusion_pipeline_handle pipeline, int index, const uint8_t* cpu_rgba,
    int img_height, int img_width)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cpu_rgba)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_controlnet_cond_rgba(index, cpu_rgba, img_height, img_width);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_controlnet_scale(
    librediffusion_pipeline_handle pipeline, int index, float scale)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_controlnet_scale(index, scale);
  });
}

/* ---- IP-Adapter ---- */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_ipadapter(
    librediffusion_config_handle config, int num_image_tokens, float scale)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_host([&]() {
    config->cpp_config.ipadapter_num_tokens = num_image_tokens;
    config->cpp_config.ipadapter_scale = scale;
    config->cpp_config.ipadapter_requested = true;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_ipadapter_image_encoder(
    librediffusion_config_handle config, const char* image_encoder_engine,
    const char* image_proj_engine)
{
  if (!valid(config))
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (!image_encoder_engine || !image_proj_engine)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_host([&]() {
    config->cpp_config.ipadapter_image_encoder_path = image_encoder_engine;
    config->cpp_config.ipadapter_image_proj_path = image_proj_engine;
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_ipadapter_image(
    librediffusion_pipeline_handle pipeline, const uint8_t* cpu_rgba, int img_height,
    int img_width)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cpu_rgba)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_ipadapter_image(cpu_rgba, img_height, img_width);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_ipadapter_tokens(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* pos_tokens,
    const librediffusion_half_t* neg_tokens, int num_tokens, int dim)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!pos_tokens)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (num_tokens <= 0 || dim <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  // Same class as L-06: the token buffer is sized num_tokens*dim from these arguments, but the
  // extended-ehs assembly reads ipadapter_num_tokens_ * config_.text_hidden_dim back out of it. A
  // `dim` that is not the pipeline's cross-attention width is an out-of-bounds device read.
  if (dim != pipeline->cpp_pipeline->config().text_hidden_dim)
  {
    std::fprintf(
        stderr,
        "[librediffusion] INVALID_DIMENSIONS: set_ipadapter_tokens got dim=%d but the pipeline's "
        "cross-attention width is %d\n",
        dim, pipeline->cpp_pipeline->config().text_hidden_dim);
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  }
  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_ipadapter_tokens(
        to_half_ptr(pos_tokens), to_half_ptr(neg_tokens), num_tokens, dim);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_ipadapter_scale(librediffusion_pipeline_handle pipeline, float scale)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  return try_catch_wrapper([&]() { pipeline->cpp_pipeline->set_ipadapter_scale(scale); });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_ipadapter_scale_vector(
    librediffusion_pipeline_handle pipeline, const float* per_layer, int num_ip_layers)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!per_layer)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_ipadapter_scale_vector(per_layer, num_ip_layers);
  });
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_num_runtime_loras(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
    return 0;
  return pipeline->cpp_pipeline->num_runtime_loras();
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_lora_scale(librediffusion_pipeline_handle pipeline, int idx, float scale)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  return try_catch_wrapper([&]() { pipeline->cpp_pipeline->set_lora_scale(idx, scale); });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_lora_scale_vector(
    librediffusion_pipeline_handle pipeline, const float* scales, int n)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!scales)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return try_catch_wrapper([&]() { pipeline->cpp_pipeline->set_lora_scale_vector(scales, n); });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_reseed(librediffusion_pipeline_handle pipeline, int64_t seed)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->reseed(seed);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_guidance_scale(
    librediffusion_pipeline_handle pipeline, float guidance)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->set_guidance_scale(guidance);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_delta(librediffusion_pipeline_handle pipeline, float delta)
{
  if (!valid(pipeline))
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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cpu_rgba_input || !cpu_rgba_output)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->img2img(cpu_rgba_input, cpu_rgba_output, width, height);
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_txt2img(
    librediffusion_pipeline_handle pipeline, uint8_t* cpu_rgba_output, int width,
    int height)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!cpu_rgba_output)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if (width <= 0 || height <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_in || !image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_in || !image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!image_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if (!x_t_latent_in || !x_0_pred_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;

  if (librediffusion_error_t e = check_inference_ready(pipeline); e != LIBREDIFFUSION_SUCCESS)
    return e;

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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  // See config_set_temporal_params: cache_interval == 0 is a SIGFPE on the next img2img, and a
  // negative cache_maxframes reads as SIZE_MAX, i.e. "cache every frame forever".
  if (cache_interval < 1 || max_cached_frames < 1)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;

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
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->disableTemporalCoherence();
  });
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_reset_temporal_state(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  return try_catch_wrapper([&]() {
    pipeline->cpp_pipeline->resetTemporalState();
  });
}

LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_get_current_frame_id(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
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
    case LIBREDIFFUSION_ERROR_CUDA_CONTEXT_LOST:
      return "CUDA context lost (unrecoverable; restart the process)";
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
  if (!valid(pipeline))
    return nullptr;

  return static_cast<librediffusion_stream_t>(pipeline->cpp_pipeline->stream_);
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_synchronize(librediffusion_pipeline_handle pipeline)
{
  if (!valid(pipeline))
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;

  cudaError_t err = cudaStreamSynchronize(pipeline->cpp_pipeline->stream_);
  if (err != cudaSuccess)
    return report_cuda_failure(err);
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
    report_cuda_failure(err);
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
      report_cuda_failure(err);
  }
}

LIBREDIFFUSION_API void* LIBREDIFFUSION_CALL librediffusion_cuda_malloc_host(size_t size)
{
  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err != cudaSuccess)
  {
    report_cuda_failure(err);
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
      report_cuda_failure(err);
  }
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_memcpy_h2d(void* dst, const void* src, size_t size)
{
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    return report_cuda_failure(err);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_memcpy_d2h(void* dst, const void* src, size_t size)
{
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    return report_cuda_failure(err);
  return LIBREDIFFUSION_SUCCESS;
}

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_device_synchronize(void)
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    return report_cuda_failure(err);
  return LIBREDIFFUSION_SUCCESS;
}

} // extern "C"
