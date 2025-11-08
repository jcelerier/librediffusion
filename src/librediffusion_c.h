/**
 * StreamDiffusion C API
 *
 * C-compatible wrapper for the StreamDiffusion C++/CUDA library.
 * Provides ABI-stable interface for cross-compiler compatibility on Windows
 * (supports both MSVC cl.exe and MinGW).
 *
 * Usage:
 *   1. Create a config with librediffusion_config_create()
 *   2. Set config options with librediffusion_config_set_*()
 *   3. Create pipeline with librediffusion_pipeline_create()
 *   4. Prepare embeddings and scheduler
 *   5. Run inference with librediffusion_img2img() or librediffusion_txt2img()
 *   6. Cleanup with librediffusion_pipeline_destroy() and librediffusion_config_destroy()
 */

#ifndef STREAM_DIFFUSION_C_H
#define STREAM_DIFFUSION_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================*/
/* Platform Detection & Export Macros                                        */
/*===========================================================================*/

#if defined(_WIN32) || defined(_WIN64)
#ifdef LIBREDIFFUSION_BUILD_DLL
/* Building the DLL */
#define LIBREDIFFUSION_API __declspec(dllexport)
#elif defined(LIBREDIFFUSION_USE_DLL)
/* Using the DLL */
#define LIBREDIFFUSION_API __declspec(dllimport)
#else
/* Static library */
#define LIBREDIFFUSION_API
#endif
#define LIBREDIFFUSION_CALL __cdecl
#else
    /* Non-Windows (Linux, macOS) */
    #if __GNUC__ >= 4
#define LIBREDIFFUSION_API __attribute__((visibility("default")))
#else
#define LIBREDIFFUSION_API
#endif
#define LIBREDIFFUSION_CALL
#endif

/*===========================================================================*/
/* Opaque Handle Types                                                       */
/*===========================================================================*/

/** Opaque handle to StreamDiffusion pipeline */
typedef struct librediffusion_pipeline_t* librediffusion_pipeline_handle;

/** Opaque handle to configuration */
typedef struct librediffusion_config_t* librediffusion_config_handle;

/** Opaque handle to CUDA tensor (for advanced usage) */
typedef struct librediffusion_tensor_t* librediffusion_tensor_handle;

/*===========================================================================*/
/* Error Codes                                                               */
/*===========================================================================*/

typedef enum librediffusion_error_t
{
  LIBREDIFFUSION_SUCCESS = 0,
  LIBREDIFFUSION_ERROR_INVALID_ARGUMENT = -1,
  LIBREDIFFUSION_ERROR_NULL_POINTER = -2,
  LIBREDIFFUSION_ERROR_OUT_OF_MEMORY = -3,
  LIBREDIFFUSION_ERROR_CUDA_ERROR = -4,
  LIBREDIFFUSION_ERROR_ENGINE_LOAD_FAILED = -5,
  LIBREDIFFUSION_ERROR_NOT_INITIALIZED = -6,
  LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS = -7,
  LIBREDIFFUSION_ERROR_FILE_NOT_FOUND = -8,
  LIBREDIFFUSION_ERROR_INTERNAL = -99
} librediffusion_error_t;

/*===========================================================================*/
/* Enumerations                                                              */
/*===========================================================================*/

typedef enum librediffusion_model_type_t
{
  MODEL_SD_15 = 0,
  MODEL_SD_TURBO = 1,
  MODEL_SDXL_TURBO = 2
} librediffusion_model_type_t;

typedef enum librediffusion_pipeline_mode_t
{
  MODE_SINGLE_FRAME = 0,
  MODE_TEMPORAL_V2V = 1
} librediffusion_pipeline_mode_t;

typedef enum librediffusion_cfg_type_t
{
  SD_CFG_NONE = 0,
  SD_CFG_FULL = 1,
  SD_CFG_SELF = 2,
  SD_CFG_INITIALIZE = 3
} librediffusion_cfg_type_t;

/*===========================================================================*/
/* Data Types                                                                */
/*===========================================================================*/

/**
 * Half-precision float represented as uint16_t for ABI compatibility.
 * Use librediffusion_float_to_half() and librediffusion_half_to_float() for conversion.
 */
typedef uint16_t librediffusion_half_t;

/**
 * CUDA stream handle (void* for ABI stability)
 */
typedef void* librediffusion_stream_t;

/*===========================================================================*/
/* Configuration API                                                         */
/*===========================================================================*/

/**
 * Create a new configuration with default values.
 *
 * @param[out] config  Pointer to receive the config handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_create(librediffusion_config_handle* config);

/**
 * Destroy a configuration and free resources.
 *
 * @param config  Config handle to destroy (safe to pass NULL)
 */
LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_config_destroy(librediffusion_config_handle config);

/**
 * Create a deep copy of a configuration.
 *
 * @param src      Source config
 * @param[out] dst Pointer to receive the cloned config handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_clone(
    librediffusion_config_handle src, librediffusion_config_handle* dst);

/* Device and model settings */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_device(librediffusion_config_handle config, int device);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_model_type(
    librediffusion_config_handle config, librediffusion_model_type_t type);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_dimensions(
    librediffusion_config_handle config, int width, int height, int latent_width,
    int latent_height);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_batch_size(
    librediffusion_config_handle config, int batch_size);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_denoising_steps(
    librediffusion_config_handle config, int steps);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_frame_buffer_size(
    librediffusion_config_handle config, int size);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_guidance_scale(
    librediffusion_config_handle config, float scale);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_delta(librediffusion_config_handle config, float delta);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_add_noise(librediffusion_config_handle config, int enabled);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_denoising_batch(
    librediffusion_config_handle config, int enabled);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_seed(librediffusion_config_handle config, uint64_t seed);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_cfg_type(
    librediffusion_config_handle config, librediffusion_cfg_type_t type);

/* Text encoder configuration */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_text_config(
    librediffusion_config_handle config, int seq_len, int hidden_dim, int pad_token);

/* SDXL-specific configuration */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_sdxl_config(
    librediffusion_config_handle config, int pooled_embedding_dim, int time_ids_dim);

/* Engine paths */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_unet_engine(
    librediffusion_config_handle config, const char* path);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_vae_encoder(
    librediffusion_config_handle config, const char* path);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_vae_decoder(
    librediffusion_config_handle config, const char* path);

/* Timestep indices (array copy) */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_timestep_indices(
    librediffusion_config_handle config, const int* indices, size_t count);

/* Pipeline mode and temporal coherence */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_pipeline_mode(
    librediffusion_config_handle config, librediffusion_pipeline_mode_t mode);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_temporal_params(
    librediffusion_config_handle config, int use_cached_attn, int use_feature_injection,
    float injection_strength, float similarity_threshold, int cache_interval,
    int cache_maxframes);
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_tome(
    librediffusion_config_handle config, int enabled, float ratio);

/* Getters for validation */
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_width(librediffusion_config_handle config);
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_height(librediffusion_config_handle config);
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_latent_width(librediffusion_config_handle config);
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_latent_height(librediffusion_config_handle config);
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_batch_size(librediffusion_config_handle config);
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_config_get_denoising_steps(librediffusion_config_handle config);

/*===========================================================================*/
/* Pipeline API                                                              */
/*===========================================================================*/

/**
 * Create a new StreamDiffusion pipeline.
 *
 * @param config       Configuration handle
 * @param[out] pipeline Pointer to receive the pipeline handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_create(
    librediffusion_config_handle config, librediffusion_pipeline_handle* pipeline);

/**
 * Destroy a pipeline and free all resources.
 *
 * @param pipeline  Pipeline handle to destroy (safe to pass NULL)
 */
LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_pipeline_destroy(librediffusion_pipeline_handle pipeline);

/**
 * Initialize CUDA context for the pipeline.
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_cuda(librediffusion_pipeline_handle pipeline);

/**
 * Initialize NPP (NVIDIA Performance Primitives).
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_npp(librediffusion_pipeline_handle pipeline);

/**
 * Load TensorRT engines.
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_engines(librediffusion_pipeline_handle pipeline);

/**
 * Initialize internal buffers.
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_buffers(librediffusion_pipeline_handle pipeline);

/**
 * Convenience function to initialize everything.
 * Calls init_cuda, init_npp, init_engines, and init_buffers.
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_init_all(librediffusion_pipeline_handle pipeline);

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
 * Parameters that CANNOT be changed (require full pipeline recreation):
 * - unet_engine_path, vae_encoder_path, vae_decoder_path
 * - model_type, mode (SINGLE_FRAME vs TEMPORAL_V2V)
 *
 * @param pipeline Pipeline handle
 * @param config   New configuration to apply
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_reinit_buffers(
    librediffusion_pipeline_handle pipeline, librediffusion_config_handle config);

/*===========================================================================*/
/* Embedding & Scheduler Preparation                                         */
/*===========================================================================*/

/**
 * Prepare prompt embeddings.
 *
 * @param pipeline      Pipeline handle
 * @param prompt_embeds Prompt embeddings [batch_size, seq_len, hidden_dim] as half
 * @param seq_len       Sequence length
 * @param hidden_dim    Hidden dimension
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_embeds(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* prompt_embeds,
    int seq_len, int hidden_dim);

/**
 * Prepare null (unconditional) embeddings for StreamV2V.
 *
 * @param pipeline    Pipeline handle
 * @param null_embeds Null embeddings [1, seq_len, hidden_dim] as half
 * @param seq_len     Sequence length
 * @param hidden_dim  Hidden dimension
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_null_embeds(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* null_embeds,
    int seq_len, int hidden_dim);

/**
 * Prepare negative (unconditional) embeddings for CFG guidance.
 *
 * @param pipeline         Pipeline handle
 * @param negative_embeds  Negative embeddings [1, seq_len, hidden_dim] as half
 * @param seq_len          Sequence length
 * @param hidden_dim       Hidden dimension
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_negative_embeds(
    librediffusion_pipeline_handle pipeline,
    const librediffusion_half_t* negative_embeds, int seq_len, int hidden_dim);

/**
 * Blend multiple embeddings with weights for prompt interpolation.
 * Creates a weighted sum: result = sum(weights[i] * embeddings[i])
 * Weights are normalized internally if they don't sum to 1.0.
 *
 * @param pipeline        Pipeline handle
 * @param embeddings      Array of embedding device pointers [num_embeddings]
 *                        Each embedding is [1, seq_len, hidden_dim] as half
 * @param weights         Blend weights (host array) [num_embeddings]
 * @param num_embeddings  Number of embeddings to blend
 * @param seq_len         Sequence length
 * @param hidden_dim      Hidden dimension
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_blend_embeds(
    librediffusion_pipeline_handle pipeline,
    const librediffusion_half_t* const* embeddings, const float* weights,
    int num_embeddings, int seq_len, int hidden_dim);

/**
 * Prepare SDXL-specific conditioning.
 *
 * @param pipeline     Pipeline handle
 * @param text_embeds  Pooled text embeddings [batch_size, pooled_dim] as half
 * @param time_ids     Time IDs [batch_size, 6] as half
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_sdxl_conditioning(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* text_embeds,
    const librediffusion_half_t* time_ids);

/**
 * Prepare scheduler parameters.
 *
 * @param pipeline          Pipeline handle
 * @param timesteps         Timestep values [num_timesteps]
 * @param alpha_prod_t_sqrt sqrt(alpha_cumprod) [num_timesteps]
 * @param beta_prod_t_sqrt  sqrt(1 - alpha_cumprod) [num_timesteps]
 * @param c_skip            Skip connection scaling [num_timesteps]
 * @param c_out             Output scaling [num_timesteps]
 * @param num_timesteps     Number of timesteps
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_prepare_scheduler(
    librediffusion_pipeline_handle pipeline, const float* timesteps,
    const float* alpha_prod_t_sqrt, const float* beta_prod_t_sqrt, const float* c_skip,
    const float* c_out, size_t num_timesteps);

/**
 * Set initial noise for testing/validation.
 *
 * @param pipeline Pipeline handle
 * @param noise    Initial noise [denoising_steps, 4, latent_h, latent_w] as half
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_init_noise(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* noise);

/**
 * Reseed the random number generator.
 *
 * @param pipeline Pipeline handle
 * @param seed     New seed value
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_reseed(librediffusion_pipeline_handle pipeline, int64_t seed);

/**
 * Update the current guidance scale.
 *
 * @param pipeline Pipeline handle
 * @param guidance New guidance value
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_guidance_scale(
    librediffusion_pipeline_handle pipeline, float guidance);

/**
 * Update the current delta parameter
 *
 * @param pipeline Pipeline handle
 * @param delta New delta value
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_delta(librediffusion_pipeline_handle pipeline, float delta);

/*===========================================================================*/
/* High-Level Inference API (CPU buffers)                                    */
/*===========================================================================*/

/**
 * Run image-to-image inference with CPU RGBA buffers.
 *
 * @param pipeline         Pipeline handle
 * @param cpu_rgba_input   Input image [height, width, 4] RGBA
 * @param cpu_rgba_output  Output image [height, width, 4] RGBA
 * @param width            Image width
 * @param height           Image height
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_img2img(
    librediffusion_pipeline_handle pipeline, const uint8_t* cpu_rgba_input,
    uint8_t* cpu_rgba_output, int width, int height);

/**
 * Run text-to-image inference with CPU RGBA buffer.
 *
 * @param pipeline         Pipeline handle
 * @param cpu_rgba_output  Output image [height, width, 4] RGBA
 * @param width            Image width
 * @param height           Image height
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_txt2img(
    librediffusion_pipeline_handle pipeline, uint8_t* cpu_rgba_output, int width,
    int height);

/*===========================================================================*/
/* Low-Level GPU Inference API                                               */
/*===========================================================================*/

/**
 * Run image-to-image on GPU memory (half precision).
 *
 * @param pipeline   Pipeline handle
 * @param image_in   Input [batch, 3, height, width] as half (device memory)
 * @param image_out  Output [batch, 3, height, width] as half (device memory)
 * @param stream     CUDA stream (NULL for default)
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_img2img_gpu_half(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* image_in,
    librediffusion_half_t* image_out, librediffusion_stream_t stream);

/**
 * Run image-to-image on GPU memory (float precision).
 *
 * @param pipeline   Pipeline handle
 * @param image_in   Input [batch, 3, height, width] as float (device memory)
 * @param image_out  Output [batch, 3, height, width] as half (device memory)
 * @param stream     CUDA stream (NULL for default)
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_img2img_gpu_float(
    librediffusion_pipeline_handle pipeline, const float* image_in,
    librediffusion_half_t* image_out, librediffusion_stream_t stream);

/**
 * Run text-to-image on GPU memory.
 *
 * @param pipeline   Pipeline handle
 * @param image_out  Output [batch, 3, height, width] as half (device memory)
 * @param stream     CUDA stream (NULL for default)
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_txt2img_gpu(
    librediffusion_pipeline_handle pipeline, librediffusion_half_t* image_out,
    librediffusion_stream_t stream);

/**
 * Run SD-Turbo text-to-image (single-step).
 *
 * @param pipeline   Pipeline handle
 * @param image_out  Output [batch, 3, height, width] as half (device memory)
 * @param stream     CUDA stream (NULL for default)
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_txt2img_sd_turbo_gpu(
    librediffusion_pipeline_handle pipeline, librediffusion_half_t* image_out,
    librediffusion_stream_t stream);

/*===========================================================================*/
/* VAE Operations                                                            */
/*===========================================================================*/

/**
 * Encode image to latent space (half precision input).
 *
 * @param pipeline   Pipeline handle
 * @param image      Input [batch, 3, height, width] as half
 * @param latent_out Output [batch, 4, latent_h, latent_w] as half
 * @param stream     CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_encode_image_half(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* image,
    librediffusion_half_t* latent_out, librediffusion_stream_t stream);

/**
 * Encode image to latent space (float precision input).
 *
 * @param pipeline   Pipeline handle
 * @param image      Input [batch, 3, height, width] as float
 * @param latent_out Output [batch, 4, latent_h, latent_w] as half
 * @param stream     CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_encode_image_float(
    librediffusion_pipeline_handle pipeline, const float* image,
    librediffusion_half_t* latent_out, librediffusion_stream_t stream);

/**
 * Decode latent to image.
 *
 * @param pipeline   Pipeline handle
 * @param latent     Input [batch, 4, latent_h, latent_w] as half
 * @param image_out  Output [batch, 3, height, width] as half
 * @param stream     CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_decode_latent(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* latent,
    librediffusion_half_t* image_out, librediffusion_stream_t stream);

/*===========================================================================*/
/* UNet Operations                                                           */
/*===========================================================================*/

/**
 * Run batched x0 prediction.
 *
 * @param pipeline      Pipeline handle
 * @param x_t_latent_in Input latents [batch, 4, latent_h, latent_w] as half
 * @param x_0_pred_out  Output predictions [batch, 4, latent_h, latent_w] as half
 * @param stream        CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_predict_x0_batch(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* x_t_latent_in,
    librediffusion_half_t* x_0_pred_out, librediffusion_stream_t stream);

/*===========================================================================*/
/* Image Format Conversion                                                   */
/*===========================================================================*/

/**
 * Convert RGBA NHWC uint8 to RGB NCHW float on GPU.
 *
 * @param pipeline      Pipeline handle
 * @param rgba_nhwc_in  Input [batch, height, width, 4] as uint8
 * @param rgb_nchw_out  Output [batch, 3, height, width] as float, normalized [-1, 1]
 * @param width         Image width
 * @param height        Image height
 * @param stream        CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_rgba_nhwc_to_nchw_float(
    librediffusion_pipeline_handle pipeline, const uint8_t* rgba_nhwc_in,
    float* rgb_nchw_out, int width, int height, librediffusion_stream_t stream);

/**
 * Convert RGBA NHWC uint8 to RGB NCHW half on GPU.
 *
 * @param pipeline      Pipeline handle
 * @param rgba_nhwc_in  Input [batch, height, width, 4] as uint8
 * @param rgb_nchw_out  Output [batch, 3, height, width] as half, normalized [-1, 1]
 * @param width         Image width
 * @param height        Image height
 * @param stream        CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_rgba_nhwc_to_nchw_half(
    librediffusion_pipeline_handle pipeline, const uint8_t* rgba_nhwc_in,
    librediffusion_half_t* rgb_nchw_out, int width, int height,
    librediffusion_stream_t stream);

/**
 * Convert RGB NCHW half to RGBA NHWC uint8 on GPU.
 *
 * @param pipeline       Pipeline handle
 * @param rgb_nchw_in    Input [batch, 3, height, width] as half, normalized [-1, 1]
 * @param rgba_nhwc_out  Output [batch, height, width, 4] as uint8
 * @param width          Image width
 * @param height         Image height
 * @param stream         CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_nchw_half_to_rgba_nhwc(
    librediffusion_pipeline_handle pipeline, const librediffusion_half_t* rgb_nchw_in,
    uint8_t* rgba_nhwc_out, int width, int height, librediffusion_stream_t stream);

/**
 * Convert RGB NCHW float to RGBA NHWC uint8 on GPU.
 *
 * @param pipeline       Pipeline handle
 * @param rgb_nchw_in    Input [batch, 3, height, width] as float, normalized [-1, 1]
 * @param rgba_nhwc_out  Output [batch, height, width, 4] as uint8
 * @param width          Image width
 * @param height         Image height
 * @param stream         CUDA stream
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_nchw_float_to_rgba_nhwc(
    librediffusion_pipeline_handle pipeline, const float* rgb_nchw_in,
    uint8_t* rgba_nhwc_out, int width, int height, librediffusion_stream_t stream);

/**
 * Resize RGBA image on GPU.
 *
 * @param pipeline      Pipeline handle
 * @param rgba_input    Input [in_height, in_width, 4] as uint8
 * @param in_width      Input width
 * @param in_height     Input height
 * @param rgba_output   Output [out_height, out_width, 4] as uint8
 * @param out_width     Output width
 * @param out_height    Output height
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL librediffusion_rgba_resize(
    librediffusion_pipeline_handle pipeline, uint8_t* rgba_input, int in_width,
    int in_height, uint8_t* rgba_output, int out_width, int out_height);

/*===========================================================================*/
/* Temporal Coherence (StreamV2V)                                            */
/*===========================================================================*/

/**
 * Enable temporal coherence for video processing.
 *
 * @param pipeline              Pipeline handle
 * @param use_feature_injection Enable feature injection
 * @param injection_strength    Blend ratio (0-1)
 * @param similarity_threshold  NN matching threshold
 * @param cache_interval        Update cache every N frames
 * @param max_cached_frames     Maximum frames to cache
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_enable_temporal_coherence(
    librediffusion_pipeline_handle pipeline, int use_feature_injection,
    float injection_strength, float similarity_threshold, int cache_interval,
    int max_cached_frames);

/**
 * Disable temporal coherence.
 *
 * @param pipeline Pipeline handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_disable_temporal_coherence(librediffusion_pipeline_handle pipeline);

/**
 * Reset temporal state (clear caches).
 *
 * @param pipeline Pipeline handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_reset_temporal_state(librediffusion_pipeline_handle pipeline);

/**
 * Get current frame ID.
 *
 * @param pipeline Pipeline handle
 * @return Current frame ID, or -1 on error
 */
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL
librediffusion_get_current_frame_id(librediffusion_pipeline_handle pipeline);

/*===========================================================================*/
/* CLIP Text Encoder API                                                     */
/*===========================================================================*/

/** Opaque handle to CLIP text encoder */
typedef struct librediffusion_clip_t* librediffusion_clip_handle;

/**
 * Create a CLIP text encoder from a TensorRT engine.
 *
 * @param engine_path  Path to the CLIP TensorRT engine file
 * @param[out] clip    Pointer to receive the CLIP handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_clip_create(const char* engine_path, librediffusion_clip_handle* clip);

/**
 * Destroy a CLIP encoder and free resources.
 *
 * @param clip  CLIP handle to destroy (safe to pass NULL)
 */
LIBREDIFFUSION_API void LIBREDIFFUSION_CALL
librediffusion_clip_destroy(librediffusion_clip_handle clip);

/**
 * Compute text embeddings from a prompt.
 *
 * @param clip         CLIP handle
 * @param prompt       Text prompt to encode
 * @param pad_token    Padding token ID (0 for SD-Turbo, 49407 for SD1.5)
 * @param stream       CUDA stream (NULL for default)
 * @param[out] embeddings  Device pointer to receive embeddings [1, seq_len, hidden_dim]
 * @return LIBREDIFFUSION_SUCCESS or error code
 *
 * @note The caller is responsible for freeing the embeddings with cudaFree
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_clip_compute_embeddings(
    librediffusion_clip_handle clip, const char* prompt, int pad_token,
    librediffusion_stream_t stream, librediffusion_half_t** embeddings);

/**
 * Compute SDXL text embeddings from a prompt (requires two CLIP encoders).
 *
 * @param clip1        First CLIP encoder handle
 * @param clip2        Second CLIP encoder handle
 * @param prompt       Text prompt to encode
 * @param batch_size   Batch size
 * @param height       Target image height
 * @param width        Target image width
 * @param stream       CUDA stream
 * @param[out] embeddings     Device pointer for embeddings [batch, seq_len, hidden_dim]
 * @param[out] pooled_embeds  Device pointer for pooled embeddings [batch, pooled_dim]
 * @param[out] time_ids       Device pointer for time IDs [batch, 6]
 * @return LIBREDIFFUSION_SUCCESS or error code
 *
 * @note The caller is responsible for freeing the output pointers with cudaFree
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_clip_compute_embeddings_sdxl(
    librediffusion_clip_handle clip1, librediffusion_clip_handle clip2,
    const char* prompt, int batch_size, int height, int width,
    librediffusion_stream_t stream, librediffusion_half_t** embeddings,
    librediffusion_half_t** pooled_embeds, librediffusion_half_t** time_ids);

/*===========================================================================*/
/* Utility Functions                                                         */
/*===========================================================================*/

/**
 * Get human-readable error message.
 *
 * @param error Error code
 * @return Static string describing the error
 */
LIBREDIFFUSION_API const char* LIBREDIFFUSION_CALL
librediffusion_error_string(librediffusion_error_t error);

/**
 * Get library version string.
 *
 * @return Static version string (e.g., "1.0.0")
 */
LIBREDIFFUSION_API const char* LIBREDIFFUSION_CALL librediffusion_version(void);

/**
 * Convert float to half-precision.
 *
 * @param value Float value
 * @return Half-precision representation as uint16_t
 */
LIBREDIFFUSION_API librediffusion_half_t LIBREDIFFUSION_CALL
librediffusion_float_to_half(float value);

/**
 * Convert half-precision to float.
 *
 * @param value Half-precision value
 * @return Float value
 */
LIBREDIFFUSION_API float LIBREDIFFUSION_CALL
librediffusion_half_to_float(librediffusion_half_t value);

/**
 * Get the CUDA stream associated with a pipeline.
 *
 * @param pipeline Pipeline handle
 * @return CUDA stream handle, or NULL on error
 */
LIBREDIFFUSION_API librediffusion_stream_t LIBREDIFFUSION_CALL
librediffusion_pipeline_get_stream(librediffusion_pipeline_handle pipeline);

/**
 * Synchronize the pipeline's CUDA stream.
 *
 * @param pipeline Pipeline handle
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_pipeline_synchronize(librediffusion_pipeline_handle pipeline);

/**
 * Get the last CUDA error code.
 *
 * @return CUDA error code (cudaError_t cast to int)
 */
LIBREDIFFUSION_API int LIBREDIFFUSION_CALL librediffusion_get_last_cuda_error(void);

/**
 * Get the last CUDA error string.
 *
 * @return Static string describing the last CUDA error
 */
LIBREDIFFUSION_API const char* LIBREDIFFUSION_CALL
librediffusion_get_last_cuda_error_string(void);

/*===========================================================================*/
/* CUDA Memory Management                                                    */
/*===========================================================================*/

/**
 * Allocate device (GPU) memory.
 *
 * @param size  Number of bytes to allocate
 * @return Device pointer, or NULL on failure
 */
LIBREDIFFUSION_API void* LIBREDIFFUSION_CALL librediffusion_cuda_malloc(size_t size);

/**
 * Free device (GPU) memory.
 *
 * @param ptr  Device pointer to free (safe to pass NULL)
 */
LIBREDIFFUSION_API void LIBREDIFFUSION_CALL librediffusion_cuda_free(void* ptr);

/**
 * Allocate pinned (page-locked) host memory.
 *
 * @param size  Number of bytes to allocate
 * @return Host pointer, or NULL on failure
 */
LIBREDIFFUSION_API void* LIBREDIFFUSION_CALL
librediffusion_cuda_malloc_host(size_t size);

/**
 * Free pinned host memory.
 *
 * @param ptr  Host pointer to free (safe to pass NULL)
 */
LIBREDIFFUSION_API void LIBREDIFFUSION_CALL librediffusion_cuda_free_host(void* ptr);

/**
 * Copy memory from host to device.
 *
 * @param dst   Device destination pointer
 * @param src   Host source pointer
 * @param size  Number of bytes to copy
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_memcpy_h2d(void* dst, const void* src, size_t size);

/**
 * Copy memory from device to host.
 *
 * @param dst   Host destination pointer
 * @param src   Device source pointer
 * @param size  Number of bytes to copy
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_memcpy_d2h(void* dst, const void* src, size_t size);

/**
 * Synchronize the default CUDA stream.
 *
 * @return LIBREDIFFUSION_SUCCESS or error code
 */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_cuda_device_synchronize(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STREAM_DIFFUSION_C_H */
