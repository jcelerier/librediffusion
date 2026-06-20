#pragma once

/**
 * LibreDiffusion Dynamic Library Loader
 * 
 * Loads all LibreDiffusion C API symbols at runtime via dlopen/LoadLibrary.
 * This allows using the library without linking against it at build time.
 * 
 * Usage:
 *   const auto& sd = sd::liblibrediffusion::instance();
 *   if (!sd.available) {
 *     // Library not found or symbols missing
 *     return;
 *   }
 *   
 *   sd_config_handle config = nullptr;
 *   sd.config_create(&config);
 *   // ... use other sd.* function pointers ...
 *   sd.config_destroy(config);
 */

#include "dylib_loader.hpp"
#include "librediffusion_c.h"

namespace sd
{

/**
 * @brief LibreDiffusion dynamic library wrapper
 * 
 * Loads all symbols from the LibreDiffusion shared library at runtime.
 * Use instance() to get the singleton, then check `available` before use.
 */
class liblibrediffusion
{
public:
  /*=========================================================================*/
  /* Configuration API                                                       */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_create);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_destroy);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_clone);

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_device);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_model_type);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_dimensions);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_batch_size);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_denoising_steps);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_frame_buffer_size);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_guidance_scale);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_delta);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_add_noise);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_denoising_batch);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_seed);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_cfg_type);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_text_config);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_sdxl_config);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_unet_engine);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_vae_encoder);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_vae_decoder);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_timestep_indices);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_pipeline_mode);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_temporal_params);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_set_tome);

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_get_width);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_get_height);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_get_latent_width);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_get_latent_height);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_get_batch_size);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, config_get_denoising_steps);

  /*=========================================================================*/
  /* Pipeline API                                                            */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_create);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_destroy);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_init_cuda);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_init_npp);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_init_engines);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_init_buffers);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_init_all);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_reinit_buffers);

  /*=========================================================================*/
  /* Embedding & Scheduler Preparation                                       */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, prepare_embeds);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, prepare_null_embeds);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, prepare_negative_embeds);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, prepare_sdxl_conditioning);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, prepare_scheduler);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, blend_embeds);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, set_init_noise);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, reseed);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, set_guidance_scale);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, set_delta);

  /*=========================================================================*/
  /* High-Level Inference (CPU buffers)                                      */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, img2img);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, txt2img);

  /*=========================================================================*/
  /* Low-Level GPU Inference                                                 */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, img2img_gpu_half);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, img2img_gpu_float);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, txt2img_gpu);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, txt2img_sd_turbo_gpu);

  /*=========================================================================*/
  /* VAE Operations                                                          */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, encode_image_half);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, encode_image_float);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, decode_latent);

  /*=========================================================================*/
  /* UNet Operations                                                         */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, predict_x0_batch);

  /*=========================================================================*/
  /* Image Format Conversion                                                 */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, rgba_nhwc_to_nchw_float);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, rgba_nhwc_to_nchw_half);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, nchw_half_to_rgba_nhwc);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, nchw_float_to_rgba_nhwc);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, rgba_resize);

  /*=========================================================================*/
  /* Temporal Coherence (StreamV2V)                                          */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, enable_temporal_coherence);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, disable_temporal_coherence);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, reset_temporal_state);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, get_current_frame_id);

  /*=========================================================================*/
  /* CLIP Text Encoder API                                                   */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, clip_create);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, clip_destroy);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, clip_compute_embeddings);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, clip_compute_embeddings_sdxl);

  /*=========================================================================*/
  /* Utility Functions                                                       */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, error_string);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, version);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, float_to_half);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, half_to_float);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_get_stream);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, pipeline_synchronize);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, get_last_cuda_error);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, get_last_cuda_error_string);

  /*=========================================================================*/
  /* CUDA Memory Management                                                  */
  /*=========================================================================*/

  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_malloc);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_free);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_malloc_host);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_free_host);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_memcpy_h2d);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_memcpy_d2h);
  LIBREDIFFUSION_SYMBOL_DEF(librediffusion, cuda_device_synchronize);

  /*=========================================================================*/
  /* Singleton Access                                                        */
  /*=========================================================================*/

  /**
   * @brief Get the singleton instance
   * @return Reference to the library loader
   * 
   * The library is loaded on first access. Check `available` before use.
   */
  static const liblibrediffusion& instance()
  {
    static const liblibrediffusion self;
    return self;
  }

  /**
   * @brief Create a new non-singleton instance
   * @return A new loader instance (useful for custom library paths)
   */
  static liblibrediffusion create(const char* library_path)
  {
    return liblibrediffusion(library_path);
  }

  /**
   * @brief Whether all required symbols were loaded successfully
   */
  bool available{true};

private:
  dylib_loader m_library;

  // Default library names to search
  static constexpr const char* default_library_names[] = {
#if defined(_WIN32) || defined(_WIN64)
    "librediffusion.dll",
#else
      "librediffusion.so",
      "librediffusion.so.0",
      "liblibrediffusion.so",
      "liblibrediffusion.so.0",
      "librediffusion",
#endif
    nullptr
  };

  /**
   * @brief Default constructor - tries default library names
   */
  liblibrediffusion()
      : m_library{default_library_names}
  {
    if (!m_library)
    {
      available = false;
      return;
    }
    load_symbols();
  }

  /**
   * @brief Constructor with explicit library path
   */
  explicit liblibrediffusion(const char* library_path)
      : m_library{library_path}
  {
    if (!m_library)
    {
      available = false;
      return;
    }
    load_symbols();
  }

  void load_symbols()
  {
    /*=====================================================================*/
    /* Configuration API                                                   */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_create);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_destroy);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_clone);

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_device);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_model_type);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_dimensions);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_batch_size);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_denoising_steps);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_frame_buffer_size);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_guidance_scale);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_delta);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_add_noise);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_denoising_batch);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_seed);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_cfg_type);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_text_config);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_sdxl_config);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_unet_engine);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_vae_encoder);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_vae_decoder);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_timestep_indices);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_pipeline_mode);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_temporal_params);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_set_tome);

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_get_width);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_get_height);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_get_latent_width);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_get_latent_height);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_get_batch_size);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, config_get_denoising_steps);

    /*=====================================================================*/
    /* Pipeline API                                                        */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_create);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_destroy);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_init_cuda);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_init_npp);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_init_engines);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_init_buffers);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_init_all);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_reinit_buffers);

    /*=====================================================================*/
    /* Embedding & Scheduler Preparation                                   */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, prepare_embeds);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, blend_embeds);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, prepare_null_embeds);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, prepare_negative_embeds);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, prepare_sdxl_conditioning);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, prepare_scheduler);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, set_init_noise);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, reseed);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, set_guidance_scale);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, set_delta);

    /*=====================================================================*/
    /* High-Level Inference (CPU buffers)                                  */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, img2img);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, txt2img);

    /*=====================================================================*/
    /* Low-Level GPU Inference                                             */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, img2img_gpu_half);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, img2img_gpu_float);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, txt2img_gpu);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, txt2img_sd_turbo_gpu);

    /*=====================================================================*/
    /* VAE Operations                                                      */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, encode_image_half);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, encode_image_float);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, decode_latent);

    /*=====================================================================*/
    /* UNet Operations                                                     */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, predict_x0_batch);

    /*=====================================================================*/
    /* Image Format Conversion                                             */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, rgba_nhwc_to_nchw_float);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, rgba_nhwc_to_nchw_half);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, nchw_half_to_rgba_nhwc);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, nchw_float_to_rgba_nhwc);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, rgba_resize);

    /*=====================================================================*/
    /* Temporal Coherence (StreamV2V)                                      */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, enable_temporal_coherence);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, disable_temporal_coherence);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, reset_temporal_state);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, get_current_frame_id);

    /*=====================================================================*/
    /* CLIP Text Encoder API                                               */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, clip_create);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, clip_destroy);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, clip_compute_embeddings);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, clip_compute_embeddings_sdxl);

    /*=====================================================================*/
    /* Utility Functions                                                   */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, error_string);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, version);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, float_to_half);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, half_to_float);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_get_stream);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, pipeline_synchronize);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, get_last_cuda_error);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, get_last_cuda_error_string);

    /*=====================================================================*/
    /* CUDA Memory Management                                              */
    /*=====================================================================*/

    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_malloc);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_free);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_malloc_host);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_free_host);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_memcpy_h2d);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_memcpy_d2h);
    LIBREDIFFUSION_SYMBOL_INIT(librediffusion, cuda_device_synchronize);
  }
};

} // namespace sd
