#include "librediffusion.hpp"
#include "tensorrt_wrappers.hpp"
#include "kernels.hpp"

namespace librediffusion
{

void LibreDiffusionPipeline::add_noise_direct(
    __half* original_samples, const __half* noise, int t_index, int N,
    cudaStream_t stream)
{
  // Access host-side copies 
  float alpha = alpha_prod_t_sqrt_host_[t_index];
  float beta = beta_prod_t_sqrt_host_[t_index];

  launch_add_noise_direct_fp16(original_samples, noise, alpha, beta, N, stream);
}


void LibreDiffusionPipeline::encode_image(
    const __half* image, __half* latent_out, cudaStream_t stream)
{
  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }

  // Call VAE encoder directly with FP16 input
  vae_encoder_->encode(
      image,      // Pass FP16 directly. Wrapper converts to persistent FP32 buffer
      latent_out, // Engine outputs FP16
      config_.batch_size, config_.height, config_.width, stream);

  // Apply VAE scaling factor (matching Python's pipeline.py:609)
  // This is required for Stable Diffusion VAE to match the latent distribution
  // expected by the UNet. Python applies: img_latent.mul_(self.vae.config.scaling_factor)
  // NOTE: TAESD (Tiny AutoEncoder for SD) has scaling_factor=1.0, full SD VAE has 0.18215
  constexpr float vae_scaling_factor = 1.0f;  // TAESD scaling factor
  const int latent_elements
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

  // Only apply scaling if needed (not for TAESD)
  if (vae_scaling_factor != 1.0f) {
    launch_scalar_mul_inplace_fp16(latent_out, vae_scaling_factor, latent_elements, stream);
  }

  // Add noise to the clean latent (matching Python's pipeline.py encode_image).
  // Python: x_t_latent = self.add_noise(img_latent, init_noise[0], 0)
  // CRITICAL: encode_image adds noise@t0 UNCONDITIONALLY in every StreamDiffusion
  // reference (upstream cumulo-autumn, the bundled fork, and daydream). The
  // `do_add_noise` flag gates only the DENOISING-LOOP noise additions (see
  // predict_x0_batch in librediffusion.unet.cpp), NOT this encode-time one. A previous
  // `config_.do_add_noise &&` guard here made noise-0 img2img return the clean
  // (alpha=1) latent instead of alpha0*latent + beta0*noise, diverging from the
  // reference (validation harness: encode cos 0.73 -> 0.99999 once corrected).
  // Still guard on scheduler init (prepare() must have run to populate alpha/beta).
  if(init_noise_ && !alpha_prod_t_sqrt_host_.empty())
  {
    // Python uses init_noise[0]
    add_noise_direct(
        latent_out,          // clean latent (img_latent)
        init_noise_->data(), // noise[0] - first timestep noise
        0,                   // t_index = 0 (first timestep)
        latent_elements, stream);
  }
}

void LibreDiffusionPipeline::encode_image(
    const float* image, __half* latent_out, cudaStream_t stream)
{
  if(stream == 0)
  {
    stream = stream_;
  }

  vae_encoder_->encode(
      image,
      latent_out, // Engine outputs FP16
      config_.batch_size, config_.height, config_.width, stream);

  const int latent_elements
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;
  // Unconditional add_noise@t0 — matches every StreamDiffusion reference (see the __half
  // overload above). do_add_noise gates only the denoising-loop noise, not this one.
  if(init_noise_ && !alpha_prod_t_sqrt_host_.empty())
  {
    add_noise_direct(
        latent_out,          // clean latent (img_latent)
        init_noise_->data(), // noise[0] - first timestep noise
        0,                   // t_index = 0 (first timestep)
        latent_elements, stream);
  }
}

void LibreDiffusionPipeline::decode_latent(
    const __half* latent, __half* image_out, cudaStream_t stream)
{
  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }

  // Divide by VAE scaling factor before decoding (matching Python's pipeline.py:620)
  // This is the inverse of the multiplication we do during encoding
  // Python: scaled_latent = x_0_pred_out / self.vae.config.scaling_factor
  constexpr float vae_scaling_factor = 1.0f; 
  const int latent_elements = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

  // For TAESD with scaling_factor=1.0, no scaling is needed, but keep the code structure
  // for compatibility if we switch to full SD VAE later
  if (vae_scaling_factor != 1.0f) {
    // Create temp buffer for scaled latent
    CUDATensor<__half> scaled_latent(latent_elements);
    scaled_latent.load_d2d(latent, latent_elements, stream);

    // Divide by scaling factor (inverse of encode)
    const float inv_scaling_factor = 1.0f / vae_scaling_factor;
    launch_scalar_mul_inplace_fp16(scaled_latent.data(), inv_scaling_factor, latent_elements, stream);

    // Call VAE decoder with scaled latent
    vae_decoder_->decode(
        scaled_latent.data(),
        image_out,
        config_.batch_size, config_.latent_height, config_.latent_width, stream);
  } else {
    vae_decoder_->decode(
        latent,  // Pass unscaled latent
        image_out,
        config_.batch_size, config_.latent_height, config_.latent_width, stream);
  }
}
}
