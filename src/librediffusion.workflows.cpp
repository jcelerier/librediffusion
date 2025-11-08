#include "librediffusion.hpp"
#include "tensorrt_wrappers.hpp"
#include "kernels.hpp"

namespace librediffusion
{

void LibreDiffusionPipeline::img2img_impl(
    const __half* image_in, __half* image_out, cudaStream_t stream)
{
  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }

  // Encode
  encode_image(image_in, vae_encoded_x_t_latent_->data(), stream);

  // See pipeline.py lines 704-712 (__call__) and 626-686 (predict_x0_batch).

  // StreamV2V: Cache x_t_latent for temporal coherence
  if(config_.mode == PipelineMode::TEMPORAL_V2V)
  {
    // Update cache every N frames (Python: if self.frame_id % self.interval == 0)
    if((temporal_state_.frame_id % config_.cache_interval) == 0)
    {
      int latent_size = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

      // Create new cache entry
      auto cached_latent = std::make_unique<CUDATensor<__half>>(latent_size);
      cached_latent->load_d2d(vae_encoded_x_t_latent_->data(), latent_size, stream);

      // Add to deque (automatically maintains maxlen)
      temporal_state_.cached_x_t_latent.push_back(std::move(cached_latent));

      // Maintain max size manually (deque doesn't have maxlen like Python)
      while(temporal_state_.cached_x_t_latent.size() > static_cast<size_t>(config_.cache_maxframes))
      {
        temporal_state_.cached_x_t_latent.pop_front();
      }
    }

    // Increment frame counter
    temporal_state_.frame_id++;
  }

  // Denoise (predict_x0_batch will handle noise internally if do_add_noise is set)
  predict_x0_batch(
      vae_encoded_x_t_latent_->data(), unet_output_x_0_pred_->data(), stream);

  // Decode
  decode_latent(unet_output_x_0_pred_->data(), image_out, stream);
}

void LibreDiffusionPipeline::img2img_impl(
    const float* image_in, __half* image_out, cudaStream_t stream)
{
  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }

  // Encode
  encode_image(image_in, vae_encoded_x_t_latent_->data(), stream);

  // Predict
  predict_x0_batch(
      vae_encoded_x_t_latent_->data(), unet_output_x_0_pred_->data(), stream);

  // Decode
  decode_latent(unet_output_x_0_pred_->data(), image_out, stream);
}

void LibreDiffusionPipeline::txt2img_impl(__half* image_out, cudaStream_t stream)
{
  // SDXL-Turbo also uses the single-step direct prediction (like SD-Turbo)
  if(config_.model_type == ModelType::SD_TURBO || config_.model_type == ModelType::SDXL_TURBO)
    return txt2img_sd_turbo_impl(image_out, stream);

  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }
  int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

  // Generate random latent
  launch_randn_fp16(vae_encoded_x_t_latent_->data(), config_.seed, latent_size, stream);

  // Denoise
  predict_x0_batch(
      vae_encoded_x_t_latent_->data(), unet_output_x_0_pred_->data(), stream);

  // Decode
  decode_latent(unet_output_x_0_pred_->data(), image_out, stream);
}

void LibreDiffusionPipeline::txt2img_sd_turbo_impl(
    __half* image_out, cudaStream_t stream)
{
  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }

  const int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

  // Generate random latent (x_t_latent)
  launch_randn_fp16(vae_encoded_x_t_latent_->data(), config_.seed, latent_size, stream);

  // Prepare UNet input (convert FP16 to FP32 for TensorRT)
  int latent_elements = config_.batch_size * 4 * config_.latent_height * config_.latent_width;
  CUDATensor<float> unet_input_latent_fp32(latent_elements);
  launch_fp16_to_fp32(
      vae_encoded_x_t_latent_->data(), unet_input_latent_fp32.data(), latent_elements,
      stream);

  // Run UNet inference
  CUDATensor<__half> model_pred(latent_size);
  if(config_.model_type == ModelType::SDXL_TURBO)
  {
    unet_->forward_sdxl(
        unet_input_latent_fp32.data(), sub_timesteps_->data(), prompt_embeds_->data(),
        text_embeds_->data(), time_ids_->data(),
        model_pred.data(), config_.batch_size, config_.latent_height, config_.latent_width,
        config_.text_seq_len, config_.text_hidden_dim,
        config_.pooled_embedding_dim, stream);
  }
  else
  {
    unet_->forward(
        unet_input_latent_fp32.data(), sub_timesteps_->data(), prompt_embeds_->data(),
        model_pred.data(), config_.batch_size, config_.latent_height, config_.latent_width,
        config_.text_seq_len,
        config_.text_hidden_dim,
        stream);
  }

  // SD-Turbo formula: x_0_pred_out = (x_t_latent - beta * model_pred) / alpha
  // Step 1: model_pred *= beta
  float beta = beta_prod_t_sqrt_host_[0];
  launch_scalar_mul_inplace_fp16(model_pred.data(), beta, latent_size, stream);

  // Step 2: x_0_pred_out = x_t_latent - model_pred
  launch_tensor_sub_fp16(
      vae_encoded_x_t_latent_->data(), model_pred.data(), unet_output_x_0_pred_->data(),
      latent_size, stream);

  // Step 3: x_0_pred_out /= alpha
  float alpha = alpha_prod_t_sqrt_host_[0];
  CUDATensor<__half> x_0_pred_final(latent_size);
  launch_scalar_div_fp16(
      unet_output_x_0_pred_->data(), x_0_pred_final.data(), alpha, latent_size, stream);

  // Decode latent to image
  decode_latent(x_0_pred_final.data(), image_out, stream);
}

void LibreDiffusionPipeline::img2img(
    const uint8_t* cpu_rgba_input, uint8_t* cpu_rgba_output, int iw, int ih)
{
  int batch_size = config_.batch_size;
  size_t rgba_input_size = batch_size * ih * iw * 4;
  size_t rgba_vae_size = batch_size * this->config_.height * this->config_.width * 4;
  size_t tensor_size = batch_size * 3 * this->config_.height * this->config_.width;

  // Allocate or resize device buffers if needed
  if(!device_rgba_input_ || device_rgba_input_->size() < rgba_input_size)
  {
    device_rgba_input_ = std::make_unique<CUDATensor<uint8_t>>(rgba_input_size);
  }
  if(!device_rgba_output_ || device_rgba_output_->size() < rgba_input_size)
  {
    device_rgba_output_ = std::make_unique<CUDATensor<uint8_t>>(rgba_input_size);
  }
  if(!device_rgba_input_vae_size_ || device_rgba_input_vae_size_->size() < rgba_vae_size)
  {
    device_rgba_input_vae_size_ = std::make_unique<CUDATensor<uint8_t>>(rgba_vae_size);
  }
  if(!device_rgba_output_vae_size_
     || device_rgba_output_vae_size_->size() < rgba_vae_size)
  {
    device_rgba_output_vae_size_ = std::make_unique<CUDATensor<uint8_t>>(rgba_vae_size);
  }
  if (!device_nchw_input_ || device_nchw_input_->size() < tensor_size) {
    device_nchw_input_ = std::make_unique<CUDATensor<float>>(tensor_size);
  }
  if (!device_nchw_output_ || device_nchw_output_->size() < tensor_size) {
    device_nchw_output_ = std::make_unique<CUDATensor<__half>>(tensor_size);
  }

  // Copy input from CPU to GPU
  cudaMemcpyAsync(
      device_rgba_input_->data(), cpu_rgba_input, rgba_input_size,
      cudaMemcpyHostToDevice, stream_);

  img_preprocess(device_rgba_input_->data(), iw, ih);

  // Run inference
  img2img_impl(device_nchw_input_->data(), device_nchw_output_->data(), stream_);

  uint8_t* device_rgba_output_correct_size
      = img_postprocess(device_nchw_output_->data(), iw, ih);

  // Copy output from GPU to CPU
  cudaMemcpyAsync(
      cpu_rgba_output, device_rgba_output_correct_size, rgba_input_size,
      cudaMemcpyDeviceToHost, stream_);

  // Wait for all operations to complete
  //cudaStreamSynchronize(stream_);
}

void LibreDiffusionPipeline::txt2img(uint8_t* cpu_rgba_output, int iw, int ih)
{
  int batch_size = config_.batch_size;
  size_t rgba_input_size = batch_size * ih * iw * 4;
  size_t rgba_vae_size = batch_size * this->config_.height * this->config_.width * 4;
  size_t tensor_size = batch_size * 3 * this->config_.height * this->config_.width;

  // Allocate or resize device buffers if needed
  if(!device_rgba_output_ || device_rgba_output_->size() < rgba_input_size)
  {
    device_rgba_output_ = std::make_unique<CUDATensor<uint8_t>>(rgba_input_size);
  }
  if(!device_rgba_output_vae_size_
     || device_rgba_output_vae_size_->size() < rgba_vae_size)
  {
    device_rgba_output_vae_size_ = std::make_unique<CUDATensor<uint8_t>>(rgba_vae_size);
  }
  if (!device_nchw_output_ || device_nchw_output_->size() < tensor_size) {
    device_nchw_output_ = std::make_unique<CUDATensor<__half>>(tensor_size);
  }

  // Run inference
  txt2img_impl(device_nchw_output_->data(), stream_);

  uint8_t* device_rgba_output_correct_size
      = img_postprocess(device_nchw_output_->data(), iw, ih);

  // Copy output from GPU to CPU
  cudaMemcpyAsync(
      cpu_rgba_output, device_rgba_output_correct_size, rgba_input_size,
      cudaMemcpyDeviceToHost, stream_);

  // Wait for all operations to complete
  cudaStreamSynchronize(stream_);
}
}
