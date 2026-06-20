#include "librediffusion.hpp"

namespace librediffusion
{

void LibreDiffusionPipeline::prepare_null_embeds(
    const __half* null_embeds, // [1, seq_len, hidden_dim]
    int seq_len, int hidden_dim)
{
  // Store null (unconditional) embeddings for StreamV2V
  // Python StreamV2V stores this as self.null_prompt_embeds = encoder_output[1]
  int null_size = 1 * seq_len * hidden_dim;
  temporal_state_.null_prompt_embeds = std::make_unique<CUDATensor<__half>>(null_size);
  temporal_state_.null_prompt_embeds->load_d2d(null_embeds, null_size, stream_);
}

void LibreDiffusionPipeline::enableTemporalCoherence(
    bool use_feature_injection, float injection_strength, float similarity_threshold,
    int cache_interval, int max_cached_frames)
{
  config_.mode = PipelineMode::TEMPORAL_V2V;
  config_.use_feature_injection = use_feature_injection;
  config_.feature_injection_strength = injection_strength;
  config_.feature_similarity_threshold = similarity_threshold;
  config_.cache_interval = cache_interval;
  config_.cache_maxframes = max_cached_frames;

  // Allocate temporal state buffers if not already allocated
  if(!temporal_state_.randn_noise)
  {
    int single_latent_size = 1 * 4 * config_.latent_height * config_.latent_width;
    temporal_state_.randn_noise = std::make_unique<CUDATensor<__half>>(single_latent_size);
    temporal_state_.warp_noise = std::make_unique<CUDATensor<__half>>(single_latent_size);

    // Initialize with current init_noise
    if(init_noise_)
    {
      cudaMemcpyAsync(
          temporal_state_.randn_noise->data(), init_noise_->data(),
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream_);
      cudaMemcpyAsync(
          temporal_state_.warp_noise->data(), init_noise_->data(),
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream_);
      cudaStreamSynchronize(stream_);
    }
  }

  temporal_state_.frame_id = 0;
}

void LibreDiffusionPipeline::disableTemporalCoherence()
{
  config_.mode = PipelineMode::SINGLE_FRAME;
}

void LibreDiffusionPipeline::resetTemporalState()
{
  temporal_state_.prev_image_tensor.reset();
  temporal_state_.prev_x_t_latent.reset();
  temporal_state_.cached_x_t_latent.clear();
  temporal_state_.cached_attentions.clear();
  temporal_state_.frame_id = 0;
}
}
