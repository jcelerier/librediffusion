#include "librediffusion.hpp"

#include "tensorrt_wrappers.hpp"
#include "kernels.hpp"

#include <cassert>
#include <numeric>
namespace librediffusion
{

void LibreDiffusionPipeline::prepare_embeds(
    const __half* prompt_embeds, // [batch_size, seq_len, hidden_dim]
    int seq_len, int hidden_dim)
{
  // Allocate prompt embeddings
  int prompt_size = config_.batch_size * seq_len * hidden_dim;
  prompt_embeds_ = std::make_unique<CUDATensor<__half>>(prompt_size);
  prompt_embeds_->load_d2d(prompt_embeds, prompt_size, stream_);
}

void LibreDiffusionPipeline::prepare_negative_embeds(
    const __half* negative_embeds, // [1, seq_len, hidden_dim]
    int seq_len, int hidden_dim)
{
  // Allocate negative (unconditional) embeddings for CFG
  // Stored as single embedding [1, seq_len, hidden_dim], repeated as needed during inference
  int size = 1 * seq_len * hidden_dim;
  assert(size > 0);
  negative_embeds_ = std::make_unique<CUDATensor<__half>>(size);
  assert(negative_embeds_.get());
  negative_embeds_->load_d2d(negative_embeds, size, stream_);
}

void LibreDiffusionPipeline::prepare_sdxl_conditioning(
    const __half* text_embeds,  // [batch_size, pooled_dim]
    const __half* time_ids)     // [batch_size, 6]
{
  // Allocate and copy text_embeds (pooled embeddings from second CLIP encoder)
  int text_embeds_size = config_.batch_size * config_.pooled_embedding_dim;
  text_embeds_ = std::make_unique<CUDATensor<__half>>(text_embeds_size);
  text_embeds_->load_d2d(text_embeds, text_embeds_size, stream_);

  // Allocate and copy time_ids [height, width, crop_top, crop_left, target_height, target_width]
  int time_ids_size = config_.batch_size * config_.time_ids_dim;
  time_ids_ = std::make_unique<CUDATensor<__half>>(time_ids_size);
  time_ids_->load_d2d(time_ids, time_ids_size, stream_);
}

void LibreDiffusionPipeline::blend_embeds(
    const __half* const* embeddings,  // Array of embedding device pointers
    const float* weights,              // Blend weights (host array)
    int num_embeddings,
    int seq_len,
    int hidden_dim)
{
  if(num_embeddings <= 0)
    return;

  // Calculate total weight for normalization
  float total_weight = 0.0f;
  for(int i = 0; i < num_embeddings; i++)
  {
    total_weight += weights[i];
  }

  // Handle zero total weight: assign equal weights
  std::vector<float> normalized_weights(num_embeddings);
  if(total_weight == 0.0f)
  {
    float equal_weight = 1.0f / num_embeddings;
    for(int i = 0; i < num_embeddings; i++)
    {
      normalized_weights[i] = equal_weight;
    }
  }
  else
  {
    for(int i = 0; i < num_embeddings; i++)
    {
      normalized_weights[i] = weights[i] / total_weight;
    }
  }

  // Allocate or reuse prompt_embeds_ for the blended result
  // Result shape: [batch_size, seq_len, hidden_dim]
  int size = config_.batch_size * seq_len * hidden_dim;
  if(!prompt_embeds_ || prompt_embeds_->size() != static_cast<size_t>(size))
  {
    prompt_embeds_ = std::make_unique<CUDATensor<__half>>(size);
  }

  // Initialize accumulator to zero
  launch_zero_fill_fp16(prompt_embeds_->data(), size, stream_);

  // Weighted sum: result = sum(normalized_weight[i] * embed[i])
  // Each embedding is [1, seq_len, hidden_dim], we accumulate and then repeat for batch
  int single_embed_size = seq_len * hidden_dim;

  // Create temporary buffer for single blended embedding
  CUDATensor<__half> temp_blend(single_embed_size);
  launch_zero_fill_fp16(temp_blend.data(), single_embed_size, stream_);

  // Accumulate weighted embeddings
  for(int i = 0; i < num_embeddings; i++)
  {
    launch_weighted_accumulate_fp16(
        temp_blend.data(), embeddings[i], normalized_weights[i],
        single_embed_size, stream_);
  }

  // Repeat the blended embedding for each batch element
  for(int b = 0; b < config_.batch_size; b++)
  {
    cudaMemcpyAsync(
        prompt_embeds_->data() + b * single_embed_size,
        temp_blend.data(),
        single_embed_size * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream_);
  }

  cudaStreamSynchronize(stream_);
}

}
