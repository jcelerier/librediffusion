#include "librediffusion.hpp"

#include "tensorrt_wrappers.hpp"
#include "kernels.hpp"

#include <cassert>
#include <numeric>
namespace librediffusion
{

// Grow-only / in-place (re)allocation for a buffer the captured CUDA graph reads BY ADDRESS. A live
// prompt/timestep edit calls these prepare_* every change; reallocating would move the device address
// and the replayed 1-step graph (whose memcpy nodes baked the capture-time pointer) would read a
// stale/freed buffer -> garbage on the next edit (this is the "any change randomly breaks" regression;
// random because cudaFree+cudaMalloc may or may not return the same address). capture_signature() does
// NOT hash these buffers, so reuse the buffer when the shape is unchanged (no recapture needed) and only
// (re)allocate + invalidate the graph on a genuine size change. Mirrors the set_controlnet_cond /
// set_ipadapter_tokens fix.
template <typename T>
static void reuse_or_realloc(
    std::unique_ptr<CUDATensor<T>>& buf, size_t n, bool& graph_ready)
{
  if(!buf || buf->size() != n)
  {
    buf = std::make_unique<CUDATensor<T>>(n);
    graph_ready = false;  // address moved -> any captured graph is stale
  }
}

void LibreDiffusionPipeline::prepare_embeds(
    const __half* prompt_embeds, // [batch_size, seq_len, hidden_dim]
    int seq_len, int hidden_dim)
{
  int prompt_size = config_.batch_size * seq_len * hidden_dim;
  reuse_or_realloc(prompt_embeds_, (size_t)prompt_size, graph_ready_);
  prompt_embeds_->load_d2d(prompt_embeds, prompt_size, stream_);
}

void LibreDiffusionPipeline::prepare_negative_embeds(
    const __half* negative_embeds, // [1, seq_len, hidden_dim]
    int seq_len, int hidden_dim)
{
  // Stored as single embedding [1, seq_len, hidden_dim], repeated as needed during inference
  int size = 1 * seq_len * hidden_dim;
  assert(size > 0);
  reuse_or_realloc(negative_embeds_, (size_t)size, graph_ready_);
  assert(negative_embeds_.get());
  negative_embeds_->load_d2d(negative_embeds, size, stream_);
}

void LibreDiffusionPipeline::prepare_sdxl_conditioning(
    const __half* text_embeds,  // [batch_size, pooled_dim]
    const __half* time_ids)     // [batch_size, 6]
{
  // text_embeds (pooled embeddings from second CLIP encoder)
  int text_embeds_size = config_.batch_size * config_.pooled_embedding_dim;
  reuse_or_realloc(text_embeds_, (size_t)text_embeds_size, graph_ready_);
  text_embeds_->load_d2d(text_embeds, text_embeds_size, stream_);

  // time_ids [height, width, crop_top, crop_left, target_height, target_width]
  int time_ids_size = config_.batch_size * config_.time_ids_dim;
  reuse_or_realloc(time_ids_, (size_t)time_ids_size, graph_ready_);
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
  reuse_or_realloc(prompt_embeds_, (size_t)size, graph_ready_);

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
