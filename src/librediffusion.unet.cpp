#include "librediffusion.hpp"
#include "tensorrt_wrappers.hpp"
#include "kernels.hpp"

#include <cassert>
#include <iostream>
namespace librediffusion
{

void LibreDiffusionPipeline::run_controlnets(
    const __half* sample_fp16, const float* timestep, const __half* ehs,
    const __half* text_embeds, const __half* time_ids,
    int unet_batch_size, int seq_len, int hidden_dim, int pooled_dim,
    const __half** out_down, const __half** out_mid, int& down_count, void* stream_void)
{
  cudaStream_t stream = (cudaStream_t)stream_void;
  const int img_h = config_.latent_height * 8, img_w = config_.latent_width * 8;
  const int cond_row = 3 * img_h * img_w;
  const bool sdxl = (config_.model_type == ModelType::SDXL_TURBO);

  const __half* net_down[ControlNetWrapper::MAX_DOWN];
  const __half* net_mid = nullptr;
  // Persistent (grow-on-first-use): a per-frame stack CUDATensor here would do an illegal cudaMalloc inside
  // CUDA-graph capture. Same shape every frame for a fixed config.
  const size_t cond_tiled_n = (size_t)unet_batch_size * cond_row;
  if(!controlnet_cond_tiled_ || controlnet_cond_tiled_->size() < cond_tiled_n)
    controlnet_cond_tiled_ = std::make_unique<CUDATensor<__half>>(cond_tiled_n);
  __half* cond_tiled = controlnet_cond_tiled_->data();
  int nd = 0;

  for(size_t k = 0; k < controlnets_.size(); k++)
  {
    if(!controlnet_cond_[k])
      throw std::runtime_error("ControlNet " + std::to_string(k) + " has no control image set "
                               "(call set_controlnet_cond[_rgba])");
    // Tile this net's single cond row to unet_batch_size.
    for(int i = 0; i < unet_batch_size; i++)
      cudaMemcpyAsync(cond_tiled + (size_t)i * cond_row, controlnet_cond_[k]->data(),
                      cond_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    controlnets_[k]->forward(
        sample_fp16, timestep, ehs, cond_tiled, controlnet_scales_[k],
        text_embeds, time_ids, unet_batch_size, config_.latent_height, config_.latent_width,
        img_h, img_w, seq_len, hidden_dim, pooled_dim, net_down, &net_mid, stream);
    nd = controlnets_[k]->numDown();

    if(k == 0)
    {
      // Single net (or first): use its buffers directly — no copy/sum needed.
      if(controlnets_.size() == 1)
      {
        for(int i = 0; i < nd; i++) out_down[i] = net_down[i];
        *out_mid = net_mid;
        down_count = nd;
        return;
      }
      // Multi-net: initialize the sum accumulators with the first net's residuals (done below).
      controlnet_sum_down_.clear();
      controlnet_sum_down_.resize(nd);
    }
    // Determine each residual's element count from the engine geometry (same tables as the wrapper).
    static const int kSD15Ch[12] = {320,320,320,320,640,640,640,1280,1280,1280,1280,1280};
    static const int kSD15Fac[12] = {1,1,1,2,2,2,4,4,4,8,8,8};
    static const int kSDXLCh[9] = {320,320,320,320,640,640,640,1280,1280};
    static const int kSDXLFac[9] = {1,1,1,2,2,2,4,4,4};
    const int* ch = sdxl ? kSDXLCh : kSD15Ch;
    const int* fac = sdxl ? kSDXLFac : kSD15Fac;
    const int midCh = 1280, midFac = sdxl ? 4 : 8;
    auto down_elems = [&](int i) {
      return (size_t)unet_batch_size * ch[i] * (config_.latent_height / fac[i])
             * (config_.latent_width / fac[i]);
    };
    size_t mid_elems = (size_t)unet_batch_size * midCh * (config_.latent_height / midFac)
                       * (config_.latent_width / midFac);
    if(k == 0)
    {
      for(int i = 0; i < nd; i++)
      {
        controlnet_sum_down_[i] = std::make_unique<CUDATensor<__half>>(down_elems(i));
        cudaMemcpyAsync(controlnet_sum_down_[i]->data(), net_down[i],
                        down_elems(i) * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      controlnet_sum_mid_ = std::make_unique<CUDATensor<__half>>(mid_elems);
      cudaMemcpyAsync(controlnet_sum_mid_->data(), net_mid, mid_elems * sizeof(__half),
                      cudaMemcpyDeviceToDevice, stream);
    }
    else
    {
      for(int i = 0; i < nd; i++)
        launch_tensor_add_fp16(controlnet_sum_down_[i]->data(), net_down[i],
                               (int)down_elems(i), stream);
      launch_tensor_add_fp16(controlnet_sum_mid_->data(), net_mid, (int)mid_elems, stream);
    }
  }

  for(int i = 0; i < nd; i++) out_down[i] = controlnet_sum_down_[i]->data();
  *out_mid = controlnet_sum_mid_->data();
  down_count = nd;
}

void LibreDiffusionPipeline::add_noise(
    const __half* original_samples, const __half* noise, __half* noisy_samples,
    int t_index, int N, cudaStream_t stream)
{
  // Access host-side copies (safe for CPU access)
  float alpha = alpha_prod_t_sqrt_host_[t_index];
  float beta = beta_prod_t_sqrt_host_[t_index];

  launch_add_noise_fp16(original_samples, noise, noisy_samples, alpha, beta, N, stream);
}

void LibreDiffusionPipeline::apply_cfg(
    const __half* noise_pred_uncond, const __half* noise_pred_text,
    __half* model_pred_out, int N, cudaStream_t stream)
{
  launch_apply_cfg_fp16(
      noise_pred_uncond, noise_pred_text, model_pred_out, config_.guidance_scale, N,
      stream);
}


void LibreDiffusionPipeline::scheduler_step_batch(
    const __half* model_pred, const __half* x_t_latent, __half* denoised_out, int idx,
    int N, cudaStream_t stream)
{
  // Safety checks
  if(alpha_prod_t_sqrt_host_.empty() || beta_prod_t_sqrt_host_.empty()
     || c_skip_host_.empty() || c_out_host_.empty())
  {
    throw std::runtime_error(
        "scheduler_step_batch: scheduler parameters not initialized. Call prepare() "
        "first.");
  }
  if(idx < 0 || idx >= config_.denoising_steps)
  {
    throw std::runtime_error(
        "scheduler_step_batch: idx out of bounds: " + std::to_string(idx)
        + " (denoising_steps=" + std::to_string(config_.denoising_steps) + ")");
  }

  // Access host-side copies (safe for CPU access)
  float alpha = alpha_prod_t_sqrt_host_[idx];
  float beta = beta_prod_t_sqrt_host_[idx];
  float c_skip = c_skip_host_[idx];
  float c_out = c_out_host_[idx];

  launch_scheduler_step_fp16(
      model_pred, x_t_latent, denoised_out, alpha, beta, c_skip, c_out,
      1, // FIXME double-check that batch size is 1 in this case by comparing with the python version
      4, // channels
      config_.latent_height, config_.latent_width, stream);
}


void LibreDiffusionPipeline::predict_x0_batch_impl_multi_step_batched(
    const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream)
{
  const int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

  // Persistent scratch (hoisted from per-call stack CUDATensors -> no per-frame cudaMalloc; same
  // device buffer reused every frame). model_pred + denoised are sized to the batched post-cfg-merge
  // size (single_size == denoised_size) so the cfg branches below WRITE into model_pred rather than
  // reallocating it. Each is a reference to its persistent buffer so all downstream uses are unchanged.
  if(!mp_x_t_latent_ || (int)mp_x_t_latent_->size() < latent_size)
    mp_x_t_latent_ = std::make_unique<CUDATensor<__half>>(latent_size);
  CUDATensor<__half>& x_t_latent = *mp_x_t_latent_;

  // denoised buffer needs to be large enough for batched denoising
  int total_batch
      = config_.batch_size + (config_.denoising_steps - 1) * config_.frame_buffer_size;
  int denoised_size = total_batch * 4 * config_.latent_height * config_.latent_width;
  if(!mp_model_pred_ || (int)mp_model_pred_->size() < denoised_size)
    mp_model_pred_ = std::make_unique<CUDATensor<__half>>(denoised_size);
  CUDATensor<__half>& model_pred = *mp_model_pred_;
  if(!mp_denoised_ || (int)mp_denoised_->size() < denoised_size)
    mp_denoised_ = std::make_unique<CUDATensor<__half>>(denoised_size);
  CUDATensor<__half>& denoised = *mp_denoised_;

  x_t_latent.load_d2d(x_t_latent_in, latent_size, stream);

  // Batch denoising mode
  if(config_.denoising_steps > 1)
  {
    assert(x_t_latent_buffer_);
    // Concat x_t_latent with buffer
    void* input_ptrs[2] = {(void*)x_t_latent.data(), (void*)x_t_latent_buffer_->data()};
    size_t input_sizes[2]
        = {latent_size * sizeof(__half), x_t_latent_buffer_->size() * sizeof(__half)};

    const size_t concat_n = (size_t)latent_size + x_t_latent_buffer_->size();
    if(!mp_concatenated_ || mp_concatenated_->size() < concat_n)
      mp_concatenated_ = std::make_unique<CUDATensor<__half>>(concat_n);
    CUDATensor<__half>& concatenated = *mp_concatenated_;
    launch_concat(input_ptrs, 2, input_sizes, concatenated.data(), stream);

    // LRD-PERF: removed leftover DEBUG sync (concat is stream-ordered before its consumer)

    int total_batch
        = config_.batch_size + (config_.denoising_steps - 1) * config_.frame_buffer_size;

    // Rotate stock_noise at the START of the call, BEFORE the UNet — matches the reference
    // pipeline.py predict_x0_batch (denoising_steps_num > 1):
    //   self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)
    // i.e. prepend init_noise[0], drop the last row. stock_noise feeds noise_pred_uncond for
    // self/initialize RCFG; without this leading rotation the uncond term is misaligned every
    // step and self-cfg diverges into garbage (validation harness: self-cfg predict_x0 rel ~124,
    // psnr ~2.5). Use a temp because src/dst overlap. Only self(2)/initialize(3) consume stock_noise.
    if(config_.guidance_scale > 1.0f && (config_.cfg_type == 2 || config_.cfg_type == 3))
    {
      const int stride_l = 4 * config_.latent_height * config_.latent_width;
      CUDATensor<__half> rotated(stock_noise_->size());
      // rotated[0] = init_noise[0]
      cudaMemcpyAsync(
          rotated.data(), init_noise_->data(), stride_l * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      // rotated[1:] = stock_noise[:-1]
      if(config_.denoising_steps > 1)
      {
        cudaMemcpyAsync(
            rotated.data() + stride_l, stock_noise_->data(),
            (config_.denoising_steps - 1) * stride_l * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
      }
      cudaMemcpyAsync(
          stock_noise_->data(), rotated.data(), stock_noise_->size() * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      cudaStreamSynchronize(stream);
    }

    // Prepare UNet inputs based on cfg_type
    int unet_batch_size = total_batch;

    // cfg_type: 0=none, 1=full, 2=self, 3=initialize
    // Use pre-allocated buffers
    __half* unet_input_latent_ptr = unet_input_latent_->data();
    float* unet_input_timestep_ptr = unet_input_timestep_->data();

    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full"
      // Concatenate latents: [latent, latent] -> [2*batch, 4, H, W]
      void* latent_ptrs[2] = {(void*)concatenated.data(), (void*)concatenated.data()};
      size_t latent_sizes[2]
          = {concatenated.size() * sizeof(__half), concatenated.size() * sizeof(__half)};
      launch_concat(latent_ptrs, 2, latent_sizes, unet_input_latent_ptr, stream);

      // Duplicate timesteps: [batch] -> [2*batch]
      cudaMemcpyAsync(
          unet_input_timestep_ptr, sub_timesteps_->data(),
          total_batch * sizeof(float), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          unet_input_timestep_ptr + total_batch, sub_timesteps_->data(),
          total_batch * sizeof(float), cudaMemcpyDeviceToDevice, stream);

      unet_batch_size = total_batch * 2;
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
    { // cfg_type="initialize"
      // Python: x_t_latent_plus_uc = concat([x_t_latent[0:1], x_t_latent], dim=0)
      // Prepend first latent element to create [1 + total_batch] elements
      int single_latent_size = 4 * config_.latent_height * config_.latent_width;

      // Copy first latent (for uncond)
      cudaMemcpyAsync(
          unet_input_latent_ptr, concatenated.data(),
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      // Copy all latents after the first
      cudaMemcpyAsync(
          unet_input_latent_ptr + single_latent_size, concatenated.data(),
          concatenated.size() * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      // Python: t_list = concat([t_list[0:1], t_list], dim=0)
      // Copy first timestep
      cudaMemcpyAsync(
          unet_input_timestep_ptr, sub_timesteps_->data(),
          1 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
      // Copy all timesteps after
      cudaMemcpyAsync(
          unet_input_timestep_ptr + 1, sub_timesteps_->data(),
          total_batch * sizeof(float), cudaMemcpyDeviceToDevice, stream);

      unet_batch_size = total_batch + 1;
    }
    else
    {
      // cfg_type="none" or "self" - no batch modification
      cudaMemcpyAsync(
          unet_input_latent_ptr, concatenated.data(),
          concatenated.size() * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          unet_input_timestep_ptr, sub_timesteps_->data(),
          total_batch * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Convert FP16 latent input to FP32 for TensorRT engine
    int latent_elements
        = unet_batch_size * 4 * config_.latent_height * config_.latent_width;
    if(!mp_unet_input_latent_fp32_ || (int)mp_unet_input_latent_fp32_->size() < latent_elements)
      mp_unet_input_latent_fp32_ = std::make_unique<CUDATensor<float>>(latent_elements);
    CUDATensor<float>& unet_input_latent_fp32 = *mp_unet_input_latent_fp32_;
    launch_fp16_to_fp32(
        unet_input_latent_ptr, unet_input_latent_fp32.data(), latent_elements,
        stream);
    // LRD-PERF: removed a redundant in-region cudaStreamSynchronize here — fp16->fp32 is stream-ordered
    // before its consumers, and an in-region sync is illegal during CUDA-graph capture.

    // Prepare encoder_hidden_states: Must match unet_batch_size
    // For CFG "full" and "initialize", need [negative, positive] embeddings
    // Use pre-allocated buffer
    __half* unet_encoder_hidden_states_ptr = unet_encoder_hidden_states_->data();
    int embedding_size = config_.text_seq_len * config_.text_hidden_dim;

    // Get negative embedding source (fallback to positive if not set)
    const __half* neg_embed_src = negative_embeds_ ? negative_embeds_->data() : prompt_embeds_->data();

    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full" - [negative × total_batch, positive × total_batch]
      // Copy negative embeddings for first half
      for(int i = 0; i < total_batch; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_ptr + i * embedding_size,
            neg_embed_src,
            embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      // Copy positive embeddings for second half
      for(int i = 0; i < total_batch; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_ptr + (total_batch + i) * embedding_size,
            prompt_embeds_->data(),
            embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
    { // cfg_type="initialize" - [negative × 1, positive × total_batch]
      // Copy negative embedding for first element
      cudaMemcpyAsync(
          unet_encoder_hidden_states_ptr,
          neg_embed_src,
          embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      // Copy positive embeddings for remaining elements
      for(int i = 0; i < total_batch; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_ptr + (1 + i) * embedding_size,
            prompt_embeds_->data(),
            embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
    }
    else
    { // cfg_type="self" or "none" - [positive × unet_batch_size]
      for(int i = 0; i < unet_batch_size; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_ptr + i * embedding_size,
            prompt_embeds_->data(),
            embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
    }

    // Use pre-allocated unet output buffer
    __half* unet_output_ptr = unet_output_buffer_->data();

    // Run UNet inference
    if(config_.model_type == ModelType::SDXL_TURBO)
    {
      // SDXL conditioning (pooled text_embeds + time_ids) must match unet_batch_size, exactly like
      // encoder_hidden_states above. prepare_sdxl_conditioning sized text_embeds_/time_ids_ to
      // config_.batch_size (the frame buffer, =1), but the BATCHED multi-step UNet runs
      // unet_batch_size rows (total_batch, x2 for full/initialize). Passing the single-row buffers
      // with batch=unet_batch_size made forward_sdxl's cudaMemcpyAsync read past the source end
      // ("Copy is larger than memobj size") -> a sticky CUDA error that surfaced later as a CUB
      // cudaErrorInvalidDevice in the scheduler step. Tile the single conditioning row to
      // unet_batch_size here. (Single-step turbo never hit this: unet_batch_size==1==batch_size.)
      const int pooled_dim = config_.pooled_embedding_dim;
      const int time_ids_dim = config_.time_ids_dim;
      const size_t tte_n = (size_t)unet_batch_size * pooled_dim;
      const size_t tti_n = (size_t)unet_batch_size * time_ids_dim;
      if(!mp_tiled_text_embeds_ || mp_tiled_text_embeds_->size() < tte_n)
        mp_tiled_text_embeds_ = std::make_unique<CUDATensor<__half>>(tte_n);
      if(!mp_tiled_time_ids_ || mp_tiled_time_ids_->size() < tti_n)
        mp_tiled_time_ids_ = std::make_unique<CUDATensor<__half>>(tti_n);
      CUDATensor<__half>& tiled_text_embeds = *mp_tiled_text_embeds_;
      CUDATensor<__half>& tiled_time_ids = *mp_tiled_time_ids_;
      for(int i = 0; i < unet_batch_size; i++)
      {
        cudaMemcpyAsync(tiled_text_embeds.data() + (size_t)i * pooled_dim, text_embeds_->data(),
                        pooled_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(tiled_time_ids.data() + (size_t)i * time_ids_dim, time_ids_->data(),
                        time_ids_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      if(controlnet_enabled_)
      {
        // SDXL + ControlNet (multi): run every net (same x_t/timestep/ehs + tiled SDXL conditioning +
        // each net's control image), SUM their 9 down + mid residuals, inject once.
        const __half* down_ptrs[ControlNetWrapper::MAX_DOWN];
        const __half* mid_ptr = nullptr;
        int down_count = 0;
        run_controlnets(unet_input_latent_ptr, unet_input_timestep_ptr, unet_encoder_hidden_states_ptr,
                        tiled_text_embeds.data(), tiled_time_ids.data(), unet_batch_size,
                        config_.text_seq_len, config_.text_hidden_dim, pooled_dim,
                        down_ptrs, &mid_ptr, down_count, stream);
        unet_->forward_controlnet_sdxl(
            unet_input_latent_fp32.data(), unet_input_timestep_ptr, unet_encoder_hidden_states_ptr,
            tiled_text_embeds.data(), tiled_time_ids.data(), down_ptrs, mid_ptr,
            unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
            config_.text_seq_len, config_.text_hidden_dim, config_.pooled_embedding_dim, stream);
      }
      else
      {
        unet_->forward_sdxl(
            unet_input_latent_fp32.data(), unet_input_timestep_ptr,
            unet_encoder_hidden_states_ptr,
            tiled_text_embeds.data(), tiled_time_ids.data(),
            unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
            config_.text_seq_len, config_.text_hidden_dim,
            config_.pooled_embedding_dim, stream);
      }
    }
    else if(controlnet_enabled_)
    {
      // ControlNet (SD1.5, multi): run every net on the SAME x_t/timestep/ehs the UNet sees (sample is
      // the FP16 latent unet_input_latent_ptr, before fp32 conversion), SUM their residuals, inject once.
      const __half* down_ptrs[ControlNetWrapper::MAX_DOWN];
      const __half* mid_ptr = nullptr;
      int down_count = 0;
      run_controlnets(unet_input_latent_ptr, unet_input_timestep_ptr, unet_encoder_hidden_states_ptr,
                      nullptr, nullptr, unet_batch_size, config_.text_seq_len,
                      config_.text_hidden_dim, 0, down_ptrs, &mid_ptr, down_count, stream);
      unet_->forward_controlnet(
          unet_input_latent_fp32.data(), unet_input_timestep_ptr, unet_encoder_hidden_states_ptr,
          down_ptrs, mid_ptr,
          unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
          config_.text_seq_len, config_.text_hidden_dim, stream);
    }
    else if(ipadapter_enabled_)
    {
      // IP-Adapter (SD1.5): assemble the EXTENDED encoder_hidden_states [unet_batch, 77+N, dim] =
      // per row, the 77 text tokens (already in unet_encoder_hidden_states_ptr) ++ N image tokens.
      // Per-row image tokens follow the SAME neg/pos arrangement as the text ehs:
      //   full       -> [neg x total_batch, pos x total_batch]
      //   initialize -> [neg x 1, pos x total_batch]
      //   none/self  -> [pos x unet_batch_size]
      const int dim = config_.text_hidden_dim;
      const int N = ipadapter_num_tokens_;
      const int text_tok = config_.text_seq_len;             // 77
      const int ext_tok = text_tok + N;                      // 81
      const int text_row = text_tok * dim, img_row = N * dim, ext_row = ext_tok * dim;
      const __half* pos = ipadapter_tokens_pos_ ? ipadapter_tokens_pos_->data() : nullptr;
      const __half* neg = ipadapter_tokens_neg_ ? ipadapter_tokens_neg_->data() : pos;
      if(!pos)
        throw std::runtime_error("IP-Adapter enabled but no image tokens set (set_ipadapter_tokens)");

      const size_t ext_n = (size_t)unet_batch_size * ext_row;
      if(!mp_ext_ehs_ || mp_ext_ehs_->size() < ext_n)
        mp_ext_ehs_ = std::make_unique<CUDATensor<__half>>(ext_n);
      CUDATensor<__half>& ext_ehs = *mp_ext_ehs_;
      const bool full = (config_.guidance_scale > 1.0f && config_.cfg_type == 1);
      const bool init = (config_.guidance_scale > 1.0f && config_.cfg_type == 3);
      for(int r = 0; r < unet_batch_size; r++)
      {
        // text part (77) for this row.
        cudaMemcpyAsync(ext_ehs.data() + (size_t)r * ext_row,
                        unet_encoder_hidden_states_ptr + (size_t)r * text_row,
                        text_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        // image part (N): pick neg vs pos to match the row's text conditioning.
        bool is_neg = full ? (r < total_batch) : (init ? (r == 0) : false);
        const __half* img = is_neg ? neg : pos;
        cudaMemcpyAsync(ext_ehs.data() + (size_t)r * ext_row + text_row, img,
                        img_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      int nlay = unet_->numIpLayers();
      if(nlay <= 0) nlay = (int)ipadapter_scale_vec_.size();
      unet_->forward_ipadapter(
          unet_input_latent_fp32.data(), unet_input_timestep_ptr, ext_ehs.data(),
          ipadapter_scale_vec_.data(), nlay,
          unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
          ext_tok, config_.text_hidden_dim, stream);
    }
    else
    {
      unet_->forward(
          unet_input_latent_fp32.data(), unet_input_timestep_ptr,
          unet_encoder_hidden_states_ptr, // Use repeated embeddings
          unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
          config_.text_seq_len,
          config_.text_hidden_dim,
          stream);
    }

    // Apply CFG if needed
    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full"
      // Split output into uncond and cond
      int single_size = total_batch * 4 * config_.latent_height * config_.latent_width;
      CUDATensor<__half> noise_pred_uncond(single_size);
      CUDATensor<__half> noise_pred_text(single_size);

      cudaMemcpyAsync(
          noise_pred_uncond.data(), unet_output_ptr, single_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          noise_pred_text.data(), unet_output_ptr + single_size,
          single_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
      // Use pre-allocated buffer
      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_tmp_->data(),
          single_size, stream);
      // Create model_pred with the result
      // model_pred is the persistent buffer (sized denoised_size >= single_size): write the CFG
      // result into it instead of reallocating (a per-frame malloc would block CUDA-graph capture).
      cudaMemcpyAsync(model_pred.data(), model_pred_tmp_->data(),
                      single_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
    { // cfg_type="initialize"
      // Python: noise_pred_text = model_pred[1:]  (skip first element which is uncond)
      // Python: self.stock_noise = concat([model_pred[0:1], self.stock_noise[1:]], dim=0)
      int single_size = total_batch * 4 * config_.latent_height * config_.latent_width;
      int single_latent_size = 4 * config_.latent_height * config_.latent_width;

      // noise_pred_text = model_pred[1:] - skip the first latent (uncond prediction)
      CUDATensor<__half> noise_pred_text(single_size);
      cudaMemcpyAsync(
          noise_pred_text.data(), unet_output_ptr + single_latent_size,
          single_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      // Update stock_noise: concat([model_pred[0:1], stock_noise[1:]], dim=0)
      // Copy model_pred[0:1] to stock_noise[0]
      cudaMemcpyAsync(
          stock_noise_->data(), unet_output_ptr,
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      // stock_noise[1:] remains unchanged (already has previous values)

      // noise_pred_uncond = stock_noise * delta
      CUDATensor<__half> noise_pred_uncond(single_size);
      cudaMemcpyAsync(
          noise_pred_uncond.data(), stock_noise_->data(), single_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      launch_scalar_mul_inplace_fp16(
          noise_pred_uncond.data(), config_.delta, single_size, stream);

      // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_tmp_->data(),
          single_size, stream);
      // model_pred is the persistent buffer (sized denoised_size >= single_size): write the CFG
      // result into it instead of reallocating (a per-frame malloc would block CUDA-graph capture).
      cudaMemcpyAsync(model_pred.data(), model_pred_tmp_->data(),
                      single_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 2)
    { // cfg_type="self"
      // noise_pred_text = model_pred (no modification needed)
      int single_size = total_batch * 4 * config_.latent_height * config_.latent_width;
      CUDATensor<__half> noise_pred_text(single_size);
      noise_pred_text.load_d2d(unet_output_ptr, single_size, stream);

      // noise_pred_uncond = stock_noise * delta
      CUDATensor<__half> noise_pred_uncond(single_size);
      cudaMemcpyAsync(
          noise_pred_uncond.data(), stock_noise_->data(), single_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      launch_scalar_mul_inplace_fp16(
          noise_pred_uncond.data(), config_.delta, single_size, stream);

      // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_tmp_->data(),
          single_size, stream);
      // model_pred is the persistent buffer (sized denoised_size >= single_size): write the CFG
      // result into it instead of reallocating (a per-frame malloc would block CUDA-graph capture).
      cudaMemcpyAsync(model_pred.data(), model_pred_tmp_->data(),
                      single_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    }
    else
    {
      // No CFG (cfg_type="none") -> model_pred (persistent buffer) = the UNet output.
      model_pred.load_d2d(unet_output_ptr, latent_elements, stream);
    }

    // Scheduler step - process each timestep separately with its own scheduler parameters
    // The model_pred and concatenated buffers contain denoising_steps batch elements
    // Each needs to be processed with its corresponding timestep's scheduler params
    // Stride is for ONE latent (one image at one timestep)
    int stride = 1 * 4 * config_.latent_height * config_.latent_width;
    int single_size = total_batch * 4 * config_.latent_height * config_.latent_width;
    for(int i = 0; i < config_.denoising_steps; i++)
    {
      int offset = i * stride;
      scheduler_step_batch(
          model_pred.data() + offset, concatenated.data() + offset,
          denoised.data() + offset,
          i, // Use the correct timestep index for scheduler parameters
          stride, stream);
    }

    // Update stock_noise for "self" and "initialize" CFG types
    // Python:
    //   if self.cfg_type == "self" or self.cfg_type == "initialize":
    //       scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
    //       delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
    //       alpha_next = concat([alpha[1:], ones_like(alpha[0:1])], dim=0)
    //       delta_x = alpha_next * delta_x
    //       beta_next = concat([beta[1:], ones_like(beta[0:1])], dim=0)
    //       delta_x = delta_x / beta_next
    //       init_noise = concat([init_noise[1:], init_noise[0:1]], dim=0)
    //       stock_noise = init_noise + delta_x
    if(config_.guidance_scale > 1.0f && (config_.cfg_type == 2 || config_.cfg_type == 3))
    {
      int batch_size = config_.denoising_steps;  // Number of timesteps

      // scaled_noise = beta * stock_noise
      CUDATensor<__half> scaled_noise(single_size);
      for(int i = 0; i < batch_size; i++)
      {
        int offset = i * stride;
        float beta = beta_prod_t_sqrt_host_[i];
        // Copy and scale: scaled_noise[i] = beta[i] * stock_noise[i]
        cudaMemcpyAsync(
            scaled_noise.data() + offset, stock_noise_->data() + offset,
            stride * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        launch_scalar_mul_inplace_fp16(scaled_noise.data() + offset, beta, stride, stream);
      }

      // delta_x = scheduler_step_batch(model_pred, scaled_noise, idx)
      CUDATensor<__half> delta_x(single_size);
      for(int i = 0; i < batch_size; i++)
      {
        int offset = i * stride;
        scheduler_step_batch(
            model_pred.data() + offset, scaled_noise.data() + offset,
            delta_x.data() + offset, i, stride, stream);
      }

      // delta_x = alpha_next * delta_x / beta_next
      // alpha_next[i] = alpha[i+1] for i < batch_size-1, else 1.0
      // beta_next[i] = beta[i+1] for i < batch_size-1, else 1.0
      for(int i = 0; i < batch_size; i++)
      {
        int offset = i * stride;
        float alpha_next = (i < batch_size - 1) ? alpha_prod_t_sqrt_host_[i + 1] : 1.0f;
        float beta_next = (i < batch_size - 1) ? beta_prod_t_sqrt_host_[i + 1] : 1.0f;
        float scale = alpha_next / beta_next;
        launch_scalar_mul_inplace_fp16(delta_x.data() + offset, scale, stride, stream);
      }

      // init_noise_rotated = concat([init_noise[1:], init_noise[0:1]], dim=0)
      // stock_noise = init_noise_rotated + delta_x
      // Compute stock_noise[i] = init_noise[(i+1) % batch_size] + delta_x[i]
      CUDATensor<__half> new_stock_noise(single_size);
      for(int i = 0; i < batch_size; i++)
      {
        int dst_offset = i * stride;
        int src_offset = ((i + 1) % batch_size) * stride;
        // new_stock_noise[i] = init_noise[(i+1) % batch_size] + delta_x[i]
        launch_add_noise_fp16(
            init_noise_->data() + src_offset, delta_x.data() + dst_offset,
            new_stock_noise.data() + dst_offset, 1.0f, 1.0f, stride, stream);
      }
      // Copy result to stock_noise
      cudaMemcpyAsync(
          stock_noise_->data(), new_stock_noise.data(), single_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
    }

    // Extract final prediction
    int final_offset = (config_.denoising_steps - 1) * stride;

    cudaMemcpyAsync(
        x_0_pred_out,
        denoised.data() + final_offset, // Pointer arithmetic already accounts for sizeof(__half)
        stride * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

    // Update x_t_latent_buffer with intermediate predictions (denoised[:-1])
    // Python: self.x_t_latent_buffer = alpha[1:] * x_0_pred[:-1] + beta[1:] * noise[1:]
    if(config_.denoising_steps > 1)
    {
      int buffer_size = (config_.denoising_steps - 1) * stride;

      if(config_.do_add_noise)
      {
        // Need to add noise: buffer = alpha * denoised + beta * noise
        // Python: self.x_t_latent_buffer = alpha[1:] * x_0_pred[:-1] + beta[1:] * noise[1:]
        // This means: buffer[i] = alpha[i+1] * denoised[i] + beta[i+1] * noise[i+1]
        // For denoising_steps=2: buffer[0] = alpha[1] * denoised[0] + beta[1] * noise[1]

        // WAIT - re-reading Python code more carefully:
        // x_0_pred[:-1] means "all but the last", so denoised[0:denoising_steps-1]
        // noise[1:] means "skip first", so noise[1:denoising_steps]
        // alpha[1:] means "skip first", so alpha[1:denoising_steps]

        // DEBUG: Check what we're putting into the buffer update
        // FIXME this does not look like the correct thing ?
        for(int i = 0; i < config_.denoising_steps - 1; i++)
        {
          int denoised_offset = i * stride;    // denoised[i]
          int noise_offset = (i + 1) * stride; // noise[i+1]
          int buffer_offset = i * stride;      // buffer[i]
          int t_index = i + 1;                 // Use alpha[i+1], beta[i+1]

          add_noise(
              denoised.data() + denoised_offset, init_noise_->data() + noise_offset,
              x_t_latent_buffer_->data() + buffer_offset, t_index, stride, stream);
        }
      }
      else
      {
        // No noise: buffer[i] = alpha[i+1] * denoised[i]  (beta*noise term is 0).
        // Python (do_add_noise=False): x_t_latent_buffer = alpha_prod_t_sqrt[1:] * x_0_pred[:-1].
        // The alpha scaling is NOT baked into the scheduler step output (denoised here IS
        // x_0_pred, the clean prediction), so we must apply alpha[i+1] explicitly. Previously
        // this raw-copied denoised[:-1] (omitting alpha<1), inflating the streaming buffer by
        // ~1/alpha and drifting the converged x0 magnitude ~5% high (validation harness:
        // cfg-none noise-0 predict_x0 rel 0.053, norm 146 vs golden 139).
        for(int i = 0; i < config_.denoising_steps - 1; i++)
        {
          int denoised_offset = i * stride; // denoised[i] = x_0_pred[i]
          int buffer_offset = i * stride;   // buffer[i]
          float alpha_next = alpha_prod_t_sqrt_host_[i + 1]; // alpha[i+1]
          cudaMemcpyAsync(
              x_t_latent_buffer_->data() + buffer_offset, denoised.data() + denoised_offset,
              stride * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
          launch_scalar_mul_inplace_fp16(
              x_t_latent_buffer_->data() + buffer_offset, alpha_next, stride, stream);
        }
      }
    }
  }
  else
  {
    // Single step denoising (denoising_steps == 1)
    std::unique_ptr<CUDATensor<__half>> unet_input_latent;
    std::unique_ptr<CUDATensor<float>> unet_input_timestep;
    int unet_batch_size = config_.batch_size;

    // cfg_type: 0=none, 1=full, 2=self, 3=initialize
    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full"
      // Concatenate latents: [uncond, cond] -> [2*batch, 4, H, W]
      int doubled_size = latent_size * 2;
      unet_input_latent = std::make_unique<CUDATensor<__half>>(doubled_size);

      void* latent_ptrs[2] = {(void*)x_t_latent.data(), (void*)x_t_latent.data()};
      size_t latent_sizes[2]
          = {latent_size * sizeof(__half), latent_size * sizeof(__half)};
      launch_concat(latent_ptrs, 2, latent_sizes, unet_input_latent->data(), stream);

      // Duplicate timesteps
      unet_input_timestep = std::make_unique<CUDATensor<float>>(config_.batch_size * 2);
      cudaMemcpyAsync(
          unet_input_timestep->data(), sub_timesteps_->data(),
          config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          unet_input_timestep->data() + config_.batch_size, sub_timesteps_->data(),
          config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);

      unet_batch_size = config_.batch_size * 2;
    }
    else
    {
      // No batch doubling
      unet_input_latent = std::make_unique<CUDATensor<__half>>(latent_size);
      unet_input_latent->load_d2d(x_t_latent.data(), latent_size, stream);
      unet_input_timestep = std::make_unique<CUDATensor<float>>(config_.batch_size);
      unet_input_timestep->load_d2d(sub_timesteps_->data(), config_.batch_size, stream);
    }

    // Convert FP16 latent input to FP32 for TensorRT engine
    int latent_elements_2
        = unet_batch_size * 4 * config_.latent_height * config_.latent_width;
    CUDATensor<float> unet_input_latent_fp32_2(latent_elements_2);
    launch_fp16_to_fp32(
        unet_input_latent->data(), unet_input_latent_fp32_2.data(), latent_elements_2,
        stream);

    // Use pre-allocated buffer
    __half* unet_output_ptr = unet_output_buffer_->data();
    int latent_elements = latent_elements_2;

    // Run UNet inference - use SDXL variant if configured
    if(config_.model_type == ModelType::SDXL_TURBO)
    {
      // Tile SDXL conditioning to unet_batch_size (see the primary batched path for rationale).
      const int pooled_dim = config_.pooled_embedding_dim;
      const int time_ids_dim = config_.time_ids_dim;
      CUDATensor<__half> tiled_text_embeds((size_t)unet_batch_size * pooled_dim);
      CUDATensor<__half> tiled_time_ids((size_t)unet_batch_size * time_ids_dim);
      for(int i = 0; i < unet_batch_size; i++)
      {
        cudaMemcpyAsync(tiled_text_embeds.data() + (size_t)i * pooled_dim, text_embeds_->data(),
                        pooled_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(tiled_time_ids.data() + (size_t)i * time_ids_dim, time_ids_->data(),
                        time_ids_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      unet_->forward_sdxl(
          unet_input_latent_fp32_2.data(), unet_input_timestep->data(),
          prompt_embeds_->data(),
          tiled_text_embeds.data(), tiled_time_ids.data(),
          unet_output_ptr, unet_batch_size,
          config_.latent_height, config_.latent_width, config_.text_seq_len, config_.text_hidden_dim,
          config_.pooled_embedding_dim, stream);
    }
    else
    {
      unet_->forward(
          unet_input_latent_fp32_2.data(), unet_input_timestep->data(),
          prompt_embeds_->data(), unet_output_ptr, unet_batch_size,
          config_.latent_height, config_.latent_width, config_.text_seq_len, config_.text_hidden_dim, stream);
    }

    // Apply CFG if needed
    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full"
      // Split and apply CFG
      CUDATensor<__half> noise_pred_uncond(latent_size);
      CUDATensor<__half> noise_pred_text(latent_size);

      cudaMemcpyAsync(
          noise_pred_uncond.data(), unet_output_ptr, latent_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          noise_pred_text.data(), unet_output_ptr + latent_size,
          latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      auto model_pred_tmp = std::make_unique<CUDATensor<__half>>(latent_size);
      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_tmp->data(),
          latent_size, stream);
      model_pred = std::move(*model_pred_tmp);
    }
    else
    {
      auto model_pred_tmp = std::make_unique<CUDATensor<__half>>(latent_elements);
      model_pred_tmp->load_d2d(unet_output_ptr, latent_elements, stream);
      model_pred = std::move(*model_pred_tmp);
    }

    // Scheduler step
    scheduler_step_batch(
        model_pred.data(), x_t_latent.data(), denoised.data(), 0, latent_size, stream);

    denoised.store_d2d(x_0_pred_out, latent_size, stream);
  }
}

void LibreDiffusionPipeline::predict_x0_batch_impl_multi_step_sequential(
    const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream)
{
  const int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;

  // Allocate temporary buffers
  CUDATensor<__half> x_t_latent(latent_size);
  CUDATensor<__half> model_pred(latent_size);

  // denoised buffer needs to be large enough for batched denoising
  int total_batch
      = config_.batch_size + (config_.denoising_steps - 1) * config_.frame_buffer_size;
  int denoised_size = total_batch * 4 * config_.latent_height * config_.latent_width;
  CUDATensor<__half> denoised(denoised_size);

  x_t_latent.load_d2d(x_t_latent_in, latent_size, stream);

  // Sequential multi-step mode: Loop through each timestep separately
  // This matches Python's pipeline.py lines 661-684

  // x_t_latent will be updated in each iteration
  CUDATensor<__half> current_latent(latent_size);
  current_latent.load_d2d(x_t_latent.data(), latent_size, stream);

  for(int idx = 0; idx < config_.denoising_steps; idx++)
  {
    // Prepare UNet inputs for this timestep
    std::unique_ptr<CUDATensor<__half>> unet_input_latent;
    std::unique_ptr<CUDATensor<float>> unet_input_timestep;
    int unet_batch_size = config_.batch_size;

    // cfg_type: 0=none, 1=full, 2=self, 3=initialize
    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full"
      // Concatenate latents: [uncond, cond] -> [2*batch, 4, H, W]
      int doubled_size = latent_size * 2;
      unet_input_latent = std::make_unique<CUDATensor<__half>>(doubled_size);

      void* latent_ptrs[2]
          = {(void*)current_latent.data(), (void*)current_latent.data()};
      size_t latent_sizes[2]
          = {latent_size * sizeof(__half), latent_size * sizeof(__half)};
      launch_concat(latent_ptrs, 2, latent_sizes, unet_input_latent->data(), stream);

      // Duplicate timestep for this iteration
      unet_input_timestep = std::make_unique<CUDATensor<float>>(config_.batch_size * 2);
      cudaMemcpyAsync(
          unet_input_timestep->data(), sub_timesteps_->data() + idx,
          config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          unet_input_timestep->data() + config_.batch_size, sub_timesteps_->data() + idx,
          config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);

      unet_batch_size = config_.batch_size * 2;
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
    { // cfg_type="initialize"
      // Python: x_t_latent_plus_uc = concat([x_t_latent[0:1], x_t_latent], dim=0)
      // For sequential mode with batch_size, prepend first latent
      int single_latent_size = 4 * config_.latent_height * config_.latent_width;
      int extended_size = latent_size + single_latent_size;
      unet_input_latent = std::make_unique<CUDATensor<__half>>(extended_size);

      // Copy first latent (for uncond)
      cudaMemcpyAsync(
          unet_input_latent->data(), current_latent.data(),
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      // Copy all latents after the first
      cudaMemcpyAsync(
          unet_input_latent->data() + single_latent_size, current_latent.data(),
          latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      // Python: t_list = concat([t_list[0:1], t_list], dim=0)
      unet_input_timestep = std::make_unique<CUDATensor<float>>(config_.batch_size + 1);
      // Copy first timestep
      cudaMemcpyAsync(
          unet_input_timestep->data(), sub_timesteps_->data() + idx,
          1 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
      // Copy all timesteps after
      cudaMemcpyAsync(
          unet_input_timestep->data() + 1, sub_timesteps_->data() + idx,
          config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);

      unet_batch_size = config_.batch_size + 1;
    }
    else
    {
      // cfg_type="none" or "self" - no batch modification
      unet_input_latent = std::make_unique<CUDATensor<__half>>(latent_size);
      unet_input_latent->load_d2d(current_latent.data(), latent_size, stream);
      unet_input_timestep = std::make_unique<CUDATensor<float>>(config_.batch_size);
      cudaMemcpyAsync(
          unet_input_timestep->data(), sub_timesteps_->data() + idx,
          config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Convert FP16 latent input to FP32 for TensorRT engine
    int latent_elements_seq
        = unet_batch_size * 4 * config_.latent_height * config_.latent_width;
    CUDATensor<float> unet_input_latent_fp32_seq(latent_elements_seq);
    launch_fp16_to_fp32(
        unet_input_latent->data(), unet_input_latent_fp32_seq.data(),
        latent_elements_seq, stream);

    // Prepare encoder_hidden_states: Must match unet_batch_size
    // For CFG "full" and "initialize", need [negative, positive] embeddings
    std::unique_ptr<CUDATensor<__half>> unet_encoder_hidden_states_seq;
    int embedding_size = config_.text_seq_len * config_.text_hidden_dim;
    int total_embedding_size = unet_batch_size * embedding_size;
    unet_encoder_hidden_states_seq
        = std::make_unique<CUDATensor<__half>>(total_embedding_size);

    // Get negative embedding source (fallback to positive if not set)
    const __half* neg_embed_src = negative_embeds_ ? negative_embeds_->data() : prompt_embeds_->data();

    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full" - [negative × batch_size, positive × batch_size]
      for(int i = 0; i < config_.batch_size; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_seq->data() + i * embedding_size,
            neg_embed_src, embedding_size * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
      }
      for(int i = 0; i < config_.batch_size; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_seq->data() + (config_.batch_size + i) * embedding_size,
            prompt_embeds_->data(), embedding_size * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
      }
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
    { // cfg_type="initialize" - [negative × 1, positive × batch_size]
      cudaMemcpyAsync(
          unet_encoder_hidden_states_seq->data(), neg_embed_src,
          embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      for(int i = 0; i < config_.batch_size; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_seq->data() + (1 + i) * embedding_size,
            prompt_embeds_->data(), embedding_size * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
      }
    }
    else
    { // cfg_type="self" or "none" - [positive × batch_size]
      for(int i = 0; i < config_.batch_size; i++)
      {
        cudaMemcpyAsync(
            unet_encoder_hidden_states_seq->data() + i * embedding_size,
            prompt_embeds_->data(), embedding_size * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
      }
    }

    // Use pre-allocated buffer
    __half* unet_output_ptr = unet_output_buffer_->data();
    int latent_elements = latent_elements_seq;

    // Run UNet inference - use SDXL variant if configured
    if(config_.model_type == ModelType::SDXL_TURBO)
    {
      // Tile SDXL conditioning to unet_batch_size (see the batched path for the full rationale:
      // text_embeds_/time_ids_ are sized config_.batch_size; the UNet runs unet_batch_size rows).
      const int pooled_dim = config_.pooled_embedding_dim;
      const int time_ids_dim = config_.time_ids_dim;
      CUDATensor<__half> tiled_text_embeds((size_t)unet_batch_size * pooled_dim);
      CUDATensor<__half> tiled_time_ids((size_t)unet_batch_size * time_ids_dim);
      for(int i = 0; i < unet_batch_size; i++)
      {
        cudaMemcpyAsync(tiled_text_embeds.data() + (size_t)i * pooled_dim, text_embeds_->data(),
                        pooled_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(tiled_time_ids.data() + (size_t)i * time_ids_dim, time_ids_->data(),
                        time_ids_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      if(controlnet_enabled_)
      {
        // SDXL + ControlNet in the sequential multi-step path (mirrors the batched site ~386):
        // run every net with the tiled SDXL conditioning, SUM residuals, inject via
        // forward_controlnet_sdxl. The SD1.5 forward_controlnet would leave text_embeds/time_ids
        // unset on this control-aware SDXL engine -> "Not all input dimensions specified".
        const __half* down_ptrs[ControlNetWrapper::MAX_DOWN];
        const __half* mid_ptr = nullptr;
        int down_count = 0;
        run_controlnets(unet_input_latent->data(), unet_input_timestep->data(),
                        unet_encoder_hidden_states_seq->data(),
                        tiled_text_embeds.data(), tiled_time_ids.data(), unet_batch_size,
                        config_.text_seq_len, config_.text_hidden_dim, pooled_dim,
                        down_ptrs, &mid_ptr, down_count, stream);
        unet_->forward_controlnet_sdxl(
            unet_input_latent_fp32_seq.data(), unet_input_timestep->data(),
            unet_encoder_hidden_states_seq->data(),
            tiled_text_embeds.data(), tiled_time_ids.data(), down_ptrs, mid_ptr,
            unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
            config_.text_seq_len, config_.text_hidden_dim, config_.pooled_embedding_dim, stream);
      }
      else
      {
        unet_->forward_sdxl(
            unet_input_latent_fp32_seq.data(), unet_input_timestep->data(),
            unet_encoder_hidden_states_seq->data(),
            tiled_text_embeds.data(), tiled_time_ids.data(),
            unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
            config_.text_seq_len, config_.text_hidden_dim,
            config_.pooled_embedding_dim, stream);
      }
    }
    else if(controlnet_enabled_)
    {
      // ControlNet (SD1.5) in the sequential multi-step path (mirrors the batched impl).
      const __half* down_ptrs[ControlNetWrapper::MAX_DOWN];
      const __half* mid_ptr = nullptr;
      int down_count = 0;
      run_controlnets(unet_input_latent->data(), unet_input_timestep->data(),
                      unet_encoder_hidden_states_seq->data(),
                      nullptr, nullptr, unet_batch_size, config_.text_seq_len,
                      config_.text_hidden_dim, 0, down_ptrs, &mid_ptr, down_count, stream);
      unet_->forward_controlnet(
          unet_input_latent_fp32_seq.data(), unet_input_timestep->data(),
          unet_encoder_hidden_states_seq->data(), down_ptrs, mid_ptr,
          unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
          config_.text_seq_len, config_.text_hidden_dim, stream);
    }
    else if(ipadapter_enabled_)
    {
      // IP-Adapter (SD1.5) in the sequential multi-step path (mirrors the batched impl 415-453).
      const int dim = config_.text_hidden_dim;
      const int N = ipadapter_num_tokens_;
      const int text_tok = config_.text_seq_len;
      const int ext_tok = text_tok + N;
      const int text_row = text_tok * dim, img_row = N * dim, ext_row = ext_tok * dim;
      const __half* pos = ipadapter_tokens_pos_ ? ipadapter_tokens_pos_->data() : nullptr;
      const __half* neg = ipadapter_tokens_neg_ ? ipadapter_tokens_neg_->data() : pos;
      if(!pos)
        throw std::runtime_error("IP-Adapter enabled but no image tokens set (set_ipadapter_tokens)");
      CUDATensor<__half> ext_ehs((size_t)unet_batch_size * ext_row);
      const bool full = (config_.guidance_scale > 1.0f && config_.cfg_type == 1);
      const bool init = (config_.guidance_scale > 1.0f && config_.cfg_type == 3);
      for(int r = 0; r < unet_batch_size; r++)
      {
        cudaMemcpyAsync(ext_ehs.data() + (size_t)r * ext_row,
                        unet_encoder_hidden_states_seq->data() + (size_t)r * text_row,
                        text_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        bool is_neg = full ? (r < config_.batch_size) : (init ? (r == 0) : false);
        const __half* img = is_neg ? neg : pos;
        cudaMemcpyAsync(ext_ehs.data() + (size_t)r * ext_row + text_row, img,
                        img_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
      int nlay = unet_->numIpLayers();
      if(nlay <= 0) nlay = (int)ipadapter_scale_vec_.size();
      unet_->forward_ipadapter(
          unet_input_latent_fp32_seq.data(), unet_input_timestep->data(), ext_ehs.data(),
          ipadapter_scale_vec_.data(), nlay,
          unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
          ext_tok, config_.text_hidden_dim, stream);
    }
    else
    {
      unet_->forward(
          unet_input_latent_fp32_seq.data(), unet_input_timestep->data(),
          unet_encoder_hidden_states_seq->data(), // Use repeated embeddings
          unet_output_ptr, unet_batch_size, config_.latent_height, config_.latent_width,
          config_.text_seq_len, config_.text_hidden_dim, stream);
    }

    // Apply CFG if needed
    CUDATensor<__half> model_pred_step(latent_size);
    if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
    { // cfg_type="full"
      // Split and apply CFG
      CUDATensor<__half> noise_pred_uncond(latent_size);
      CUDATensor<__half> noise_pred_text(latent_size);

      cudaMemcpyAsync(
          noise_pred_uncond.data(), unet_output_ptr, latent_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(
          noise_pred_text.data(), unet_output_ptr + latent_size,
          latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_step.data(),
          latent_size, stream);
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
    { // cfg_type="initialize"
      // Python: noise_pred_text = model_pred[1:]  (skip first element which is uncond)
      // Python: self.stock_noise = concat([model_pred[0:1], self.stock_noise[1:]], dim=0)
      int single_latent_size = 4 * config_.latent_height * config_.latent_width;

      // noise_pred_text = model_pred[1:] - skip the first latent (uncond prediction)
      CUDATensor<__half> noise_pred_text(latent_size);
      cudaMemcpyAsync(
          noise_pred_text.data(), unet_output_ptr + single_latent_size,
          latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      // Update stock_noise: concat([model_pred[0:1], stock_noise[1:]], dim=0)
      // For sequential mode, stock_noise has shape [batch_size, 4, H, W]
      // Copy model_pred[0:1] to stock_noise[0]
      cudaMemcpyAsync(
          stock_noise_->data(), unet_output_ptr,
          single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

      // noise_pred_uncond = stock_noise * delta
      CUDATensor<__half> noise_pred_uncond(latent_size);
      cudaMemcpyAsync(
          noise_pred_uncond.data(), stock_noise_->data(), latent_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      launch_scalar_mul_inplace_fp16(
          noise_pred_uncond.data(), config_.delta, latent_size, stream);

      // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_step.data(),
          latent_size, stream);
    }
    else if(config_.guidance_scale > 1.0f && config_.cfg_type == 2)
    { // cfg_type="self"
      // noise_pred_text = model_pred (no modification needed)
      CUDATensor<__half> noise_pred_text(latent_size);
      noise_pred_text.load_d2d(unet_output_ptr, latent_size, stream);

      // noise_pred_uncond = stock_noise * delta
      CUDATensor<__half> noise_pred_uncond(latent_size);
      cudaMemcpyAsync(
          noise_pred_uncond.data(), stock_noise_->data(), latent_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
      launch_scalar_mul_inplace_fp16(
          noise_pred_uncond.data(), config_.delta, latent_size, stream);

      // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
      apply_cfg(
          noise_pred_uncond.data(), noise_pred_text.data(), model_pred_step.data(),
          latent_size, stream);
    }
    else
    {
      // No CFG (cfg_type="none" or guidance_scale <= 1.0)
      model_pred_step.load_d2d(unet_output_ptr, latent_size, stream);
    }

    // Scheduler step to get x_0_pred for this iteration
    CUDATensor<__half> x_0_pred_step(latent_size);
    scheduler_step_batch(
        model_pred_step.data(), current_latent.data(), x_0_pred_step.data(),
        idx, // Use current timestep index
        latent_size, stream);

    // Update stock_noise for "self" and "initialize" CFG types (after scheduler step)
    // Python logic for stock_noise update:
    //   scaled_noise = beta * stock_noise
    //   delta_x = scheduler_step_batch(model_pred, scaled_noise, idx)
    //   alpha_next = alpha[idx+1] if idx < denoising_steps-1 else 1.0
    //   beta_next = beta[idx+1] if idx < denoising_steps-1 else 1.0
    //   delta_x = alpha_next * delta_x / beta_next
    //   init_noise_next = init_noise[(idx+1) % denoising_steps]
    //   stock_noise = init_noise_next + delta_x
    if(config_.guidance_scale > 1.0f && (config_.cfg_type == 2 || config_.cfg_type == 3))
    {
      int stride = latent_size;
      float beta = beta_prod_t_sqrt_host_[idx];

      // scaled_noise = beta * stock_noise
      CUDATensor<__half> scaled_noise(stride);
      cudaMemcpyAsync(
          scaled_noise.data(), stock_noise_->data(),
          stride * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      launch_scalar_mul_inplace_fp16(scaled_noise.data(), beta, stride, stream);

      // delta_x = scheduler_step_batch(model_pred, scaled_noise, idx)
      CUDATensor<__half> delta_x(stride);
      scheduler_step_batch(
          model_pred_step.data(), scaled_noise.data(),
          delta_x.data(), idx, stride, stream);

      // delta_x = alpha_next * delta_x / beta_next
      float alpha_next = (idx < config_.denoising_steps - 1) ? alpha_prod_t_sqrt_host_[idx + 1] : 1.0f;
      float beta_next = (idx < config_.denoising_steps - 1) ? beta_prod_t_sqrt_host_[idx + 1] : 1.0f;
      float scale = alpha_next / beta_next;
      launch_scalar_mul_inplace_fp16(delta_x.data(), scale, stride, stream);

      // stock_noise = init_noise[(idx+1) % denoising_steps] + delta_x
      // For sequential mode, init_noise has shape [denoising_steps, 4, H, W]
      int single_latent_size = 4 * config_.latent_height * config_.latent_width;
      int src_idx = (idx + 1) % config_.denoising_steps;
      int src_offset = src_idx * single_latent_size;

      launch_add_noise_fp16(
          init_noise_->data() + src_offset, delta_x.data(),
          stock_noise_->data(), 1.0f, 1.0f, stride, stream);
    }

    // If not the last timestep, prepare noisy latent for next iteration
    if(idx < config_.denoising_steps - 1)
    {
      if(config_.do_add_noise)
      {
        // Python: x_t_latent = alpha[idx+1] * x_0_pred + beta[idx+1] * randn_like(x_0_pred)
        int next_t_idx = idx + 1;

        // Generate fresh random noise for next timestep in FP16
        CUDATensor<__half> fresh_noise_fp16(latent_size);
        // Use a different seed for each timestep to get fresh noise
        unsigned long long noise_seed = config_.seed + idx + 1;
        launch_randn_fp16(fresh_noise_fp16.data(), noise_seed, latent_size, stream);

        // Apply: current_latent = alpha[idx+1] * x_0_pred + beta[idx+1] * fresh_noise
        add_noise(
            x_0_pred_step.data(), fresh_noise_fp16.data(), current_latent.data(),
            next_t_idx, latent_size, stream);
      }
      else
      {
        // No noise: just scale with alpha
        // current_latent = alpha[idx+1] * x_0_pred
        int next_t_idx = idx + 1;
        float alpha = alpha_prod_t_sqrt_host_[next_t_idx];

        // Copy and scale x_0_pred by alpha
        current_latent.load_d2d(x_0_pred_step.data(), latent_size, stream);
        launch_scalar_mul_inplace_fp16(
            current_latent.data(), alpha, latent_size, stream);
      }
    }
    else
    {
      // Last timestep: copy x_0_pred to output
      // FIXME why doesn't this work here? we're copying to current_latent and from current_latent to x_0_pred_out just after
      // cudaMemcpyAsync(
      //     x_0_pred_out, x_0_pred_step.data(), latent_size, cudaMemcpyDeviceToDevice,
      //     stream);
      current_latent.load_d2d(x_0_pred_step.data(), latent_size, stream);
    }
  }

  current_latent.store_d2d(x_0_pred_out, latent_size, stream);
}

// ---- CUDA-graph capture/replay wrapper around the 1-step device work ----
// capture_signature(): recapture key. Bakes everything the graph fixes — config geometry + the persistent
// buffer device addresses the body reads/writes. If any address moves (resolution change, engine swap,
// IP token resize), the signature changes -> recapture. Mirrors the Python allocate_buffers() graph-reset rule.
uint64_t LibreDiffusionPipeline::capture_signature() const
{
  auto mix = [](uint64_t h, uint64_t v) { return (h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2))); };
  auto ptr = [&](const void* p) { return mix(0, reinterpret_cast<uint64_t>(p)); };
  uint64_t h = 1469598103934665603ULL;
  h = mix(h, (uint64_t)config_.batch_size);
  h = mix(h, (uint64_t)config_.latent_height);
  h = mix(h, (uint64_t)config_.latent_width);
  h = mix(h, (uint64_t)config_.cfg_type);
  h = mix(h, (uint64_t)(int)config_.model_type);
  h = mix(h, (uint64_t)controlnet_enabled_);
  h = mix(h, (uint64_t)ipadapter_enabled_);
  h = mix(h, (uint64_t)ipadapter_num_tokens_);
  h = mix(h, (uint64_t)config_.text_seq_len);
  h = mix(h, (uint64_t)config_.text_hidden_dim);
  // persistent buffers the body touches:
  if(graph_in_staging_) h = mix(h, ptr(graph_in_staging_->data()));
  if(predict_x0_batch_x_t_latent) h = mix(h, ptr(predict_x0_batch_x_t_latent->data()));
  if(predict_x0_batch_unet_input_latent) h = mix(h, ptr(predict_x0_batch_unet_input_latent->data()));
  if(predict_x0_batch_unet_output) h = mix(h, ptr(predict_x0_batch_unet_output->data()));
  if(predict_x0_batch_model_pred) h = mix(h, ptr(predict_x0_batch_model_pred->data()));
  if(predict_x0_batch_denoised) h = mix(h, ptr(predict_x0_batch_denoised->data()));
  if(ip_ext_ehs_) h = mix(h, ptr(ip_ext_ehs_->data()));
  if(controlnet_cond_tiled_) h = mix(h, ptr(controlnet_cond_tiled_->data()));
  // Per-net cond source buffers the captured tiling-copy reads (now updated in place, so stable; a
  // resize still moves the address -> a changed signature -> recapture, which is the correct response).
  for(const auto& c : controlnet_cond_) if(c) h = mix(h, ptr(c->data()));
  return h;
}

void LibreDiffusionPipeline::predict_x0_batch_impl_single_step(
    const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream)
{
  // Graph path: only for 1-step, non-V2V, cfg-none (graph_enabled_ already gates 1-step+non-V2V; cfg-none
  // because the cfg-full/self/init post-process allocates per-frame scratch -> not capturable). All other
  // configs fall through to a direct body call (per-call enqueue), unchanged behavior.
  if(!graph_enabled_ || config_.cfg_type != 0)
  {
    run_single_step_body(x_t_latent_in, x_0_pred_out, stream);
    return;
  }

  const int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;
  if(!graph_in_staging_ || (int)graph_in_staging_->size() < latent_size)
  {
    graph_in_staging_ = std::make_unique<CUDATensor<__half>>(latent_size);
    graph_ready_ = false;  // staging buffer (re)allocated -> any prior graph is stale
  }
  // Stage the volatile caller input into the FIXED buffer the graph reads from (outside the captured region).
  graph_in_staging_->load_d2d(x_t_latent_in, latent_size, stream);

  const uint64_t sig = capture_signature();
  if(!graph_ready_ || sig != graph_sig_)
  {
    if(graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if(graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }
    // Warm enqueue (TRT requires a real enqueue before capture; also primes shapes/addresses). This run
    // also produces a VALID output for THIS frame, so we return after capturing.
    run_single_step_body(graph_in_staging_->data(), x_0_pred_out, stream);
    cudaStreamSynchronize(stream);
    cudaError_t cap = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if(cap != cudaSuccess)
    {
      // Capture unavailable -> permanent fallback to per-call enqueue (already produced this frame above).
      std::cerr << "Note: CUDA graph capture unavailable (" << cudaGetErrorString(cap)
                << "); using per-call enqueue" << std::endl;
      graph_enabled_ = false;
      return;
    }
    run_single_step_body(graph_in_staging_->data(), x_0_pred_out, stream);
    cudaError_t end = cudaStreamEndCapture(stream, &graph_);
    if(end != cudaSuccess || !graph_)
    {
      std::cerr << "Note: CUDA graph capture aborted (" << cudaGetErrorString(end)
                << "); using per-call enqueue" << std::endl;
      if(graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
      graph_enabled_ = false;
      cudaGetLastError();  // clear sticky capture error so this frame still returns success (warm run did it)
      return;
    }
    // Instantiate the captured graph. This can FAIL even when endCapture succeeded — e.g. the captured
    // body recorded a node that can't be instantiated (observed with the ControlNet/IP control-aware
    // engines). Its return was previously UNCHECKED, so the sticky CUDA error surfaced as an opaque
    // CUDA_ERROR (-4) from the caller. Check it + fall back to per-call enqueue (the warm run above
    // already produced THIS frame's output, so correctness is preserved).
    cudaError_t inst = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if(inst != cudaSuccess || !graph_exec_)
    {
      std::cerr << "Note: CUDA graph instantiate failed (" << cudaGetErrorString(inst)
                << "); using per-call enqueue" << std::endl;
      if(graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
      if(graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }
      graph_enabled_ = false;
      cudaGetLastError();  // clear the sticky error so the caller doesn't see a spurious CUDA_ERROR
      return;
    }
    // Recompute the signature AFTER the warm run, NOT the pre-warm-run `sig`. The warm run lazily
    // allocates persistent buffers the body touches (controlnet_cond_tiled_, the per-net cond buffers,
    // ip_ext_ehs_), so their device addresses only exist now. Storing the pre-warm-run sig made the
    // very next frame's signature differ -> a forced recapture, and that recapture frame faulted
    // (ControlNet illegal memory access). Storing the post-alloc sig makes frame 2 match -> replay.
    graph_sig_ = capture_signature();
    graph_ready_ = true;
    return;  // warm run already wrote x_0_pred_out for this frame
  }
  cudaGraphLaunch(graph_exec_, stream);
}

void LibreDiffusionPipeline::run_single_step_body(
    const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream)
{
  const int latent_size
      = config_.batch_size * 4 * config_.latent_height * config_.latent_width;
  // denoised buffer needs to be large enough for batched denoising
  const int total_batch
      = config_.batch_size + (config_.denoising_steps - 1) * config_.frame_buffer_size;

  predict_x0_batch_x_t_latent->load_d2d(x_t_latent_in, latent_size, stream);

  // Single-step mode for non-batch (use_denoising_batch=False, denoising_steps=1)
  // This should behave like the single-step within denoising batch block
  int unet_batch_size = config_.batch_size;

  // cfg_type: 0=none, 1=full, 2=self, 3=initialize
  if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
  { // cfg_type="full"
    void* latent_ptrs[2]
        = {(void*)predict_x0_batch_x_t_latent->data(),
           (void*)predict_x0_batch_x_t_latent->data()};
    size_t latent_sizes[2]
        = {latent_size * sizeof(__half), latent_size * sizeof(__half)};
    launch_concat(
        latent_ptrs, 2, latent_sizes, predict_x0_batch_unet_input_latent->data(),
        stream);

    cudaMemcpyAsync(
        predict_x0_batch_unet_input_timestep->data(), sub_timesteps_->data(),
        config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(
        predict_x0_batch_unet_input_timestep->data() + config_.batch_size,
        sub_timesteps_->data(), config_.batch_size * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);

    unet_batch_size = config_.batch_size * 2;
  }
  else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
  { // cfg_type="initialize"
    // Python: x_t_latent_plus_uc = concat([x_t_latent[0:1], x_t_latent], dim=0)
    int single_latent_size = 4 * config_.latent_height * config_.latent_width;

    // Copy first latent (for uncond)
    cudaMemcpyAsync(
        predict_x0_batch_unet_input_latent->data(),
        predict_x0_batch_x_t_latent->data(),
        single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    // Copy all latents after the first
    cudaMemcpyAsync(
        predict_x0_batch_unet_input_latent->data() + single_latent_size,
        predict_x0_batch_x_t_latent->data(),
        latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

    // Python: t_list = concat([t_list[0:1], t_list], dim=0)
    // Copy first timestep
    cudaMemcpyAsync(
        predict_x0_batch_unet_input_timestep->data(), sub_timesteps_->data(),
        1 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    // Copy all timesteps after
    cudaMemcpyAsync(
        predict_x0_batch_unet_input_timestep->data() + 1, sub_timesteps_->data(),
        config_.batch_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    unet_batch_size = config_.batch_size + 1;
  }
  else
  {
    // cfg_type="none" or "self" - no batch modification
    assert(predict_x0_batch_unet_input_latent);
    assert(predict_x0_batch_x_t_latent);
    assert(predict_x0_batch_unet_input_timestep);
    assert(sub_timesteps_);

    predict_x0_batch_unet_input_latent->load_d2d(
        predict_x0_batch_x_t_latent->data(), latent_size, stream);
    predict_x0_batch_unet_input_timestep->load_d2d(
        sub_timesteps_->data(), config_.batch_size, stream);
  }

  const int latent_elements_single
      = unet_batch_size * 4 * config_.latent_height * config_.latent_width;

  launch_fp16_to_fp32(
      predict_x0_batch_unet_input_latent->data(),
      predict_x0_batch_unet_input_latent_fp32_single->data(), latent_elements_single,
      stream);

  // Prepare encoder_hidden_states: Must match unet_batch_size
  // For CFG "full" and "initialize", need [negative, positive] embeddings
  int embedding_size = config_.text_seq_len * config_.text_hidden_dim;

  // Get negative embedding source (fallback to positive if not set)
  const __half* neg_embed_src = negative_embeds_ ? negative_embeds_->data() : prompt_embeds_->data();

  if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
  { // cfg_type="full" - [negative × batch_size, positive × batch_size]
    for(int i = 0; i < config_.batch_size; i++)
    {
      cudaMemcpyAsync(
          predict_x0_batch_unet_encoder_hidden_states_single->data() + i * embedding_size,
          neg_embed_src, embedding_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
    }
    for(int i = 0; i < config_.batch_size; i++)
    {
      cudaMemcpyAsync(
          predict_x0_batch_unet_encoder_hidden_states_single->data()
              + (config_.batch_size + i) * embedding_size,
          prompt_embeds_->data(), embedding_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
    }
  }
  else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
  { // cfg_type="initialize" - [negative × 1, positive × batch_size]
    cudaMemcpyAsync(
        predict_x0_batch_unet_encoder_hidden_states_single->data(), neg_embed_src,
        embedding_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    for(int i = 0; i < config_.batch_size; i++)
    {
      cudaMemcpyAsync(
          predict_x0_batch_unet_encoder_hidden_states_single->data()
              + (1 + i) * embedding_size,
          prompt_embeds_->data(), embedding_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
    }
  }
  else
  { // cfg_type="self" or "none" - [positive × batch_size]
    for(int i = 0; i < config_.batch_size; i++)
    {
      cudaMemcpyAsync(
          predict_x0_batch_unet_encoder_hidden_states_single->data() + i * embedding_size,
          prompt_embeds_->data(), embedding_size * sizeof(__half),
          cudaMemcpyDeviceToDevice, stream);
    }
  }

  // Run UNet inference - use SDXL variant if configured
  if(config_.model_type == ModelType::SDXL_TURBO)
  {
    // Tile SDXL conditioning to unet_batch_size (config.batch_size, x2 for full/initialize) — see the
    // batched path. Single-step turbo with batch_size==1 was the only previously-exercised SDXL case,
    // so this only mattered once batch_size>1 / cfg doubling landed.
    const int pooled_dim = config_.pooled_embedding_dim;
    const int time_ids_dim = config_.time_ids_dim;
    CUDATensor<__half> tiled_text_embeds((size_t)unet_batch_size * pooled_dim);
    CUDATensor<__half> tiled_time_ids((size_t)unet_batch_size * time_ids_dim);
    for(int i = 0; i < unet_batch_size; i++)
    {
      cudaMemcpyAsync(tiled_text_embeds.data() + (size_t)i * pooled_dim, text_embeds_->data(),
                      pooled_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(tiled_time_ids.data() + (size_t)i * time_ids_dim, time_ids_->data(),
                      time_ids_dim * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    }
    if(controlnet_enabled_)
    {
      // SDXL + ControlNet in the single-step path (mirrors the batched site ~386): run every net
      // with the tiled SDXL conditioning, SUM residuals, inject via forward_controlnet_sdxl. The
      // SD1.5 forward_controlnet would leave text_embeds/time_ids unset on this control-aware SDXL
      // engine -> "Not all input dimensions specified".
      const __half* down_ptrs[ControlNetWrapper::MAX_DOWN];
      const __half* mid_ptr = nullptr;
      int down_count = 0;
      run_controlnets(predict_x0_batch_unet_input_latent->data(),
                      predict_x0_batch_unet_input_timestep->data(),
                      predict_x0_batch_unet_encoder_hidden_states_single->data(),
                      tiled_text_embeds.data(), tiled_time_ids.data(), unet_batch_size,
                      config_.text_seq_len, config_.text_hidden_dim, pooled_dim,
                      down_ptrs, &mid_ptr, down_count, stream);
      unet_->forward_controlnet_sdxl(
          predict_x0_batch_unet_input_latent_fp32_single->data(),
          predict_x0_batch_unet_input_timestep->data(),
          predict_x0_batch_unet_encoder_hidden_states_single->data(),
          tiled_text_embeds.data(), tiled_time_ids.data(), down_ptrs, mid_ptr,
          predict_x0_batch_unet_output->data(), unet_batch_size, config_.latent_height,
          config_.latent_width, config_.text_seq_len, config_.text_hidden_dim,
          config_.pooled_embedding_dim, stream);
    }
    else
    {
      unet_->forward_sdxl(
          predict_x0_batch_unet_input_latent_fp32_single->data(),
          predict_x0_batch_unet_input_timestep->data(),
          predict_x0_batch_unet_encoder_hidden_states_single->data(),
          tiled_text_embeds.data(), tiled_time_ids.data(),
          predict_x0_batch_unet_output->data(), unet_batch_size, config_.latent_height,
          config_.latent_width, config_.text_seq_len, config_.text_hidden_dim,
          config_.pooled_embedding_dim, stream);
    }
  }
  else if(controlnet_enabled_)
  {
    // ControlNet (SD1.5) in the single-step path (mirrors the batched impl): run every net on the SAME
    // fp16 latent/timestep/ehs the UNet sees, SUM residuals, inject via forward_controlnet. Without this
    // branch a control-aware unet.engine hits plain forward() -> "Not all input dimensions specified".
    const __half* down_ptrs[ControlNetWrapper::MAX_DOWN];
    const __half* mid_ptr = nullptr;
    int down_count = 0;
    run_controlnets(predict_x0_batch_unet_input_latent->data(),
                    predict_x0_batch_unet_input_timestep->data(),
                    predict_x0_batch_unet_encoder_hidden_states_single->data(),
                    nullptr, nullptr, unet_batch_size, config_.text_seq_len,
                    config_.text_hidden_dim, 0, down_ptrs, &mid_ptr, down_count, stream);
    unet_->forward_controlnet(
        predict_x0_batch_unet_input_latent_fp32_single->data(),
        predict_x0_batch_unet_input_timestep->data(),
        predict_x0_batch_unet_encoder_hidden_states_single->data(),
        down_ptrs, mid_ptr,
        predict_x0_batch_unet_output->data(), unet_batch_size, config_.latent_height,
        config_.latent_width, config_.text_seq_len, config_.text_hidden_dim, stream);
  }
  else if(ipadapter_enabled_)
  {
    // IP-Adapter (SD1.5) in the single-step path (mirrors the batched impl 415-453): build the EXTENDED
    // encoder_hidden_states [unet_batch, 77+N, dim] = per row text tokens ++ N image tokens (neg/pos to
    // match the row's text conditioning), then forward_ipadapter at seq_len = 77+N. Without this branch a
    // detected IP unet (encoder_hidden_states [-1,81,768]) hits plain forward() at 77 -> shape mismatch.
    const int dim = config_.text_hidden_dim;
    const int N = ipadapter_num_tokens_;
    const int text_tok = config_.text_seq_len;
    const int ext_tok = text_tok + N;
    const int text_row = text_tok * dim, img_row = N * dim, ext_row = ext_tok * dim;
    const __half* pos = ipadapter_tokens_pos_ ? ipadapter_tokens_pos_->data() : nullptr;
    const __half* neg = ipadapter_tokens_neg_ ? ipadapter_tokens_neg_->data() : pos;
    if(!pos)
      throw std::runtime_error("IP-Adapter enabled but no image tokens set (set_ipadapter_tokens)");
    // Persistent (was a per-frame stack CUDATensor): a per-call cudaMalloc is illegal inside CUDA-graph
    // capture. Grow-on-first-use, same shape every frame for a fixed config (see ipadapter_scale_buffer_).
    const size_t ext_n = (size_t)unet_batch_size * ext_row;
    if(!ip_ext_ehs_ || ip_ext_ehs_->size() < ext_n)
    {
      ip_ext_ehs_ = std::make_unique<CUDATensor<__half>>(ext_n);
      graph_ready_ = false;  // buffer (re)allocated -> invalidate any captured graph (address moved)
    }
    __half* ext_ehs = ip_ext_ehs_->data();
    const bool full = (config_.guidance_scale > 1.0f && config_.cfg_type == 1);
    const bool init = (config_.guidance_scale > 1.0f && config_.cfg_type == 3);
    for(int r = 0; r < unet_batch_size; r++)
    {
      cudaMemcpyAsync(ext_ehs + (size_t)r * ext_row,
                      predict_x0_batch_unet_encoder_hidden_states_single->data() + (size_t)r * text_row,
                      text_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      bool is_neg = full ? (r < config_.batch_size) : (init ? (r == 0) : false);
      const __half* img = is_neg ? neg : pos;
      cudaMemcpyAsync(ext_ehs + (size_t)r * ext_row + text_row, img,
                      img_row * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    }
    int nlay = unet_->numIpLayers();
    if(nlay <= 0) nlay = (int)ipadapter_scale_vec_.size();
    unet_->forward_ipadapter(
        predict_x0_batch_unet_input_latent_fp32_single->data(),
        predict_x0_batch_unet_input_timestep->data(), ext_ehs,
        ipadapter_scale_vec_.data(), nlay,
        predict_x0_batch_unet_output->data(), unet_batch_size, config_.latent_height,
        config_.latent_width, ext_tok, config_.text_hidden_dim, stream);
  }
  else
  {
    unet_->forward(
        predict_x0_batch_unet_input_latent_fp32_single->data(),
        predict_x0_batch_unet_input_timestep->data(),
        predict_x0_batch_unet_encoder_hidden_states_single
            ->data(), // Use repeated embeddings
        predict_x0_batch_unet_output->data(), unet_batch_size, config_.latent_height,
        config_.latent_width, config_.text_seq_len, config_.text_hidden_dim, stream);
  }

  // StreamV2V: Cache and apply attention feature injection
  if(config_.mode == PipelineMode::TEMPORAL_V2V)
  {
    // Store current attention outputs in cache (every N frames)
    if((temporal_state_.frame_id % config_.cache_interval) == 0)
    {
      const auto& attn_buffers = unet_->getAttentionBuffers();

      if(!attn_buffers.empty())
      {
        // Create new cache entry
        TemporalState::AttentionCache new_cache;
        new_cache.frame_id = temporal_state_.frame_id;
        new_cache.attention_layers.reserve(16);

        // Copy all 16 attention outputs
        for(size_t i = 0; i < attn_buffers.size(); i++)
        {
          size_t attn_size = attn_buffers[i]->size();
          auto cached_attn = std::make_unique<CUDATensor<__half>>(attn_size);
          cached_attn->load_d2d(attn_buffers[i]->data(), attn_size, stream);
          new_cache.attention_layers.push_back(std::move(cached_attn));
        }

        // Add to cache deque
        temporal_state_.cached_attentions.push_back(std::move(new_cache));

        // Maintain max size
        while(temporal_state_.cached_attentions.size()
              > static_cast<size_t>(config_.cache_maxframes))
        {
          temporal_state_.cached_attentions.pop_front();
        }
      }
    }

    // Apply feature injection if we have cached attentions
    if(config_.use_feature_injection && !temporal_state_.cached_attentions.empty())
    {
      const auto& current_attns = unet_->getAttentionBuffers();
      const auto& cached_attns
          = temporal_state_.cached_attentions.back().attention_layers;

      // Apply feature injection per layer (only if buffers are ready)
      if(!current_attns.empty() && !cached_attns.empty())
      {
        for(size_t layer = 0; layer < current_attns.size() && layer < cached_attns.size();
            layer++)
        {
          // Get dimensions from tensor sizes
          size_t current_size = current_attns[layer]->size();
          size_t cached_size = cached_attns[layer]->size();

          // Skip if sizes don't match
          if(current_size != cached_size)
            continue;

          int seq_len = config_.latent_height * config_.latent_width;
          // Infer hidden_dim from total size
          int hidden_dim = current_size / (config_.batch_size * seq_len);

          int batch = config_.batch_size;
          int cached_seq_len = seq_len;

          // Allocate temp buffers
          CUDATensor<float> similarities(batch * seq_len * cached_seq_len);
          CUDATensor<__half> nn_features(current_size);

          // 1. Compute cosine similarities
          launch_cosine_similarity(
              current_attns[layer]->data(), cached_attns[layer]->data(),
              similarities.data(), batch, seq_len, cached_seq_len, hidden_dim, stream);

          // 2. Find nearest neighbors
          launch_nearest_neighbor(
              current_attns[layer]->data(), cached_attns[layer]->data(),
              similarities.data(), nn_features.data(),
              config_.feature_similarity_threshold, batch, seq_len, cached_seq_len,
              hidden_dim, stream);

          // 3. Blend current with nearest neighbors (in-place)
          launch_blend_features(
              current_attns[layer]->data(), nn_features.data(),
              current_attns[layer]->data(), config_.feature_injection_strength,
              current_size, stream);
        }
      }
    }
  }

  if(config_.guidance_scale > 1.0f && config_.cfg_type == 1)
  { // cfg_type="full"
    CUDATensor<__half> noise_pred_uncond(latent_size);
    CUDATensor<__half> noise_pred_text(latent_size);

    cudaMemcpyAsync(
        noise_pred_uncond.data(), predict_x0_batch_unet_output->data(),
        latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(
        noise_pred_text.data(), predict_x0_batch_unet_output->data() + latent_size,
        latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

    apply_cfg(
        noise_pred_uncond.data(), noise_pred_text.data(),
        predict_x0_batch_model_pred->data(), latent_size, stream);
  }
  else if(config_.guidance_scale > 1.0f && config_.cfg_type == 3)
  { // cfg_type="initialize"
    // Python: noise_pred_text = model_pred[1:]  (skip first element which is uncond)
    // Python: self.stock_noise = concat([model_pred[0:1], self.stock_noise[1:]], dim=0)
    int single_latent_size = 4 * config_.latent_height * config_.latent_width;

    // noise_pred_text = model_pred[1:] - skip the first latent (uncond prediction)
    CUDATensor<__half> noise_pred_text(latent_size);
    cudaMemcpyAsync(
        noise_pred_text.data(), predict_x0_batch_unet_output->data() + single_latent_size,
        latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

    // Update stock_noise: concat([model_pred[0:1], stock_noise[1:]], dim=0)
    // Copy model_pred[0:1] to stock_noise[0]
    cudaMemcpyAsync(
        stock_noise_->data(), predict_x0_batch_unet_output->data(),
        single_latent_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

    // noise_pred_uncond = stock_noise * delta
    CUDATensor<__half> noise_pred_uncond(latent_size);
    cudaMemcpyAsync(
        noise_pred_uncond.data(), stock_noise_->data(), latent_size * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    launch_scalar_mul_inplace_fp16(
        noise_pred_uncond.data(), config_.delta, latent_size, stream);

    // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
    apply_cfg(
        noise_pred_uncond.data(), noise_pred_text.data(),
        predict_x0_batch_model_pred->data(), latent_size, stream);
  }
  else if(config_.guidance_scale > 1.0f && config_.cfg_type == 2)
  { // cfg_type="self"
    // noise_pred_text = model_pred (no modification needed)
    CUDATensor<__half> noise_pred_text(latent_size);
    noise_pred_text.load_d2d(predict_x0_batch_unet_output->data(), latent_size, stream);

    // noise_pred_uncond = stock_noise * delta
    CUDATensor<__half> noise_pred_uncond(latent_size);
    cudaMemcpyAsync(
        noise_pred_uncond.data(), stock_noise_->data(), latent_size * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    launch_scalar_mul_inplace_fp16(
        noise_pred_uncond.data(), config_.delta, latent_size, stream);

    // Apply CFG: output = uncond + guidance_scale * (cond - uncond)
    apply_cfg(
        noise_pred_uncond.data(), noise_pred_text.data(),
        predict_x0_batch_model_pred->data(), latent_size, stream);
  }
  else
  {
    // No CFG (cfg_type="none" or guidance_scale <= 1.0)
    predict_x0_batch_model_pred->load_d2d(
        predict_x0_batch_unet_output->data(), predict_x0_batch_unet_output->size(),
        stream);
  }

  scheduler_step_batch(
      predict_x0_batch_model_pred->data(), predict_x0_batch_x_t_latent->data(),
      predict_x0_batch_denoised->data(), 0, latent_size, stream);

  predict_x0_batch_denoised->store_d2d(x_0_pred_out, latent_size, stream);
}

void LibreDiffusionPipeline::predict_x0_batch(
    const __half* x_t_latent_in, __half* x_0_pred_out, cudaStream_t stream)
{
  // Use internal stream_ if stream is 0 (default/null stream)
  if(stream == 0)
  {
    stream = stream_;
  }

  if(config_.denoising_steps > 1)
  {
    if(config_.use_denoising_batch)
    {
      predict_x0_batch_impl_multi_step_batched(x_t_latent_in, x_0_pred_out, stream);
    }
    else
    {
      predict_x0_batch_impl_multi_step_sequential(x_t_latent_in, x_0_pred_out, stream);
    }
  }
  else
  {
    predict_x0_batch_impl_single_step(x_t_latent_in, x_0_pred_out, stream);
  }
}
}
