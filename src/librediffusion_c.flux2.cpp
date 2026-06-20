/** FLUX.2-klein-4B C-API implementation (with streaming). */
#include "librediffusion.flux2.hpp"
#include "librediffusion_c.h"
#include "kernels.hpp"
#include "qwen_tokenizer_c.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace librediffusion;

struct librediffusion_flux2
{
  std::unique_ptr<Flux2Pipeline> pipe;
};

extern "C" {

librediffusion_flux2_handle librediffusion_flux2_create(
    const char* transformer_engine, const char* qwen_engine, const char* vae_decoder_engine,
    const char* vae_encoder_engine)
{
  try
  {
    Flux2EnginePaths p;
    p.transformer = transformer_engine ? transformer_engine : "";
    p.qwen = qwen_engine ? qwen_engine : "";
    p.vae_decoder = vae_decoder_engine ? vae_decoder_engine : "";
    p.vae_encoder = vae_encoder_engine ? vae_encoder_engine : "";
    auto* h = new librediffusion_flux2;
    h->pipe = std::make_unique<Flux2Pipeline>(p);
    return h;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "flux2_create failed: %s\n", e.what());
    return nullptr;
  }
}

void librediffusion_flux2_destroy(librediffusion_flux2_handle h)
{
  delete h;
}

librediffusion_error_t librediffusion_flux2_encode_text(
    librediffusion_flux2_handle h, const void* input_ids, const void* attention_mask, void* ehs_out,
    int Lt)
{
  if(!h || !h->pipe) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    h->pipe->encode_text(
        (const int64_t*)input_ids, (const int64_t*)attention_mask, (__nv_bfloat16*)ehs_out, Lt, 0);
    cudaStreamSynchronize(0);
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "flux2_encode_text failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

librediffusion_error_t librediffusion_flux2_txt2img(
    librediffusion_flux2_handle h, const void* init_noise, const void* ehs, const void* bn_mean,
    const void* bn_std, int Th, int Tw, int Lt, int num_steps,
    unsigned char* rgba_host, float* out_final_latent_host)
{
  if(!h || !h->pipe) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    const int Lp = Th * Tw;
    const int H = Th * 2 * 8, W = Tw * 2 * 8;
    cudaStream_t stream = 0;

    // build ids on device
    float *img_ids = nullptr, *txt_ids = nullptr;
    cudaMalloc(&img_ids, (long)Lp * 4 * sizeof(float));
    cudaMalloc(&txt_ids, (long)Lt * 4 * sizeof(float));
    launch_klein_img_ids(img_ids, Th, Tw, stream);
    launch_klein_txt_ids(txt_ids, Lt, stream);

    unsigned char* rgba_dev = nullptr;
    cudaMalloc(&rgba_dev, (long)H * W * 4);
    __nv_bfloat16* final_lat = nullptr;
    if(out_final_latent_host)
      cudaMalloc(&final_lat, (long)Lp * 128 * sizeof(__nv_bfloat16));

    h->pipe->denoise_decode(
        (const __nv_bfloat16*)init_noise, (const __nv_bfloat16*)ehs, img_ids, txt_ids,
        (const float*)bn_mean, (const float*)bn_std, Lp, Lt, Th, Tw, num_steps,
        rgba_dev, final_lat, stream);

    cudaMemcpy(rgba_host, rgba_dev, (long)H * W * 4, cudaMemcpyDeviceToHost);
    if(out_final_latent_host && final_lat)
    {
      // bf16 -> fp32 on device, then copy
      float* f32 = nullptr;
      cudaMalloc(&f32, (long)Lp * 128 * sizeof(float));
      launch_klein_bf16_to_fp32(final_lat, f32, (long)Lp * 128, stream);
      cudaStreamSynchronize(stream);
      cudaMemcpy(out_final_latent_host, f32, (long)Lp * 128 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(f32);
    }

    cudaFree(img_ids); cudaFree(txt_ids); cudaFree(rgba_dev);
    if(final_lat) cudaFree(final_lat);
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "flux2_txt2img failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

librediffusion_error_t librediffusion_flux2_txt2img_ref(
    librediffusion_flux2_handle h, const void* init_noise, const void* ref_tokens, const void* ehs,
    const void* ref_ids, const void* bn_mean, const void* bn_std, int Th, int Tw, int Lt,
    int num_steps, unsigned char* rgba_host, float* out_final_latent_host)
{
  if(!h || !h->pipe) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    const int Lp = Th * Tw;
    const int H = Th * 2 * 8, W = Tw * 2 * 8;
    cudaStream_t stream = 0;

    float *img_ids = nullptr, *txt_ids = nullptr;
    cudaMalloc(&img_ids, (long)Lp * 4 * sizeof(float));
    cudaMalloc(&txt_ids, (long)Lt * 4 * sizeof(float));
    launch_klein_img_ids(img_ids, Th, Tw, stream);
    launch_klein_txt_ids(txt_ids, Lt, stream);

    unsigned char* rgba_dev = nullptr;
    cudaMalloc(&rgba_dev, (long)H * W * 4);
    __nv_bfloat16* final_lat = nullptr;
    if(out_final_latent_host)
      cudaMalloc(&final_lat, (long)Lp * 128 * sizeof(__nv_bfloat16));

    h->pipe->denoise_decode_ref(
        (const __nv_bfloat16*)init_noise, (const __nv_bfloat16*)ref_tokens, (const __nv_bfloat16*)ehs,
        img_ids, (const float*)ref_ids, txt_ids, (const float*)bn_mean, (const float*)bn_std,
        Lp, Lt, Th, Tw, num_steps, rgba_dev, final_lat, stream);

    cudaMemcpy(rgba_host, rgba_dev, (long)H * W * 4, cudaMemcpyDeviceToHost);
    if(out_final_latent_host && final_lat)
    {
      float* f32 = nullptr;
      cudaMalloc(&f32, (long)Lp * 128 * sizeof(float));
      launch_klein_bf16_to_fp32(final_lat, f32, (long)Lp * 128, stream);
      cudaStreamSynchronize(stream);
      cudaMemcpy(out_final_latent_host, f32, (long)Lp * 128 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(f32);
    }
    cudaFree(img_ids); cudaFree(txt_ids); cudaFree(rgba_dev);
    if(final_lat) cudaFree(final_lat);
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "flux2_txt2img_ref failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

} // extern "C"

// ===================== streaming =====================

struct librediffusion_flux2_stream
{
  std::unique_ptr<Flux2Pipeline> pipe;
  int Th = 36, Tw = 20, Lt = 512, num_steps = 2;
  unsigned long long seed = 52ull;

  // Owned NON-default CUDA stream. Using stream 0 (the legacy default) implicitly serializes against
  // EVERY other stream device-wide, so klein diffusion would block the render thread's RIFE stream.
  // A dedicated non-blocking stream lets diffusion overlap with foreground interpolation/IO.
  cudaStream_t stream = nullptr;

  // device buffers held resident for the streaming loop
  float* bn_mean = nullptr;     // [128] fp32
  float* bn_std = nullptr;      // [128] fp32
  float* img_ids = nullptr;     // [Lp,4] fp32  (noisy-latent ids, fixed)
  float* txt_ids = nullptr;     // [Lt,4] fp32  (fixed)
  __nv_bfloat16* init_noise = nullptr;   // [Lp,128] bf16 fixed-seed pure noise
  __nv_bfloat16* ehs = nullptr;          // [Lt,7680] bf16 CACHED encoder output
  // per-frame reference scratch
  __nv_bfloat16* ref_tokens = nullptr;   // [Lp,128] bf16
  float* ref_ids = nullptr;              // [Lp,4] fp32
  unsigned char* in_rgba_dev = nullptr;  // [H*W*4]
  unsigned char* out_rgba_dev = nullptr; // [H*W*4]

  bool have_prompt = false;
  std::string cur_prompt;
  bool have_reference = false;  // ref_tokens/ref_ids are valid (set via set_reference)
};

extern "C" {

librediffusion_flux2_stream_handle librediffusion_flux2_stream_create(
    const char* transformer_engine, const char* qwen_engine, const char* vae_decoder_engine,
    const char* vae_encoder_engine, const char* tokenizer_json, int Th, int Tw,
    unsigned long long seed)
{
  try
  {
    if(!vae_encoder_engine || !vae_encoder_engine[0])
      throw std::runtime_error("stream needs a vae_encoder engine (reference path)");
    if(!tokenizer_json || qwen_tokenizer_load(tokenizer_json) != 0)
      throw std::runtime_error("failed to load qwen tokenizer.json");

    Flux2EnginePaths p;
    p.transformer = transformer_engine ? transformer_engine : "";
    p.qwen = qwen_engine ? qwen_engine : "";
    p.vae_decoder = vae_decoder_engine ? vae_decoder_engine : "";
    p.vae_encoder = vae_encoder_engine;

    auto* s = new librediffusion_flux2_stream;
    s->pipe = std::make_unique<Flux2Pipeline>(p);
    s->Th = Th; s->Tw = Tw; s->seed = seed;
    // LOW-priority non-blocking stream: background diffusion yields GPU scheduling to the render
    // thread's high-priority RIFE stream (see librediffusion.rife.cpp). Non-blocking so it never
    // serializes against other streams the way the default stream 0 does.
    {
      int lo = 0, hi = 0;
      cudaDeviceGetStreamPriorityRange(&lo, &hi); // lo = numerically-largest (lowest) priority
      if(cudaStreamCreateWithPriority(&s->stream, cudaStreamNonBlocking, lo) != cudaSuccess)
        cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
    }
    const int Lp = Th * Tw;
    const int H = Th * 2 * 8, W = Tw * 2 * 8;

    cudaMalloc(&s->bn_mean, 128 * sizeof(float));
    cudaMalloc(&s->bn_std, 128 * sizeof(float));
    cudaMalloc(&s->img_ids, (long)Lp * 4 * sizeof(float));
    cudaMalloc(&s->txt_ids, (long)s->Lt * 4 * sizeof(float));
    cudaMalloc(&s->init_noise, (long)Lp * 128 * sizeof(__nv_bfloat16));
    cudaMalloc(&s->ehs, (long)s->Lt * 7680 * sizeof(__nv_bfloat16));
    cudaMalloc(&s->ref_tokens, (long)Lp * 128 * sizeof(__nv_bfloat16));
    cudaMalloc(&s->ref_ids, (long)Lp * 4 * sizeof(float));
    cudaMalloc(&s->in_rgba_dev, (long)H * W * 4);
    cudaMalloc(&s->out_rgba_dev, (long)H * W * 4);

    // fixed ids + fixed-seed pure noise (built once; never reseeded per frame)
    launch_klein_img_ids(s->img_ids, Th, Tw, s->stream);
    launch_klein_txt_ids(s->txt_ids, s->Lt, s->stream);
    launch_klein_randn_bf16(s->init_noise, s->seed, (long)Lp * 128, s->stream);
    cudaStreamSynchronize(s->stream);
    return s;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "flux2_stream_create failed: %s\n", e.what());
    return nullptr;
  }
}

void librediffusion_flux2_stream_destroy(librediffusion_flux2_stream_handle s)
{
  if(!s) return;
  cudaFree(s->bn_mean); cudaFree(s->bn_std); cudaFree(s->img_ids); cudaFree(s->txt_ids);
  cudaFree(s->init_noise); cudaFree(s->ehs); cudaFree(s->ref_tokens); cudaFree(s->ref_ids);
  cudaFree(s->in_rgba_dev); cudaFree(s->out_rgba_dev);
  if(s->stream) cudaStreamDestroy(s->stream);
  delete s;
}

void librediffusion_flux2_stream_set_steps(librediffusion_flux2_stream_handle s, int num_steps)
{
  if(s && num_steps > 0) s->num_steps = num_steps;
}

/* Caller sets the bn buffers (model VAE batch-norm) once after create. host fp32 [128] each. */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_flux2_stream_set_bn(
    librediffusion_flux2_stream_handle s, const float* bn_mean_host, const float* bn_std_host)
{
  if(!s || !bn_mean_host || !bn_std_host) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  cudaMemcpy(s->bn_mean, bn_mean_host, 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(s->bn_std, bn_std_host, 128 * sizeof(float), cudaMemcpyHostToDevice);
  return LIBREDIFFUSION_SUCCESS;
}

int librediffusion_flux2_stream_set_prompt(librediffusion_flux2_stream_handle s, const char* prompt)
{
  if(!s || !prompt) return -1;
  if(s->have_prompt && s->cur_prompt == prompt) return 0; // unchanged -> reuse cached embeds
  try
  {
    std::vector<int64_t> ids(s->Lt), mask(s->Lt);
    int n = qwen_tokenizer_encode_chat(prompt, s->Lt, ids.data(), mask.data());
    if(n < 0) return -1;

    int64_t *d_ids = nullptr, *d_mask = nullptr;
    cudaMalloc(&d_ids, (long)s->Lt * 8);
    cudaMalloc(&d_mask, (long)s->Lt * 8);
    cudaMemcpy(d_ids, ids.data(), (long)s->Lt * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), (long)s->Lt * 8, cudaMemcpyHostToDevice);
    s->pipe->encode_text(d_ids, d_mask, s->ehs, s->Lt, s->stream);  // CACHED in s->ehs
    cudaStreamSynchronize(s->stream);
    cudaFree(d_ids); cudaFree(d_mask);

    s->cur_prompt = prompt;
    s->have_prompt = true;
    return n;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "stream_set_prompt failed: %s\n", e.what());
    return -1;
  }
}

/* VAE-encode a reference frame -> cached ref_tokens/ref_ids. Call ONLY when the reference
 * image changed (the node hashes it); then run frames via flux2_stream_frame_cached, which
 * skips this (expensive) VAE-encoder pass. */
librediffusion_error_t librediffusion_flux2_stream_set_reference(
    librediffusion_flux2_stream_handle s, const unsigned char* input_rgba)
{
  if(!s || !input_rgba) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    const int Th = s->Th, Tw = s->Tw;
    const int H = Th * 2 * 8, W = Tw * 2 * 8;
    cudaStream_t stream = s->stream;
    cudaMemcpyAsync(s->in_rgba_dev, input_rgba, (long)H * W * 4, cudaMemcpyHostToDevice, stream);
    // VAE-encode reference frame -> reference tokens + reference ids (T offset = 10 for image 0)
    s->pipe->encode_reference(
        s->in_rgba_dev, s->bn_mean, s->bn_std, Th, Tw, 10.0f, s->ref_tokens, s->ref_ids, stream);
    cudaStreamSynchronize(stream);
    s->have_reference = true;
    static int enc_count = 0;
    if((++enc_count % 64) == 1)
      fprintf(stderr, "flux2: encode_reference (count=%d)\n", enc_count);
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "stream_set_reference failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

/* Run one frame using the CACHED reference (set via set_reference). No VAE-encode here. */
librediffusion_error_t librediffusion_flux2_stream_frame_cached(
    librediffusion_flux2_stream_handle s, unsigned char* output_rgba)
{
  if(!s || !output_rgba) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(!s->have_prompt || !s->have_reference) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    const int Th = s->Th, Tw = s->Tw, Lp = Th * Tw;
    const int H = Th * 2 * 8, W = Tw * 2 * 8;
    cudaStream_t stream = s->stream;
    // denoise from fixed-seed pure noise + cached encoder embeds + CACHED reference tokens
    s->pipe->denoise_decode_ref(
        s->init_noise, s->ref_tokens, s->ehs, s->img_ids, s->ref_ids, s->txt_ids,
        s->bn_mean, s->bn_std, Lp, s->Lt, Th, Tw, s->num_steps, s->out_rgba_dev, nullptr, stream);
    // out_rgba_dev is produced on s->stream (non-blocking). The D->H copy must wait for it: async
    // on the SAME stream + sync that stream (a default-stream cudaMemcpy would NOT order against a
    // non-blocking stream -> stale read).
    cudaMemcpyAsync(output_rgba, s->out_rgba_dev, (long)H * W * 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "stream_frame_cached failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

librediffusion_error_t librediffusion_flux2_stream_frame(
    librediffusion_flux2_stream_handle s, const unsigned char* input_rgba,
    unsigned char* output_rgba)
{
  if(!s || !input_rgba || !output_rgba) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(!s->have_prompt) return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    const int Th = s->Th, Tw = s->Tw, Lp = Th * Tw;
    const int H = Th * 2 * 8, W = Tw * 2 * 8;
    cudaStream_t stream = s->stream;

    cudaMemcpyAsync(s->in_rgba_dev, input_rgba, (long)H * W * 4, cudaMemcpyHostToDevice, stream);

    // VAE-encode reference frame -> reference tokens + reference ids (T offset = 10 for image 0)
    s->pipe->encode_reference(
        s->in_rgba_dev, s->bn_mean, s->bn_std, Th, Tw, 10.0f, s->ref_tokens, s->ref_ids, stream);

    // denoise from fixed-seed pure noise + cached encoder embeds + reference tokens
    s->pipe->denoise_decode_ref(
        s->init_noise, s->ref_tokens, s->ehs, s->img_ids, s->ref_ids, s->txt_ids,
        s->bn_mean, s->bn_std, Lp, s->Lt, Th, Tw, s->num_steps, s->out_rgba_dev, nullptr, stream);

    // D->H copy must wait for the non-blocking stream (see frame_cached note).
    cudaMemcpyAsync(output_rgba, s->out_rgba_dev, (long)H * W * 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "stream_frame failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

void librediffusion_flux2_stream_dims(librediffusion_flux2_stream_handle s, int* H, int* W)
{
  if(!s) return;
  if(H) *H = s->Th * 2 * 8;
  if(W) *W = s->Tw * 2 * 8;
}

} // extern "C"
