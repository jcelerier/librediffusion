/** FLUX.3-image-klein C-API. */
#include "librediffusion.flux3.hpp"
#include "create_error.hpp"
#include "cuda_error_state.hpp"
#include "device_guard.hpp"
#include "kernels.hpp"
#include "librediffusion_c.h"
#include "qwen_tokenizer_c.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace librediffusion;

namespace
{
constexpr int kMaxText = 640;

struct DevBuf
{
  void* p{};
  size_t cap{};
  void* get(size_t bytes)
  {
    if(cap < bytes)
    {
      cudaFree(p);
      p = nullptr;
      cap = 0;
      if(cudaMalloc(&p, bytes) != cudaSuccess)
        throw std::runtime_error("flux3: cudaMalloc failed");
      cap = bytes;
    }
    return p;
  }
  // grows to bytes, keeping the first `keep` bytes
  void* grow_keep(size_t bytes, size_t keep, cudaStream_t stream)
  {
    if(cap >= bytes)
      return p;
    void* n = nullptr;
    if(cudaMalloc(&n, bytes) != cudaSuccess)
      throw std::runtime_error("flux3: cudaMalloc failed");
    if(keep)
      cudaMemcpyAsync(n, p, keep, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFree(p);
    p = n;
    cap = bytes;
    return p;
  }
  ~DevBuf() { cudaFree(p); }
};

librediffusion_error_t run(int device, const char* what, auto&& f)
{
  try
  {
    DeviceGuard guard{device};
    f();
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::invalid_argument& e)
  {
    fprintf(stderr, "flux3 %s: %s\n", what, e.what());
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  }
  catch(const std::exception& e)
  {
    librediffusion::drain_cuda_error();
    fprintf(stderr, "flux3 %s: %s\n", what, e.what());
    return LIBREDIFFUSION_ERROR_CUDA_ERROR;
  }
}
}

struct librediffusion_flux3
{
  std::unique_ptr<Flux3Pipeline> pipe;
  int device{};
  cudaStream_t stream{};

  DevBuf bn_mean, bn_std, ehs, ids, mask, cond, ref_tmp, noise, rgba_in, rgba_out, f32;
  bool have_bn{}, have_prompt{};
  std::string prompt;
  int lt{};
  std::vector<std::pair<int, int>> refs; // (th, tw) of each reference, tokens concatenated in cond
  size_t cond_tokens{};
  int noise_th{-1}, noise_tw{-1};
  unsigned long long seed{1};
  int steps{16};
  float alpha{kFlux3DefaultAlpha};
  std::vector<float> times; // explicit, terminal included; empty -> steps/alpha

  ~librediffusion_flux3()
  {
    if(stream)
      cudaStreamDestroy(stream);
  }
};

extern "C" {

librediffusion_flux3_handle librediffusion_flux3_create(
    const char* transformer_engine, const char* qwen_engine, const char* vae_decoder_engine,
    const char* vae_encoder_engine, const char* tokenizer_json, int device)
{
  const int dev = resolve_device(device);
  if(dev < 0)
  {
    set_create_error(LIBREDIFFUSION_ERROR_INVALID_ARGUMENT);
    return nullptr;
  }
  try
  {
    DeviceGuard guard{dev};
    if(!transformer_engine || !qwen_engine || !vae_decoder_engine)
      throw std::invalid_argument("transformer, qwen and vae_decoder engines are required");
    if(tokenizer_json && tokenizer_json[0] && qwen_tokenizer_load(tokenizer_json) != 0)
      throw std::runtime_error("failed to load qwen tokenizer.json");
    Flux3EnginePaths p{transformer_engine, qwen_engine, vae_decoder_engine,
                       vae_encoder_engine ? vae_encoder_engine : ""};
    auto h = std::make_unique<librediffusion_flux3>();
    h->device = dev;
    h->pipe = std::make_unique<Flux3Pipeline>(p);
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    if(cudaStreamCreateWithPriority(&h->stream, cudaStreamNonBlocking, lo) != cudaSuccess)
      cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking);
    set_create_error(LIBREDIFFUSION_SUCCESS);
    return h.release();
  }
  catch(const std::exception& e)
  {
    librediffusion::drain_cuda_error();
    fprintf(stderr, "flux3_create failed: %s\n", e.what());
    set_create_error(e);
    return nullptr;
  }
}

void librediffusion_flux3_destroy(librediffusion_flux3_handle h)
{
  if(!h)
    return;
  DeviceGuard guard{h->device};
  delete h;
}

librediffusion_error_t
librediffusion_flux3_set_bn(librediffusion_flux3_handle h, const float* bn_mean, const float* bn_std)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!bn_mean || !bn_std)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  return run(h->device, "set_bn", [&] {
    cudaMemcpy(h->bn_mean.get(128 * 4), bn_mean, 128 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(h->bn_std.get(128 * 4), bn_std, 128 * 4, cudaMemcpyHostToDevice);
    h->have_bn = true;
  });
}

int librediffusion_flux3_set_prompt(librediffusion_flux3_handle h, const char* prompt)
{
  if(!h || !prompt)
    return -1;
  if(h->have_prompt && h->prompt == prompt)
    return h->lt;
  std::vector<int64_t> ids(kMaxText), mask(kMaxText);
  const int n = qwen_tokenizer_encode_chat_nothink(prompt, kMaxText, ids.data(), mask.data());
  if(n <= 0)
    return -1;
  const int lt = (n + kFlux3TextPad - 1) / kFlux3TextPad * kFlux3TextPad;
  auto err = run(h->device, "set_prompt", [&] {
    auto* d_ids = h->ids.get(lt * 8);
    auto* d_mask = h->mask.get(lt * 8);
    cudaMemcpyAsync(d_ids, ids.data(), lt * 8, cudaMemcpyHostToDevice, h->stream);
    cudaMemcpyAsync(d_mask, mask.data(), lt * 8, cudaMemcpyHostToDevice, h->stream);
    auto* ehs = (__nv_bfloat16*)h->ehs.get((size_t)lt * kFlux3TextDim * 2);
    h->pipe->encode_text((const int64_t*)d_ids, (const int64_t*)d_mask, ehs, lt, h->stream);
    cudaStreamSynchronize(h->stream);
  });
  if(err != LIBREDIFFUSION_SUCCESS)
  {
    h->have_prompt = false;
    return -1;
  }
  h->prompt = prompt;
  h->lt = lt;
  h->have_prompt = true;
  return lt;
}

librediffusion_error_t librediffusion_flux3_encode_reference(
    librediffusion_flux3_handle h, const unsigned char* rgba, int th, int tw, void* cond_out)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!rgba || !cond_out || th <= 0 || tw <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(!h->have_bn)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  return run(h->device, "encode_reference", [&] {
    const size_t bytes = (size_t)th * 16 * tw * 16 * 4;
    auto* d = h->rgba_in.get(bytes);
    cudaMemcpyAsync(d, rgba, bytes, cudaMemcpyHostToDevice, h->stream);
    h->pipe->encode_reference(
        (const unsigned char*)d, (const float*)h->bn_mean.p, (const float*)h->bn_std.p, th, tw,
        (__nv_bfloat16*)cond_out, h->stream);
  });
}

librediffusion_error_t librediffusion_flux3_add_reference(
    librediffusion_flux3_handle h, const unsigned char* rgba, int H, int W)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!rgba)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(H <= 0 || W <= 0 || H % 16 || W % 16)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  const int th = H / 16, tw = W / 16;
  const size_t tok = (size_t)th * tw;
  if(h->cond_tokens + tok > (size_t)librediffusion_flux3_max_reference_tokens(h))
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  void* tmp = nullptr;
  try
  {
    DeviceGuard guard{h->device};
    tmp = h->ref_tmp.get(tok * 128 * 2);
  }
  catch(const std::exception&)
  {
    return LIBREDIFFUSION_ERROR_OUT_OF_MEMORY;
  }
  auto err = librediffusion_flux3_encode_reference(h, rgba, th, tw, tmp);
  if(err != LIBREDIFFUSION_SUCCESS)
    return err;
  return run(h->device, "add_reference", [&] {
    const size_t row = 128 * 2, keep = h->cond_tokens * row;
    char* cond = (char*)h->cond.grow_keep(keep + tok * row, keep, h->stream);
    cudaMemcpyAsync(cond + keep, tmp, tok * row, cudaMemcpyDeviceToDevice, h->stream);
    cudaStreamSynchronize(h->stream);
    h->refs.emplace_back(th, tw);
    h->cond_tokens += tok;
  });
}

librediffusion_error_t librediffusion_flux3_set_reference(
    librediffusion_flux3_handle h, const unsigned char* rgba, int H, int W)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  h->refs.clear();
  h->cond_tokens = 0;
  return rgba ? librediffusion_flux3_add_reference(h, rgba, H, W) : LIBREDIFFUSION_SUCCESS;
}

int librediffusion_flux3_reference_count(librediffusion_flux3_handle h)
{
  return h ? (int)h->refs.size() : 0;
}

int librediffusion_flux3_max_reference_tokens(librediffusion_flux3_handle h)
{
  return h ? h->pipe->max_reference_tokens() : 0;
}

librediffusion_error_t
librediffusion_flux3_set_steps(librediffusion_flux3_handle h, int steps, float alpha)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(steps < 1 || !(alpha > 0.f))
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  h->steps = steps;
  h->alpha = alpha;
  return LIBREDIFFUSION_SUCCESS;
}

librediffusion_error_t
librediffusion_flux3_set_times(librediffusion_flux3_handle h, const float* times, int n)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  h->times.clear();
  if(n <= 0 || !times)
    return LIBREDIFFUSION_SUCCESS;
  for(int i = 0; i < n; ++i)
    h->times.push_back(std::clamp(times[i], 0.f, 1.f));
  h->times.push_back(0.f);
  return LIBREDIFFUSION_SUCCESS;
}

void librediffusion_flux3_set_seed(librediffusion_flux3_handle h, unsigned long long seed)
{
  if(h && h->seed != seed)
  {
    h->seed = seed;
    h->noise_th = h->noise_tw = -1;
  }
}

librediffusion_error_t
librediffusion_flux3_generate(librediffusion_flux3_handle h, int H, int W, unsigned char* rgba_out)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!rgba_out)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(H <= 0 || W <= 0 || H % 16 || W % 16)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  if(!h->have_bn || !h->have_prompt)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  return run(h->device, "generate", [&] {
    const int th = H / 16, tw = W / 16;
    const size_t n = (size_t)th * tw * 128;
    auto* noise = h->noise.get(n * 2);
    if(th != h->noise_th || tw != h->noise_tw)
    {
      launch_klein_randn_bf16(noise, h->seed, (long)n, h->stream);
      h->noise_th = th;
      h->noise_tw = tw;
    }
    Flux3Pipeline::Request r;
    r.noise = (const __nv_bfloat16*)noise;
    if(!h->refs.empty())
    {
      r.cond = (const __nv_bfloat16*)h->cond.p;
      r.refs = h->refs;
    }
    r.ehs = (const __nv_bfloat16*)h->ehs.p;
    r.lt = h->lt;
    r.th = th;
    r.tw = tw;
    r.schedule = h->times.empty() ? flux3_schedule(h->steps, h->alpha) : h->times;
    r.bn_mean = (const float*)h->bn_mean.p;
    r.bn_std = (const float*)h->bn_std.p;
    const size_t bytes = (size_t)H * W * 4;
    r.rgba_out = (unsigned char*)h->rgba_out.get(bytes);
    h->pipe->denoise_decode(r, h->stream);
    cudaMemcpy(rgba_out, r.rgba_out, bytes, cudaMemcpyDeviceToHost);
  });
}

librediffusion_error_t librediffusion_flux3_encode_text(
    librediffusion_flux3_handle h, const void* input_ids, const void* attention_mask, void* ehs_out,
    int Lt)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!input_ids || !attention_mask || !ehs_out)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  return run(h->device, "encode_text", [&] {
    h->pipe->encode_text(
        (const int64_t*)input_ids, (const int64_t*)attention_mask, (__nv_bfloat16*)ehs_out, Lt,
        h->stream);
    cudaStreamSynchronize(h->stream);
  });
}

librediffusion_error_t librediffusion_flux3_denoise_decode(
    librediffusion_flux3_handle h, const void* noise, const void* cond, const int* ref_hw, int n_refs,
    const void* ehs, int Lt, int th, int tw, const float* times, int n, unsigned char* rgba_out,
    float* final_tokens, float* velocity0)
{
  if(!h)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!noise || !ehs || !times || n < 2 || !rgba_out || th <= 0 || tw <= 0
     || (cond && (!ref_hw || n_refs <= 0)))
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(!h->have_bn)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  return run(h->device, "denoise_decode", [&] {
    const size_t ntok = (size_t)th * tw * 128;
    Flux3Pipeline::Request r;
    r.noise = (const __nv_bfloat16*)noise;
    r.cond = (const __nv_bfloat16*)cond;
    if(cond)
      for(int i = 0; i < n_refs; ++i)
        r.refs.emplace_back(ref_hw[2 * i], ref_hw[2 * i + 1]);
    r.ehs = (const __nv_bfloat16*)ehs;
    r.lt = Lt;
    r.th = th;
    r.tw = tw;
    r.schedule.assign(times, times + n);
    r.bn_mean = (const float*)h->bn_mean.p;
    r.bn_std = (const float*)h->bn_std.p;
    const size_t bytes = (size_t)th * 16 * tw * 16 * 4;
    r.rgba_out = (unsigned char*)h->rgba_out.get(bytes);
    // f32 scratch: [final_tokens fp32 | velocity0 bf16 | velocity0 fp32]
    auto* scratch = (char*)h->f32.get(ntok * 4 + ntok * 2 + ntok * 4);
    r.final_tokens = final_tokens ? (float*)scratch : nullptr;
    r.velocity0 = velocity0 ? (__nv_bfloat16*)(scratch + ntok * 4) : nullptr;
    h->pipe->denoise_decode(r, h->stream);
    cudaMemcpy(rgba_out, r.rgba_out, bytes, cudaMemcpyDeviceToHost);
    if(final_tokens)
      cudaMemcpy(final_tokens, scratch, ntok * 4, cudaMemcpyDeviceToHost);
    if(velocity0)
    {
      float* v32 = (float*)(scratch + ntok * 6);
      launch_klein_bf16_to_fp32(r.velocity0, v32, (long)ntok, h->stream);
      cudaStreamSynchronize(h->stream);
      cudaMemcpy(velocity0, v32, ntok * 4, cudaMemcpyDeviceToHost);
    }
  });
}

} // extern "C"
