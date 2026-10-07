/** SANA-Streaming V2V C-API implementation (in-tree DiT engine).
 *
 * Exposes the validated GDN + DiT forward through the librediffusion DLL.
 * create/dit_forward/free/selftest are implemented; run_v2v/encode_prompt are
 * declared as the documented surface and return NOT_INITIALIZED until the TRT
 * VAE/gemma + streaming-scheduler wiring is landed (deferred — see report).
 */
#include "librediffusion_c.h"

#include "librediffusion.sana.hpp"

#include <cstdio>
#include <memory>
#include <vector>

using namespace librediffusion;

struct librediffusion_sana
{
  std::unique_ptr<SanaDiT> dit;
};

extern "C" {

librediffusion_sana_handle librediffusion_sana_create(const char* weights_dir)
{
  try
  {
    // Construct the DiT first: if its ctor throws (missing weights / alloc failure),
    // no handle has been allocated yet, so nothing leaks.
    auto dit = std::make_unique<SanaDiT>(weights_dir ? weights_dir : "");
    auto* h = new librediffusion_sana;
    h->dit = std::move(dit);
    return h;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_create failed: %s\n", e.what());
    return nullptr;
  }
}

void librediffusion_sana_free(librediffusion_sana_handle h)
{
  delete h;
}

/* Full-model self-test: runs x_embedder -> 20 blocks -> final on the resident
 * golden input and reports rel-L2 vs gold_model_out.bin (the no-regression gate). */
librediffusion_error_t
librediffusion_sana_selftest(librediffusion_sana_handle h, float* out_rel_l2)
{
  if(!h || !h->dit)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  try
  {
    std::vector<float> out;
    float rl2 = -1.f;
    if(h->dit->full_forward(out, rl2) != 0)
      return LIBREDIFFUSION_ERROR_INTERNAL;
    if(out_rel_l2)
      *out_rel_l2 = rl2;
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_selftest failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

/* One DiT denoise forward: x_embed (device, N*C fp32, post x_embedder+pos)
 * -> noise_out (device, N*128 fp32). */
librediffusion_error_t librediffusion_sana_dit_forward(
    librediffusion_sana_handle h, const float* x_embed_dev, float* noise_out_dev)
{
  if(!h || !h->dit || !x_embed_dev || !noise_out_dev)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    return h->dit->dit_forward(x_embed_dev, noise_out_dev) == 0
               ? LIBREDIFFUSION_SUCCESS
               : LIBREDIFFUSION_ERROR_INTERNAL;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_dit_forward failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

/* Set the TRT engine dir (vae/*.plan, gemma/*.plan) + the v2v data dir
 * (latents_mean/std.bin, init_noise.bin). Required before encode_prompt/run_v2v. */
librediffusion_error_t librediffusion_sana_set_engines(
    librediffusion_sana_handle h, const char* trt_dir, const char* data_dir)
{
  if(!h || !h->dit || !trt_dir || !data_dir)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return h->dit->set_trt(trt_dir, data_dir) == 0 ? LIBREDIFFUSION_SUCCESS
                                                 : LIBREDIFFUSION_ERROR_INTERNAL;
}

/* Tokenized prompt (int64 ids+attention_mask, len 300) -> gemma -> stored embeds.
 * Tokenization is the caller's step (the gemma tokenizer is not in-tree). */
librediffusion_error_t librediffusion_sana_encode_prompt_ids(
    librediffusion_sana_handle h, const long long* ids, const long long* mask)
{
  if(!h || !h->dit || !ids || !mask)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  try
  {
    return h->dit->encode_prompt_ids(ids, mask) == 0 ? LIBREDIFFUSION_SUCCESS
                                                     : LIBREDIFFUSION_ERROR_INTERNAL;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_encode_prompt_ids failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

/* Directly set (300,2304) gemma text embeds (host) — reproduces a reference prompt exactly. */
librediffusion_error_t
librediffusion_sana_set_text_embeds(librediffusion_sana_handle h, const float* embeds)
{
  if(!h || !h->dit || !embeds)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  return h->dit->set_text_embeds(embeds) == 0 ? LIBREDIFFUSION_SUCCESS
                                              : LIBREDIFFUSION_ERROR_INTERNAL;
}

librediffusion_error_t
librediffusion_sana_encode_prompt(librediffusion_sana_handle h, const char* /*text*/)
{
  // Text tokenization isn't in-tree; use librediffusion_sana_encode_prompt_ids with
  // caller-provided gemma token ids.
  return h ? LIBREDIFFUSION_ERROR_NOT_INITIALIZED : LIBREDIFFUSION_ERROR_NULL_POINTER;
}

librediffusion_error_t librediffusion_sana_rollout_selftest(
    librediffusion_sana_handle h, const char* data_dir, float* out_rel_l2)
{
  if(!h || !h->dit || !data_dir)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  try
  {
    float rl2 = -1.f;
    if(h->dit->run_rollout(data_dir, rl2) != 0)
      return LIBREDIFFUSION_ERROR_INTERNAL;
    if(out_rel_l2)
      *out_rel_l2 = rl2;
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_rollout_selftest failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

librediffusion_error_t librediffusion_sana_rollout_selftest_sc(
    librediffusion_sana_handle h, const char* data_dir, float* out_rel_l2)
{
  if(!h || !h->dit || !data_dir)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  try
  {
    float rl2 = -1.f;
    if(h->dit->run_rollout_sc(data_dir, rl2) != 0)
      return LIBREDIFFUSION_ERROR_INTERNAL;
    if(out_rel_l2)
      *out_rel_l2 = rl2;
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_rollout_selftest_sc failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

/* Full V2V: in_frames RGB8 (121*480*832*3) -> out_frames RGB8 (same). Pipeline:
 * preprocess -> VAE-encode -> self-contained 5-chunk DiT rollout -> VAE-decode.
 * Requires set_engines + encode_prompt_ids first. */
librediffusion_error_t librediffusion_sana_run_v2v(
    librediffusion_sana_handle h, const unsigned char* in_frames, int n_in,
    unsigned char* out_frames, int* n_out)
{
  if(!h || !h->dit || !in_frames || !out_frames || !n_out)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  try
  {
    return h->dit->run_v2v(in_frames, n_in, out_frames, n_out) == 0
               ? LIBREDIFFUSION_SUCCESS
               : LIBREDIFFUSION_ERROR_INTERNAL;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "sana_run_v2v failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

} // extern "C"
