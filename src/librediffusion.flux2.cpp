/**
 * FLUX.2-klein-4B C++ pipeline: instruct-from-noise single-image txt2img.
 *
 * Flow (verified vs diffusers Flux2KleinPipeline, SHA f3d42be):
 *   tokenize(prompt) [streaming; the base path takes golden input_ids] -> Qwen3 (3 layers stacked -> 7680)
 *   -> build 4-col img_ids / txt_ids -> pack pure-noise latent [B,Lp,128]
 *   -> for step in [0..N): transformer(velocity) ; x += (sigma_{i+1}-sigma_i)*velocity   (Euler)
 *   -> unpack -> bn denormalize -> unpatchify -> VAE decode -> RGBA.
 *
 * No CFG (guidance=1.0, distilled). The base path uses no reference image (pure noise); the
 * reference-token path (concat image tokens to the seq) is designed-compatible (extend Lp + concat ids).
 */
#include "librediffusion.flux2.hpp"

#include "kernels.hpp"
#include "tensorrt_wrappers_klein.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace librediffusion
{

static float compute_empirical_mu(int image_seq_len, int num_steps)
{
  const double a1 = 8.73809524e-05, b1 = 1.89833333;
  const double a2 = 0.00016927, b2 = 0.45666666;
  double L = image_seq_len;
  if(L > 4300.0)
    return (float)(a2 * L + b2);
  double m200 = a2 * L + b2;
  double m10 = a1 * L + b1;
  double a = (m200 - m10) / 190.0;
  double b = m200 - 200.0 * a;
  return (float)(a * num_steps + b);
}

// FlowMatchEuler dynamic exponential-shift sigmas, length N+1 (terminal 0).
std::vector<float> klein_sigmas(int num_steps, int image_seq_len)
{
  float mu = compute_empirical_mu(image_seq_len, num_steps);
  double emu = std::exp((double)mu);
  std::vector<float> sig(num_steps + 1);
  for(int i = 0; i < num_steps; ++i)
  {
    // linspace(1, 1/N, N)
    double s = 1.0 + (double)i * ((1.0 / num_steps) - 1.0) / (double)(num_steps - 1 > 0 ? num_steps - 1 : 1);
    if(num_steps == 1)
      s = 1.0;
    double shifted = emu / (emu + (1.0 / s - 1.0));
    sig[i] = (float)shifted;
  }
  sig[num_steps] = 0.0f;
  return sig;
}

Flux2Pipeline::Flux2Pipeline(const Flux2EnginePaths& paths)
{
  transformer_ = std::make_unique<Flux2TransformerWrapper>(paths.transformer);
  qwen_ = std::make_unique<Qwen3EncoderWrapper>(paths.qwen);
  vae_dec_ = std::make_unique<KleinVAEDecoderWrapper>(paths.vae_decoder);
  if(!paths.vae_encoder.empty())
    vae_enc_ = std::make_unique<KleinVAEEncoderWrapper>(paths.vae_encoder);
}

Flux2Pipeline::~Flux2Pipeline() { free_scratch(); }

// Grow-only device scratch: reuse the existing buffer when it is already large enough, otherwise
// (re)allocate. Reallocation moves the address, so any consumer that bakes the pointer (none here —
// these are per-call working buffers on a per-call stream) would need to react; for klein the buffers
// are simply reused each keyframe at a fixed geometry, so steady state does zero cudaMalloc.
void* Flux2Pipeline::grow_scratch(Scratch& s, size_t bytes)
{
  if(s.cap < bytes)
  {
    if(s.p) cudaFree(s.p);
    s.p = nullptr;
    cudaError_t e = cudaMalloc(&s.p, bytes);
    if(e != cudaSuccess)
    {
      s.cap = 0;
      throw std::runtime_error(
          std::string("Flux2Pipeline: scratch cudaMalloc failed (") + cudaGetErrorString(e) + ")");
    }
    s.cap = bytes;
  }
  return s.p;
}

void Flux2Pipeline::free_scratch()
{
  for(Scratch* s : {&sc_x_, &sc_vel_, &sc_full_in_, &sc_full_ids_, &sc_unpacked_, &sc_patched_,
                    &sc_vae_lat_, &sc_img_, &sc_ts_})
  {
    if(s->p) cudaFree(s->p);
    s->p = nullptr;
    s->cap = 0;
  }
}

// Run encoder seam only: input_ids/mask (device int64 [1,Lt]) -> encoder_hidden_states (device bf16 [1,Lt,7680]).
void Flux2Pipeline::encode_text(
    const int64_t* input_ids, const int64_t* attention_mask, __nv_bfloat16* ehs_out, int Lt,
    cudaStream_t stream)
{
  qwen_->forward(input_ids, attention_mask, ehs_out, 1, Lt, 7680, stream);
}

// Full denoise+decode. init_noise: device bf16 packed [1,Lp,128] (pure noise). ehs device bf16 [1,Lt,7680].
// bn_mean/bn_std: device fp32 [128]. img_ids/txt_ids: device fp32. Writes rgba_out (device uint8 [H*W*4]).
// out_final_latent (optional, device bf16 [1,Lp,128]) captures the post-Euler latent for seam validation.
void Flux2Pipeline::denoise_decode(
    const __nv_bfloat16* init_noise, const __nv_bfloat16* ehs, const float* img_ids,
    const float* txt_ids, const float* bn_mean, const float* bn_std,
    int Lp, int Lt, int Th, int Tw, int num_steps,
    unsigned char* rgba_out, __nv_bfloat16* out_final_latent, cudaStream_t stream)
{
  const int B = 1, C = 32;
  const long packed_n = (long)Lp * 128;
  const int H = Th * 2 * 8, W = Tw * 2 * 8;
  // persistent grow-only working buffers (reused across keyframes; freed in the destructor)
  __nv_bfloat16* x = (__nv_bfloat16*)grow_scratch(sc_x_, packed_n * sizeof(__nv_bfloat16));
  __nv_bfloat16* vel = (__nv_bfloat16*)grow_scratch(sc_vel_, packed_n * sizeof(__nv_bfloat16));
  __nv_bfloat16* unpacked = (__nv_bfloat16*)grow_scratch(sc_unpacked_, (long)128 * Th * Tw * sizeof(__nv_bfloat16));
  __nv_bfloat16* vae_lat = (__nv_bfloat16*)grow_scratch(sc_vae_lat_, (long)C * (Th * 2) * (Tw * 2) * sizeof(__nv_bfloat16));
  __nv_bfloat16* img = (__nv_bfloat16*)grow_scratch(sc_img_, (long)3 * H * W * sizeof(__nv_bfloat16));
  float* ts_dev = (float*)grow_scratch(sc_ts_, sizeof(float));

  cudaMemcpyAsync(x, init_noise, packed_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

  auto sig = klein_sigmas(num_steps, Lp);
  for(int step = 0; step < num_steps; ++step)
  {
    float ts_host = sig[step];
    cudaMemcpyAsync(ts_dev, &ts_host, sizeof(float), cudaMemcpyHostToDevice, stream);
    transformer_->forward(x, ehs, ts_dev, img_ids, txt_ids, vel, B, Lp, Lt, 7680, stream);
    float dt = sig[step + 1] - sig[step];
    launch_klein_euler_axpy(x, vel, dt, packed_n, stream);
  }
  if(out_final_latent)
    cudaMemcpyAsync(out_final_latent, x, packed_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

  // unpack [1,Lp,128] -> [1,128,Th,Tw]
  launch_klein_unpack(x, unpacked, B, 128, Th, Tw, stream);
  // bn denormalize (denorm=1)
  launch_klein_bn(unpacked, unpacked, bn_mean, bn_std, B, 128, Th * Tw, 1, stream);
  // unpatchify [1,128,Th,Tw] -> [1,32,Th*2,Tw*2]
  launch_klein_unpatchify(unpacked, vae_lat, B, 32, Th, Tw, stream);
  // decode
  vae_dec_->decode(vae_lat, img, B, Th * 2, Tw * 2, stream);
  // to rgba uint8
  launch_klein_chw_to_rgba_u8(img, rgba_out, H, W, stream);

  cudaStreamSynchronize(stream);
  // working buffers are persistent (grow-only members) -> no per-call free
}

// ===== streaming with a reference image ============================================

// VAE-encode a reference frame -> packed reference tokens + reference RoPE ids.
// Mirrors pipeline._encode_vae_image: VAE.encode (argmax/mode) -> patchify -> bn-normalize -> pack.
void Flux2Pipeline::encode_reference(
    const unsigned char* ref_rgba, const float* bn_mean, const float* bn_std,
    int Th, int Tw, float t_offset, __nv_bfloat16* ref_tokens_out, float* ref_ids_out,
    cudaStream_t stream)
{
  if(!vae_enc_)
    throw std::runtime_error("klein: reference path needs a vae_encoder engine");
  const int B = 1, C = 32;
  const int H = Th * 2 * 8, W = Tw * 2 * 8;   // reference frame resolution == output resolution
  const int Lh = Th * 2, Lw = Tw * 2;         // VAE latent spatial dims

  // persistent grow-only working buffers (reused; freed in the destructor)
  __nv_bfloat16* img = (__nv_bfloat16*)grow_scratch(sc_img_, (long)3 * H * W * sizeof(__nv_bfloat16));
  __nv_bfloat16* vae_lat = (__nv_bfloat16*)grow_scratch(sc_vae_lat_, (long)C * Lh * Lw * sizeof(__nv_bfloat16));
  __nv_bfloat16* patched = (__nv_bfloat16*)grow_scratch(sc_patched_, (long)128 * Th * Tw * sizeof(__nv_bfloat16));

  // RGBA uint8 -> CHW bf16 [-1,1]
  launch_klein_rgba_u8_to_chw_m1p1(ref_rgba, img, H, W, stream);
  // VAE encode -> [1,32,Lh,Lw]
  vae_enc_->encode(img, vae_lat, B, H, W, stream);
  // patchify [1,32,Lh,Lw] -> [1,128,Th,Tw]
  launch_klein_patchify(vae_lat, patched, B, C, Lh, Lw, stream);
  // bn normalize (denorm=0)
  launch_klein_bn(patched, patched, bn_mean, bn_std, B, 128, Th * Tw, 0, stream);
  // pack [1,128,Th,Tw] -> [1,Th*Tw,128]
  launch_klein_pack(patched, ref_tokens_out, B, 128, Th, Tw, stream);
  // reference ids (T=t_offset)
  launch_klein_ref_ids(ref_ids_out, Th, Tw, t_offset, stream);
  cudaStreamSynchronize(stream);  // working buffers are persistent members; ensure reuse-safety before return
}

void Flux2Pipeline::denoise_decode_ref(
    const __nv_bfloat16* init_noise, const __nv_bfloat16* ref_tokens, const __nv_bfloat16* ehs,
    const float* img_ids, const float* ref_ids, const float* txt_ids, const float* bn_mean,
    const float* bn_std, int Lp, int Lt, int Th, int Tw, int num_steps,
    unsigned char* rgba_out, __nv_bfloat16* out_final_latent, cudaStream_t stream,
    float strength, const float* sigmas, int n_sigmas, const float* inpaint_mask)
{
  const int B = 1, C = 32;
  const long packed_n = (long)Lp * 128;       // noisy-latent token count
  const int Limg = 2 * Lp;                     // concatenated image tokens (noisy | ref)
  const long full_n = (long)Limg * 128;

  const int H = Th * 2 * 8, W = Tw * 2 * 8;
  // persistent grow-only working buffers (reused across keyframes; freed in the destructor)
  __nv_bfloat16* x = (__nv_bfloat16*)grow_scratch(sc_x_, packed_n * sizeof(__nv_bfloat16));
  __nv_bfloat16* full_in = (__nv_bfloat16*)grow_scratch(sc_full_in_, full_n * sizeof(__nv_bfloat16));
  __nv_bfloat16* vel = (__nv_bfloat16*)grow_scratch(sc_vel_, full_n * sizeof(__nv_bfloat16));  // velocity for all Limg tokens
  float* full_ids = (float*)grow_scratch(sc_full_ids_, (long)Limg * 4 * sizeof(float));
  __nv_bfloat16* unpacked = (__nv_bfloat16*)grow_scratch(sc_unpacked_, (long)128 * Th * Tw * sizeof(__nv_bfloat16));
  __nv_bfloat16* vae_lat = (__nv_bfloat16*)grow_scratch(sc_vae_lat_, (long)C * (Th * 2) * (Tw * 2) * sizeof(__nv_bfloat16));
  __nv_bfloat16* img = (__nv_bfloat16*)grow_scratch(sc_img_, (long)3 * H * W * sizeof(__nv_bfloat16));
  float* ts_dev = (float*)grow_scratch(sc_ts_, sizeof(float));

  // concatenate ids ONCE (fixed across steps): [img_ids | ref_ids]. ids are fp32 [Lp,4], so this
  // is a plain device-to-device copy of each half (NOT the bf16 concat kernel).
  cudaMemcpyAsync(full_ids, img_ids, (long)Lp * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(full_ids + (long)Lp * 4, ref_ids, (long)Lp * 4 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);

  // Build the sigma schedule (high->low, terminal 0). Either the caller's explicit native-scale list
  // (sigmas[0..n_sigmas-1]) or the dynamic-shift default truncated by `strength`.
  std::vector<float> sig;
  int nsteps;
  if(sigmas && n_sigmas > 0)
  {
    nsteps = n_sigmas;
    sig.assign(sigmas, sigmas + n_sigmas);
    sig.push_back(0.0f);
  }
  else
  {
    nsteps = num_steps;
    sig = klein_sigmas(num_steps, Lp);
    if(strength < 1.0f && sig[0] > 1e-6f)
    {
      const float scale = strength / sig[0];  // map [sig0..0] -> [strength..0]
      for(auto& v : sig) v *= scale;
    }
  }
  // sig[0] is the start noise level (=img2img strength). <1 -> start from a noised reference
  // x_s=(1-s)*ref + s*noise (smaller s preserves more of the reference); ==1 -> pure-noise edit.
  const float start = sig[0];
  if(start < 1.0f)
    launch_klein_lerp(x, ref_tokens, init_noise, start, packed_n, stream);
  else
    cudaMemcpyAsync(x, init_noise, packed_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

  for(int step = 0; step < nsteps; ++step)
  {
    // rebuild concatenated hidden states each step (noisy x changes; ref tokens fixed).
    launch_klein_concat_seq(x, ref_tokens, full_in, packed_n, packed_n, stream);

    float ts_host = sig[step];
    cudaMemcpyAsync(ts_dev, &ts_host, sizeof(float), cudaMemcpyHostToDevice, stream);
    transformer_->forward(full_in, ehs, ts_dev, full_ids, txt_ids, vel, B, Limg, Lt, 7680, stream);

    // Euler step uses ONLY the first Lp velocity rows (pipeline.py:1018 noise_pred[:, :Lp, :]).
    float dt = sig[step + 1] - sig[step];
    launch_klein_euler_axpy(x, vel, dt, packed_n, stream);

    // Inpaint: keep the UNMASKED region on the reference's noise trajectory, regenerate only where the
    // mask is 1. x = mask*x + (1-mask)*scale_noise(ref, sigma_next) (Flux2KleinInpaintPipeline:1209-1220).
    if(inpaint_mask)
      launch_klein_inpaint_blend(x, ref_tokens, init_noise, inpaint_mask, sig[step + 1], Lp, 128, stream);
  }
  if(out_final_latent)
    cudaMemcpyAsync(out_final_latent, x, packed_n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

  launch_klein_unpack(x, unpacked, B, 128, Th, Tw, stream);
  launch_klein_bn(unpacked, unpacked, bn_mean, bn_std, B, 128, Th * Tw, 1, stream);
  launch_klein_unpatchify(unpacked, vae_lat, B, 32, Th, Tw, stream);
  vae_dec_->decode(vae_lat, img, B, Th * 2, Tw * 2, stream);
  launch_klein_chw_to_rgba_u8(img, rgba_out, H, W, stream);

  cudaStreamSynchronize(stream);
  // working buffers are persistent (grow-only members) -> no per-call free
}

} // namespace librediffusion
