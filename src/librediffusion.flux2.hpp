/** FLUX.2-klein-4B C++ pipeline. See librediffusion.flux2.cpp. */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace librediffusion
{
class Flux2TransformerWrapper;
class Qwen3EncoderWrapper;
class KleinVAEEncoderWrapper;
class KleinVAEDecoderWrapper;

struct Flux2EnginePaths
{
  std::string transformer;
  std::string qwen;
  std::string vae_decoder;
  std::string vae_encoder; // optional (reference image)
};

// FlowMatchEuler dynamic exponential-shift sigma schedule (length num_steps+1, terminal 0).
std::vector<float> klein_sigmas(int num_steps, int image_seq_len);

class Flux2Pipeline
{
public:
  explicit Flux2Pipeline(const Flux2EnginePaths& paths);
  ~Flux2Pipeline();
  Flux2Pipeline(const Flux2Pipeline&) = delete;
  Flux2Pipeline& operator=(const Flux2Pipeline&) = delete;

  void encode_text(
      const int64_t* input_ids, const int64_t* attention_mask, __nv_bfloat16* ehs_out, int Lt,
      cudaStream_t stream);

  void denoise_decode(
      const __nv_bfloat16* init_noise, const __nv_bfloat16* ehs, const float* img_ids,
      const float* txt_ids, const float* bn_mean, const float* bn_std,
      int Lp, int Lt, int Th, int Tw, int num_steps,
      unsigned char* rgba_out, __nv_bfloat16* out_final_latent, cudaStream_t stream);

  // ---- streaming (instruct-from-noise with a reference image) ----

  // VAE-encode a reference frame to packed reference tokens [1, Lp, 128] (bf16) + build its
  // reference RoPE ids [Lp,4] (fp32, T=t_offset). ref_rgba: DEVICE uint8 RGBA [H*W*4].
  // ref_tokens_out / ref_ids_out are caller-allocated DEVICE buffers (Lp*128 bf16 / Lp*4 fp32).
  void encode_reference(
      const unsigned char* ref_rgba, const float* bn_mean, const float* bn_std,
      int Th, int Tw, float t_offset, __nv_bfloat16* ref_tokens_out, float* ref_ids_out,
      cudaStream_t stream);

  // Full denoise+decode with reference tokens concatenated to the noisy-latent sequence.
  // init_noise: [1,Lp,128] pure noise. ref_tokens: [1,Lp,128] (from encode_reference).
  // img_ids/ref_ids: [Lp,4] each (noisy-latent ids + reference ids). The transformer sees the
  // concatenated image sequence [noisy | ref] (2*Lp tokens); only the first Lp velocity rows are
  // used for the Euler step (pipeline.py:1018). Output Lp tokens decoded to rgba_out.
  void denoise_decode_ref(
      const __nv_bfloat16* init_noise, const __nv_bfloat16* ref_tokens, const __nv_bfloat16* ehs,
      const float* img_ids, const float* ref_ids, const float* txt_ids, const float* bn_mean,
      const float* bn_std, int Lp, int Lt, int Th, int Tw, int num_steps,
      unsigned char* rgba_out, __nv_bfloat16* out_final_latent, cudaStream_t stream);

  Flux2TransformerWrapper* transformer() { return transformer_.get(); }
  KleinVAEEncoderWrapper* vae_encoder() { return vae_enc_.get(); }

private:
  std::unique_ptr<Flux2TransformerWrapper> transformer_;
  std::unique_ptr<Qwen3EncoderWrapper> qwen_;
  std::unique_ptr<KleinVAEEncoderWrapper> vae_enc_;
  std::unique_ptr<KleinVAEDecoderWrapper> vae_dec_;
};

} // namespace librediffusion
