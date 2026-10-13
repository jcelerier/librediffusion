/** SANA-Streaming TRT engine wrapper — runs the LTX-2 VAE (bf16) encoder/decoder
 *  and the gemma-2 text encoder in-process via the TensorRT C++ runtime, on device
 *  buffers. Used by librediffusion_sana_run_v2v (VAE-encode -> DiT rollout ->
 *  VAE-decode) and encode_prompt (gemma). Plain C++ interface (no CUDA/TRT types
 *  leaked). Engines are loaded from `<trt_dir>/vae/*.plan` and `<trt_dir>/gemma/*.plan`.
 */
#pragma once

#include <string>

namespace librediffusion
{

class SanaTRT
{
public:
  /** trt_dir contains vae/ltx2_vae_{encoder,decoder}_bf16.plan and gemma/gemma_encoder.plan. */
  explicit SanaTRT(const std::string& trt_dir);
  ~SanaTRT();

  /** Deserialize the run_v2v VAE engines (encoder-full + decoder-16f) now and keep them
   *  resident, so the first run_v2v call does not pay the ~2.4 GB deserialize cost. */
  int warm_vae();

  SanaTRT(const SanaTRT&) = delete;
  SanaTRT& operator=(const SanaTRT&) = delete;

  /** pixels_dev: fp32 (1,3,25,480,832) device -> moments_dev: fp32 (1,256,4,15,26) device. */
  int vae_encode(const float* pixels_dev, float* moments_dev);
  /** latent_dev: fp32 (1,128,3,15,26) device -> frames_dev: fp32 (1,3,17,480,832) device. */
  int vae_decode(const float* latent_dev, float* frames_dev);
  /** full-clip pixels (1,3,121,480,832) fp32 -> moments (1,256,16,15,26) fp32. */
  int vae_encode_full(const float* pixels_dev, float* moments_dev);
  /** full-clip latent (1,128,16,15,26) fp32 -> frames (1,3,121,480,832) fp32. */
  int vae_decode_16f(const float* latent_dev, float* frames_dev);
  /** ids/mask: int64 (1,300) device -> hidden_dev: fp32 (1,300,2304) device. Loads then frees the
   *  ~10.5 GB gemma engine around the call to bound VRAM. */
  int gemma_encode(
      const long long* ids_dev, const long long* mask_dev, float* hidden_dev);

private:
  struct Impl;
  Impl* p_;
};

} // namespace librediffusion
