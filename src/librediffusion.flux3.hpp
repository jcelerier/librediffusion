/** FLUX.3-image-klein C++ pipeline (distilled flow-map checkpoint). See librediffusion.flux3.cpp. */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace librediffusion
{
class Flux3TransformerWrapper;
class Qwen3EncoderWrapper;
class KleinVAEEncoderWrapper;
class KleinVAEDecoderWrapper;

struct Flux3EnginePaths
{
  std::string transformer;
  std::string qwen;
  std::string vae_decoder;
  std::string vae_encoder; // optional (reference images)
};

constexpr int kFlux3TextDim = 20480;
constexpr int kFlux3TextPad = 80;
constexpr float kFlux3DefaultAlpha = 6.93f;

// t_i = alpha*s/(1+(alpha-1)*s), s = linspace(1,0,steps+1) in fp32 (sampling.py:timesteps).
std::vector<float> flux3_schedule(int steps, float alpha);

// (0, row*0.5, col*0.5, l) row-major over th x tw.
std::vector<float> flux3_image_ids(int th, int tw, float l);
// (0, 0, 0, pos) for pos in [0, lt).
std::vector<float> flux3_text_ids(int lt);
// ids [L,4] -> rotation tables [L,64,2,2] built in fp64 (dit.py:rope_embed).
std::vector<float> flux3_rope_table(const std::vector<float>& ids);

class Flux3Pipeline
{
public:
  explicit Flux3Pipeline(const Flux3EnginePaths& paths);
  ~Flux3Pipeline();
  Flux3Pipeline(const Flux3Pipeline&) = delete;
  Flux3Pipeline& operator=(const Flux3Pipeline&) = delete;

  // input_ids/attention_mask: device int64 [1,Lt], Lt a multiple of 80. ehs_out: device bf16 [1,Lt,20480].
  void encode_text(
      const int64_t* input_ids, const int64_t* attention_mask, __nv_bfloat16* ehs_out, int Lt,
      cudaStream_t stream);

  // ref_rgba: device uint8 [16*th x 16*tw x 4]. cond_out: device bf16 [th*tw,128].
  void encode_reference(
      const unsigned char* ref_rgba, const float* bn_mean, const float* bn_std, int th, int tw,
      __nv_bfloat16* cond_out, cudaStream_t stream);

  struct Request
  {
    const __nv_bfloat16* noise{};  // device [Li,128], Li = th*tw
    // references concatenated in order, each [th_i*tw_i,128]; RoPE l axis = -(1+i). Empty: text-to-image.
    const __nv_bfloat16* cond{};
    std::vector<std::pair<int, int>> refs; // (th_i, tw_i)
    const __nv_bfloat16* ehs{};    // device [Lt,20480]
    int lt{};
    int th{}, tw{};
    std::vector<float> schedule;   // high->low, terminal included (size steps+1)
    // schedule[0] < 1 starts from (1-t0)*start + t0*noise; start defaults to cond (same geometry only).
    const __nv_bfloat16* start{};
    const float* bn_mean{};        // device [128]
    const float* bn_std{};
    unsigned char* rgba_out{};     // device uint8 [16*th x 16*tw x 4]
    float* final_tokens{};         // optional device fp32 [Li,128]
    __nv_bfloat16* velocity0{};    // optional device bf16 [Li,128]: first-step velocity (seam check)
  };
  void denoise_decode(const Request& r, cudaStream_t stream);

  int max_reference_tokens() const;

private:
  std::unique_ptr<Flux3TransformerWrapper> transformer_;
  std::unique_ptr<Qwen3EncoderWrapper> qwen_;
  std::unique_ptr<KleinVAEEncoderWrapper> vae_enc_;
  std::unique_ptr<KleinVAEDecoderWrapper> vae_dec_;

  struct Scratch { void* p{}; size_t cap{}; };
  Scratch sc_x_, sc_xb_, sc_vel_, sc_unpacked_, sc_patched_, sc_vae_lat_, sc_img_, sc_tr_;
  Scratch sc_rope_img_, sc_rope_cond_, sc_rope_txt_;
  int rope_th_{-1}, rope_tw_{-1}, rope_lt_{-1};
  std::vector<std::pair<int, int>> rope_refs_;
  bool rope_refs_valid_{};
  void* grow_scratch(Scratch& s, size_t bytes);
  void free_scratch();
  void prepare_rope(int th, int tw, const std::vector<std::pair<int, int>>& refs, int lt, cudaStream_t stream);
};

} // namespace librediffusion
