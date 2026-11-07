/**
 * FLUX.3-image-klein: tokens are 128-ch 2x2-patchified latents of the FLUX.2 autoencoder, so the
 * encode/decode seams are klein.2's. The DiT takes image, reference (image_cond) and text tokens;
 * sampling is a flow map: v = DiT(x, t -> r), x += (r - t) * v, with x kept in fp32.
 */
#include "librediffusion.flux3.hpp"

#include "kernels.hpp"
#include "tensorrt_wrappers_klein.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace librediffusion
{

std::vector<float> flux3_schedule(int steps, float alpha)
{
  if(steps < 1)
    throw std::invalid_argument("flux3: steps must be >= 1");
  // torch.linspace(1, 0, n): start + i*step for the first half, end - (n-1-i)*step for the rest.
  const int n = steps + 1;
  const float step = (0.f - 1.f) / (float)steps;
  std::vector<float> t(n);
  for(int i = 0; i < n; ++i)
  {
    const float s = i < n / 2 ? 1.f + step * (float)i : 0.f - step * (float)(n - 1 - i);
    t[i] = alpha * s / (1.f + (alpha - 1.f) * s);
  }
  return t;
}

std::vector<float> flux3_image_ids(int th, int tw, float l)
{
  std::vector<float> ids((size_t)th * tw * 4);
  for(int y = 0; y < th; ++y)
    for(int x = 0; x < tw; ++x)
    {
      float* p = &ids[((size_t)y * tw + x) * 4];
      p[0] = 0.f;
      p[1] = (float)y * 0.5f;
      p[2] = (float)x * 0.5f;
      p[3] = l;
    }
  return ids;
}

std::vector<float> flux3_text_ids(int lt)
{
  std::vector<float> ids((size_t)lt * 4, 0.f);
  for(int i = 0; i < lt; ++i)
    ids[(size_t)i * 4 + 3] = (float)i;
  return ids;
}

std::vector<float> flux3_rope_table(const std::vector<float>& ids)
{
  constexpr int kAxes = 4, kAxisDim = 32, kTheta = 10000;
  const size_t L = ids.size() / 4;
  std::vector<float> out(L * 64 * 4);
  for(size_t tok = 0; tok < L; ++tok)
    for(int a = 0; a < kAxes; ++a)
    {
      const double pos = ids[tok * 4 + a];
      for(int k = 0; k < kAxisDim / 2; ++k)
      {
        const double freq = 1.0 / std::pow((double)kTheta, (double)(2 * k) / kAxisDim);
        const double ang = pos * freq;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        float* p = &out[(tok * 64 + a * (kAxisDim / 2) + k) * 4];
        p[0] = c;
        p[1] = -s;
        p[2] = s;
        p[3] = c;
      }
    }
  return out;
}

Flux3Pipeline::Flux3Pipeline(const Flux3EnginePaths& paths)
{
  transformer_ = std::make_unique<Flux3TransformerWrapper>(paths.transformer);
  qwen_ = std::make_unique<Qwen3EncoderWrapper>(paths.qwen);
  vae_dec_ = std::make_unique<KleinVAEDecoderWrapper>(paths.vae_decoder);
  if(!paths.vae_encoder.empty())
    vae_enc_ = std::make_unique<KleinVAEEncoderWrapper>(paths.vae_encoder);
}

Flux3Pipeline::~Flux3Pipeline() { free_scratch(); }

int Flux3Pipeline::max_reference_tokens() const { return transformer_->max_cond_tokens(); }

void* Flux3Pipeline::grow_scratch(Scratch& s, size_t bytes)
{
  if(s.cap < bytes)
  {
    if(s.p) cudaFree(s.p);
    s.p = nullptr;
    if(cudaError_t e = cudaMalloc(&s.p, bytes); e != cudaSuccess)
    {
      s.cap = 0;
      throw std::runtime_error(
          std::string("Flux3Pipeline: scratch cudaMalloc failed (") + cudaGetErrorString(e) + ")");
    }
    s.cap = bytes;
  }
  return s.p;
}

void Flux3Pipeline::free_scratch()
{
  for(Scratch* s : {&sc_x_, &sc_xb_, &sc_vel_, &sc_unpacked_, &sc_patched_, &sc_vae_lat_, &sc_img_,
                    &sc_tr_, &sc_rope_img_, &sc_rope_cond_, &sc_rope_txt_})
  {
    if(s->p) cudaFree(s->p);
    s->p = nullptr;
    s->cap = 0;
  }
  rope_th_ = rope_tw_ = rope_lt_ = -1;
  rope_refs_valid_ = false;
}

void Flux3Pipeline::prepare_rope(
    int th, int tw, const std::vector<std::pair<int, int>>& refs, int lt, cudaStream_t stream)
{
  auto upload = [&](Scratch& s, const std::vector<float>& host) {
    void* p = grow_scratch(s, std::max<size_t>(host.size(), 1) * sizeof(float));
    if(!host.empty())
      cudaMemcpyAsync(p, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  };
  if(th != rope_th_ || tw != rope_tw_)
  {
    upload(sc_rope_img_, flux3_rope_table(flux3_image_ids(th, tw, 0.f)));
    rope_th_ = th;
    rope_tw_ = tw;
  }
  if(!rope_refs_valid_ || refs != rope_refs_)
  {
    std::vector<float> ids;
    for(size_t i = 0; i < refs.size(); ++i)
    {
      auto r = flux3_image_ids(refs[i].first, refs[i].second, -(1.f + (float)i));
      ids.insert(ids.end(), r.begin(), r.end());
    }
    upload(sc_rope_cond_, flux3_rope_table(ids));
    rope_refs_ = refs;
    rope_refs_valid_ = true;
  }
  if(lt != rope_lt_)
  {
    upload(sc_rope_txt_, flux3_rope_table(flux3_text_ids(lt)));
    rope_lt_ = lt;
  }
  // host vectors die at scope exit; the copies must land first
  cudaStreamSynchronize(stream);
}

void Flux3Pipeline::encode_text(
    const int64_t* input_ids, const int64_t* attention_mask, __nv_bfloat16* ehs_out, int Lt,
    cudaStream_t stream)
{
  if(Lt <= 0 || Lt % kFlux3TextPad)
    throw std::invalid_argument("flux3: text length must be a positive multiple of 80");
  qwen_->forward(input_ids, attention_mask, ehs_out, 1, Lt, kFlux3TextDim, stream);
}

void Flux3Pipeline::encode_reference(
    const unsigned char* ref_rgba, const float* bn_mean, const float* bn_std, int th, int tw,
    __nv_bfloat16* cond_out, cudaStream_t stream)
{
  if(!vae_enc_)
    throw std::runtime_error("flux3: reference images need a vae_encoder engine");
  const int H = th * 16, W = tw * 16, lh = th * 2, lw = tw * 2;
  auto* img = (__nv_bfloat16*)grow_scratch(sc_img_, (size_t)3 * H * W * sizeof(__nv_bfloat16));
  auto* lat = (__nv_bfloat16*)grow_scratch(sc_vae_lat_, (size_t)32 * lh * lw * sizeof(__nv_bfloat16));
  auto* patched = (__nv_bfloat16*)grow_scratch(sc_patched_, (size_t)128 * th * tw * sizeof(__nv_bfloat16));
  launch_klein_rgba_u8_to_chw_m1p1(ref_rgba, img, H, W, stream);
  vae_enc_->encode(img, lat, 1, H, W, stream);
  launch_klein_patchify(lat, patched, 1, 32, lh, lw, stream);
  launch_klein_bn(patched, patched, bn_mean, bn_std, 1, 128, th * tw, 0, stream);
  launch_klein_pack(patched, cond_out, 1, 128, th, tw, stream);
  cudaStreamSynchronize(stream);
}

void Flux3Pipeline::denoise_decode(const Request& r, cudaStream_t stream)
{
  const int steps = (int)r.schedule.size() - 1;
  if(steps < 1)
    throw std::invalid_argument("flux3: schedule needs at least two entries");
  const int Li = r.th * r.tw;
  int Lc = 0;
  if(r.cond)
    for(auto [cth, ctw] : r.refs)
      Lc += cth * ctw;
  const long n = (long)Li * 128;
  const int H = r.th * 16, W = r.tw * 16;

  prepare_rope(r.th, r.tw, Lc ? r.refs : std::vector<std::pair<int, int>>{}, r.lt, stream);

  auto* x = (float*)grow_scratch(sc_x_, n * sizeof(float));
  auto* xb = (__nv_bfloat16*)grow_scratch(sc_xb_, n * sizeof(__nv_bfloat16));
  auto* vel = (__nv_bfloat16*)grow_scratch(sc_vel_, n * sizeof(__nv_bfloat16));
  auto* tr = (float*)grow_scratch(sc_tr_, (size_t)2 * steps * sizeof(float));
  auto* unpacked = (__nv_bfloat16*)grow_scratch(sc_unpacked_, n * sizeof(__nv_bfloat16));
  auto* lat = (__nv_bfloat16*)grow_scratch(sc_vae_lat_, (size_t)32 * r.th * 2 * r.tw * 2 * sizeof(__nv_bfloat16));
  auto* img = (__nv_bfloat16*)grow_scratch(sc_img_, (size_t)3 * H * W * sizeof(__nv_bfloat16));

  // (t_i, t_{i+1}) for every step, uploaded once
  std::vector<float> tr_host(2 * steps);
  for(int i = 0; i < steps; ++i)
  {
    tr_host[2 * i] = r.schedule[i];
    tr_host[2 * i + 1] = r.schedule[i + 1];
  }
  cudaMemcpyAsync(tr, tr_host.data(), tr_host.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

  const float t0 = r.schedule[0];
  const __nv_bfloat16* start = r.start ? r.start : (Lc ? r.cond : nullptr);
  if(t0 < 1.f && start)
  {
    if(!r.start && (r.refs[0].first != r.th || r.refs[0].second != r.tw))
      throw std::invalid_argument("flux3: partial denoise from the reference needs matching geometry");
    launch_klein_lerp(xb, start, r.noise, t0, n, stream);
  }
  else
    cudaMemcpyAsync(xb, r.noise, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
  launch_klein_bf16_to_fp32(xb, x, n, stream);

  // empty reference stream: any valid address, TRT may reject null even at zero volume
  const __nv_bfloat16* cond = Lc ? r.cond : xb;
  const float* rope_cond = Lc ? (const float*)sc_rope_cond_.p : (const float*)sc_rope_img_.p;
  for(int i = 0; i < steps; ++i)
  {
    transformer_->forward(
        xb, cond, r.ehs, (const float*)sc_rope_img_.p, rope_cond,
        (const float*)sc_rope_txt_.p, tr + 2 * i, tr + 2 * i + 1, vel, Li, Lc, r.lt, stream);
    if(i == 0 && r.velocity0)
      cudaMemcpyAsync(r.velocity0, vel, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    launch_flux3_euler(x, xb, vel, r.schedule[i + 1] - r.schedule[i], n, stream);
  }
  if(r.final_tokens)
    cudaMemcpyAsync(r.final_tokens, x, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

  launch_klein_unpack(xb, unpacked, 1, 128, r.th, r.tw, stream);
  launch_klein_bn(unpacked, unpacked, r.bn_mean, r.bn_std, 1, 128, Li, 1, stream);
  launch_klein_unpatchify(unpacked, lat, 1, 32, r.th, r.tw, stream);
  vae_dec_->decode(lat, img, 1, r.th * 2, r.tw * 2, stream);
  launch_klein_chw_to_rgba_u8(img, r.rgba_out, H, W, stream);
  cudaStreamSynchronize(stream);
}

} // namespace librediffusion
