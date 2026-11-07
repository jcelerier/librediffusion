/**
 * FLUX.3-image-klein C++ harness against the goldens of make_flux3_golden.py.
 *
 * Per case dir: tokenizer ids, encode_text, encode_reference (edit cases), first-step velocity,
 * final tokens and decoded image vs the reference, then the end-to-end C-API path and timing.
 *
 * Usage: flux3_harness <engine_dir> <tokenizer.json> <case_dir>...
 *   engine_dir holds transformer.plan qwen.plan vae_decoder.plan vae_encoder.plan bn_mean.bin bn_std.bin
 *   FLUX3_TRANSFORMER=transformer_fp8.plan picks another transformer; an fp8 one gets FP8 tolerances.
 */
#include "librediffusion_c.h"
#include "npy.hpp"
#include "qwen_tokenizer_c.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<int64_t> load_i64(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if(!f)
    throw std::runtime_error("cannot open " + path);
  char pre[10];
  f.read(pre, 10);
  uint16_t hlen;
  std::memcpy(&hlen, pre + 8, 2);
  std::string hdr(hlen, '\0');
  f.read(hdr.data(), hlen);
  if(hdr.find("<i8") == std::string::npos)
    throw std::runtime_error("expected int64 in " + path);
  std::vector<int64_t> out;
  int64_t v;
  while(f.read((char*)&v, 8))
    out.push_back(v);
  return out;
}

static std::vector<float> load_bin(const std::string& path, size_t n)
{
  std::vector<float> v(n);
  std::ifstream f(path, std::ios::binary);
  if(!f.read((char*)v.data(), n * 4))
    throw std::runtime_error("cannot read " + path);
  return v;
}

static uint16_t f2bf(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t h)
{
  uint32_t x = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

struct Stats
{
  double cos, max_abs, mean_abs;
};
static Stats compare(const float* a, const float* b, size_t n)
{
  double dot = 0, na = 0, nb = 0, mx = 0, sum = 0;
  for(size_t i = 0; i < n; i++)
  {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
    double d = std::abs((double)a[i] - b[i]);
    mx = std::max(mx, d);
    sum += d;
  }
  return {dot / (std::sqrt(na * nb) + 1e-30), mx, sum / (double)n};
}

static void* upload_bf16(const std::vector<float>& v)
{
  std::vector<uint16_t> b(v.size());
  for(size_t i = 0; i < v.size(); i++)
    b[i] = f2bf(v[i]);
  void* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(b.size(), 1) * 2);
  cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
  return d;
}

// golden image [3,H,W] in [-1,1] -> PSNR in 8-bit space
static double psnr_vs_chw(const std::vector<unsigned char>& rgba, const npy::Array& img, int H, int W)
{
  double mse = 0;
  for(int c = 0; c < 3; c++)
    for(int i = 0; i < H * W; i++)
    {
      double g = std::clamp((img.data[(size_t)c * H * W + i] + 1.0) * 127.5, 0.0, 255.0);
      double d = rgba[(size_t)i * 4 + c] - g;
      mse += d * d;
    }
  mse /= 3.0 * H * W;
  return 10 * std::log10(255.0 * 255.0 / (mse + 1e-12));
}

static void write_ppm(const std::string& path, const std::vector<unsigned char>& rgba, int H, int W)
{
  std::ofstream o(path, std::ios::binary);
  o << "P6\n" << W << " " << H << "\n255\n";
  for(int i = 0; i < H * W; i++)
    o.write((const char*)&rgba[(size_t)i * 4], 3);
}

int main(int argc, char** argv)
{
  if(argc < 4)
  {
    fprintf(stderr, "usage: flux3_harness <engine_dir> <tokenizer.json> <case_dir>...\n");
    return 1;
  }
  const std::string E = argv[1];
  const char* tr_env = std::getenv("FLUX3_TRANSFORMER");
  const std::string tr = tr_env && *tr_env ? tr_env : "transformer.plan";
  const bool fp8 = tr.find("fp8") != std::string::npos;
  const double v0_tol = fp8 ? 0.99 : 0.999, fin_tol = fp8 ? 0.95 : 0.99, img_tol = fp8 ? 20.0 : 25.0;
  const std::string sfx = fp8 ? "_fp8" : "";
  printf("transformer %s\n", tr.c_str());
  auto h = librediffusion_flux3_create(
      (E + "/" + tr).c_str(), (E + "/qwen.plan").c_str(), (E + "/vae_decoder.plan").c_str(),
      (E + "/vae_encoder.plan").c_str(), argv[2], 0);
  if(!h)
  {
    fprintf(stderr, "create failed\n");
    return 2;
  }
  auto bn_mean = load_bin(E + "/bn_mean.bin", 128), bn_std = load_bin(E + "/bn_std.bin", 128);
  librediffusion_flux3_set_bn(h, bn_mean.data(), bn_std.data());

  bool all = true;
  auto gate = [&](const char* what, bool ok) {
    all = all && ok;
    return ok ? "PASS" : "FAIL";
  };

  for(int ci = 3; ci < argc; ci++)
  {
    const std::string G = argv[ci];
    auto gp = [&](const char* n) { return G + "/" + n + ".npy"; };
    printf("\n== %s\n", fs::path(G).filename().c_str());

    auto ids = load_i64(gp("input_ids"));
    auto mask = load_i64(gp("attention_mask"));
    const int Lt = (int)ids.size();
    auto txt = npy::load(gp("txt_tokens"));
    auto noise = npy::load(gp("noise"));
    auto sched = npy::load(gp("schedule"));
    auto v0 = npy::load(gp("step0_v"));
    auto fin = npy::load(gp("final_tokens"));
    auto img = npy::load(gp("image"));
    const int H = (int)img.shape[1], W = (int)img.shape[2], th = H / 16, tw = W / 16;
    const size_t ntok = (size_t)th * tw * 128;
    const bool edit = fs::exists(gp("cond_tokens"));

    // tokenizer
    {
      std::string chat;
      std::ifstream cf(G + "/chat.txt");
      chat.assign(std::istreambuf_iterator<char>(cf), {});
      const std::string pre = "<|im_start|>user\n", post = "<|im_end|>\n<|im_start|>assistant\n";
      std::string prompt = chat.substr(pre.size(), chat.size() - pre.size() - post.size());
      std::vector<int64_t> ti(Lt), tm(Lt);
      qwen_tokenizer_encode_chat_nothink(prompt.c_str(), Lt, ti.data(), tm.data());
      bool ok = ti == ids && tm == mask;
      printf("[tokenizer] ids %s\n", gate("tok", ok));
    }

    // encode_text
    void *d_ids, *d_mask, *d_ehs;
    cudaMalloc(&d_ids, Lt * 8);
    cudaMalloc(&d_mask, Lt * 8);
    cudaMalloc(&d_ehs, (size_t)Lt * 20480 * 2);
    cudaMemcpy(d_ids, ids.data(), Lt * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), Lt * 8, cudaMemcpyHostToDevice);
    librediffusion_flux3_encode_text(h, d_ids, d_mask, d_ehs, Lt);
    {
      std::vector<uint16_t> b((size_t)Lt * 20480);
      cudaMemcpy(b.data(), d_ehs, b.size() * 2, cudaMemcpyDeviceToHost);
      std::vector<float> f(b.size());
      for(size_t i = 0; i < b.size(); i++)
        f[i] = bf2f(b[i]);
      auto s = compare(f.data(), txt.data.data(), f.size());
      printf("[encode_text] cos %.6f max %.4g mean %.4g %s\n", s.cos, s.max_abs, s.mean_abs,
             gate("txt", s.cos > 0.999));
    }

    // encode_reference, per reference (multi-reference cases: ref{i}_rgb + ref_hw)
    void* d_cond_gold = nullptr;
    std::vector<int> ref_hw;
    std::vector<std::vector<unsigned char>> refs_rgba;
    if(edit)
    {
      if(fs::exists(gp("ref_hw")))
        for(int64_t v : load_i64(gp("ref_hw")))
          ref_hw.push_back((int)v);
      else
        ref_hw = {th, tw};
      const int nref = (int)ref_hw.size() / 2;
      auto cg = npy::load(gp("cond_tokens"));
      size_t off = 0;
      for(int k = 0; k < nref; k++)
      {
        const int rth = ref_hw[2 * k], rtw = ref_hw[2 * k + 1], RH = rth * 16, RW = rtw * 16;
        auto ref = npy::load(nref == 1 ? gp("ref_rgb") : G + "/ref" + std::to_string(k) + "_rgb.npy");
        auto& rgba_k = refs_rgba.emplace_back((size_t)RH * RW * 4, 255);
        for(int c = 0; c < 3; c++)
          for(int i = 0; i < RH * RW; i++)
            rgba_k[(size_t)i * 4 + c]
                = (unsigned char)std::lround((ref.data[(size_t)c * RH * RW + i] + 1.0) * 127.5);
        const size_t rn = (size_t)rth * rtw * 128;
        void* d_cond;
        cudaMalloc(&d_cond, rn * 2);
        librediffusion_flux3_encode_reference(h, rgba_k.data(), rth, rtw, d_cond);
        std::vector<uint16_t> b(rn);
        cudaMemcpy(b.data(), d_cond, rn * 2, cudaMemcpyDeviceToHost);
        std::vector<float> f(rn);
        for(size_t i = 0; i < rn; i++)
          f[i] = bf2f(b[i]);
        auto s = compare(f.data(), cg.data.data() + off, rn);
        printf("[encode_reference %d %dx%d] cos %.6f max %.4g mean %.4g %s\n", k, RW, RH, s.cos,
               s.max_abs, s.mean_abs, gate("ref", s.cos > 0.999));
        cudaFree(d_cond);
        off += rn;
      }
      d_cond_gold = upload_bf16(cg.data);
    }

    // denoise + decode from golden inputs
    void* d_noise = upload_bf16(noise.data);
    void* d_txt = upload_bf16(txt.data);
    std::vector<unsigned char> rgba((size_t)H * W * 4);
    std::vector<float> fin_cpp(ntok), v0_cpp(ntok);
    auto t0 = std::chrono::steady_clock::now();
    auto err = librediffusion_flux3_denoise_decode(
        h, d_noise, d_cond_gold, ref_hw.data(), (int)ref_hw.size() / 2, d_txt, Lt, th, tw,
        sched.data.data(),
        (int)sched.data.size(), rgba.data(), fin_cpp.data(), v0_cpp.data());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if(err != LIBREDIFFUSION_SUCCESS)
    {
      printf("denoise_decode error %d\n", err);
      all = false;
      continue;
    }
    auto sv = compare(v0_cpp.data(), v0.data.data(), ntok);
    printf("[velocity0] cos %.6f max %.4g mean %.4g %s\n", sv.cos, sv.max_abs, sv.mean_abs,
           gate("v0", sv.cos > v0_tol));
    auto sf = compare(fin_cpp.data(), fin.data.data(), ntok);
    printf("[final_tokens] cos %.6f max %.4g mean %.4g %s\n", sf.cos, sf.max_abs, sf.mean_abs,
           gate("fin", sf.cos > fin_tol));
    double p = psnr_vs_chw(rgba, img, H, W);
    printf("[image] PSNR %.2f dB %s  (%zu steps, %.1f ms first call)\n", p, gate("img", p >= img_tol),
           sched.data.size() - 1, ms);
    write_ppm(G + "/cpp_flux3" + sfx + ".ppm", rgba, H, W);

    // end-to-end C-API (own noise, so only timing + a visual)
    {
      std::string chat;
      std::ifstream cf(G + "/chat.txt");
      chat.assign(std::istreambuf_iterator<char>(cf), {});
      std::string prompt = chat.substr(17, chat.size() - 17 - 33);
      librediffusion_flux3_set_prompt(h, prompt.c_str());
      librediffusion_flux3_set_reference(h, nullptr, 0, 0);
      for(size_t k = 0; k < refs_rgba.size(); k++)
        if(auto e = librediffusion_flux3_add_reference(
               h, refs_rgba[k].data(), ref_hw[2 * k] * 16, ref_hw[2 * k + 1] * 16);
           e != LIBREDIFFUSION_SUCCESS)
        {
          printf("add_reference %zu error %d\n", k, e);
          all = false;
        }
      if(edit)
        printf("[references] %d, budget %d tokens\n", librediffusion_flux3_reference_count(h),
               librediffusion_flux3_max_reference_tokens(h));
      std::vector<float> times(sched.data.begin(), sched.data.end() - 1);
      librediffusion_flux3_set_times(h, times.data(), (int)times.size());
      std::vector<unsigned char> out((size_t)H * W * 4);
      librediffusion_flux3_generate(h, H, W, out.data());
      const int N = 3;
      auto a = std::chrono::steady_clock::now();
      for(int i = 0; i < N; i++)
        librediffusion_flux3_generate(h, H, W, out.data());
      double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / N;
      printf("[generate] %.1f ms/image (%.2f fps)\n", gms, 1000.0 / gms);
      write_ppm(G + "/cpp_flux3_generate" + sfx + ".ppm", out, H, W);
    }

    cudaFree(d_ids);
    cudaFree(d_mask);
    cudaFree(d_ehs);
    cudaFree(d_noise);
    cudaFree(d_txt);
    cudaFree(d_cond_gold);
  }

  librediffusion_flux3_destroy(h);
  printf("\n=== FLUX.3 GATE: %s ===\n", all ? "PASS" : "FAIL");
  return all ? 0 : 3;
}
