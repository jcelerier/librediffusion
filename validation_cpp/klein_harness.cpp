/**
 * FLUX.2-klein-4B C++ validation harness (Phase 2 GATE).
 *
 * Drives the librediffusion_flux2_* C-API against the Phase-0 Python golden:
 *   1) encode_text seam: golden input_ids -> C++ Qwen3 engine -> compare encoder_hidden_states[1,Lt,7680]
 *      cosine vs golden 005_textencode__encoder_hidden_states_7680.
 *   2) full txt2img: golden init noise (110_transformer__in_hidden_step0) + golden encoder_hidden_states
 *      -> 2-step Euler + decode -> compare final packed latent (vs 121_scheduler__latent_after_step1)
 *      and decoded image PSNR (vs 999_output__final_image).
 *
 * The Qwen tokenizer chat-template is a Phase-4 item; here we feed the golden input_ids to validate the
 * encoder engine + the whole denoise/decode numerically against the golden.
 *
 * Usage: klein_harness <golden_dir> <transformer.plan> <qwen.plan> <vae_decoder.plan>
 */
#include "librediffusion_c.h"
#include "npy.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// minimal int npy reader (int32/int64) -> vector<int64_t>
static std::vector<int64_t> load_int_npy(const std::string& path, std::vector<size_t>& shape)
{
  std::ifstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("cannot open " + path);
  char magic[6]; f.read(magic, 6);
  uint8_t maj, min; f.read((char*)&maj, 1); f.read((char*)&min, 1);
  uint16_t hlen; f.read((char*)&hlen, 2);
  std::string hdr(hlen, '\0'); f.read(&hdr[0], hlen);
  // descr
  auto dpos = hdr.find("'descr'");
  std::string descr = hdr.substr(hdr.find('\'', dpos + 7) + 1);
  descr = descr.substr(0, descr.find('\''));
  // shape
  auto spos = hdr.find("'shape'");
  auto lp = hdr.find('(', spos), rp = hdr.find(')', spos);
  std::string sh = hdr.substr(lp + 1, rp - lp - 1);
  shape.clear();
  size_t n = 1, i = 0;
  while(i < sh.size())
  {
    while(i < sh.size() && !isdigit(sh[i])) i++;
    if(i >= sh.size()) break;
    size_t j = i; while(j < sh.size() && isdigit(sh[j])) j++;
    size_t d = std::stoul(sh.substr(i, j - i)); shape.push_back(d); n *= d; i = j;
  }
  std::vector<int64_t> out(n);
  if(descr == "<i4" || descr == "i4")
  {
    std::vector<int32_t> tmp(n); f.read((char*)tmp.data(), n * 4);
    for(size_t k = 0; k < n; k++) out[k] = tmp[k];
  }
  else if(descr == "<i8" || descr == "i8")
  {
    f.read((char*)out.data(), n * 8);
  }
  else if(descr == "<f4" || descr == "f4")
  {
    std::vector<float> tmp(n); f.read((char*)tmp.data(), n * 4);
    for(size_t k = 0; k < n; k++) out[k] = (int64_t)llround(tmp[k]);
  }
  else throw std::runtime_error("int npy dtype " + descr);
  return out;
}

static double cosine(const float* a, const float* b, size_t n)
{
  double dot = 0, na = 0, nb = 0;
  for(size_t i = 0; i < n; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

// bf16 helpers (host): store float32 as bf16 truncation
static uint16_t f2bf16(float f)
{
  uint32_t x; std::memcpy(&x, &f, 4);
  uint32_t r = x + 0x7fff + ((x >> 16) & 1);
  return (uint16_t)(r >> 16);
}
static float bf162f(uint16_t h)
{
  uint32_t x = (uint32_t)h << 16; float f; std::memcpy(&f, &x, 4); return f;
}

int main(int argc, char** argv)
{
  if(argc < 5) { fprintf(stderr, "usage: klein_harness <golden_dir> <transformer.plan> <qwen.plan> <vae_decoder.plan>\n"); return 1; }
  std::string G = argv[1];
  auto gp = [&](const std::string& n) { return G + "/" + n + ".npy"; };

  // ---- load golden ----
  std::vector<size_t> idsh;
  auto input_ids = load_int_npy(gp("000_textencode__input_ids"), idsh);
  auto attn_mask = load_int_npy(gp("001_textencode__attention_mask"), idsh);
  int Lt = (int)idsh[1];
  auto ehs_gold = npy::load(gp("005_textencode__encoder_hidden_states_7680"));  // [1,Lt,7680]
  auto init_noise = npy::load(gp("110_transformer__in_hidden_step0"));          // [1,Lp,128]
  auto final_gold = npy::load(gp("121_scheduler__latent_after_step1"));         // [1,Lp,128]
  auto img_gold = npy::load(gp("999_output__final_image"));                     // [H,W,3] u8-as-f32
  auto bn_mean = npy::load(gp("020_vae__bn_mean"));                             // [128]
  auto bn_std = npy::load(gp("022_vae__bn_std"));                               // [128]

  int Lp = (int)init_noise.shape[1];
  int Th = 36, Tw = 20;  // 320x576 -> latent 72x40 -> tokens 36x20
  int H = (int)img_gold.shape[0], W = (int)img_gold.shape[1];
  printf("golden: Lt=%d Lp=%d Th=%d Tw=%d H=%d W=%d\n", Lt, Lp, Th, Tw, H, W);

  // ---- create pipeline ----
  auto h = librediffusion_flux2_create(argv[2], argv[3], argv[4], nullptr, 0);
  if(!h) { fprintf(stderr, "create failed\n"); return 2; }

  // ===== SEAM 1: Qwen encode_text =====
  int64_t *d_ids, *d_mask; cudaMalloc(&d_ids, Lt * 8); cudaMalloc(&d_mask, Lt * 8);
  cudaMemcpy(d_ids, input_ids.data(), Lt * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, attn_mask.data(), Lt * 8, cudaMemcpyHostToDevice);
  uint16_t* d_ehs; cudaMalloc(&d_ehs, (long)Lt * 7680 * 2);
  librediffusion_flux2_encode_text(h, d_ids, d_mask, d_ehs, Lt);
  std::vector<uint16_t> ehs_bf(Lt * 7680);
  cudaMemcpy(ehs_bf.data(), d_ehs, (long)Lt * 7680 * 2, cudaMemcpyDeviceToHost);
  std::vector<float> ehs_cpp(Lt * 7680);
  for(size_t i = 0; i < ehs_cpp.size(); i++) ehs_cpp[i] = bf162f(ehs_bf[i]);
  double ehs_cos = cosine(ehs_cpp.data(), ehs_gold.data.data(), ehs_cpp.size());
  printf("[SEAM encode_text] encoder_hidden_states cosine = %.5f  GATE>=0.99: %s\n",
         ehs_cos, ehs_cos >= 0.99 ? "PASS" : "FAIL");

  // ===== SEAM 2+3: full txt2img (use GOLDEN ehs to isolate the transformer/decode path) =====
  // upload golden ehs (bf16) + golden init noise (bf16) + bn (fp32)
  std::vector<uint16_t> ehsg_bf(ehs_gold.size());
  for(size_t i = 0; i < ehsg_bf.size(); i++) ehsg_bf[i] = f2bf16(ehs_gold.data[i]);
  uint16_t* d_ehsg; cudaMalloc(&d_ehsg, ehsg_bf.size() * 2);
  cudaMemcpy(d_ehsg, ehsg_bf.data(), ehsg_bf.size() * 2, cudaMemcpyHostToDevice);

  std::vector<uint16_t> noise_bf(init_noise.size());
  for(size_t i = 0; i < noise_bf.size(); i++) noise_bf[i] = f2bf16(init_noise.data[i]);
  uint16_t* d_noise; cudaMalloc(&d_noise, noise_bf.size() * 2);
  cudaMemcpy(d_noise, noise_bf.data(), noise_bf.size() * 2, cudaMemcpyHostToDevice);

  float *d_mean, *d_std; cudaMalloc(&d_mean, 128 * 4); cudaMalloc(&d_std, 128 * 4);
  cudaMemcpy(d_mean, bn_mean.data.data(), 128 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_std, bn_std.data.data(), 128 * 4, cudaMemcpyHostToDevice);

  std::vector<unsigned char> rgba(H * W * 4);
  std::vector<float> final_cpp(Lp * 128);
  librediffusion_flux2_txt2img(h, d_noise, d_ehsg, d_mean, d_std, Th, Tw, Lt, 2,
                               rgba.data(), final_cpp.data());

  double lat_cos = cosine(final_cpp.data(), final_gold.data.data(), final_cpp.size());
  printf("[SEAM final latent] cosine = %.5f  GATE>=0.99: %s\n", lat_cos, lat_cos >= 0.99 ? "PASS" : "FAIL");

  // PSNR of decoded image vs golden
  double mse = 0;
  for(int i = 0; i < H * W; i++)
    for(int c = 0; c < 3; c++)
    {
      double cpp = rgba[i * 4 + c];
      double gld = img_gold.data[(size_t)i * 3 + c];
      mse += (cpp - gld) * (cpp - gld);
    }
  mse /= (double)(H * W * 3);
  double psnr = 20 * std::log10(255.0) - 10 * std::log10(mse + 1e-12);
  printf("[SEAM decode] image PSNR = %.2f dB  GATE>=25: %s\n", psnr, psnr >= 25 ? "PASS" : "FAIL");

  // write a PNG for visual check (simple PPM->png skip; write raw rgba as ppm)
  {
    std::ofstream o(std::string(argv[1]) + "/cpp_klein_out.ppm", std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for(int i = 0; i < H * W; i++) { o.put(rgba[i*4]); o.put(rgba[i*4+1]); o.put(rgba[i*4+2]); }
  }

  // ===== timing: engine-resident single-image fps (encode_text + 2-step denoise + decode) =====
  {
    // warmup
    librediffusion_flux2_encode_text(h, d_ids, d_mask, d_ehs, Lt);
    librediffusion_flux2_txt2img(h, d_noise, d_ehsg, d_mean, d_std, Th, Tw, Lt, 2, rgba.data(), nullptr);
    cudaDeviceSynchronize();
    const int N = 10;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for(int i = 0; i < N; i++)
    {
      librediffusion_flux2_encode_text(h, d_ids, d_mask, d_ehs, Lt);
      librediffusion_flux2_txt2img(h, d_noise, d_ehsg, d_mean, d_std, Th, Tw, Lt, 2, rgba.data(), nullptr);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= N;
    printf("[TIMING] full pipeline (qwen + 2-step transformer + decode) = %.1f ms/image = %.2f fps\n",
           ms, 1000.0 / ms);
  }

  librediffusion_flux2_destroy(h);
  bool all = ehs_cos >= 0.99 && lat_cos >= 0.99 && psnr >= 25;
  printf("\n=== PHASE 2 GATE: %s ===\n", all ? "PASS" : "PARTIAL");
  return 0;
}
