/*
 * SD/SDXL streaming-FPS harness — measures sustained img2img streaming throughput + pacing
 * for the SD1.5 / SDXL pipeline (the counterpart to klein_fps_harness, which is klein-only).
 *
 * Reproduces the ossia StreamDiffusion node's per-frame work: encode_image -> predict_x0_batch
 * -> decode_latent -> synchronize, looped for a fixed wall-clock window. Reports average FPS
 * AND frame-pacing percentiles (p50/p95/p99) — average hides stutter; p99 is the real bar.
 *
 * Engine/config are taken from a golden cell (same dir layout as harness.cpp) so the run
 * matches a validated configuration exactly (model, steps, cfg, resolution, seed).
 *
 * Usage:
 *   sd_fps_harness --cell <golden_cell_dir> --engines <engine_set_dir> --clip <clip.engine>
 *                  [--seconds 30] [--label NAME]
 */
#include "librediffusion_c.h"

#include <cuda_runtime.h>

#include "npy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

#define CK(x)                                                              \
  do {                                                                     \
    int _rc = (x);                                                         \
    if(_rc != 0) {                                                         \
      std::cerr << "FAIL " << #x << " rc=" << _rc << "\n";                 \
      std::exit(2);                                                        \
    }                                                                      \
  } while(0)

// minimal json read of the cell config (reuse json_min via harness would pull dump deps;
// we only need a handful of scalar fields, so parse with a tiny helper).
#include "json_min.hpp"

struct Cfg {
  std::vector<int> t_index_list;
  std::string cfg_type = "none";
  float guidance_scale = 1.0f, delta = 1.0f;
  bool do_add_noise = true, use_denoising_batch = true;
  int width = 512, height = 512, seed = 1234;
  std::string prompt;
};

static librediffusion_cfg_type_t cfg_enum(const std::string& s) {
  if(s == "none") return SD_CFG_NONE;
  if(s == "full") return SD_CFG_FULL;
  if(s == "self") return SD_CFG_SELF;
  if(s == "initialize") return SD_CFG_INITIALIZE;
  throw std::runtime_error("unknown cfg_type " + s);
}

static std::vector<librediffusion_half_t> floats_to_half(const std::vector<float>& f) {
  std::vector<librediffusion_half_t> h(f.size());
  for(size_t i = 0; i < f.size(); i++) h[i] = librediffusion_float_to_half(f[i]);
  return h;
}

static bool load_golden(const fs::path& cell, const std::string& boundary,
                        const std::string& role, npy::Array& out) {
  for(auto& e : fs::directory_iterator(cell)) {
    auto fn = e.path().filename().string();
    if(fn.find("_" + boundary + "__" + role + ".npy") != std::string::npos) {
      out = npy::load(e.path().string());
      return true;
    }
  }
  return false;
}

static void pacing_report(std::vector<double>& iv_ms, double window_s, const std::string& label) {
  if(iv_ms.empty()) { std::printf("  (no frames)\n"); return; }
  std::sort(iv_ms.begin(), iv_ms.end());
  auto pct = [&](double p) { return iv_ms[(size_t)std::min<double>(iv_ms.size()-1, p*iv_ms.size())]; };
  double sum = 0; for(double v : iv_ms) sum += v;
  double mean = sum / iv_ms.size();
  double p50 = pct(0.50), p95 = pct(0.95), p99 = pct(0.99);
  size_t frames = iv_ms.size();
  double avg_fps = frames / window_s;
  std::printf("=== [%s] streaming FPS over %.1fs ===\n", label.c_str(), window_s);
  std::printf("  frames=%zu   AVG fps=%.2f   (mean frame=%.2f ms)\n", frames, avg_fps, mean);
  std::printf("  --- frame pacing (interval ms) ---\n");
  std::printf("    p50=%.2f  p95=%.2f  p99=%.2f ms   => p99 fps=%.1f (worst-1%% sustained)\n",
              p50, p95, p99, 1000.0/p99);
}

int main(int argc, char** argv) {
  std::string cell_s, eng_s, clip_s, label = "sd";
  double seconds = 30.0;
  int hidden_dim_override = 0, pad_token_override = -1;
  int denoise_batch_override = -1;  // -1 = take the cell's value
  std::string cn_engine; float cn_scale = 1.0f;
  for(int i = 1; i < argc; i++) {
    if(!std::strcmp(argv[i], "--controlnet") && i + 1 < argc) { cn_engine = argv[++i]; continue; }
    if(!std::strcmp(argv[i], "--cn-scale")   && i + 1 < argc) { cn_scale = std::atof(argv[++i]); continue; }
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if(a == "--cell") cell_s = nxt();
    else if(a == "--engines") eng_s = nxt();
    else if(a == "--clip") clip_s = nxt();
    else if(a == "--seconds") seconds = std::stod(nxt());
    else if(a == "--label") label = nxt();
    else if(a == "--hidden-dim") hidden_dim_override = std::stoi(nxt());
    else if(a == "--pad-token") pad_token_override = std::stoi(nxt());
    else if(a == "--denoise-batch") denoise_batch_override = std::stoi(nxt());
  }
  if(cell_s.empty() || eng_s.empty() || clip_s.empty()) {
    std::cerr << "usage: sd_fps_harness --cell <dir> --engines <dir> --clip <clip.engine> "
                 "[--seconds 30] [--label NAME]\n";
    return 1;
  }
  fs::path cell = cell_s, eng = eng_s;

  // --- parse cell config ---
  auto j = jsonm::parse_file((cell / "config.json").string());
  Cfg c;
  for(auto& t : j->at("t_index_list")->arr) c.t_index_list.push_back(t->as_int());
  c.cfg_type = j->at("cfg_type")->as_str();
  c.guidance_scale = (float)j->at("guidance_scale")->as_num();
  c.delta = (float)j->at("delta")->as_num();
  c.do_add_noise = j->at("do_add_noise")->as_bool();
  c.use_denoising_batch = j->at("use_denoising_batch")->as_bool();
  c.width = j->at("width")->as_int();
  c.height = j->at("height")->as_int();
  c.seed = j->at("seed")->as_int();
  c.prompt = j->at("prompt")->as_str();

  const int latent_w = c.width / 8, latent_h = c.height / 8;
  const int steps = (int)c.t_index_list.size();
  const int batch = 1; // frame_buffer / config batch_size
  const int latent_elems = batch * 4 * latent_h * latent_w;

  // hidden_dim from golden prompt_embeds last dim (or --hidden-dim override for engines w/o a matching golden)
  int hidden_dim = 768;
  { npy::Array pe; if(load_golden(cell, "static", "prompt_embeds", pe) && !pe.shape.empty())
      hidden_dim = (int)pe.shape.back(); }
  if(hidden_dim_override) hidden_dim = hidden_dim_override;
  const int pad_token = pad_token_override >= 0 ? pad_token_override : 49407;

  // --- config ---
  librediffusion_config_handle cfg = nullptr;

  CK(librediffusion_config_create(&cfg));
  CK(librediffusion_config_set_device(cfg, 0));
  CK(librediffusion_config_set_model_type(cfg, MODEL_SD_15));
  CK(librediffusion_config_set_dimensions(cfg, c.width, c.height, latent_w, latent_h));
  CK(librediffusion_config_set_batch_size(cfg, batch));
  CK(librediffusion_config_set_denoising_steps(cfg, steps));
  CK(librediffusion_config_set_frame_buffer_size(cfg, 1));
  CK(librediffusion_config_set_guidance_scale(cfg, c.guidance_scale));
  CK(librediffusion_config_set_delta(cfg, c.delta));
  CK(librediffusion_config_set_add_noise(cfg, c.do_add_noise ? 1 : 0));
  // LRD_FPS_GRAPH=1 -> enable the .so's internal 1-step CUDA-graph capture/replay (the production path).
  if(std::getenv("LRD_FPS_GRAPH")) CK(librediffusion_config_set_cuda_graph(cfg, 1));
  const bool db = denoise_batch_override >= 0 ? (denoise_batch_override != 0)
                                             : c.use_denoising_batch;
  CK(librediffusion_config_set_denoising_batch(cfg, db ? 1 : 0));
  CK(librediffusion_config_set_seed(cfg, (uint64_t)c.seed));
  CK(librediffusion_config_set_cfg_type(cfg, cfg_enum(c.cfg_type)));
  CK(librediffusion_config_set_text_config(cfg, 77, hidden_dim, pad_token));
  CK(librediffusion_config_set_unet_engine(cfg, (eng / "unet.engine").string().c_str()));
  // ControlNet: control-aware unet.engine (already set above) + a separate controlnet.engine.
  // The per-frame cond upload is part of the cost, so it is done inside the timed frame lambda.
  int cn_index = -1;
  if(!cn_engine.empty())
  {
    cn_index = librediffusion_config_add_controlnet(cfg, cn_engine.c_str(), cn_scale);
    if(cn_index < 0)
      throw std::runtime_error("config_add_controlnet failed for " + cn_engine);
  }
  CK(librediffusion_config_set_vae_encoder(cfg, (eng / "vae_encoder.engine").string().c_str()));
  CK(librediffusion_config_set_vae_decoder(cfg, (eng / "vae_decoder.engine").string().c_str()));
  CK(librediffusion_config_set_timestep_indices(cfg, c.t_index_list.data(), c.t_index_list.size()));

  // --- pipeline ---
  librediffusion_pipeline_handle pipe = nullptr;
  CK(librediffusion_pipeline_create(cfg, &pipe));
  CK(librediffusion_pipeline_init_all(pipe));
  librediffusion_stream_t stream = librediffusion_pipeline_get_stream(pipe);

  // --- embeddings ---
  librediffusion_clip_handle clip = nullptr;
  librediffusion_half_t* prompt_embeds = nullptr;
  CK(librediffusion_clip_create(clip_s.c_str(), 0, &clip));
  CK(librediffusion_clip_compute_embeddings(clip, c.prompt.c_str(), pad_token, stream, &prompt_embeds));
  CK(librediffusion_prepare_embeds(pipe, prompt_embeds, 77, hidden_dim));

  // --- scheduler tables (inject golden statics; required before predict_x0_batch) ---
  {
    npy::Array sub_t, alpha, beta, cskip, cout;
    if(!(load_golden(cell, "static", "sub_timesteps_tensor", sub_t)
         && load_golden(cell, "static", "alpha_prod_t_sqrt", alpha)
         && load_golden(cell, "static", "beta_prod_t_sqrt", beta)
         && load_golden(cell, "static", "c_skip", cskip)
         && load_golden(cell, "static", "c_out", cout)))
      throw std::runtime_error("cell missing scheduler static tensors");
    CK(librediffusion_prepare_scheduler(pipe, sub_t.data.data(), alpha.data.data(),
                                        beta.data.data(), cskip.data.data(),
                                        cout.data.data(), sub_t.size()));
  }

  // --- init noise (golden) — required when do_add_noise ---
  librediffusion_half_t* d_init_noise = nullptr;
  {
    npy::Array init_noise;
    if(load_golden(cell, "static", "init_noise", init_noise)) {
      auto h = floats_to_half(init_noise.data);
      d_init_noise = (librediffusion_half_t*)librediffusion_cuda_malloc(h.size()*sizeof(librediffusion_half_t));
      CK(librediffusion_cuda_memcpy_h2d(d_init_noise, h.data(), h.size()*sizeof(librediffusion_half_t)));
      CK(librediffusion_set_init_noise(pipe, d_init_noise));
    }
  }

  // --- input frame: use the golden encode_image input (a real image), tiled to batch ---
  npy::Array g_in;
  if(!load_golden(cell, "encode_image", "in_image", g_in))
    throw std::runtime_error("cell missing encode_image/in_image (need an img2img cell)");
  const size_t one_img = (size_t)3 * c.height * c.width;
  std::vector<float> tiled((size_t)batch * one_img);
  for(int b = 0; b < batch; b++)
    std::memcpy(tiled.data() + b*one_img, g_in.data.data(), one_img*sizeof(float));
  void* d_img_in = librediffusion_cuda_malloc(tiled.size()*sizeof(float));
  CK(librediffusion_cuda_memcpy_h2d(d_img_in, tiled.data(), tiled.size()*sizeof(float)));

  void* d_latent = librediffusion_cuda_malloc((size_t)latent_elems*sizeof(librediffusion_half_t));
  void* d_x0     = librediffusion_cuda_malloc((size_t)latent_elems*sizeof(librediffusion_half_t));
  void* d_img_out= librediffusion_cuda_malloc((size_t)(3*c.height*c.width)*sizeof(librediffusion_half_t));

  // Per-frame work. CUDA-graph capture/replay (when LRD_FPS_GRAPH=1) is now handled INSIDE the .so
  // (librediffusion_config_set_cuda_graph above), inside predict_x0_batch — so the harness just calls the
  // C-API every frame and the pipeline transparently replays the captured graph for the predict step.
  // A control map derived from the golden input frame. Content is irrelevant to throughput; what
  // matters is that the upload + ControlNet forward happen every frame, as they do in the node.
  std::vector<uint8_t> cn_rgba;
  if(cn_index >= 0) {
    cn_rgba.resize((size_t)c.height * c.width * 4);
    for(size_t p = 0, n = (size_t)c.height * c.width; p < n; p++) {
      auto q = [&](int ch){ float v = g_in.data[(size_t)ch*n + p]; return (uint8_t)std::clamp((v*0.5f+0.5f)*255.f, 0.f, 255.f); };
      cn_rgba[p*4+0]=q(0); cn_rgba[p*4+1]=q(1); cn_rgba[p*4+2]=q(2); cn_rgba[p*4+3]=255;
    }
  }

  auto frame = [&]() {
    if(cn_index >= 0)
      CK(librediffusion_set_controlnet_cond_rgba(pipe, cn_index, cn_rgba.data(), c.height, c.width));
    CK(librediffusion_encode_image_float(pipe, (const float*)d_img_in, (librediffusion_half_t*)d_latent, stream));
    CK(librediffusion_predict_x0_batch(pipe, (const librediffusion_half_t*)d_latent, (librediffusion_half_t*)d_x0, stream));
    CK(librediffusion_decode_latent(pipe, (const librediffusion_half_t*)d_x0, (librediffusion_half_t*)d_img_out, stream));
    CK(librediffusion_pipeline_synchronize(pipe));
  };

  // warmup (5 frames — stabilize clocks/caches + let the .so capture its graph on first predict)
  std::fprintf(stderr, "[sd_fps] warmup...\n"); std::fflush(stderr);
  for(int i = 0; i < 5; i++) frame();
  std::fprintf(stderr, "[sd_fps] warmup done, timing %.1fs...\n", seconds); std::fflush(stderr);

  // timed window
  std::vector<double> iv_ms;
  auto t0 = clk::now();
  auto prev = t0;
  double elapsed = 0;
  while(elapsed < seconds) {
    frame();
    auto now = clk::now();
    iv_ms.push_back(std::chrono::duration<double, std::milli>(now - prev).count());
    prev = now;
    elapsed = std::chrono::duration<double>(now - t0).count();
  }
  std::fprintf(stderr, "[sd_fps] loop done, %zu frames in %.2fs\n", iv_ms.size(), elapsed);
  std::fflush(stderr);
  pacing_report(iv_ms, elapsed, label);

  // correctness sanity: dump the final output image stats (finite? in range?) — env LRD_FPS_DUMPSTATS
  if(std::getenv("LRD_FPS_DUMPSTATS")) {
    size_t n = (size_t)3 * c.height * c.width;
    std::vector<librediffusion_half_t> h(n);
    CK(librediffusion_cuda_memcpy_d2h(h.data(), d_img_out, n*sizeof(librediffusion_half_t)));
    double mn=1e30, mx=-1e30, sum=0; int nan=0;
    for(size_t i=0;i<n;i++){ float v = librediffusion_half_to_float(h[i]); if(v!=v){nan++;continue;} mn=std::min(mn,(double)v); mx=std::max(mx,(double)v); sum+=v; }
    std::printf("  [stats] min=%.3f max=%.3f mean=%.4f nan=%d (n=%zu)\n", mn, mx, sum/n, nan, n);
  }
  std::fflush(stdout);

  // INVALIDATION TEST: LRD_FPS_RECONFIG=WxH -> reinit_buffers to a new resolution on the LIVE pipeline (graph
  // must recapture for the new geometry), run a few frames, confirm finite output. Exercises the reinit_buffers
  // graph-reset fix. Requires the engine profile to cover the new resolution (e.g. engines-sd15 -> 512x768).
  if(const char* rc = std::getenv("LRD_FPS_RECONFIG")) {
    int nW=0, nH=0; if(sscanf(rc, "%dx%d", &nW, &nH) == 2 && nW>0 && nH>0) {
      std::fprintf(stderr, "[sd_fps] reconfigure -> %dx%d (reinit_buffers; graph must recapture)\n", nW, nH);
      CK(librediffusion_config_set_dimensions(cfg, nW, nH, nW/8, nH/8));
      CK(librediffusion_pipeline_reinit_buffers(pipe, cfg));
      // re-stage a synthetic input at the new resolution + re-alloc the I/O buffers
      size_t one2 = (size_t)3*nH*nW; std::vector<float> t2((size_t)batch*one2);
      for(size_t i=0;i<t2.size();i++) t2[i] = ((i*131+7)&0xff)/127.5f - 1.0f;
      void* in2 = librediffusion_cuda_malloc(t2.size()*sizeof(float));
      CK(librediffusion_cuda_memcpy_h2d(in2, t2.data(), t2.size()*sizeof(float)));
      int lat2 = batch*4*(nH/8)*(nW/8);
      void* lt2 = librediffusion_cuda_malloc((size_t)lat2*sizeof(librediffusion_half_t));
      void* x02 = librediffusion_cuda_malloc((size_t)lat2*sizeof(librediffusion_half_t));
      void* out2= librediffusion_cuda_malloc((size_t)3*nH*nW*sizeof(librediffusion_half_t));
      // re-prepare embeds (same prompt) — prepare_embeds copies into the (reallocated) buffers
      CK(librediffusion_prepare_embeds(pipe, prompt_embeds, 77, hidden_dim));
      for(int i=0;i<15;i++){
        CK(librediffusion_encode_image_float(pipe,(const float*)in2,(librediffusion_half_t*)lt2,stream));
        CK(librediffusion_predict_x0_batch(pipe,(const librediffusion_half_t*)lt2,(librediffusion_half_t*)x02,stream));
        CK(librediffusion_decode_latent(pipe,(const librediffusion_half_t*)x02,(librediffusion_half_t*)out2,stream));
        CK(librediffusion_pipeline_synchronize(pipe));
      }
      size_t n2=(size_t)3*nH*nW; std::vector<librediffusion_half_t> h2(n2);
      CK(librediffusion_cuda_memcpy_d2h(h2.data(), out2, n2*sizeof(librediffusion_half_t)));
      double mn=1e30,mx=-1e30,sum=0; int nan=0;
      for(size_t i=0;i<n2;i++){ float v=librediffusion_half_to_float(h2[i]); if(v!=v){nan++;continue;} mn=std::min(mn,(double)v); mx=std::max(mx,(double)v); sum+=v; }
      std::printf("  [reconfig %dx%d] frames=15 min=%.3f max=%.3f mean=%.4f nan=%d  %s\n",
                  nW,nH, mn,mx,sum/n2,nan, (nan==0 && mx>mn) ? "OK" : "BAD");
      librediffusion_cuda_free(in2); librediffusion_cuda_free(lt2);
      librediffusion_cuda_free(x02); librediffusion_cuda_free(out2);
      std::fflush(stdout);
    }
  }

  librediffusion_cuda_free(d_img_in);
  librediffusion_cuda_free(d_latent);
  librediffusion_cuda_free(d_x0);
  librediffusion_cuda_free(d_img_out);
  librediffusion_cuda_free(prompt_embeds);
  if(d_init_noise) librediffusion_cuda_free(d_init_noise);
  librediffusion_clip_destroy(clip);
  librediffusion_pipeline_destroy(pipe);
  librediffusion_config_destroy(cfg);
  return 0;
}
