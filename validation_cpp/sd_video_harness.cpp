/*
 * SD / SD-Turbo / SDXL img2img over a real video, configured from an engine bundle the way the ossia
 * StreamDiffusion node configures it: model-type defaults, bundle.json (embedding_dim,
 * unet_timestep_scale), the node's precomputed scheduler tables, CLIP / CLIP2 engines from the bundle,
 * and librediffusion_img2img on host RGBA frames, which is the node's per-frame call (upload, encode,
 * denoise, decode, download, synchronize). No golden cell needed, so any bundle can be benchmarked.
 *
 * Usage:
 *   sd_video_harness --bundle DIR --model sd15|sd-turbo|sdxl --video FRAMES.rgba --size WxH
 *                    --timesteps "32, 45" --prompt TEXT
 *                    [--warmup 10] [--frames N] [--seed 42] [--denoise-batch 0|1] [--add-noise 0|1]
 *                    [--out OUT.rgba] [--json OUT.json] [--label NAME]
 *                    [--cuda-graph 0|1]   (default 1, like the node)
 *                    [--rife PLAN --rife-exp N]
 *
 * FRAMES.rgba is raw concatenated WxH RGBA8 frames, e.g.
 *   ffmpeg -i in.mp4 -vf scale=W:H:force_original_aspect_ratio=increase,crop=W:H -pix_fmt rgba -f rawvideo FRAMES.rgba
 * The video is looped if --warmup + --frames exceeds it. Only the img2img call is timed; --out
 * writes each timed output frame (same format) for encoding afterwards.
 *
 * --rife-exp N adds RIFE after each diffused frame, as the node's producer does: prev -> cur is
 * interpolated into 2^N frames, timed after the img2img call, and --out gets all of them (encode at
 * 2^N times the source rate). The first timed frame has no predecessor and is repeated 2^N times.
 */
#include "librediffusion_c.h"

#include "bundle_manifest.hpp"
#include "schedulers/lcm_dreamshaper_v7.hpp"
#include "schedulers/schedule.hpp"
#include "schedulers/sd-turbo.hpp"
#include "schedulers/sdxl-turbo.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

#define CK(x)                                                                         \
  do {                                                                                \
    auto _rc = (x);                                                                   \
    if(_rc != LIBREDIFFUSION_SUCCESS) {                                               \
      std::fprintf(stderr, "FAIL %s rc=%d (%s)\n", #x, (int)_rc,                     \
                   librediffusion_get_last_cuda_error_string());                      \
      std::exit(2);                                                                   \
    }                                                                                 \
  } while(0)

namespace
{
struct Args
{
  std::string bundle, model, video, timesteps, prompt, out, json, label;
  int width = 0, height = 0, warmup = 10, frames = 0, seed = 42;
  int denoise_batch = -1; // -1: on for multi-step, like the presets
  int add_noise = 0;      // the node presets' img2img default
  int cuda_graph = 1;     // like the node: requested, the library gates it
  std::string rife;       // rife_ifnet_fp16.plan
  int rife_exp = 0;
};

Args parse(int argc, char** argv)
{
  Args a;
  for(int i = 1; i < argc; i++)
  {
    std::string k = argv[i];
    auto v = [&]() -> std::string {
      if(i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if(k == "--bundle") a.bundle = v();
    else if(k == "--model") a.model = v();
    else if(k == "--video") a.video = v();
    else if(k == "--timesteps") a.timesteps = v();
    else if(k == "--prompt") a.prompt = v();
    else if(k == "--out") a.out = v();
    else if(k == "--json") a.json = v();
    else if(k == "--label") a.label = v();
    else if(k == "--warmup") a.warmup = std::stoi(v());
    else if(k == "--frames") a.frames = std::stoi(v());
    else if(k == "--seed") a.seed = std::stoi(v());
    else if(k == "--denoise-batch") a.denoise_batch = std::stoi(v());
    else if(k == "--add-noise") a.add_noise = std::stoi(v());
    else if(k == "--cuda-graph") a.cuda_graph = std::stoi(v());
    else if(k == "--rife") a.rife = v();
    else if(k == "--rife-exp") a.rife_exp = std::stoi(v());
    else if(k == "--size")
    {
      if(std::sscanf(v().c_str(), "%dx%d", &a.width, &a.height) != 2)
        throw std::runtime_error("--size expects WxH");
    }
    else
      throw std::runtime_error("unknown argument " + k);
  }
  if(a.bundle.empty() || a.model.empty() || a.video.empty() || a.timesteps.empty()
     || a.width <= 0 || a.height <= 0)
    throw std::runtime_error(
        "required: --bundle --model --video --size --timesteps (see the header comment)");
  return a;
}

// "32, 45" / "32 45" -> {32, 45}
std::vector<int> parse_indices(const std::string& s)
{
  std::vector<int> out;
  std::string tok;
  for(char c : s + ",")
  {
    if(c == ',' || c == ' ')
    {
      if(!tok.empty())
        out.push_back(std::clamp(std::stoi(tok), 0, 49));
      tok.clear();
    }
    else
      tok += c;
  }
  return out;
}

double percentile(std::vector<double> v, double p)
{
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
}
}

int main(int argc, char** argv)
try
{
  const Args a = parse(argc, argv);
  const std::vector<int> indices = parse_indices(a.timesteps);
  if(indices.empty())
    throw std::runtime_error("no timesteps");
  const int steps = (int)indices.size();

  // ---- model-type defaults: these mirror StreamDiffusion::createConfiguration ----
  librediffusion_model_type_t model_type;
  int hidden_dim, pad_token;
  std::span<const int> table_t;
  std::span<const streamdiffusion::TimestepParams> table_p;
  if(a.model == "sdxl")
  {
    model_type = MODEL_SDXL_TURBO;
    hidden_dim = 2048;
    pad_token = 0;
    table_t = streamdiffusion::SCHEDULER_STABILITYAI_SDXL_TURBO::TIMESTEP_VALUES;
    table_p = streamdiffusion::SCHEDULER_STABILITYAI_SDXL_TURBO::TIMESTEP_PARAMS;
  }
  else if(a.model == "sd-turbo")
  {
    model_type = MODEL_SD_TURBO;
    hidden_dim = 1024;
    pad_token = 0;
    table_t = streamdiffusion::SCHEDULER_STABILITYAI_SD_TURBO::TIMESTEP_VALUES;
    table_p = streamdiffusion::SCHEDULER_STABILITYAI_SD_TURBO::TIMESTEP_PARAMS;
  }
  else if(a.model == "sd15")
  {
    model_type = MODEL_SD_15;
    hidden_dim = 768;
    pad_token = 49407;
    if(int d = librediffusion::bundle_embedding_dim(a.bundle); d > 0)
      hidden_dim = d;
    table_t = streamdiffusion::SCHEDULER_SIMIANLUO_LCM_DREAMSHAPER_V7::TIMESTEP_VALUES;
    table_p = streamdiffusion::SCHEDULER_SIMIANLUO_LCM_DREAMSHAPER_V7::TIMESTEP_PARAMS;
  }
  else
    throw std::runtime_error("--model must be sd15, sd-turbo or sdxl");
  const float t_scale = librediffusion::bundle_unet_timestep_scale(a.bundle);
  const bool denoise_batch = a.denoise_batch >= 0 ? a.denoise_batch != 0 : steps > 1;

  streamdiffusion::ScheduleArrays sched;
  streamdiffusion::build_schedule(table_t, table_p, indices, t_scale, sched);
  std::fprintf(stderr, "[sd_video] %s %dx%d steps=%d unet_timestep_scale=%g unet t =", a.model.c_str(),
               a.width, a.height, steps, t_scale);
  for(float t : sched.timesteps)
    std::fprintf(stderr, " %g", t);
  std::fprintf(stderr, "\n");

  // ---- config ----
  const std::string B = a.bundle + "/";
  librediffusion_config_handle cfg = nullptr;
  CK(librediffusion_config_create(&cfg));
  CK(librediffusion_config_set_device(cfg, 0));
  CK(librediffusion_config_set_model_type(cfg, model_type));
  CK(librediffusion_config_set_pipeline_mode(cfg, MODE_SINGLE_FRAME));
  CK(librediffusion_config_set_dimensions(cfg, a.width, a.height, a.width / 8, a.height / 8));
  CK(librediffusion_config_set_batch_size(cfg, 1));
  CK(librediffusion_config_set_denoising_steps(cfg, steps));
  CK(librediffusion_config_set_guidance_scale(cfg, 0.f));
  CK(librediffusion_config_set_delta(cfg, 1.f));
  CK(librediffusion_config_set_add_noise(cfg, a.add_noise));
  CK(librediffusion_config_set_denoising_batch(cfg, denoise_batch ? 1 : 0));
  CK(librediffusion_config_set_cfg_type(cfg, SD_CFG_NONE));
  CK(librediffusion_config_set_seed(cfg, (uint64_t)a.seed));
  // the node enables the in-library CUDA graph exactly for 1-step cfg-none single-frame runs
  CK(librediffusion_config_set_cuda_graph(cfg, a.cuda_graph));
  CK(librediffusion_config_set_text_config(cfg, 77, hidden_dim, pad_token));
  if(model_type == MODEL_SDXL_TURBO)
    CK(librediffusion_config_set_sdxl_config(cfg, 1280, 6));
  CK(librediffusion_config_set_unet_engine(cfg, (B + "unet.engine").c_str()));
  CK(librediffusion_config_set_vae_encoder(cfg, (B + "vae_encoder.engine").c_str()));
  CK(librediffusion_config_set_vae_decoder(cfg, (B + "vae_decoder.engine").c_str()));
  CK(librediffusion_config_set_timestep_indices(cfg, indices.data(), indices.size()));

  librediffusion_pipeline_handle pipe = nullptr;
  CK(librediffusion_pipeline_create(cfg, &pipe));
  CK(librediffusion_pipeline_init_all(pipe));

  // ---- prompt (same calls as StreamDiffusion::updatePromptEmbedding) ----
  librediffusion_clip_handle clip1 = nullptr, clip2 = nullptr;
  librediffusion_half_t *embeds = nullptr, *pooled = nullptr, *time_ids = nullptr;
  CK(librediffusion_clip_create((B + "clip.engine").c_str(), 0, &clip1));
  if(model_type == MODEL_SDXL_TURBO)
  {
    CK(librediffusion_clip_create((B + "clip2.engine").c_str(), 0, &clip2));
    CK(librediffusion_clip_compute_embeddings_sdxl(
        clip1, clip2, a.prompt.c_str(), 1, a.height, a.width, nullptr, &embeds, &pooled,
        &time_ids));
    CK(librediffusion_prepare_sdxl_conditioning(pipe, pooled, time_ids));
  }
  else
    CK(librediffusion_clip_compute_embeddings(clip1, a.prompt.c_str(), pad_token, nullptr, &embeds));
  CK(librediffusion_prepare_embeds(pipe, embeds, 77, hidden_dim));
  CK(librediffusion_prepare_scheduler(
      pipe, sched.timesteps.data(), sched.alpha_prod_t_sqrt.data(), sched.beta_prod_t_sqrt.data(),
      sched.c_skip.data(), sched.c_out.data(), sched.size()));
  if(sched.has_noise_schedule())
    CK(librediffusion_prepare_noise_schedule(
        pipe, sched.noise_alpha_prod_t_sqrt.data(), sched.noise_beta_prod_t_sqrt.data(),
        sched.size()));

  // ---- video ----
  const size_t frame_bytes = (size_t)a.width * a.height * 4;
  std::vector<uint8_t> video;
  {
    std::ifstream f{a.video, std::ios::binary};
    if(!f)
      throw std::runtime_error("cannot open " + a.video);
    video.assign(std::istreambuf_iterator<char>{f}, {});
  }
  const size_t n_video = video.size() / frame_bytes;
  if(n_video == 0 || video.size() % frame_bytes)
    throw std::runtime_error(a.video + " is not a whole number of WxH RGBA frames");
  const int n_timed = a.frames > 0 ? a.frames : (int)n_video;

  std::ofstream out;
  if(!a.out.empty())
  {
    out.open(a.out, std::ios::binary);
    if(!out)
      throw std::runtime_error("cannot write " + a.out);
  }

  // ---- RIFE (optional) ----
  librediffusion_rife_handle rife = nullptr;
  std::vector<uint8_t> rife_frames, prev_out;
  if(a.rife_exp > 0)
  {
    if(a.rife.empty())
      throw std::runtime_error("--rife-exp needs --rife PLAN");
    rife = librediffusion_rife_create(a.rife.c_str(), 0);
    if(!rife)
      throw std::runtime_error("cannot load RIFE engine " + a.rife);
    librediffusion_rife_set_enabled(rife, 1);
    librediffusion_rife_set_interpolation_exp(rife, a.rife_exp);
    rife_frames.resize(librediffusion_rife_required_out_bytes(rife, a.height, a.width));
  }
  const int sub = 1 << a.rife_exp;

  std::vector<uint8_t> rgba_out(frame_bytes);
  std::vector<double> ms, rife_ms;
  ms.reserve(n_timed);
  const int total = a.warmup + n_timed;
  for(int i = 0; i < total; i++)
  {
    const uint8_t* in = video.data() + (size_t)(i % n_video) * frame_bytes;
    const auto t0 = clk::now();
    CK(librediffusion_img2img(pipe, in, rgba_out.data(), a.width, a.height));
    const auto t1 = clk::now();
    int n_sub = 0;
    if(rife && !prev_out.empty())
      CK(librediffusion_rife_interpolate_sized(
          rife, prev_out.data(), rgba_out.data(), a.height, a.width, rife_frames.data(),
          rife_frames.size(), &n_sub));
    const auto t2 = clk::now();
    if(rife)
      prev_out = rgba_out;
    if(i < a.warmup)
      continue;
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if(rife)
      rife_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    if(out)
    {
      if(n_sub == sub)
        out.write((const char*)rife_frames.data(), (std::streamsize)(sub * frame_bytes));
      else
        for(int k = 0; k < sub; k++)
          out.write((const char*)rgba_out.data(), frame_bytes);
    }
  }

  double sum = 0;
  for(double v : ms)
    sum += v;
  const double mean = sum / ms.size();
  const double fps = 1000. / mean, p50 = percentile(ms, .5), p99 = percentile(ms, .99);
  std::printf(
      "%s: %zu frames  mean %.2f ms  -> %.1f fps   p50 %.2f ms  p99 %.2f ms (%.1f fps)\n",
      a.label.empty() ? a.bundle.c_str() : a.label.c_str(), ms.size(), mean, fps, p50, p99,
      1000. / p99);
  double rife_mean = 0, key_fps = fps, display_fps = fps;
  if(rife)
  {
    for(double v : rife_ms)
      rife_mean += v;
    rife_mean /= rife_ms.size();
    key_fps = 1000. / (mean + rife_mean);
    display_fps = key_fps * sub;
    std::printf("  + RIFE x%d: %.2f ms per frame -> %.1f diffused fps, %.1f displayed fps\n", sub,
                rife_mean, key_fps, display_fps);
  }
  if(!a.json.empty())
  {
    std::ofstream j{a.json};
    j << "{\"label\":\"" << a.label << "\",\"bundle\":\"" << a.bundle << "\",\"model\":\""
      << a.model << "\",\"width\":" << a.width << ",\"height\":" << a.height
      << ",\"steps\":" << steps << ",\"timesteps\":\"" << a.timesteps
      << "\",\"unet_timestep_scale\":" << t_scale << ",\"denoise_batch\":" << denoise_batch
      << ",\"frames\":" << ms.size() << ",\"mean_ms\":" << mean << ",\"fps\":" << fps
      << ",\"p50_ms\":" << p50 << ",\"p99_ms\":" << p99 << ",\"rife_exp\":" << a.rife_exp
      << ",\"rife_ms\":" << rife_mean << ",\"key_fps\":" << key_fps
      << ",\"display_fps\":" << display_fps << "}\n";
  }

  if(rife)
    librediffusion_rife_destroy(rife);
  librediffusion_cuda_free(embeds);
  librediffusion_cuda_free(pooled);
  librediffusion_cuda_free(time_ids);
  librediffusion_clip_destroy(clip1);
  librediffusion_clip_destroy(clip2);
  librediffusion_pipeline_destroy(pipe);
  librediffusion_config_destroy(cfg);
  return 0;
}
catch(const std::exception& e)
{
  std::fprintf(stderr, "sd_video_harness: %s\n", e.what());
  return 1;
}
