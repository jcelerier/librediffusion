// C++ validation harness — mirrors the Python validation matrix, driving librediffusion
// through its C API and comparing the C++ TRT inference against the Python pure-PyTorch
// goldens at the seams the C API exposes (encode_image, predict_x0_batch, decode_image,
// final output).
//
// Per the user's design (see memory cpp-harness-design):
//   - text embeddings are computed FROM SCRATCH in C++ via the CLIP engine + tokenizer
//   - the scheduler tables are computed C++-side (read from precomputed headers / engine)
//   - init_noise is injected from the golden cell (the C++ RNG cannot reproduce torch's,
//     so to compare denoise output we must seed the same noise)
//   - comparison uses the latent profile (cosine>=0.999, rel<=0.05) for latents and the
//     image profile (psnr>=30, ssim>=0.95) for decoded images — fp16-TRT-vs-fp32-torch.
//
// Usage:
//   harness --cell <golden_cell_dir> --engines <engine_set_dir> [--clip <clip.engine>]
//           [--clip2 <clip2.engine>] --out <dump_dir>
//   harness --matrix <goldens/matrix> --engine-index <index.json> --out <dump_root>
//
// This first version drives the OUTER seams the C API exposes. The inner per-step
// unet_step / scheduler_step_batch seams live inside predict_x0_batch and are not yet
// reachable through the C API; comparing them needs a small C-API dump-hook extension
// (tracked as a follow-up).

#include "librediffusion_c.h"

#include "compare.hpp"
#include "json_min.hpp"
#include "npy.hpp"
#include "png_min.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h> // _exit
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#define CK(call)                                                                 \
  do                                                                             \
  {                                                                              \
    librediffusion_error_t _e = (call);                                          \
    if(_e != LIBREDIFFUSION_SUCCESS)                                             \
    {                                                                            \
      std::cerr << "C-API error " << _e << " (" << librediffusion_error_string(_e) \
                << ") at " << #call << " [" << __FILE__ << ":" << __LINE__ << "]\n"; \
      throw std::runtime_error("C-API call failed");                            \
    }                                                                            \
  } while(0)

static std::vector<librediffusion_half_t> floats_to_half(const std::vector<float>& f)
{
  std::vector<librediffusion_half_t> h(f.size());
  for(size_t i = 0; i < f.size(); i++)
    h[i] = librediffusion_float_to_half(f[i]);
  return h;
}

static std::vector<float> half_to_floats(const std::vector<librediffusion_half_t>& h)
{
  std::vector<float> f(h.size());
  for(size_t i = 0; i < h.size(); i++)
    f[i] = librediffusion_half_to_float(h[i]);
  return f;
}

// Golden cell config, parsed from config.json.
struct CellConfig
{
  std::string name;
  std::string model;
  std::vector<int> t_index_list;
  std::string mode;       // img2img | txt2img
  std::string cfg_type;   // none|self|initialize|full
  float guidance_scale = 1.2f;
  float delta = 1.0f;
  bool do_add_noise = true;
  bool use_denoising_batch = true;
  std::string scheduler;  // lcm|tcd
  int width = 512, height = 512;
  int seed = 1234;
  std::string prompt, negative_prompt, input_image;
  int n_runs = 1;
  float controlnet_scale = 0.6f;  // matches the controlnet golden (CONTROL_SCALE)
};

static CellConfig load_cell_config(const fs::path& cell)
{
  auto j = jsonm::parse_file((cell / "config.json").string());
  CellConfig c;
  c.name = j->at("name")->as_str();
  c.model = j->at("model_id_or_path")->as_str();
  for(auto& t : j->at("t_index_list")->arr)
    c.t_index_list.push_back(t->as_int());
  c.mode = j->at("mode")->as_str();
  c.cfg_type = j->at("cfg_type")->as_str();
  c.guidance_scale = (float)j->at("guidance_scale")->as_num();
  c.delta = (float)j->at("delta")->as_num();
  c.do_add_noise = j->at("do_add_noise")->as_bool();
  c.use_denoising_batch = j->at("use_denoising_batch")->as_bool();
  c.scheduler = j->at("scheduler")->as_str();
  c.width = j->at("width")->as_int();
  c.height = j->at("height")->as_int();
  c.seed = j->at("seed")->as_int();
  c.prompt = j->at("prompt")->as_str();
  c.negative_prompt = j->has("negative_prompt") ? j->at("negative_prompt")->as_str() : "";
  c.input_image = j->has("input_image") && !j->at("input_image")->is_null()
                      ? j->at("input_image")->as_str() : "";
  c.n_runs = j->has("n_runs") ? j->at("n_runs")->as_int() : 1;
  return c;
}

static librediffusion_cfg_type_t cfg_enum(const std::string& s)
{
  if(s == "none") return SD_CFG_NONE;
  if(s == "full") return SD_CFG_FULL;
  if(s == "self") return SD_CFG_SELF;
  if(s == "initialize") return SD_CFG_INITIALIZE;
  throw std::runtime_error("unknown cfg_type " + s);
}

// Load a golden tensor (.npy) given a cell dir + role substring (searches manifest order).
static bool load_golden(const fs::path& cell, const std::string& boundary,
                        const std::string& role, npy::Array& out)
{
  // Files are named NNN_<boundary>__<role>.npy
  for(auto& e : fs::directory_iterator(cell))
  {
    auto fn = e.path().filename().string();
    if(fn.find("_" + boundary + "__" + role + ".npy") != std::string::npos)
    {
      out = npy::load(e.path().string());
      return true;
    }
  }
  return false;
}

// Read a PNG into an RGBA8 buffer using the library's own resize/convert is overkill;
// for img2img we need the input image as RGBA. We rely on the golden's dumped
// encode_image input tensor instead of decoding the PNG, so the harness has no PNG dep.

struct SeamResult
{
  std::string boundary;
  std::string role;
  bool compared = false;
  bool pass = false;
  std::string detail;
};

// ---------------------------------------------------------------------------
// run one cell
// ---------------------------------------------------------------------------

struct EngineSet
{
  std::string unet, vae_encoder, vae_decoder, clip, clip2;
  std::vector<std::string> controlnets;  // 0+ controlnet engines (controlnet.engine + extras via --controlnet2)
  bool is_sdxl = false;
  int pad_token = 49407; // SD1.5 default; SD-Turbo/SDXL-Turbo use 0
};

static std::vector<SeamResult> run_cell(
    const fs::path& cell, const EngineSet& eng, const fs::path& out_dir, bool verbose)
{
  std::vector<SeamResult> results;
  CellConfig c = load_cell_config(cell);
  fs::create_directories(out_dir);

  if(verbose)
    std::cerr << "[cell] " << c.name << "  mode=" << c.mode << " cfg=" << c.cfg_type
              << " steps=" << c.t_index_list.size() << " " << c.width << "x" << c.height
              << " n_runs=" << c.n_runs << "\n";

  const int latent_w = c.width / 8, latent_h = c.height / 8;
  const int steps = (int)c.t_index_list.size();
  // config_.batch_size is the FRAME-buffer size (number of NEW frames per call = 1 for a
  // single stream), NOT the number of denoising steps. The pipeline builds the UNet batch
  // internally as total_batch = batch_size + (denoising_steps-1)*frame_buffer_size, so for
  // a 2-step single stream the UNet sees batch 2. encode_image/decode/predict all take a
  // single (batch_size=1) frame; the streaming buffer supplies the rest.
  const int frame_buffer = 1;
  const int batch = frame_buffer;            // config batch_size
  const int unet_total_batch = batch + (steps - 1) * frame_buffer;  // for reference/dumps

  // --- config ---
  librediffusion_config_handle cfg = nullptr;
  CK(librediffusion_config_create(&cfg));
  CK(librediffusion_config_set_device(cfg, 0));
  CK(librediffusion_config_set_model_type(
      cfg, eng.is_sdxl ? MODEL_SDXL_TURBO : MODEL_SD_15));
  CK(librediffusion_config_set_dimensions(cfg, c.width, c.height, latent_w, latent_h));
  CK(librediffusion_config_set_batch_size(cfg, batch));
  CK(librediffusion_config_set_denoising_steps(cfg, steps));
  CK(librediffusion_config_set_frame_buffer_size(cfg, 1));
  CK(librediffusion_config_set_guidance_scale(cfg, c.guidance_scale));
  CK(librediffusion_config_set_delta(cfg, c.delta));
  CK(librediffusion_config_set_add_noise(cfg, c.do_add_noise ? 1 : 0));
  CK(librediffusion_config_set_denoising_batch(cfg, c.use_denoising_batch ? 1 : 0));
  CK(librediffusion_config_set_seed(cfg, (uint64_t)c.seed));
  CK(librediffusion_config_set_cfg_type(cfg, cfg_enum(c.cfg_type)));
  // LRD_CUDA_GRAPH=1 -> exercise the .so's 1-step CUDA-graph path while validating seams (numerics must match
  // the golden with graph on, proving capture/replay is correct end-to-end). No-op for non-1-step cells.
  if(std::getenv("LRD_CUDA_GRAPH") && librediffusion_config_set_cuda_graph)
    CK(librediffusion_config_set_cuda_graph(cfg, 1));

  // hidden_dim is read from the golden prompt_embeds last dim (768 SD1.5, 1024 SD2.1/turbo,
  // 2048 SDXL) so we never hardcode per-model. pad_token comes from the caller (--pad-token):
  // 49407 for SD1.5, 0 for SD-Turbo/SDXL-Turbo.
  int hidden_dim = eng.is_sdxl ? 2048 : 768;
  {
    npy::Array pe;
    if(load_golden(cell, "static", "prompt_embeds", pe) && pe.shape.size() >= 1)
      hidden_dim = (int)pe.shape.back();
  }
  CK(librediffusion_config_set_text_config(cfg, 77, hidden_dim, eng.pad_token));
  if(eng.is_sdxl)
    CK(librediffusion_config_set_sdxl_config(cfg, 1280, 6));

  CK(librediffusion_config_set_unet_engine(cfg, eng.unet.c_str()));
  CK(librediffusion_config_set_vae_encoder(cfg, eng.vae_encoder.c_str()));
  CK(librediffusion_config_set_vae_decoder(cfg, eng.vae_decoder.c_str()));

  // Optional: on-device IP-Adapter image encoder. When LRD_IPADAPTER_ONDEVICE points at a raw RGBA
  // style image (and the engine dir has clip_image_encoder.engine + ip_image_proj.engine), configure
  // the on-device encoder so we can feed the RAW image instead of golden tokens (end-to-end gate).
  const char* ip_ondevice = std::getenv("LRD_IPADAPTER_ONDEVICE");
  if(ip_ondevice && librediffusion_config_set_ipadapter_image_encoder)
  {
    std::string ed = std::filesystem::path(eng.unet).parent_path().string();
    std::string enc = ed + "/clip_image_encoder.engine";
    std::string proj = ed + "/ip_image_proj.engine";
    CK(librediffusion_config_set_ipadapter_image_encoder(cfg, enc.c_str(), proj.c_str()));
  }
  for(size_t ci = 0; ci < eng.controlnets.size(); ci++)
  {
    // LRD_CN_SCALE overrides the cell's conditioning scale (debug probe: scale 0 isolates whether
    // a divergence comes from the residual path or from the control-aware UNet itself).
    float cn_scale = c.controlnet_scale;
    if(const char* e = std::getenv("LRD_CN_SCALE")) cn_scale = (float)std::atof(e);
    int cn_idx = librediffusion_config_add_controlnet(cfg, eng.controlnets[ci].c_str(), cn_scale);
    if(cn_idx < 0)
      throw std::runtime_error("config_add_controlnet failed");
  }
  CK(librediffusion_config_set_timestep_indices(
      cfg, c.t_index_list.data(), c.t_index_list.size()));

  // --- pipeline ---
  librediffusion_pipeline_handle pipe = nullptr;
  CK(librediffusion_pipeline_create(cfg, &pipe));
  CK(librediffusion_pipeline_init_all(pipe));
  librediffusion_stream_t stream = librediffusion_pipeline_get_stream(pipe);

  // --- embeddings: compute from scratch via the CLIP engine(s) ---
  // The prompt embedding is a single [1,77,hidden] device tensor; prepare_embeds copies
  // config_.batch_size*77*hidden d2d, so we tile the single embedding to `batch` rows first.
  librediffusion_clip_handle clip = nullptr;
  librediffusion_clip_handle clip2 = nullptr;  // SDXL second encoder
  librediffusion_half_t* prompt_embeds = nullptr;
  librediffusion_half_t* prompt_embeds_tiled = nullptr;
  librediffusion_half_t* sdxl_pooled = nullptr;   // SDXL pooled text_embeds (1280)
  librediffusion_half_t* sdxl_time_ids = nullptr; // SDXL time_ids (6)

  CK(librediffusion_clip_create(eng.clip.c_str(), 0, &clip));

  if(eng.is_sdxl)
  {
    // SDXL: dual-CLIP. clip_compute_embeddings_sdxl returns the concatenated 2048-dim
    // encoder_hidden_states + pooled text_embeds (1280) + time_ids (6), all device pointers.
    // Then prepare_embeds(2048) + prepare_sdxl_conditioning(pooled, time_ids).
    CK(librediffusion_clip_create(eng.clip2.c_str(), 0, &clip2));
    CK(librediffusion_clip_compute_embeddings_sdxl(
        clip, clip2, c.prompt.c_str(), 1, c.height, c.width, stream,
        &prompt_embeds, &sdxl_pooled, &sdxl_time_ids));
  }
  else
  {
    CK(librediffusion_clip_compute_embeddings(clip, c.prompt.c_str(), eng.pad_token, stream,
                                              &prompt_embeds));
  }

  const size_t one_embed = (size_t)77 * hidden_dim;
  // Stage the single embedding to host, then build a tiled host buffer and upload once.
  std::vector<librediffusion_half_t> host_one(one_embed);
  CK(librediffusion_cuda_memcpy_d2h(host_one.data(), prompt_embeds,
                                    one_embed * sizeof(librediffusion_half_t)));
  std::vector<librediffusion_half_t> host_tiled((size_t)batch * one_embed);
  for(int b = 0; b < batch; b++)
    std::memcpy(host_tiled.data() + b * one_embed, host_one.data(),
                one_embed * sizeof(librediffusion_half_t));
  prompt_embeds_tiled = (librediffusion_half_t*)librediffusion_cuda_malloc(
      (size_t)batch * one_embed * sizeof(librediffusion_half_t));
  CK(librediffusion_cuda_memcpy_h2d(prompt_embeds_tiled, host_tiled.data(),
                                    host_tiled.size() * sizeof(librediffusion_half_t)));
  CK(librediffusion_prepare_embeds(pipe, prompt_embeds_tiled, 77, hidden_dim));

  if(eng.is_sdxl)
  {
    // pooled text_embeds + time_ids are single-row from clip_compute_embeddings_sdxl;
    // prepare_sdxl_conditioning copies config_.batch_size * pooled_dim / *6 d2d, so tile to batch.
    auto tile_to_batch = [&](librediffusion_half_t* src, int dim) -> librediffusion_half_t* {
      std::vector<librediffusion_half_t> h1(dim), hb((size_t)batch * dim);
      CK(librediffusion_cuda_memcpy_d2h(h1.data(), src, dim * sizeof(librediffusion_half_t)));
      for(int b = 0; b < batch; b++)
        std::memcpy(hb.data() + (size_t)b * dim, h1.data(), dim * sizeof(librediffusion_half_t));
      auto* d = (librediffusion_half_t*)librediffusion_cuda_malloc(
          (size_t)batch * dim * sizeof(librediffusion_half_t));
      CK(librediffusion_cuda_memcpy_h2d(d, hb.data(), (size_t)batch * dim * sizeof(librediffusion_half_t)));
      return d;
    };
    librediffusion_half_t* pooled_tiled = tile_to_batch(sdxl_pooled, 1280);
    librediffusion_half_t* time_ids_tiled = tile_to_batch(sdxl_time_ids, 6);
    CK(librediffusion_prepare_sdxl_conditioning(pipe, pooled_tiled, time_ids_tiled));
    librediffusion_cuda_free(pooled_tiled);
    librediffusion_cuda_free(time_ids_tiled);
  }

  if(c.cfg_type == "full" || c.cfg_type == "initialize")
  {
    // Negative embeds. For SDXL the negative encoder_hidden_states also come from the dual-CLIP
    // path (2048-dim); reuse clip_compute_embeddings_sdxl for the negative prompt and take the
    // hidden_states (the pooled/time_ids negative half is handled inside prepare for cfg).
    librediffusion_half_t* neg = nullptr;
    if(eng.is_sdxl)
    {
      librediffusion_half_t* neg_pooled = nullptr;
      librediffusion_half_t* neg_time = nullptr;
      CK(librediffusion_clip_compute_embeddings_sdxl(
          clip, clip2, c.negative_prompt.c_str(), 1, c.height, c.width, stream,
          &neg, &neg_pooled, &neg_time));
      CK(librediffusion_prepare_negative_embeds(pipe, neg, 77, hidden_dim));
      librediffusion_cuda_free(neg_pooled);
      librediffusion_cuda_free(neg_time);
    }
    else
    {
      CK(librediffusion_clip_compute_embeddings(clip, c.negative_prompt.c_str(), eng.pad_token,
                                                stream, &neg));
      CK(librediffusion_prepare_negative_embeds(pipe, neg, 77, hidden_dim));
    }
    librediffusion_cuda_free(neg);
  }

  // --- scheduler: inject the golden's tables (computed from scratch C++-side is a TODO;
  //     for now reuse the golden scheduler statics so the seam comparison isolates the
  //     UNet/VAE compute rather than scheduler-table reproduction). ---
  {
    npy::Array sub_t, alpha, beta, cskip, cout;
    bool ok = load_golden(cell, "static", "sub_timesteps_tensor", sub_t)
              && load_golden(cell, "static", "alpha_prod_t_sqrt", alpha)
              && load_golden(cell, "static", "beta_prod_t_sqrt", beta)
              && load_golden(cell, "static", "c_skip", cskip)
              && load_golden(cell, "static", "c_out", cout);
    if(!ok)
      throw std::runtime_error("cell missing scheduler static tensors: " + c.name);
    CK(librediffusion_prepare_scheduler(pipe, sub_t.data.data(), alpha.data.data(),
                                        beta.data.data(), cskip.data.data(),
                                        cout.data.data(), sub_t.size()));
  }

  // --- init noise: inject golden init_noise so RNG matches ---
  // set_init_noise does a DEVICE-to-device copy of init_noise_->size() elements, so the
  // input must be a device pointer with at least that many half elements. The golden
  // init_noise is [batch,4,lh,lw] fp32; upload it (tiled/truncated to the internal size).
  librediffusion_half_t* d_init_noise = nullptr;
  {
    npy::Array init_noise;
    if(load_golden(cell, "static", "init_noise", init_noise))
    {
      auto h = floats_to_half(init_noise.data);
      d_init_noise = (librediffusion_half_t*)librediffusion_cuda_malloc(
          h.size() * sizeof(librediffusion_half_t));
      CK(librediffusion_cuda_memcpy_h2d(d_init_noise, h.data(),
                                        h.size() * sizeof(librediffusion_half_t)));
      CK(librediffusion_set_init_noise(pipe, d_init_noise));
    }
  }

  // --- ControlNet: inject each net's golden control image (isolate engine+injection from preproc).
  // Net 0 cond = static/controlnet_cond; net i>0 = static/controlnet_cond<i> (e.g. controlnet_cond1).
  std::vector<librediffusion_half_t*> d_cn_conds;
  for(size_t ci = 0; ci < eng.controlnets.size(); ci++)
  {
    std::string role = (ci == 0) ? "controlnet_cond" : ("controlnet_cond" + std::to_string(ci));
    npy::Array cond;  // [1,3,H,W] fp32 in [0,1]
    if(!load_golden(cell, "static", role, cond))
      throw std::runtime_error("controlnet " + std::to_string(ci) + " golden missing static/" + role
                               + ": " + c.name);
    auto h = floats_to_half(cond.data);
    auto* d = (librediffusion_half_t*)librediffusion_cuda_malloc(
        h.size() * sizeof(librediffusion_half_t));
    CK(librediffusion_cuda_memcpy_h2d(d, h.data(), h.size() * sizeof(librediffusion_half_t)));
    CK(librediffusion_set_controlnet_cond(pipe, (int)ci, d, c.height, c.width));
    d_cn_conds.push_back(d);
  }

  // --- IP-Adapter: feed the golden image tokens + per-layer scale (engine auto-detected as IP). ---
  librediffusion_half_t* d_ip_pos = nullptr;
  librediffusion_half_t* d_ip_neg = nullptr;
  {
    npy::Array pos, neg, scale;  // pos/neg [N,dim] fp32; scale [L] fp32
    if(load_golden(cell, "static", "ipadapter_tokens_pos", pos))
    {
      int N = (int)pos.shape[0], dim = (int)pos.shape[1];
      if(ip_ondevice && librediffusion_set_ipadapter_image)
      {
        // On-device path: read the raw RGBA style image and let the pipeline compute the tokens via
        // the CLIP image encoder + projection engines (the gate for the end-to-end node path).
        std::ifstream sf(ip_ondevice, std::ios::binary);
        std::vector<uint8_t> rgba((size_t)c.height * c.width * 4);
        sf.read((char*)rgba.data(), rgba.size());
        CK(librediffusion_set_ipadapter_image(pipe, rgba.data(), c.height, c.width));
        std::printf("[ipadapter] on-device tokens from %s (%dx%d)\n", ip_ondevice, c.width, c.height);
      }
      else
      {
        auto hp = floats_to_half(pos.data);
        d_ip_pos = (librediffusion_half_t*)librediffusion_cuda_malloc(hp.size() * sizeof(librediffusion_half_t));
        CK(librediffusion_cuda_memcpy_h2d(d_ip_pos, hp.data(), hp.size() * sizeof(librediffusion_half_t)));
        if(load_golden(cell, "static", "ipadapter_tokens_neg", neg))
        {
          auto hn = floats_to_half(neg.data);
          d_ip_neg = (librediffusion_half_t*)librediffusion_cuda_malloc(hn.size() * sizeof(librediffusion_half_t));
          CK(librediffusion_cuda_memcpy_h2d(d_ip_neg, hn.data(), hn.size() * sizeof(librediffusion_half_t)));
        }
        CK(librediffusion_set_ipadapter_tokens(pipe, d_ip_pos, d_ip_neg, N, dim));
      }
      if(load_golden(cell, "static", "ipadapter_scale", scale))
        CK(librediffusion_set_ipadapter_scale_vector(pipe, scale.data.data(), (int)scale.data.size()));
    }
  }

  // --- drive the seams ---
  // We compare three outer seams the C API exposes: encode_image (img2img only),
  // predict_x0_batch, decode_image, and the final image.
  const int latent_elems = batch * 4 * latent_h * latent_w;
  const int img_elems = batch * 3 * c.height * c.width;

  // device buffers
  auto dmalloc = [](size_t bytes) { return librediffusion_cuda_malloc(bytes); };

  std::vector<float> golden_in_image_f;
  npy::Array g_in_image;
  bool have_in_image = (c.mode == "img2img")
                       && load_golden(cell, "encode_image", "in_image", g_in_image);

  // Streaming convergence: run n_runs-1 warmup frames, dump only the last.
  for(int run = 0; run < c.n_runs; run++)
  {
    const bool dump = (run == c.n_runs - 1);

    if(c.mode == "img2img")
    {
      if(!have_in_image)
        throw std::runtime_error("img2img cell missing encode_image/in_image golden: " + c.name);

      // upload input image (golden in_image is [1,3,H,W] fp32, normalized [-1,1]).
      // encode_image feeds config_.batch_size (=batch) images to the VAE engine, so tile
      // the single golden image to `batch` rows; the denoising batch uses the same frame.
      const size_t one_img = (size_t)3 * c.height * c.width;
      std::vector<float> tiled_img((size_t)batch * one_img);
      for(int b = 0; b < batch; b++)
        std::memcpy(tiled_img.data() + b * one_img, g_in_image.data.data(),
                    one_img * sizeof(float));
      void* d_img_in = dmalloc(tiled_img.size() * sizeof(float));
      CK(librediffusion_cuda_memcpy_h2d(d_img_in, tiled_img.data(),
                                        tiled_img.size() * sizeof(float)));

      // encode_image (float input -> half latent)
      void* d_latent = dmalloc((size_t)latent_elems * sizeof(librediffusion_half_t));
      CK(librediffusion_encode_image_float(pipe, (const float*)d_img_in,
                                           (librediffusion_half_t*)d_latent, stream));
      CK(librediffusion_pipeline_synchronize(pipe));

      if(dump)
      {
        std::vector<librediffusion_half_t> hbuf(latent_elems);
        CK(librediffusion_cuda_memcpy_d2h(hbuf.data(), d_latent,
                                          latent_elems * sizeof(librediffusion_half_t)));
        auto fbuf = half_to_floats(hbuf);
        npy::save((out_dir / "cpp_encode_image__out_latent.npy").string(), fbuf.data(),
                  {(size_t)batch, 4, (size_t)latent_h, (size_t)latent_w});
      }

      // predict_x0_batch (half latent -> half x0_pred)
      void* d_x0 = dmalloc((size_t)latent_elems * sizeof(librediffusion_half_t));
      CK(librediffusion_predict_x0_batch(pipe, (const librediffusion_half_t*)d_latent,
                                         (librediffusion_half_t*)d_x0, stream));
      CK(librediffusion_pipeline_synchronize(pipe));

      if(dump)
      {
        std::vector<librediffusion_half_t> hbuf(latent_elems);
        CK(librediffusion_cuda_memcpy_d2h(hbuf.data(), d_x0,
                                          latent_elems * sizeof(librediffusion_half_t)));
        auto fbuf = half_to_floats(hbuf);
        npy::save((out_dir / "cpp_predict_x0_batch__out_x_0_pred.npy").string(),
                  fbuf.data(), {(size_t)batch, 4, (size_t)latent_h, (size_t)latent_w});
      }

      // decode_image (half latent -> half image). Decode only the first latent (B=1 output).
      void* d_img_out = dmalloc((size_t)(3 * c.height * c.width) * sizeof(librediffusion_half_t));
      CK(librediffusion_decode_latent(pipe, (const librediffusion_half_t*)d_x0,
                                      (librediffusion_half_t*)d_img_out, stream));
      CK(librediffusion_pipeline_synchronize(pipe));

      if(dump)
      {
        std::vector<librediffusion_half_t> hbuf(3 * c.height * c.width);
        CK(librediffusion_cuda_memcpy_d2h(hbuf.data(), d_img_out,
                                          hbuf.size() * sizeof(librediffusion_half_t)));
        auto fbuf = half_to_floats(hbuf);
        npy::save((out_dir / "cpp_decode_image__out_image.npy").string(), fbuf.data(),
                  {1, 3, (size_t)c.height, (size_t)c.width});
        pngmin::write_nchw_norm((out_dir / "cpp_output.png").string(), fbuf.data(),
                                c.height, c.width);
      }

      librediffusion_cuda_free(d_img_in);
      librediffusion_cuda_free(d_latent);
      librediffusion_cuda_free(d_x0);
      librediffusion_cuda_free(d_img_out);
    }
    else
    {
      // txt2img: high-level GPU path; just produce the final image for the perceptual gate.
      void* d_img_out = dmalloc((size_t)img_elems * sizeof(librediffusion_half_t));
      CK(librediffusion_txt2img_gpu(pipe, (librediffusion_half_t*)d_img_out, stream));
      CK(librediffusion_pipeline_synchronize(pipe));
      if(dump)
      {
        std::vector<librediffusion_half_t> hbuf(3 * c.height * c.width);
        CK(librediffusion_cuda_memcpy_d2h(hbuf.data(), d_img_out,
                                          hbuf.size() * sizeof(librediffusion_half_t)));
        auto fbuf = half_to_floats(hbuf);
        npy::save((out_dir / "cpp_decode_image__out_image.npy").string(), fbuf.data(),
                  {1, 3, (size_t)c.height, (size_t)c.width});
        pngmin::write_nchw_norm((out_dir / "cpp_output.png").string(), fbuf.data(),
                                c.height, c.width);
      }
      librediffusion_cuda_free(d_img_out);
    }
  }

  librediffusion_cuda_free(prompt_embeds);
  librediffusion_cuda_free(prompt_embeds_tiled);
  if(sdxl_pooled)
    librediffusion_cuda_free(sdxl_pooled);
  if(sdxl_time_ids)
    librediffusion_cuda_free(sdxl_time_ids);
  if(d_init_noise)
    librediffusion_cuda_free(d_init_noise);
  librediffusion_clip_destroy(clip);
  if(clip2)
    librediffusion_clip_destroy(clip2);
  librediffusion_pipeline_destroy(pipe);
  librediffusion_config_destroy(cfg);

  // --- compare dumped seams against goldens ---
  auto compare_seam = [&](const std::string& boundary, const std::string& role,
                          const std::string& cpp_file) {
    SeamResult sr;
    sr.boundary = boundary;
    sr.role = role;
    npy::Array golden;
    if(!load_golden(cell, boundary, role, golden))
    {
      sr.detail = "no golden";
      results.push_back(sr);
      return;
    }
    fs::path cpp_path = out_dir / cpp_file;
    if(!fs::exists(cpp_path))
    {
      sr.detail = "no cpp dump";
      results.push_back(sr);
      return;
    }
    npy::Array cppv = npy::load(cpp_path.string());
    sr.compared = true;
    auto prof = cmp::boundary_to_profile(boundary, role);
    if(prof == cmp::Profile::Image)
    {
      // golden out_image is fp32 normalized [-1,1] (decode_image) or uint8 (output).
      double dr = 2.0; // [-1,1]
      cmp::PerceptualResult pr;
      pr.psnr = cmp::psnr(golden.data, cppv.data, dr);
      pr.ssim = cmp::ssim_global(golden.data, cppv.data, dr);
      auto v = cmp::judge_image(pr);
      sr.pass = v.pass;
      sr.detail = v.detail;
    }
    else
    {
      auto nr = cmp::numeric(golden.data, cppv.data);
      auto v = cmp::judge_numeric(nr, prof);
      sr.pass = v.pass;
      sr.detail = v.detail;
    }
    results.push_back(sr);
  };

  if(c.mode == "img2img")
  {
    compare_seam("encode_image", "out_latent", "cpp_encode_image__out_latent.npy");
    compare_seam("predict_x0_batch", "out_x_0_pred", "cpp_predict_x0_batch__out_x_0_pred.npy");
    compare_seam("decode_image", "out_image", "cpp_decode_image__out_image.npy");
  }
  else
  {
    compare_seam("decode_image", "out_image", "cpp_decode_image__out_image.npy");
  }

  return results;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage()
{
  std::cerr <<
      "Usage:\n"
      "  harness --cell <golden_cell_dir> --engines <engine_set_dir>\n"
      "          [--clip <clip.engine>] [--clip2 <clip2.engine>] [--sdxl] --out <dir>\n";
}

int main(int argc, char** argv)
{
  std::map<std::string, std::string> args;
  bool sdxl = false, verbose = true;
  for(int i = 1; i < argc; i++)
  {
    std::string a = argv[i];
    if(a == "--sdxl") { sdxl = true; continue; }
    if(a == "--quiet") { verbose = false; continue; }
    if(a.rfind("--", 0) == 0 && i + 1 < argc)
      args[a.substr(2)] = argv[++i];
  }

  if(!args.count("cell") || !args.count("engines") || !args.count("out"))
  {
    usage();
    return 2;
  }

  fs::path cell = args["cell"];
  fs::path engdir = args["engines"];
  fs::path out = args["out"];

  EngineSet eng;
  eng.is_sdxl = sdxl;
  // pad_token: 49407 for SD1.5, 0 for SD-Turbo/SDXL-Turbo (driver passes --pad-token).
  eng.pad_token = args.count("pad-token") ? std::stoi(args["pad-token"]) : (sdxl ? 0 : 49407);
  eng.unet = (engdir / "unet.engine").string();
  eng.vae_encoder = (engdir / "vae_encoder.engine").string();
  eng.vae_decoder = (engdir / "vae_decoder.engine").string();
  eng.clip = args.count("clip") ? args["clip"] : (engdir / "clip.engine").string();
  if(sdxl)
    eng.clip2 = args.count("clip2") ? args["clip2"] : (engdir / "clip2.engine").string();
  // ControlNet: auto-enable if a controlnet.engine sits in the engine dir (the unet.engine must also
  // be the control-aware variant — the pipeline checks hasControlInputs()). Extra nets (multi-net sum)
  // via --controlnet2 <path> [--controlnet3 <path>] — each needs a matching static/controlnet_cond<i>.
  if(fs::exists(engdir / "controlnet.engine"))
  {
    eng.controlnets.push_back((engdir / "controlnet.engine").string());
    for(const char* k : {"controlnet2", "controlnet3"})
      if(args.count(k))
        eng.controlnets.push_back(args[k]);
    std::cout << "ControlNet: " << eng.controlnets.size() << " net(s)\n";
    for(auto& p : eng.controlnets) std::cout << "  " << p << "\n";
  }

  std::cout << "librediffusion " << librediffusion_version() << "\n";

  try
  {
    auto res = run_cell(cell, eng, out, verbose);
    int npass = 0, ncmp = 0;
    for(auto& r : res)
    {
      std::cout << (r.compared ? (r.pass ? "  PASS " : "  FAIL ") : "  SKIP ")
                << r.boundary << "/" << r.role << "  " << r.detail << "\n";
      if(r.compared) { ncmp++; if(r.pass) npass++; }
    }
    std::cout << "=== " << npass << "/" << ncmp << " seams pass for " << cell.filename().string()
              << " ===\n";
    std::cout.flush();
    // TensorRT/CUDA can crash in static destructors at normal process exit (atexit
    // ordering). Results are already computed + flushed, so bypass teardown with _exit
    // to get a deterministic exit code.
    _exit((ncmp > 0 && npass == ncmp) ? 0 : 1);
  }
  catch(const std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    std::cerr.flush();
    _exit(3);
  }
}
