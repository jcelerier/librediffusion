/*
 * txt2img inner-loop probe.
 *
 * For txt2img cells the harness only ever compares the final decode, so a divergence anywhere in
 * predict_x0_batch surfaces as one blurry PSNR with no localisation — which is why the hyper-sd15
 * batch-0 cells (16.94 dB vs a 32.9 dB baseline) could not be pinned down.
 *
 * This does for txt2img what the harness already does for img2img: inject the golden scheduler
 * tables, prompt embedding and init_noise, feed the GOLDEN start latent into predict_x0_batch, and
 * compare against the golden x_0_pred. Everything upstream of the loop is thereby held fixed, so a
 * mismatch here is the loop's own arithmetic and nothing else.
 *
 * Usage: txt2img_loop_probe --cell <golden_cell_dir> --engines <engine_dir> [--pad-token N]
 */
#include "librediffusion_c.h"

#include "npy.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CK(x)                                                          \
  do {                                                                 \
    int _rc = (x);                                                     \
    if(_rc != 0) {                                                     \
      std::fprintf(stderr, "FAIL %s rc=%d\n", #x, _rc);                \
      std::exit(2);                                                    \
    }                                                                  \
  } while(0)

namespace
{
std::vector<librediffusion_half_t> to_half(const std::vector<float>& f)
{
  std::vector<librediffusion_half_t> h(f.size());
  for(size_t i = 0; i < f.size(); i++)
    h[i] = librediffusion_float_to_half(f[i]);
  return h;
}

std::vector<float> from_half(const std::vector<librediffusion_half_t>& h)
{
  std::vector<float> f(h.size());
  for(size_t i = 0; i < h.size(); i++)
    f[i] = librediffusion_half_to_float(h[i]);
  return f;
}

// Same latent metrics the harness reports, so the numbers are directly comparable.
void report(const char* what, const std::vector<float>& a, const std::vector<float>& b)
{
  const size_t n = std::min(a.size(), b.size());
  double dot = 0, na = 0, nb = 0, num = 0, den = 0, maxabs = 0;
  for(size_t i = 0; i < n; i++)
  {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
    const double d = std::fabs((double)a[i] - b[i]);
    num += d * d;
    den += (double)a[i] * a[i];
    if(d > maxabs)
      maxabs = d;
  }
  const double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
  const double rel = den > 0 ? std::sqrt(num) / std::sqrt(den) : 0.0;
  const bool pass = cos >= 0.999 && rel <= 0.05;
  std::printf("  %-34s cos=%.6f rel=%.6f maxabs=%.6f  %s\n", what, cos, rel, maxabs,
              pass ? "PASS" : "FAIL");
}

std::string arg(int c, char** v, const char* k, const char* dflt = "")
{
  for(int i = 1; i + 1 < c; i++)
    if(std::strcmp(v[i], k) == 0)
      return v[i + 1];
  return dflt;
}

// Golden files are NNN_<boundary>__<role>.npy; the index shifts with cfg mode, so match by role.
bool load_role(const fs::path& cell, const std::string& suffix, npy::Array& out)
{
  for(auto& e : fs::directory_iterator(cell))
  {
    const std::string fn = e.path().filename().string();
    if(fn.size() > suffix.size() && fn.rfind(suffix) == fn.size() - suffix.size())
    {
      out = npy::load(e.path().string());
      return true;
    }
  }
  return false;
}
} // namespace

int main(int argc, char** argv)
{
  const std::string cell_s = arg(argc, argv, "--cell");
  const std::string eng_s = arg(argc, argv, "--engines");
  const int pad_token = std::atoi(arg(argc, argv, "--pad-token", "49407").c_str());
  const int denoise_batch = std::atoi(arg(argc, argv, "--denoise-batch", "0").c_str());
  if(cell_s.empty() || eng_s.empty())
  {
    std::fprintf(stderr, "usage: --cell <dir> --engines <dir> [--pad-token N]\n");
    return 1;
  }
  const fs::path cell{cell_s}, eng{eng_s};

  npy::Array pe, init_noise, sub_t, alpha, beta, cskip, cout_, start, want;
  if(!load_role(cell, "static__prompt_embeds.npy", pe)
     || !load_role(cell, "static__init_noise.npy", init_noise)
     || !load_role(cell, "static__sub_timesteps_tensor.npy", sub_t)
     || !load_role(cell, "static__alpha_prod_t_sqrt.npy", alpha)
     || !load_role(cell, "static__beta_prod_t_sqrt.npy", beta)
     || !load_role(cell, "static__c_skip.npy", cskip)
     || !load_role(cell, "static__c_out.npy", cout_)
     || !load_role(cell, "predict_x0_batch__in_x_t_latent.npy", start)
     || !load_role(cell, "predict_x0_batch__out_x_0_pred.npy", want))
  {
    std::fprintf(stderr, "cell is missing one of the tensors this probe needs\n");
    return 1;
  }

  const int hidden = (int)pe.shape.back();
  const int steps = (int)sub_t.size();
  const int latent_h = (int)start.shape[2], latent_w = (int)start.shape[3];
  const int batch = (int)start.shape[0];
  std::printf("cell   %s\n", cell.filename().string().c_str());
  std::printf("steps=%d batch=%d latent=%dx%d hidden=%d denoise_batch=%d\n\n", steps, batch,
              latent_w, latent_h, hidden, denoise_batch);

  librediffusion_config_handle cfg = nullptr;
  CK(librediffusion_config_create(&cfg));
  CK(librediffusion_config_set_device(cfg, 0));
  CK(librediffusion_config_set_model_type(cfg, MODEL_SD_15));
  CK(librediffusion_config_set_dimensions(cfg, latent_w * 8, latent_h * 8, latent_w, latent_h));
  CK(librediffusion_config_set_batch_size(cfg, batch));
  CK(librediffusion_config_set_denoising_steps(cfg, steps));
  CK(librediffusion_config_set_frame_buffer_size(cfg, 1));
  CK(librediffusion_config_set_guidance_scale(cfg, 1.0f));
  CK(librediffusion_config_set_delta(cfg, 1.0f));
  CK(librediffusion_config_set_add_noise(cfg, 0));
  CK(librediffusion_config_set_denoising_batch(cfg, denoise_batch));
  CK(librediffusion_config_set_seed(cfg, 1234));
  CK(librediffusion_config_set_cfg_type(cfg, SD_CFG_NONE));
  CK(librediffusion_config_set_text_config(cfg, 77, hidden, pad_token));
  const std::string unet = (eng / "unet.engine").string();
  const std::string vae_e = (eng / "vae_encoder.engine").string();
  const std::string vae_d = (eng / "vae_decoder.engine").string();
  CK(librediffusion_config_set_unet_engine(cfg, unet.c_str()));
  CK(librediffusion_config_set_vae_encoder(cfg, vae_e.c_str()));
  CK(librediffusion_config_set_vae_decoder(cfg, vae_d.c_str()));
  std::vector<int> tidx(steps);
  for(int i = 0; i < steps; i++)
    tidx[i] = i;
  CK(librediffusion_config_set_timestep_indices(cfg, tidx.data(), tidx.size()));

  librediffusion_pipeline_handle pipe = nullptr;
  CK(librediffusion_pipeline_create(cfg, &pipe));
  CK(librediffusion_pipeline_init_all(pipe));
  librediffusion_stream_t stream = librediffusion_pipeline_get_stream(pipe);

  // Golden prompt embedding, tiled to the batch extent prepare_embeds copies.
  {
    auto h = to_half(pe.data);
    const size_t one = (size_t)77 * hidden;
    std::vector<librediffusion_half_t> tiled((size_t)batch * one);
    for(int b = 0; b < batch; b++)
      std::memcpy(tiled.data() + b * one, h.data(), one * sizeof(librediffusion_half_t));
    auto* d = (librediffusion_half_t*)librediffusion_cuda_malloc(
        tiled.size() * sizeof(librediffusion_half_t));
    CK(librediffusion_cuda_memcpy_h2d(d, tiled.data(),
                                      tiled.size() * sizeof(librediffusion_half_t)));
    CK(librediffusion_prepare_embeds(pipe, d, 77, hidden));
  }

  CK(librediffusion_prepare_scheduler(pipe, sub_t.data.data(), alpha.data.data(),
                                      beta.data.data(), cskip.data.data(), cout_.data.data(),
                                      sub_t.size()));
  {
    auto h = to_half(init_noise.data);
    auto* d = (librediffusion_half_t*)librediffusion_cuda_malloc(
        h.size() * sizeof(librediffusion_half_t));
    CK(librediffusion_cuda_memcpy_h2d(d, h.data(), h.size() * sizeof(librediffusion_half_t)));
    CK(librediffusion_set_init_noise(pipe, d));
  }

  // The seam: golden start latent in, x_0_pred out.
  const size_t elems = start.data.size();
  auto hs = to_half(start.data);
  auto* d_in
      = (librediffusion_half_t*)librediffusion_cuda_malloc(elems * sizeof(librediffusion_half_t));
  auto* d_out
      = (librediffusion_half_t*)librediffusion_cuda_malloc(elems * sizeof(librediffusion_half_t));
  CK(librediffusion_cuda_memcpy_h2d(d_in, hs.data(), elems * sizeof(librediffusion_half_t)));
  CK(librediffusion_predict_x0_batch(pipe, d_in, d_out, stream));
  CK(librediffusion_pipeline_synchronize(pipe));

  std::vector<librediffusion_half_t> hout(elems);
  CK(librediffusion_cuda_memcpy_d2h(hout.data(), d_out, elems * sizeof(librediffusion_half_t)));
  const auto got = from_half(hout);

  std::printf("seams (golden inputs injected, so this is the loop alone):\n");
  report("predict_x0_batch/out_x_0_pred", want.data, got);

  // Round-trip the golden's own x_0_pred through fp16 as the achievable floor: anything at or near
  // this is precision, anything far from it is arithmetic.
  {
    auto floor_ = from_half(to_half(want.data));
    report("[fp16 floor: golden vs fp16(golden)]", want.data, floor_);
  }

  librediffusion_pipeline_destroy(pipe);
  librediffusion_config_destroy(cfg);
  std::fflush(stdout);
  ::_exit(0); // TRT static-destructor teardown is unsafe; the harness does the same
}
