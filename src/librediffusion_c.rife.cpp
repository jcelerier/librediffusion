/** RIFE frame-interpolation C-API implementation (shared/model-agnostic). */
#include "librediffusion.rife.hpp"
#include "librediffusion_c.h"

#include <cstdio>
#include <memory>

using namespace librediffusion;

struct librediffusion_rife
{
  std::unique_ptr<RifeInterpolator> interp;
  bool enabled{false};
  int exp{1};
};

extern "C" {

librediffusion_rife_handle librediffusion_rife_create(const char* engine_path)
{
  try
  {
    auto* h = new librediffusion_rife;
    h->interp = std::make_unique<RifeInterpolator>(engine_path ? engine_path : "");
    return h;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "rife_create failed: %s\n", e.what());
    return nullptr;
  }
}

void librediffusion_rife_destroy(librediffusion_rife_handle h)
{
  delete h;
}

void librediffusion_rife_set_enabled(librediffusion_rife_handle h, int enabled)
{
  if(h)
    h->enabled = (enabled != 0);
}

int librediffusion_rife_is_enabled(librediffusion_rife_handle h)
{
  return (h && h->enabled) ? 1 : 0;
}

void librediffusion_rife_set_interpolation_exp(librediffusion_rife_handle h, int exp)
{
  if(h)
    h->exp = exp < 0 ? 0 : exp;
}

int librediffusion_rife_get_interpolation_exp(librediffusion_rife_handle h)
{
  return h ? h->exp : 0;
}

librediffusion_error_t librediffusion_rife_interpolate(
    librediffusion_rife_handle h, const unsigned char* prev_rgba, const unsigned char* cur_rgba,
    int H, int W, unsigned char* out_frames, int* out_count)
{
  if(!h || !h->interp || !prev_rgba || !cur_rgba || !out_frames)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    int eff_exp = h->enabled ? h->exp : 0;
    int n = h->interp->interpolate(prev_rgba, cur_rgba, H, W, eff_exp, out_frames);
    if(out_count)
      *out_count = n;
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "rife_interpolate failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

librediffusion_error_t librediffusion_rife_interpolate_gpu(
    librediffusion_rife_handle h, const unsigned char* prev_rgba_dev,
    const unsigned char* cur_rgba_dev, int H, int W, unsigned char* out_frames_dev, int* out_count)
{
  if(!h || !h->interp || !prev_rgba_dev || !cur_rgba_dev || !out_frames_dev)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  try
  {
    int eff_exp = h->enabled ? h->exp : 0;
    int n = h->interp->interpolate_gpu(
        prev_rgba_dev, cur_rgba_dev, H, W, eff_exp, out_frames_dev, 0);
    if(out_count)
      *out_count = n;
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "rife_interpolate_gpu failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

} // extern "C"
