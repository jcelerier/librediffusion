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
  // Clamp BOTH ways. This used to clamp only from below, so any value was stored verbatim: a
  // readback of 2147483647 was possible, and `for(i = 0; i < exp; ++i) total_out *= 2` is signed
  // overflow (UB) from exp = 31 up. 2^LIBREDIFFUSION_RIFE_MAX_EXP frames per real frame is already
  // far beyond anything a display pipeline can consume.
  if(!h)
    return;
  if(exp < 0)
    exp = 0;
  else if(exp > LIBREDIFFUSION_RIFE_MAX_EXP)
    exp = LIBREDIFFUSION_RIFE_MAX_EXP;
  h->exp = exp;
}

int librediffusion_rife_required_out_bytes(librediffusion_rife_handle h, int H, int W)
{
  if(!h || H <= 0 || W <= 0)
    return 0;
  const int eff_exp = h->enabled ? h->exp : 0;
  return (1 << eff_exp) * H * W * 4;
}

int librediffusion_rife_get_interpolation_exp(librediffusion_rife_handle h)
{
  return h ? h->exp : 0;
}

namespace
{
// L-05: interpolate() writes 2^exp * H*W*4 bytes into a caller buffer that carried no length, and
// the frame count comes from state the caller set in a DIFFERENT call (set_interpolation_exp). Any
// host that raises the interpolation factor without re-sizing its output buffer overflows it —
// measured at 98 304 bytes past a 32 768-byte buffer for exp 1 -> 3. The _sized entry points let the
// caller declare the capacity so the mismatch is an error code instead.
librediffusion_error_t check_out_capacity(
    librediffusion_rife_handle h, int H, int W, size_t out_capacity_bytes)
{
  const int eff_exp = h->enabled ? h->exp : 0;
  const size_t need = (size_t)(1 << eff_exp) * (size_t)H * (size_t)W * 4u;
  if(out_capacity_bytes < need)
  {
    fprintf(
        stderr,
        "rife_interpolate: out_frames capacity %zu bytes is too small — exp=%d at %dx%d needs %zu "
        "bytes (%d frames)\n",
        out_capacity_bytes, eff_exp, W, H, need, 1 << eff_exp);
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  }
  return LIBREDIFFUSION_SUCCESS;
}
} // namespace

librediffusion_error_t librediffusion_rife_interpolate_sized(
    librediffusion_rife_handle h, const unsigned char* prev_rgba, const unsigned char* cur_rgba,
    int H, int W, unsigned char* out_frames, size_t out_capacity_bytes, int* out_count)
{
  if(!h || !h->interp || !prev_rgba || !cur_rgba || !out_frames)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(H <= 0 || W <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  if(librediffusion_error_t e = check_out_capacity(h, H, W, out_capacity_bytes);
     e != LIBREDIFFUSION_SUCCESS)
    return e;
  return librediffusion_rife_interpolate(h, prev_rgba, cur_rgba, H, W, out_frames, out_count);
}

librediffusion_error_t librediffusion_rife_interpolate_gpu_sized(
    librediffusion_rife_handle h, const unsigned char* prev_rgba_dev,
    const unsigned char* cur_rgba_dev, int H, int W, unsigned char* out_frames_dev,
    size_t out_capacity_bytes, int* out_count)
{
  if(!h || !h->interp || !prev_rgba_dev || !cur_rgba_dev || !out_frames_dev)
    return LIBREDIFFUSION_ERROR_INVALID_ARGUMENT;
  if(H <= 0 || W <= 0)
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  if(librediffusion_error_t e = check_out_capacity(h, H, W, out_capacity_bytes);
     e != LIBREDIFFUSION_SUCCESS)
    return e;
  return librediffusion_rife_interpolate_gpu(
      h, prev_rgba_dev, cur_rgba_dev, H, W, out_frames_dev, out_count);
}

/* DEPRECATED, kept for ABI compatibility: no capacity is declared, so the library cannot check that
 * out_frames is big enough for the 2^exp frames it is about to write. Prefer the _sized form. */
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
