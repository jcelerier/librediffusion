/** img2img-turbo skip-VAE C-API implementation. */
#include "librediffusion.img2img_turbo.hpp"
#include "librediffusion_c.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>

using namespace librediffusion;

struct librediffusion_img2img_turbo
{
  std::unique_ptr<Img2ImgTurboPipeline> pipe;
};

extern "C" {

librediffusion_img2img_turbo_handle librediffusion_img2img_turbo_create(
    const char* unet_engine, const char* vae_encoder_engine, const char* vae_decoder_engine)
{
  try
  {
    Img2ImgTurboEngines p;
    p.unet = unet_engine ? unet_engine : "";
    p.vae_encoder = vae_encoder_engine ? vae_encoder_engine : "";
    p.vae_decoder = vae_decoder_engine ? vae_decoder_engine : "";
    auto h = std::make_unique<librediffusion_img2img_turbo>();
    h->pipe = std::make_unique<Img2ImgTurboPipeline>(p);  // may throw (engine load) -> h freed by RAII
    return h.release();
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "img2img_turbo_create failed: %s\n", e.what());
    return nullptr;
  }
}

void librediffusion_img2img_turbo_destroy(librediffusion_img2img_turbo_handle h)
{
  delete h;
}

librediffusion_error_t librediffusion_img2img_turbo_forward(
    librediffusion_img2img_turbo_handle h, const void* image_dev, const void* ehs_dev, void* out_dev,
    librediffusion_stream_t stream)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!image_dev || !ehs_dev || !out_dev)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  try
  {
    cudaStream_t s = (cudaStream_t)stream;
    h->pipe->forward((const float*)image_dev, (const float*)ehs_dev, (float*)out_dev, s);
    cudaError_t e = cudaStreamSynchronize(s);
    if(e != cudaSuccess)
    {
      fprintf(stderr, "img2img_turbo_forward cuda: %s\n", cudaGetErrorString(e));
      return LIBREDIFFUSION_ERROR_CUDA_ERROR;
    }
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "img2img_turbo_forward failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

namespace
{
// L-03: the unsized _frame entry points copy H_*W_*4 bytes out of `in_rgba`, H_*W_*4 bytes INTO
// `out_rgba` and 77*1024 floats out of `ehs` — none of which the caller ever declared. The _sized
// variants below carry those declarations so the library can refuse a mismatch instead of walking
// off the end of three host buffers.
librediffusion_error_t check_frame_sizes(
    librediffusion_img2img_turbo_handle h, size_t in_bytes, size_t ehs_elements, size_t out_bytes)
{
  const size_t need = (size_t)h->pipe->frameHeight() * h->pipe->frameWidth() * 4;
  const size_t need_ehs = (size_t)librediffusion::Img2ImgTurboPipeline::kEhsElements;
  if(in_bytes != need || out_bytes != need)
  {
    fprintf(
        stderr,
        "img2img_turbo: buffer size mismatch — model is %dx%d (%zu bytes per frame), caller "
        "declared in=%zu out=%zu\n",
        h->pipe->frameWidth(), h->pipe->frameHeight(), need, in_bytes, out_bytes);
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  }
  if(ehs_elements != need_ehs)
  {
    fprintf(
        stderr,
        "img2img_turbo: embedding size mismatch — the model reads %zu floats [1,77,1024], caller "
        "declared %zu\n",
        need_ehs, ehs_elements);
    return LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS;
  }
  return LIBREDIFFUSION_SUCCESS;
}
} // namespace

librediffusion_error_t librediffusion_img2img_turbo_frame_sized(
    librediffusion_img2img_turbo_handle h, const unsigned char* in_rgba, size_t in_bytes,
    const float* ehs, size_t ehs_elements, unsigned char* out_rgba, size_t out_bytes)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!in_rgba || !ehs || !out_rgba)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  if(librediffusion_error_t e = check_frame_sizes(h, in_bytes, ehs_elements, out_bytes);
     e != LIBREDIFFUSION_SUCCESS)
    return e;
  try
  {
    h->pipe->forward_rgba(in_rgba, ehs, out_rgba, 0);  // host bytes in/out; syncs the stream
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "img2img_turbo_frame failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

librediffusion_error_t librediffusion_img2img_turbo_frame_dev_sized(
    librediffusion_img2img_turbo_handle h, const unsigned char* in_rgba, size_t in_bytes,
    const librediffusion_half_t* ehs_dev, unsigned char* out_rgba, size_t out_bytes)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!in_rgba || !ehs_dev || !out_rgba)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
  // ehs is a DEVICE buffer here; its length is the engine's and is not the caller's to get wrong.
  if(librediffusion_error_t e = check_frame_sizes(
         h, in_bytes, (size_t)librediffusion::Img2ImgTurboPipeline::kEhsElements, out_bytes);
     e != LIBREDIFFUSION_SUCCESS)
    return e;
  try
  {
    h->pipe->forward_rgba_dev(in_rgba, ehs_dev, out_rgba, 0);  // device fp16 ehs; syncs the stream
    return LIBREDIFFUSION_SUCCESS;
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "img2img_turbo_frame_dev failed: %s\n", e.what());
    return LIBREDIFFUSION_ERROR_INTERNAL;
  }
}

int librediffusion_img2img_turbo_frame_bytes(librediffusion_img2img_turbo_handle h)
{
  if(!h || !h->pipe)
    return 0;
  return h->pipe->frameHeight() * h->pipe->frameWidth() * 4;
}

int librediffusion_img2img_turbo_ehs_elements(librediffusion_img2img_turbo_handle h)
{
  if(!h || !h->pipe)
    return 0;
  return librediffusion::Img2ImgTurboPipeline::kEhsElements;
}

/* DEPRECATED, kept for ABI compatibility: these declare no sizes, so the library CANNOT check that
 * the caller's buffers are big enough for what it copies. Prefer the _sized variants above. */
librediffusion_error_t librediffusion_img2img_turbo_frame(
    librediffusion_img2img_turbo_handle h, const unsigned char* in_rgba, const float* ehs,
    unsigned char* out_rgba)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  const size_t n = (size_t)h->pipe->frameHeight() * h->pipe->frameWidth() * 4;
  return librediffusion_img2img_turbo_frame_sized(
      h, in_rgba, n, ehs, (size_t)librediffusion::Img2ImgTurboPipeline::kEhsElements, out_rgba, n);
}

librediffusion_error_t librediffusion_img2img_turbo_frame_dev(
    librediffusion_img2img_turbo_handle h, const unsigned char* in_rgba,
    const librediffusion_half_t* ehs_dev, unsigned char* out_rgba)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  const size_t n = (size_t)h->pipe->frameHeight() * h->pipe->frameWidth() * 4;
  return librediffusion_img2img_turbo_frame_dev_sized(h, in_rgba, n, ehs_dev, out_rgba, n);
}

} // extern "C"
