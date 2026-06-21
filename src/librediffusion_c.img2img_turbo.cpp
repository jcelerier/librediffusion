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

librediffusion_error_t librediffusion_img2img_turbo_frame(
    librediffusion_img2img_turbo_handle h, const unsigned char* in_rgba, const float* ehs,
    unsigned char* out_rgba)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!in_rgba || !ehs || !out_rgba)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
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

librediffusion_error_t librediffusion_img2img_turbo_frame_dev(
    librediffusion_img2img_turbo_handle h, const unsigned char* in_rgba,
    const librediffusion_half_t* ehs_dev, unsigned char* out_rgba)
{
  if(!h || !h->pipe)
    return LIBREDIFFUSION_ERROR_NOT_INITIALIZED;
  if(!in_rgba || !ehs_dev || !out_rgba)
    return LIBREDIFFUSION_ERROR_NULL_POINTER;
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

} // extern "C"
