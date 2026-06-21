/** img2img-turbo skip-VAE C++ pipeline implementation. */
#include "librediffusion.img2img_turbo.hpp"

#include "kernels.hpp"
#include "model_cache.hpp"

#include <stdexcept>

namespace librediffusion
{
namespace
{
std::unique_ptr<nvinfer1::IExecutionContext>
makeContext(std::shared_ptr<CachedTensorRTEngine>& eng, const std::string& path)
{
  eng = getCachedEngine(path);
  if(!eng || !eng->isValid())
    throw std::runtime_error("img2img-turbo: failed to load engine: " + path);
  auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(eng->createExecutionContext());
  if(!ctx)
    throw std::runtime_error("img2img-turbo: failed to create context: " + path);
  return ctx;
}

nvinfer1::Dims dims3(int a, int b, int c)
{
  nvinfer1::Dims d; d.nbDims = 3; d.d[0] = a; d.d[1] = b; d.d[2] = c; return d;
}
nvinfer1::Dims dims4(int a, int b, int c, int e)
{
  nvinfer1::Dims d; d.nbDims = 4; d.d[0] = a; d.d[1] = b; d.d[2] = c; d.d[3] = e; return d;
}

void bindIn(nvinfer1::IExecutionContext* c, const char* name, const void* ptr, nvinfer1::Dims d)
{
  if(!c->setInputShape(name, d))
    throw std::runtime_error(std::string("img2img-turbo: setInputShape failed: ") + name);
  if(!c->setTensorAddress(name, const_cast<void*>(ptr)))
    throw std::runtime_error(std::string("img2img-turbo: setTensorAddress(in) failed: ") + name);
}
void bindOut(nvinfer1::IExecutionContext* c, const char* name, void* ptr)
{
  if(!c->setTensorAddress(name, ptr))
    throw std::runtime_error(std::string("img2img-turbo: setTensorAddress(out) failed: ") + name);
}
} // namespace

Img2ImgTurboPipeline::Img2ImgTurboPipeline(const Img2ImgTurboEngines& p)
{
  c_enc_ = makeContext(e_enc_, p.vae_encoder);
  c_unet_ = makeContext(e_unet_, p.unet);
  c_dec_ = makeContext(e_dec_, p.vae_decoder);
  alloc();
}

Img2ImgTurboPipeline::~Img2ImgTurboPipeline() = default;

void Img2ImgTurboPipeline::alloc()
{
  latent_ = std::make_unique<CUDATensor<float>>((size_t)1 * 4 * lh_ * lw_);
  enc0_ = std::make_unique<CUDATensor<float>>((size_t)1 * 128 * H_ * W_);
  enc1_ = std::make_unique<CUDATensor<float>>((size_t)1 * 128 * (H_ / 2) * (W_ / 2));
  enc2_ = std::make_unique<CUDATensor<float>>((size_t)1 * 256 * (H_ / 4) * (W_ / 4));
  enc3_ = std::make_unique<CUDATensor<float>>((size_t)1 * 512 * lh_ * lw_);
  model_pred_ = std::make_unique<CUDATensor<float>>((size_t)1 * 4 * lh_ * lw_);
  x0_ = std::make_unique<CUDATensor<float>>((size_t)1 * 4 * lh_ * lw_);
  // forward_rgba scratch
  rgba_in_ = std::make_unique<CUDATensor<unsigned char>>((size_t)H_ * W_ * 4);
  rgba_out_ = std::make_unique<CUDATensor<unsigned char>>((size_t)H_ * W_ * 4);
  image_ = std::make_unique<CUDATensor<float>>((size_t)1 * 3 * H_ * W_);
  ehs_ = std::make_unique<CUDATensor<float>>((size_t)1 * 77 * 1024);
  out_ = std::make_unique<CUDATensor<float>>((size_t)1 * 3 * H_ * W_);
}

void Img2ImgTurboPipeline::forward(const float* image, const float* ehs, float* out, cudaStream_t stream)
{
  // 1) VAE encode -> latent + 4 skip activations
  {
    auto* c = c_enc_.get();
    bindIn(c, "image", image, dims4(1, 3, H_, W_));
    bindOut(c, "latent", latent_->data());
    bindOut(c, "enc0", enc0_->data());
    bindOut(c, "enc1", enc1_->data());
    bindOut(c, "enc2", enc2_->data());
    bindOut(c, "enc3", enc3_->data());
    if(!c->enqueueV3(stream))
      throw std::runtime_error("img2img-turbo: vae encode enqueueV3 failed");
  }

  // 2) UNet (single call, t=999 baked) -> model_pred
  {
    auto* c = c_unet_.get();
    bindIn(c, "latent", latent_->data(), dims4(1, 4, lh_, lw_));
    bindIn(c, "ehs", ehs, dims3(1, 77, 1024));
    bindOut(c, "model_pred", model_pred_->data());
    if(!c->enqueueV3(stream))
      throw std::runtime_error("img2img-turbo: unet enqueueV3 failed");
  }

  // 3) closed-form 1-step DDPM x0
  launch_ddpm_1step_x0(
      latent_->data(), model_pred_->data(), acp_, (long)4 * lh_ * lw_, x0_->data(), stream);

  // 4) VAE decode with skips (REVERSED: s0=enc3, s1=enc2, s2=enc1, s3=enc0) -> image[-1,1]
  {
    auto* c = c_dec_.get();
    bindIn(c, "latent_scaled", x0_->data(), dims4(1, 4, lh_, lw_));
    bindIn(c, "s0", enc3_->data(), dims4(1, 512, lh_, lw_));
    bindIn(c, "s1", enc2_->data(), dims4(1, 256, H_ / 4, W_ / 4));
    bindIn(c, "s2", enc1_->data(), dims4(1, 128, H_ / 2, W_ / 2));
    bindIn(c, "s3", enc0_->data(), dims4(1, 128, H_, W_));
    bindOut(c, "image", out);
    if(!c->enqueueV3(stream))
      throw std::runtime_error("img2img-turbo: vae decode enqueueV3 failed");
  }
}

void Img2ImgTurboPipeline::run_rgba(
    const unsigned char* in_rgba, unsigned char* out_rgba, cudaStream_t stream)
{
  const size_t rgba_bytes = (size_t)H_ * W_ * 4;
  // upload RGBA8 -> device, convert to RGB fp32 CHW [0,1] (ToTensor; alpha dropped)
  cudaMemcpyAsync(rgba_in_->data(), in_rgba, rgba_bytes, cudaMemcpyHostToDevice, stream);
  launch_rgba_to_chw01_f32(rgba_in_->data(), image_->data(), H_, W_, stream);
  // run the device pipeline (encode -> unet -> x0 -> decode), out_ is fp32 CHW [-1,1]. ehs_ pre-filled.
  forward(image_->data(), ehs_->data(), out_->data(), stream);
  // convert RGB fp32 CHW [-1,1] -> RGBA8, download to host
  launch_chw_m1p1_to_rgba_f32(out_->data(), rgba_out_->data(), H_, W_, stream);
  cudaMemcpyAsync(out_rgba, rgba_out_->data(), rgba_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

void Img2ImgTurboPipeline::forward_rgba(
    const unsigned char* in_rgba, const float* ehs_host, unsigned char* out_rgba, cudaStream_t stream)
{
  cudaMemcpyAsync(
      ehs_->data(), ehs_host, (size_t)1 * 77 * 1024 * sizeof(float), cudaMemcpyHostToDevice, stream);
  run_rgba(in_rgba, out_rgba, stream);
}

void Img2ImgTurboPipeline::forward_rgba_dev(
    const unsigned char* in_rgba, const void* ehs_dev_fp16, unsigned char* out_rgba,
    cudaStream_t stream)
{
  launch_fp16_to_fp32(ehs_dev_fp16, ehs_->data(), 77 * 1024, stream);
  run_rgba(in_rgba, out_rgba, stream);
}

} // namespace librediffusion
