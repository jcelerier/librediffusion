#include "librediffusion.hpp"
#include "kernels.hpp"
#include "nchw.hpp"
#include "iostream"

namespace librediffusion
{
// Convert RGBA NHWC byte image to float NCHW tensor on GPU
void LibreDiffusionPipeline::rgba_nhwc_to_nchw_gpu(
    const uint8_t* rgba_nhwc_in,
    float* rgb_nchw_out,
    int width, int height,
    cudaStream_t stream)
{
  if (!stream) stream = stream_;

  int batch_size = config_.batch_size;
  size_t rgb_nhwc_size = batch_size * height * width * 3;

  // Allocate or resize temporary buffer if needed
  if (!rgb_nhwc_tmp_fp32_ || rgb_nhwc_tmp_fp32_->size() < rgb_nhwc_size) {
    rgb_nhwc_tmp_fp32_ = std::make_unique<CUDATensor<float>>(rgb_nhwc_size);
  }

  // Step 1: Convert RGBA byte to RGB float with normalization
  launch_rgba_to_rgb_normalized_fp32(
      rgba_nhwc_in, rgb_nhwc_tmp_fp32_->data(), batch_size, height, width, stream);

  cudaStreamSynchronize(stream);
  // Step 2: Use Cutlass to convert NHWC to NCHW layout
  launch_nhwc_to_nchw_rgb(batch_size, height, width, rgb_nhwc_tmp_fp32_->data(), rgb_nchw_out, stream);
}

// Convert RGBA NHWC byte image to half NCHW tensor on GPU
void LibreDiffusionPipeline::rgba_nhwc_to_nchw_gpu(
    const uint8_t* rgba_nhwc_in,
    __half* rgb_nchw_out,
    int width, int height,
    cudaStream_t stream)
{
  if (!stream) stream = stream_;

  int batch_size = config_.batch_size;
  size_t rgb_nhwc_size = batch_size * height * width * 3;

  // Allocate or resize temporary buffer if needed
  if (!rgb_nhwc_tmp_fp16_ || rgb_nhwc_tmp_fp16_->size() < rgb_nhwc_size) {
    rgb_nhwc_tmp_fp16_ = std::make_unique<CUDATensor<__half>>(rgb_nhwc_size);
  }

  // Step 1: Convert RGBA byte to RGB half with normalization
  launch_rgba_to_rgb_normalized_fp16(
      rgba_nhwc_in, rgb_nhwc_tmp_fp16_->data(), batch_size, height, width, stream);

  // Step 2: Use Cutlass to convert NHWC to NCHW layout
  launch_nhwc_to_nchw_rgb(batch_size, height, width, rgb_nhwc_tmp_fp16_->data(), rgb_nchw_out, stream);
}

// Convert half NCHW tensor to RGBA NHWC byte image on GPU
void LibreDiffusionPipeline::nchw_to_rgba_nhwc_gpu(
    const __half* rgb_nchw_in,
    uint8_t* rgba_nhwc_out,
    int width, int height,
    cudaStream_t stream)
{
  if (!stream) stream = stream_;

  int batch_size = config_.batch_size;
  size_t rgb_nhwc_size = batch_size * height * width * 3;

  // Allocate or resize temporary buffer if needed
  if (!rgb_nhwc_tmp_fp16_ || rgb_nhwc_tmp_fp16_->size() < rgb_nhwc_size) {
    rgb_nhwc_tmp_fp16_ = std::make_unique<CUDATensor<__half>>(rgb_nhwc_size);
  }

  // Step 1: Use Cutlass to convert NCHW to NHWC layout
  launch_nchw_to_nhwc_rgb(batch_size, height, width, rgb_nchw_in, rgb_nhwc_tmp_fp16_->data(), stream);

  // Step 2: Convert RGB half to RGBA byte with denormalization
  launch_rgb_to_rgba_denormalized_fp16(
      rgb_nhwc_tmp_fp16_->data(), rgba_nhwc_out, batch_size, height, width, stream);
}

// Convert float NCHW tensor to RGBA NHWC byte image on GPU
void LibreDiffusionPipeline::nchw_to_rgba_nhwc_gpu(
    const float* rgb_nchw_in,
    uint8_t* rgba_nhwc_out,
    int width, int height,
    cudaStream_t stream)
{
  if (!stream) stream = stream_;

  int batch_size = config_.batch_size;
  size_t rgb_nhwc_size = batch_size * height * width * 3;

  // Allocate or resize temporary buffer if needed
  if (!rgb_nhwc_tmp_fp32_ || rgb_nhwc_tmp_fp32_->size() < rgb_nhwc_size) {
    rgb_nhwc_tmp_fp32_ = std::make_unique<CUDATensor<float>>(rgb_nhwc_size);
  }

  // Step 1: Use Cutlass to convert NCHW to NHWC layout
  launch_nchw_to_nhwc_rgb(batch_size, height, width, rgb_nchw_in, rgb_nhwc_tmp_fp32_->data(), stream);

  // Step 2: Convert RGB float to RGBA byte with denormalization
  launch_rgb_to_rgba_denormalized_fp32(
      rgb_nhwc_tmp_fp32_->data(), rgba_nhwc_out, batch_size, height, width, stream);
}

void LibreDiffusionPipeline::rgba_resize(
    Npp8u* device_rgba_input_, int iw, int ih, Npp8u* device_rgba_resized_, int ow,
    int oh)
{
  // 1. Calculate Steps (Strides)
  // NPP requires the distance in bytes between the start of consecutive rows.
  // For tightly packed RGBA, this is width * 4 channels.
  int nSrcStep = iw * 4 * sizeof(Npp8u);
  int nDstStep = ow * 4 * sizeof(Npp8u);

  // 2. Define Sizes and ROIs (Regions of Interest)
  NppiSize oSrcSize = {iw, ih};
  NppiRect oSrcRectROI = {0, 0, iw, ih}; // Full image

  NppiSize oDstSize = {ow, oh};
  NppiRect oDstRectROI = {0, 0, ow, oh}; // Full image target

  // 3. Choose Interpolation
  int eInterpolation = NPPI_INTER_LINEAR;

  return;
  // 5. Execute Resize
  NppStatus status = nppiResize_8u_C4R_Ctx(
      device_rgba_input_, nSrcStep, oSrcSize, oSrcRectROI, device_rgba_resized_,
      nDstStep, oDstSize, oDstRectROI, eInterpolation, this->npp_stream_);

  // 6. Error Handling
  if(status != NPP_NO_ERROR)
  {
    std::cerr << "NPP Resize Error: " << status << std::endl;
    // In production, handle this gracefully or throw
  }
}
float*
LibreDiffusionPipeline::img_preprocess(const uint8_t* device_rgba_input, int iw, int ih)
{
  // FIXME batch not handled
  uint8_t* device_rgba_input_correct_size{};
  if((ih * iw) == (this->config_.height * this->config_.width))
  {
    device_rgba_input_correct_size = device_rgba_input_->data();
  }
  else
  {
    this->rgba_resize(
        device_rgba_input_->data(), iw, ih, device_rgba_input_vae_size_->data(),
        this->config_.width, this->config_.height);
    device_rgba_input_correct_size = device_rgba_input_vae_size_->data();
  }

  // Convert RGBA NHWC to float NCHW on GPU
  rgba_nhwc_to_nchw_gpu(
      device_rgba_input_correct_size, device_nchw_input_->data(), config_.width,
      config_.height, stream_);

  return device_nchw_input_->data();
}

uint8_t*
LibreDiffusionPipeline::img_postprocess(__half* device_rgba_output, int iw, int ih)
{
  // FIXME batch not handled
  // Convert output from NCHW to RGBA NHWC on GPU
  nchw_to_rgba_nhwc_gpu(
      device_nchw_output_->data(), device_rgba_output_vae_size_->data(), config_.width,
      config_.height, stream_);

  uint8_t* device_rgba_output_correct_size{};
  if(ih * iw == this->config_.height * this->config_.width)
  {
    device_rgba_output_correct_size = device_rgba_output_vae_size_->data();
  }
  else
  {
    this->rgba_resize(
        device_rgba_output_vae_size_->data(), this->config_.width, this->config_.height,
        device_rgba_output_->data(), iw, ih);
    device_rgba_output_correct_size = device_rgba_output_->data();
  }
  return device_rgba_output_correct_size;
}
}
