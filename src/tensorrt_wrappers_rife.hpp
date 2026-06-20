/**
 * RIFE (IFNet) frame-interpolation TensorRT wrapper (fp16 I/O).
 *
 * Model-agnostic: operates purely on decoded RGB frames, so it benefits ANY pipeline
 * (FLUX/klein, SDXL, SD1.5, v2v). Reference: hzwer/ECCV2022-RIFE (TensorForger fork,
 * flownet.safetensors). Reimplemented from FluxRT interpolation_model.py.
 *
 * Engine (built strongly-typed fp16):
 *   - frames: [B,6,H,W] fp16   (= cat([img0,img1], dim=1), two RGB frames in [0,1])
 *   - mid:    [B,3,H,W] fp16   (the 0.5-midpoint interpolated frame)
 *  Dynamic profile: H,W in [64,1024], B in [1,7]. GridSample is native (opset-17, no plugin).
 *
 * Callers pass DEVICE pointers; the wrapper binds them directly (no internal staging copies).
 */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "NvInfer.h"

namespace librediffusion
{
class CachedTensorRTEngine;

class RifeWrapper
{
public:
  explicit RifeWrapper(const std::string& engine_path);
  ~RifeWrapper();
  RifeWrapper(const RifeWrapper&) = delete;
  RifeWrapper& operator=(const RifeWrapper&) = delete;

  // frames DEVICE fp16 [B,6,H,W] -> mid DEVICE fp16 [B,3,H,W]. Produces B midpoints, one per
  // consecutive (img0,img1) pair packed in the channel dim.
  void interpolate(const __half* frames, __half* mid, int batch, int H, int W,
                   cudaStream_t stream);

private:
  std::shared_ptr<CachedTensorRTEngine> cached_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void loadEngine(const std::string& engine_path);
};

} // namespace librediffusion
