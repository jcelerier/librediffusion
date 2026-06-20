/**
 * RIFE frame interpolation — SHARED, model-agnostic module.
 *
 * Runs on decoded RGB frames AFTER any pipeline's decode step, so it multiplies the DISPLAYED
 * fps for ALL methods (FLUX/klein, SDXL, SD1.5, v2v) equally. It does NOT change throughput.
 *
 * Algorithm (mirrors FluxRT model_inference_subprocess.py:interpolate_frames): given the previous
 * and current real frames, recursively insert midpoints via IFNet. interpolation_exp = E yields
 * 2^E displayed frames per real frame: (2^E - 1) interpolated + 1 real (the current frame).
 *
 * I/O is RGBA uint8 on the HOST (the universal seam), so callers don't need to share device
 * layouts. The recursion is host-orchestrated; all heavy work (IFNet incl. grid_sample warp) runs
 * in the TRT engine, with tiny CUDA glue kernels (kernels_rife.cu) for layout/interleave.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace librediffusion
{
class RifeWrapper;

class RifeInterpolator
{
public:
  // engine_path -> rife_ifnet_fp16.plan. Throws on load failure.
  explicit RifeInterpolator(const std::string& engine_path);
  ~RifeInterpolator();
  RifeInterpolator(const RifeInterpolator&) = delete;
  RifeInterpolator& operator=(const RifeInterpolator&) = delete;

  // Interpolate between two consecutive real RGBA frames (HOST uint8 [H*W*4] each).
  // exp >= 1 -> writes (2^exp) frames to out_frames in DISPLAY order:
  //   [mid_1 ... mid_(2^exp - 1), cur]   (i.e. drops the leading prev endpoint, keeps cur last),
  // matching the reference (frames[1:]). exp == 0 -> writes just `cur` (passthrough).
  // out_frames must hold (2^exp) * H * W * 4 bytes. Returns the number of frames written.
  int interpolate(
      const uint8_t* prev_rgba, const uint8_t* cur_rgba, int H, int W, int exp,
      uint8_t* out_frames);

  // GPU variant: prev/cur RGBA uint8 on DEVICE; out_frames RGBA uint8 on DEVICE
  // [(2^exp)*H*W*4]. Same display ordering. Caller owns/sizes out_frames.
  int interpolate_gpu(
      const uint8_t* prev_rgba_dev, const uint8_t* cur_rgba_dev, int H, int W, int exp,
      uint8_t* out_frames_dev, cudaStream_t stream);

private:
  std::unique_ptr<RifeWrapper> rife_;
  cudaStream_t stream_{};

  // Persistent device scratch (allocated lazily, reused across calls, freed in dtor). Per-call
  // cudaMalloc/cudaFree is a CONTEXT-SYNCHRONIZING, lock-taking op — hundreds/sec on the saturated
  // producer serialized CUDA process-wide + showed up as the top non-kernel cost (619 cudaFree, max
  // 5.7ms). Pre-allocating eliminates that. Buffers are grown if a larger H/W/exp is requested.
  void* scratch_prev_dev_{};   // [H*W*4] u8 input prev
  void* scratch_cur_dev_{};    // [H*W*4] u8 input cur
  void* scratch_out_dev_{};    // [2^exp * H*W*4] u8 output frames
  void* scratch_seq_a_{};      // [(2^exp+1)*3*H*W] fp16 ping
  void* scratch_seq_b_{};      // [(2^exp+1)*3*H*W] fp16 pong
  void* scratch_packed_{};     // [maxB*6*H*W] fp16
  void* scratch_mids_{};       // [maxB*3*H*W] fp16
  long  scratch_cap_ = -1;     // a monotonically-growing capacity key (bytes of the largest buffer set)
  int   scratch_H_ = 0, scratch_W_ = 0, scratch_exp_ = 0;
  void ensureScratch(int H, int W, int exp);
  void freeScratch();
};

} // namespace librediffusion
