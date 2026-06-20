/** RIFE frame interpolation — shared model-agnostic module. See librediffusion.rife.hpp. */
#include "librediffusion.rife.hpp"

#include "kernels.hpp"
#include "tensorrt_wrappers_rife.hpp"

#include <cuda_fp16.h>

#include <stdexcept>

namespace librediffusion
{

RifeInterpolator::RifeInterpolator(const std::string& engine_path)
{
  rife_ = std::make_unique<RifeWrapper>(engine_path);
  // RIFE is the FOREGROUND per-tick interpolation; give it a HIGH-priority non-blocking stream so the
  // GPU scheduler favors it over the (low-priority) background klein diffusion stream -> the render
  // thread's interpolation isn't starved while a ~200ms diffusion keyframe is in flight.
  int lo = 0, hi = 0;
  cudaDeviceGetStreamPriorityRange(&lo, &hi); // hi is the numerically-smallest (highest) priority
  if(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, hi) != cudaSuccess)
    cudaStreamCreate(&stream_);
}

void RifeInterpolator::freeScratch()
{
  cudaFree(scratch_prev_dev_); scratch_prev_dev_ = nullptr;
  cudaFree(scratch_cur_dev_);  scratch_cur_dev_ = nullptr;
  cudaFree(scratch_out_dev_);  scratch_out_dev_ = nullptr;
  cudaFree(scratch_seq_a_);    scratch_seq_a_ = nullptr;
  cudaFree(scratch_seq_b_);    scratch_seq_b_ = nullptr;
  cudaFree(scratch_packed_);   scratch_packed_ = nullptr;
  cudaFree(scratch_mids_);     scratch_mids_ = nullptr;
  scratch_H_ = scratch_W_ = scratch_exp_ = 0;
}

// (Re)allocate the persistent device scratch ONLY when the geometry grows. Steady state (fixed H/W/exp)
// = zero cudaMalloc/cudaFree per interpolate() call -> no per-frame CUDA context-lock churn.
void RifeInterpolator::ensureScratch(int H, int W, int exp)
{
  if(exp < 1) exp = 1;
  if(scratch_seq_a_ && H == scratch_H_ && W == scratch_W_ && exp <= scratch_exp_)
    return;  // existing buffers already big enough
  freeScratch();
  const long per = (long)3 * H * W;
  const long hw4 = (long)H * W * 4;
  int total_out = 1; for(int i = 0; i < exp; ++i) total_out *= 2;   // 2^exp
  const int max_frames = total_out + 1;                            // 2^exp + 1 (both endpoints)
  int maxB = 1; for(int i = 1; i < exp; ++i) maxB *= 2;            // (count-1=1) * 2^(exp-1)
  // Checked allocation: on OOM, free whatever already succeeded and throw rather than dereferencing
  // a null scratch pointer later (freeScratch is null-safe, so partial state cleans up correctly).
  auto checked_malloc = [&](void** p, long bytes) {
    cudaError_t e = cudaMalloc(p, (size_t)bytes);
    if(e != cudaSuccess)
    {
      freeScratch();
      throw std::runtime_error(std::string("RifeInterpolator: scratch cudaMalloc failed (")
                               + cudaGetErrorString(e) + ")");
    }
  };
  checked_malloc(&scratch_prev_dev_, hw4);
  checked_malloc(&scratch_cur_dev_,  hw4);
  checked_malloc(&scratch_out_dev_,  (long)total_out * hw4);
  checked_malloc(&scratch_seq_a_,    (long)max_frames * per * sizeof(__half));
  checked_malloc(&scratch_seq_b_,    (long)max_frames * per * sizeof(__half));
  checked_malloc(&scratch_packed_,   (long)maxB * 6 * H * W * sizeof(__half));
  checked_malloc(&scratch_mids_,     (long)maxB * per * sizeof(__half));
  scratch_H_ = H; scratch_W_ = W; scratch_exp_ = exp;
}

RifeInterpolator::~RifeInterpolator()
{
  freeScratch();
  if(stream_)
    cudaStreamDestroy(stream_);
}

// Core GPU recursion: seq_dev holds `count` RGB-CHW-fp16 frames [count,3,H,W]; on return,
// seq_dev holds (2^exp*(count-1)+1) frames, fully subdivided, in temporal order (endpoints kept).
// Returns the final frame count. Uses two ping-pong device buffers sized for the max level.
// packed/mids are PRE-ALLOCATED by the caller (RifeInterpolator scratch) — no malloc/free here.
static int subdivide_chw(
    RifeWrapper& rife, __half* seq_a, __half* seq_b, int count, int H, int W, int exp,
    cudaStream_t stream, __half** out_buf, __half* packed, __half* mids)
{
  const long per = (long)3 * H * W;
  __half* cur = seq_a;
  __half* other = seq_b;

  int n = count; // number of frames currently in `cur`
  for(int level = 0; level < exp; ++level)
  {
    int B = n - 1; // number of pairs / midpoints this level (cur has n frames)
    // pack pairs: prevs = cur[0..B-1], nexts = cur[1..B]  (contiguous overlapping slices)
    launch_rife_pack_pairs(cur, cur + per, packed, B, H, W, stream);
    rife.interpolate(packed, mids, B, H, W, stream);
    // interleave: other[2n-1] = {cur frames at even slots, mids at odd slots}
    launch_rife_interleave(cur, mids, other, n, H, W, stream);
    n = 2 * n - 1; // new frame count after inserting B midpoints
    __half* t = cur; cur = other; other = t;
  }

  *out_buf = cur;
  return n;
}

int RifeInterpolator::interpolate_gpu(
    const uint8_t* prev_rgba_dev, const uint8_t* cur_rgba_dev, int H, int W, int exp,
    uint8_t* out_frames_dev, cudaStream_t stream)
{
  if(!stream)
    stream = stream_;
  const long per = (long)3 * H * W;
  const long hw4 = (long)H * W * 4;

  if(exp <= 0)
  {
    // passthrough: just the current frame
    cudaMemcpyAsync(out_frames_dev, cur_rgba_dev, hw4, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    return 1;
  }

  int total_out = 1;
  for(int i = 0; i < exp; ++i)
    total_out *= 2; // 2^exp

  // Persistent scratch (no per-call malloc/free): seq_a/seq_b ping-pong + packed/mids for the engine.
  ensureScratch(H, W, exp);
  __half* seq_a = static_cast<__half*>(scratch_seq_a_);
  __half* seq_b = static_cast<__half*>(scratch_seq_b_);
  launch_rife_rgba_to_chw01(prev_rgba_dev, seq_a, H, W, stream);
  launch_rife_rgba_to_chw01(cur_rgba_dev, seq_a + per, H, W, stream);

  __half* final_buf = nullptr;
  int n = subdivide_chw(*rife_, seq_a, seq_b, 2, H, W, exp, stream, &final_buf,
                        static_cast<__half*>(scratch_packed_), static_cast<__half*>(scratch_mids_));
  // n == 2^exp + 1 (both endpoints). Display order drops the leading prev endpoint -> keep [1:n).
  // Convert frames 1..n-1 to RGBA into out_frames_dev (total_out = n-1 frames).
  for(int f = 1; f < n; ++f)
  {
    launch_rife_chw01_to_rgba(
        final_buf + (long)f * per, out_frames_dev + (long)(f - 1) * hw4, H, W, stream);
  }
  cudaStreamSynchronize(stream);
  return total_out;
}

int RifeInterpolator::interpolate(
    const uint8_t* prev_rgba, const uint8_t* cur_rgba, int H, int W, int exp, uint8_t* out_frames)
{
  const long hw4 = (long)H * W * 4;
  int total_out = 1;
  for(int i = 0; i < (exp > 0 ? exp : 0); ++i)
    total_out *= 2;
  if(exp <= 0)
    total_out = 1;

  // Persistent device scratch for the H->D inputs + D->H output (no per-call malloc/free).
  ensureScratch(H, W, exp > 0 ? exp : 1);
  uint8_t* prev_dev = static_cast<uint8_t*>(scratch_prev_dev_);
  uint8_t* cur_dev = static_cast<uint8_t*>(scratch_cur_dev_);
  uint8_t* out_dev = static_cast<uint8_t*>(scratch_out_dev_);
  cudaMemcpyAsync(prev_dev, prev_rgba, hw4, cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(cur_dev, cur_rgba, hw4, cudaMemcpyHostToDevice, stream_);

  int n = interpolate_gpu(prev_dev, cur_dev, H, W, exp, out_dev, stream_);

  // D->H of the result. Async + per-stream sync (a blocking cudaMemcpy serializes the whole context).
  cudaMemcpyAsync(out_frames, out_dev, (long)n * hw4, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  return n;
}

} // namespace librediffusion
