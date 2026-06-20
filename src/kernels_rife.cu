/**
 * RIFE (IFNet) frame-interpolation CUDA helpers (fp16). Model-agnostic.
 *
 * The IFNet itself (incl. warp/grid_sample) runs entirely inside the TRT engine — GridSample is
 * native at opset 17, so NO CUDA grid_sample fallback was needed. These kernels only handle the
 * host-orchestrated plumbing around the engine:
 *   - RGBA uint8 (NHWC [H,W,4]) <-> RGB fp16 CHW [3,H,W] in [0,1]   (RIFE's I/O domain)
 *   - pack two RGB CHW frames into the engine's 6-channel "frames" input
 *   - interleave-scatter for the recursive-midpoint subdivision (frames[0::2]=old, [1::2]=mids)
 *
 * Layout/contract verified against the Python golden (gen_golden.py): frames in [0,1], CHW,
 * channel order RGB, midpoint = engine(cat([img0,img1],dim=1)).
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <kernels.hpp>

using half = __half;

// RGBA uint8 NHWC [H,W,4] -> RGB fp16 CHW [3,H,W] in [0,1] (alpha dropped).
__global__ void k_rife_rgba_to_chw01(const unsigned char* rgba, half* chw, int H, int W)
{
  long total = (long)H * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  for(int ch = 0; ch < 3; ++ch)
  {
    float v = (float)rgba[idx * 4 + ch] * (1.0f / 255.0f);
    chw[(long)ch * total + idx] = __float2half(v);
  }
}

void launch_rife_rgba_to_chw01(const void* rgba, void* chw, int H, int W, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)H * W;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_rife_rgba_to_chw01<<<blocks, threads, 0, s>>>(
      (const unsigned char*)rgba, (half*)chw, H, W);
}

// RGB fp16 CHW [3,H,W] in [0,1] -> RGBA uint8 NHWC [H,W,4] (alpha=255).
__global__ void k_rife_chw01_to_rgba(const half* chw, unsigned char* rgba, int H, int W)
{
  long total = (long)H * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  for(int ch = 0; ch < 3; ++ch)
  {
    float v = __half2float(chw[(long)ch * total + idx]);
    v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    rgba[idx * 4 + ch] = (unsigned char)(v * 255.0f + 0.5f);
  }
  rgba[idx * 4 + 3] = 255;
}

void launch_rife_chw01_to_rgba(const void* chw, void* rgba, int H, int W, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)H * W;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_rife_chw01_to_rgba<<<blocks, threads, 0, s>>>(
      (const half*)chw, (unsigned char*)rgba, H, W);
}

// Pack B consecutive (img0,img1) pairs into the engine's "frames" [B,6,H,W].
// prevs/nexts are [B,3,H,W]; out[b, 0:3] = prevs[b], out[b, 3:6] = nexts[b].
__global__ void k_rife_pack_pairs(const half* prevs, const half* nexts, half* out, int B, int H, int W)
{
  long hw = (long)H * W;
  long per = 3 * hw;          // per-frame element count
  long total = (long)B * per; // total over one of prevs/nexts
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int b = (int)(idx / per);
  long r = idx - (long)b * per; // [0, 3*hw)
  long ob = (long)b * 6 * hw;
  out[ob + r] = prevs[idx];           // channels 0..2
  out[ob + per + r] = nexts[idx];     // channels 3..5
}

void launch_rife_pack_pairs(
    const void* prevs, const void* nexts, void* out, int B, int H, int W, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * 3 * H * W;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_rife_pack_pairs<<<blocks, threads, 0, s>>>(
      (const half*)prevs, (const half*)nexts, (half*)out, B, H, W);
}

// Interleave-scatter one subdivision level: dst has 2*B-1 frames; dst[2k]=src[k], dst[2k+1]=mids[k].
// src: [B,3,H,W]; mids: [B-1,3,H,W]; dst: [2B-1,3,H,W].
__global__ void k_rife_interleave(const half* src, const half* mids, half* dst, int B, int H, int W)
{
  long per = (long)3 * H * W;
  long total_src = (long)B * per;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  // first pass: copy src to even slots
  if(idx < total_src)
  {
    int k = (int)(idx / per);
    long r = idx - (long)k * per;
    dst[(long)(2 * k) * per + r] = src[idx];
  }
  // second pass region: copy mids to odd slots
  long total_mids = (long)(B - 1) * per;
  long midx = idx - total_src;
  if(midx >= 0 && midx < total_mids)
  {
    int k = (int)(midx / per);
    long r = midx - (long)k * per;
    dst[(long)(2 * k + 1) * per + r] = mids[midx];
  }
}

void launch_rife_interleave(
    const void* src, const void* mids, void* dst, int B, int H, int W, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long per = (long)3 * H * W;
  long total = (long)B * per + (long)(B - 1) * per;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_rife_interleave<<<blocks, threads, 0, s>>>(
      (const half*)src, (const half*)mids, (half*)dst, B, H, W);
}
