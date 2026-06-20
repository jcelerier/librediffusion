/**
 * FLUX.2-klein-4B specific CUDA kernels (bf16).
 *
 * All seam-validated against the Phase-0 Python golden (diffusers Flux2KleinPipeline, SHA f3d42be).
 * Layout facts (verified from pipeline_flux2_klein.py):
 *   - VAE latent: [B,32,Lh,Lw]  (Lh=H/8, Lw=W/8)
 *   - patchify: [B,32,Lh,Lw] view [B,32,Lh/2,2,Lw/2,2] permute(0,1,3,5,2,4) -> [B,128,Lh/2,Lw/2]
 *   - bn normalize (encode):  (x - bn_mean[c]) / bn_std[c]   over the 128 patchified channels
 *   - bn denormalize (decode): x * bn_std[c] + bn_mean[c]
 *   - pack:   [B,128,Th,Tw] reshape [B,128,Th*Tw] permute(0,2,1) -> [B,Th*Tw,128]   (Th=Lh/2,Tw=Lw/2)
 *   - unpack: inverse of pack (scatter by ids; here ids are dense row-major so = permute back)
 *   - encoder stack: 3 layers [B,Lt,2560] -> stack dim1 -> permute(0,2,1,3) -> reshape [B,Lt,7680]
 *       i.e. ehs[b,s, c*2560 + d] = layer_c[b,s,d]
 *   - rope ids (4 col, T,H,W,L): img latent ids = (0, h, w, 0) row-major over Th x Tw;
 *       txt ids = (0,0,0, l) for l in 0..Lt-1.   (cartesian_prod order)
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <kernels.hpp>

using bf16 = __nv_bfloat16;

// ---- stack 3 qwen layers -> 7680 -------------------------------------------------------------
// in: layer0,layer1,layer2 each [B,Lt,2560]; out [B,Lt,7680]
__global__ void k_stack3_7680(
    const bf16* l0, const bf16* l1, const bf16* l2, bf16* out, int B, int Lt, int D)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * Lt * D;
  if(idx >= total) return;
  int d = idx % D;
  long bs = idx / D;  // b*Lt + s
  // out[bs, 0*D + d] = l0; 1*D+d = l1; 2*D+d = l2
  long ob = bs * (3 * D);
  out[ob + 0 * D + d] = l0[idx];
  out[ob + 1 * D + d] = l1[idx];
  out[ob + 2 * D + d] = l2[idx];
}

void launch_klein_stack3_7680(
    const void* l0, const void* l1, const void* l2, void* out, int B, int Lt, int D,
    void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * Lt * D;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_stack3_7680<<<blocks, threads, 0, s>>>(
      (const bf16*)l0, (const bf16*)l1, (const bf16*)l2, (bf16*)out, B, Lt, D);
}

// ---- patchify [B,32,Lh,Lw] -> [B,128,Lh/2,Lw/2] ---------------------------------------------
// view [B,C,Lh/2,2,Lw/2,2] permute(0,1,3,5,2,4) reshape [B,C*4,Lh/2,Lw/2]
// out channel = c*4 + (ph*2 + pw); out[b, c*4+ph*2+pw, th, tw] = in[b,c, th*2+ph, tw*2+pw]
__global__ void k_patchify(const bf16* in, bf16* out, int B, int C, int Lh, int Lw)
{
  int Th = Lh / 2, Tw = Lw / 2;
  long total = (long)B * C * Lh * Lw;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  // decode input index [b,c,y,x]
  int x = idx % Lw;
  long t = idx / Lw;
  int y = t % Lh;
  t /= Lh;
  int c = t % C;
  int b = t / C;
  int th = y / 2, ph = y % 2;
  int tw = x / 2, pw = x % 2;
  int oc = c * 4 + ph * 2 + pw;
  long oidx = (((long)b * (C * 4) + oc) * Th + th) * Tw + tw;
  out[oidx] = in[idx];
}

void launch_klein_patchify(const void* in, void* out, int B, int C, int Lh, int Lw, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * C * Lh * Lw;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_patchify<<<blocks, threads, 0, s>>>((const bf16*)in, (bf16*)out, B, C, Lh, Lw);
}

// ---- unpatchify [B,128,Th,Tw] -> [B,32,Th*2,Tw*2] -------------------------------------------
// inverse: in[b, c*4+ph*2+pw, th, tw] -> out[b,c, th*2+ph, tw*2+pw]
__global__ void k_unpatchify(const bf16* in, bf16* out, int B, int C, int Th, int Tw)
{
  int Lh = Th * 2, Lw = Tw * 2;
  long total = (long)B * (C * 4) * Th * Tw;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int tw = idx % Tw;
  long t = idx / Tw;
  int th = t % Th;
  t /= Th;
  int oc = t % (C * 4);
  int b = t / (C * 4);
  int c = oc / 4;
  int p = oc % 4;
  int ph = p / 2, pw = p % 2;
  int y = th * 2 + ph, x = tw * 2 + pw;
  long oidx = (((long)b * C + c) * Lh + y) * Lw + x;
  out[oidx] = in[idx];
}

void launch_klein_unpatchify(const void* in, void* out, int B, int C, int Th, int Tw, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * (C * 4) * Th * Tw;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_unpatchify<<<blocks, threads, 0, s>>>((const bf16*)in, (bf16*)out, B, C, Th, Tw);
}

// ---- pack [B,128,Th,Tw] -> [B,Th*Tw,128] ----------------------------------------------------
// reshape [B,128,Th*Tw] permute(0,2,1): out[b, hw, c] = in[b,c,hw]
__global__ void k_pack(const bf16* in, bf16* out, int B, int C, int Th, int Tw)
{
  int HW = Th * Tw;
  long total = (long)B * C * HW;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int hw = idx % HW;
  long t = idx / HW;
  int c = t % C;
  int b = t / C;
  long oidx = ((long)b * HW + hw) * C + c;
  out[oidx] = in[idx];
}

void launch_klein_pack(const void* in, void* out, int B, int C, int Th, int Tw, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * C * Th * Tw;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_pack<<<blocks, threads, 0, s>>>((const bf16*)in, (bf16*)out, B, C, Th, Tw);
}

// ---- unpack [B,Th*Tw,128] -> [B,128,Th,Tw] --------------------------------------------------
__global__ void k_unpack(const bf16* in, bf16* out, int B, int C, int Th, int Tw)
{
  int HW = Th * Tw;
  long total = (long)B * HW * C;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int c = idx % C;
  long t = idx / C;
  int hw = t % HW;
  int b = t / HW;
  long oidx = ((long)b * C + c) * HW + hw;
  out[oidx] = in[idx];
}

void launch_klein_unpack(const void* in, void* out, int B, int C, int Th, int Tw, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * Th * Tw * C;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_unpack<<<blocks, threads, 0, s>>>((const bf16*)in, (bf16*)out, B, C, Th, Tw);
}

// ---- bn normalize / denormalize over the 128 patchified channels ----------------------------
// data layout [B,128,Th,Tw]; mean/std are [128] fp32.
__global__ void k_bn(const bf16* in, bf16* out, const float* mean, const float* std_, int B, int C,
                     int HW, bool denorm)
{
  long total = (long)B * C * HW;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int c = (idx / HW) % C;
  float v = __bfloat162float(in[idx]);
  float r = denorm ? (v * std_[c] + mean[c]) : ((v - mean[c]) / std_[c]);
  out[idx] = __float2bfloat16(r);
}

void launch_klein_bn(const void* in, void* out, const void* mean, const void* std_, int B, int C,
                     int HW, int denorm, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)B * C * HW;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_bn<<<blocks, threads, 0, s>>>(
      (const bf16*)in, (bf16*)out, (const float*)mean, (const float*)std_, B, C, HW, denorm != 0);
}

// ---- build 4-col RoPE ids (fp32) ------------------------------------------------------------
// img ids: row-major over Th x Tw, (T=0,H=h,W=w,L=0)
__global__ void k_img_ids(float* out, int Th, int Tw)
{
  long total = (long)Th * Tw;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int h = idx / Tw, w = idx % Tw;
  out[idx * 4 + 0] = 0.f;
  out[idx * 4 + 1] = (float)h;
  out[idx * 4 + 2] = (float)w;
  out[idx * 4 + 3] = 0.f;
}

void launch_klein_img_ids(void* out, int Th, int Tw, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)Th * Tw;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_img_ids<<<blocks, threads, 0, s>>>((float*)out, Th, Tw);
}

// txt ids: (T=0,H=0,W=0,L=l)
__global__ void k_txt_ids(float* out, int Lt)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= Lt) return;
  out[idx * 4 + 0] = 0.f;
  out[idx * 4 + 1] = 0.f;
  out[idx * 4 + 2] = 0.f;
  out[idx * 4 + 3] = (float)idx;
}

void launch_klein_txt_ids(void* out, int Lt, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  int threads = 256;
  int blocks = (Lt + threads - 1) / threads;
  k_txt_ids<<<blocks, threads, 0, s>>>((float*)out, Lt);
}

// ---- Euler axpy: x += dt * v   (bf16, fp32 accumulate) --------------------------------------
__global__ void k_euler_axpy(bf16* x, const bf16* v, float dt, long N)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= N) return;
  float xv = __bfloat162float(x[idx]);
  float vv = __bfloat162float(v[idx]);
  x[idx] = __float2bfloat16(xv + dt * vv);
}

void launch_klein_euler_axpy(void* x, const void* v, float dt, long N, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);
  k_euler_axpy<<<blocks, threads, 0, s>>>((bf16*)x, (const bf16*)v, dt, N);
}

// ---- PCG32 randn (bf16), index-deterministic (matches pcg.hpp / deterministic_noise.py) -----
__host__ __device__ static inline unsigned int pcg_rotr32(unsigned int v, unsigned int r)
{
  return (v >> r) | (v << ((32u - r) & 31u));
}
// PCG32 advance with a PER-STREAM increment `inc` (matches pcg.hpp). The increment MUST be added
// at every step and MUST encode the stream selector, otherwise every element draws the same value.
__host__ __device__ static inline unsigned int pcg_step(unsigned long long& state,
                                                        unsigned long long inc)
{
  unsigned long long old = state;
  state = old * 6364136223846793005ULL + inc;
  unsigned int xorshifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
  unsigned int rot = (unsigned int)(old >> 59u);
  return pcg_rotr32(xorshifted, rot);
}
// Seed a unique PCG32 stream. s0 = global seed, s1 = per-element stream id (here: the flat index).
// inc is returned so the caller threads it through every subsequent pcg_step (as pcg.hpp does).
__host__ __device__ static inline void pcg_seed(unsigned long long& state, unsigned long long& inc,
                                                unsigned long long s0, unsigned long long s1)
{
  state = 0u;
  inc = (s1 << 1u) | 1u;
  pcg_step(state, inc);
  state += s0;
  pcg_step(state, inc);
}

__global__ void k_randn_bf16(bf16* out, unsigned long long seed, long N)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= N) return;
  unsigned long long state, inc;
  pcg_seed(state, inc, seed, (unsigned long long)idx);
  unsigned int u0 = pcg_step(state, inc);
  unsigned int u1 = pcg_step(state, inc);
  float f0 = ((float)u0 + 1.0f) * 2.3283064365386963e-10f;  // (u+1)*2^-32
  float f1 = ((float)u1 + 1.0f) * 2.3283064365386963e-10f;
  float r = sqrtf(-2.0f * logf(f0));
  float z0 = r * cosf(6.283185307179586f * f1);
  out[idx] = __float2bfloat16(z0);
}

void launch_klein_randn_bf16(void* out, unsigned long long seed, long N, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);
  k_randn_bf16<<<blocks, threads, 0, s>>>((bf16*)out, seed, N);
}

// ---- fp32 <-> bf16 and image post (bf16 [-1,1] CHW -> uint8 RGBA HWC) -----------------------
__global__ void k_bf16_to_fp32(const bf16* in, float* out, long N)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) out[idx] = __bfloat162float(in[idx]);
}
void launch_klein_bf16_to_fp32(const void* in, void* out, long N, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);
  k_bf16_to_fp32<<<blocks, threads, 0, s>>>((const bf16*)in, (float*)out, N);
}

__global__ void k_fp32_to_bf16(const float* in, bf16* out, long N)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) out[idx] = __float2bfloat16(in[idx]);
}
void launch_klein_fp32_to_bf16(const void* in, void* out, long N, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);
  k_fp32_to_bf16<<<blocks, threads, 0, s>>>((const float*)in, (bf16*)out, N);
}

// image: in bf16 [B,3,H,W] in [-1,1]; out uint8 RGBA [B,H,W,4]
__global__ void k_chw_to_rgba_u8(const bf16* in, unsigned char* out, int H, int W)
{
  long total = (long)H * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  for(int c = 0; c < 3; ++c)
  {
    // ((c*H + y)*W + x) == c*total + idx  (idx = y*W + x); avoids the div/mod for y,x
    float v = __bfloat162float(in[(long)c * total + idx]);
    v = (v * 0.5f + 0.5f);
    v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    out[(idx * 4) + c] = (unsigned char)(v * 255.0f + 0.5f);
  }
  out[idx * 4 + 3] = 255;
}
void launch_klein_chw_to_rgba_u8(const void* in, void* out, int H, int W, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)H * W;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_chw_to_rgba_u8<<<blocks, threads, 0, s>>>((const bf16*)in, (unsigned char*)out, H, W);
}

// ===== streaming / reference image ================================================

// reference frame: in uint8 RGBA [H,W,4] -> out bf16 CHW [3,H,W] in [-1,1]  (Flux2ImageProcessor:
// preprocess() does ToTensor()/255 then 2x-1; alpha dropped). Inverse of k_chw_to_rgba_u8.
__global__ void k_rgba_u8_to_chw_m1p1(const unsigned char* in, bf16* out, int H, int W)
{
  long total = (long)H * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  for(int c = 0; c < 3; ++c)
  {
    float v = (float)in[idx * 4 + c] * (1.0f / 255.0f);
    v = v * 2.0f - 1.0f;
    out[(long)c * total + idx] = __float2bfloat16(v);  // c*total+idx == ((c*H+y)*W+x); no div/mod
  }
}
void launch_klein_rgba_u8_to_chw_m1p1(const void* in, void* out, int H, int W, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)H * W;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_rgba_u8_to_chw_m1p1<<<blocks, threads, 0, s>>>((const unsigned char*)in, (bf16*)out, H, W);
}

// reference-image 4-col RoPE ids: row-major over Th x Tw, (T=t_offset, H=h, W=w, L=0).
// t_offset = scale + scale*i for the i-th reference image (pipeline.py:411, scale=10).
__global__ void k_ref_ids(float* out, int Th, int Tw, float t_offset)
{
  long total = (long)Th * Tw;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total) return;
  int h = idx / Tw, w = idx % Tw;
  out[idx * 4 + 0] = t_offset;
  out[idx * 4 + 1] = (float)h;
  out[idx * 4 + 2] = (float)w;
  out[idx * 4 + 3] = 0.f;
}
void launch_klein_ref_ids(void* out, int Th, int Tw, float t_offset, void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long total = (long)Th * Tw;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);
  k_ref_ids<<<blocks, threads, 0, s>>>((float*)out, Th, Tw, t_offset);
}

// concatenate two packed token sequences along the sequence dim:
// a [B, La, C] , b [B, Lb, C] -> out [B, La+Lb, C]   (B==1 in streaming)
__global__ void k_concat_seq_bf16(const bf16* a, const bf16* b, bf16* out, long Na, long Nb)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = Na + Nb;
  if(idx >= tot) return;
  out[idx] = (idx < Na) ? a[idx] : b[idx - Na];
}
void launch_klein_concat_seq(const void* a, const void* b, void* out, long Na, long Nb,
                             void* stream_ptr)
{
  cudaStream_t s = (cudaStream_t)stream_ptr;
  long tot = Na + Nb;
  int threads = 256;
  int blocks = (int)((tot + threads - 1) / threads);
  k_concat_seq_bf16<<<blocks, threads, 0, s>>>((const bf16*)a, (const bf16*)b, (bf16*)out, Na, Nb);
}
