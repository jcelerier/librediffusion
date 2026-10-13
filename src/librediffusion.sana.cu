/** SANA-Streaming DiT — pure CUDA algorithm (kernels + thin host launchers).
 *
 *  The DiT architecture and the Gated-Delta-Net (GDN) linear-attention recurrence
 *  are ported from NVIDIA SANA-Streaming (https://github.com/NVlabs/Sana),
 *  licensed Apache-2.0. This is an independent C++/CUDA re-implementation,
 *  validated numerically against the reference (see PR description). Model
 *  weights are research-only under the NVIDIA Open Model License and are NOT
 *  included or distributed here — they are loaded at runtime from a user path.
 *
 *  This translation unit is compiled by nvcc and holds ONLY the device kernels plus a
 *  `launch_*` host wrapper for each (grid/block derived exactly as the original inline
 *  launches did). ALL C++/TRT/GEMM-library/header-map orchestration lives in the pure-C++
 *  units librediffusion.sana.cpp / librediffusion.sana_trt.cpp, which call these
 *  launchers via librediffusion.sana_kernels.hpp. Pure CUDA algorithm only here.
 *
 *  All internals live under `librediffusion::sana` to avoid symbol collisions
 *  with the library's other CUDA units (kernels.cu etc.).
 */
#include "librediffusion.sana_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace librediffusion
{
namespace sana
{

// Grid-size helper: number of t-wide blocks needed to cover n elements (ceil-div).
static inline int gr(long n, int t = 256)
{
  return (int)((n + t - 1) / t);
}

// ================= generic kernels (verbatim from validated dit_common.cuh) =================
// Add a per-column bias vector b[No] into the (M,No) row-major matrix Y, in place.
__global__ void k_add_bias(float* Y, const float* b, int M, int No)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * No)
    return;
  Y[i] += b[i % No];
}
// LayerNorm (no affine) over the D-wide feature axis of each of the M rows: one
// threadblock per row, warp-shuffle + shared-mem reductions for mean and variance.
__global__ void k_layernorm(float* Y, const float* X, int M, int D, float eps)
{
  int m = blockIdx.x;
  if(m >= M)
    return;
  const float* x = X + (long)m * D;
  float* y = Y + (long)m * D;
  __shared__ float sm, sv;
  float s = 0;
  for(int i = threadIdx.x; i < D; i += blockDim.x)
    s += x[i];
  for(int o = 16; o; o >>= 1)
    s += __shfl_down_sync(0xffffffff, s, o);
  __shared__ float ps[32];
  if((threadIdx.x & 31) == 0)
    ps[threadIdx.x >> 5] = s;
  __syncthreads();
  if(threadIdx.x == 0)
  {
    float t = 0;
    for(int i = 0; i < (blockDim.x + 31) / 32; i++)
      t += ps[i];
    sm = t / D;
  }
  __syncthreads();
  float mu = sm;
  float v = 0;
  for(int i = threadIdx.x; i < D; i += blockDim.x)
  {
    float d = x[i] - mu;
    v += d * d;
  }
  for(int o = 16; o; o >>= 1)
    v += __shfl_down_sync(0xffffffff, v, o);
  if((threadIdx.x & 31) == 0)
    ps[threadIdx.x >> 5] = v;
  __syncthreads();
  if(threadIdx.x == 0)
  {
    float t = 0;
    for(int i = 0; i < (blockDim.x + 31) / 32; i++)
      t += ps[i];
    sv = t / D;
  }
  __syncthreads();
  float inv = rsqrtf(sv + eps);
  for(int i = threadIdx.x; i < D; i += blockDim.x)
    y[i] = (x[i] - mu) * inv;
}
// RMSNorm with per-feature weight W over the D-wide axis of each of the M rows
// (y = x / sqrt(mean(x^2)+eps) * W): one threadblock per row.
__global__ void
k_rmsnorm(float* Y, const float* X, const float* W, int M, int D, float eps)
{
  int m = blockIdx.x;
  if(m >= M)
    return;
  const float* x = X + (long)m * D;
  float* y = Y + (long)m * D;
  __shared__ float sv;
  float v = 0;
  for(int i = threadIdx.x; i < D; i += blockDim.x)
    v += x[i] * x[i];
  for(int o = 16; o; o >>= 1)
    v += __shfl_down_sync(0xffffffff, v, o);
  __shared__ float ps[32];
  if((threadIdx.x & 31) == 0)
    ps[threadIdx.x >> 5] = v;
  __syncthreads();
  if(threadIdx.x == 0)
  {
    float t = 0;
    for(int i = 0; i < (blockDim.x + 31) / 32; i++)
      t += ps[i];
    sv = t / D;
  }
  __syncthreads();
  float inv = rsqrtf(sv + eps);
  for(int i = threadIdx.x; i < D; i += blockDim.x)
    y[i] = x[i] * inv * W[i];
}
// Adaptive-LayerNorm modulation: Y = X*(1+scale) + shift, with per-feature scale/shift
// (D-vectors broadcast across all M rows). The DiT block's adaLN-zero conditioning.
__global__ void
k_adaln(float* Y, const float* X, const float* shift, const float* scale, int M, int D)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * D)
    return;
  int c = i % D;
  Y[i] = X[i] * (1.f + scale[c]) + shift[c];
}
// Gated residual add: Y = A + gate*Bm, with a per-feature gate (D-vector). Used for
// the adaLN-zero residual (x = x + gate_msa * attn_out, etc.).
__global__ void
k_add_gate(float* Y, const float* A, const float* gate, const float* Bm, int M, int D)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * D)
    return;
  int c = i % D;
  Y[i] = A[i] + gate[c] * Bm[i];
}
// Elementwise add of two length-n buffers: Y = A + Bm.
__global__ void k_add(float* Y, const float* A, const float* Bm, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
    Y[i] = A[i] + Bm[i];
}
// Apply the SiLU activation x*sigmoid(x) to each of the n elements of Y, in place.
__global__ void k_silu_inplace(float* Y, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
  {
    float x = Y[i];
    Y[i] = x / (1.f + expf(-x));
  }
}
// Output-gate: multiply Y elementwise by SiLU(og) (Y *= og*sigmoid(og)), over n elements.
__global__ void k_gate_silu(float* Y, const float* og, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
  {
    float g = og[i];
    Y[i] *= g / (1.f + expf(-g));
  }
}
// Per-row inverse-RMS scalar: out[m] = 1/sqrt(mean(x[m]^2)+eps) for each of the M rows.
// Used to normalize q/k in the GDN attention path (one scalar per token).
__global__ void k_inv_rms(float* out, const float* X, int M, int D, float eps)
{
  int m = blockIdx.x;
  if(m >= M)
    return;
  const float* x = X + (long)m * D;
  __shared__ float ssv;
  float v = 0;
  for(int i = threadIdx.x; i < D; i += blockDim.x)
    v += x[i] * x[i];
  for(int o = 16; o; o >>= 1)
    v += __shfl_down_sync(0xffffffff, v, o);
  __shared__ float ps[32];
  if((threadIdx.x & 31) == 0)
    ps[threadIdx.x >> 5] = v;
  __syncthreads();
  if(threadIdx.x == 0)
  {
    float t = 0;
    for(int i = 0; i < (blockDim.x + 31) / 32; i++)
      t += ps[i];
    out[m] = rsqrtf(t / D + eps);
  }
  (void)ssv;
}
// Numerically-stable row-wise softmax over `cols` of each of `rows` rows, in place.
// If valid>0, columns j>=valid are masked out (used to ignore padded text tokens).
__global__ void k_softmax(float* S, int rows, int cols, int valid)
{
  int r = blockIdx.x;
  if(r >= rows)
    return;
  float* s = S + (long)r * cols;
  float mx = -1e30f;
  for(int j = threadIdx.x; j < cols; j += blockDim.x)
  {
    float v = (valid > 0 && j >= valid) ? -1e4f : s[j];
    mx = fmaxf(mx, v);
  }
  for(int o = 16; o; o >>= 1)
    mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
  __shared__ float pm[32], ps[32];
  if((threadIdx.x & 31) == 0)
    pm[threadIdx.x >> 5] = mx;
  __syncthreads();
  __shared__ float M_, Z_;
  if(threadIdx.x == 0)
  {
    float t = -1e30f;
    for(int i = 0; i < (blockDim.x + 31) / 32; i++)
      t = fmaxf(t, pm[i]);
    M_ = t;
  }
  __syncthreads();
  float z = 0;
  for(int j = threadIdx.x; j < cols; j += blockDim.x)
  {
    float v = (valid > 0 && j >= valid) ? -1e4f : s[j];
    z += expf(v - M_);
  }
  for(int o = 16; o; o >>= 1)
    z += __shfl_down_sync(0xffffffff, z, o);
  if((threadIdx.x & 31) == 0)
    ps[threadIdx.x >> 5] = z;
  __syncthreads();
  if(threadIdx.x == 0)
  {
    float t = 0;
    for(int i = 0; i < (blockDim.x + 31) / 32; i++)
      t += ps[i];
    Z_ = t;
  }
  __syncthreads();
  float inv = 1.f / Z_;
  for(int j = threadIdx.x; j < cols; j += blockDim.x)
  {
    float v = (valid > 0 && j >= valid) ? -1e4f : s[j];
    s[j] = expf(v - M_) * inv;
  }
}
// Reshape token-major X(M,C) into head-major O(HEADS,M,HD): scatters each token's
// C=HEADS*HD features so cuBLAS batched-GEMM can treat each head as an independent matrix.
__global__ void k_split_heads(float* O, const float* X, int M)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * C)
    return;
  int d = i % HD;
  int h = (i / HD) % HEADS;
  int m = i / C;
  O[((long)h * M + m) * HD + d] = X[i];
}
// Inverse of k_split_heads: gather head-major X(HEADS,M,HD) back to token-major O(M,C).
__global__ void k_merge_heads(float* O, const float* X, int M)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * C)
    return;
  int d = i % HD;
  int h = (i / HD) % HEADS;
  int m = i / C;
  O[i] = X[((long)h * M + m) * HD + d];
}
// Apply rotary position embedding to token-major X(M,HEADS,HD) using per-token cos/sin
// tables (M,HD); rotates each adjacent dim pair (d, d^1). Output Y is token-major.
__global__ void
k_rope_bnhd(float* Y, const float* X, const float* cos, const float* sin, int M)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * C)
    return;
  int d = i % HD;
  int h = (i / HD) % HEADS;
  int m = i / C;
  int dp = d ^ 1;
  long base = ((long)m * HEADS + h) * HD;
  float cc = cos[(long)m * HD + d], ss = sin[(long)m * HD + d];
  Y[i] = X[base + d] * cc + X[base + dp] * ss;
}
// ================= GDN kernel (optimised: row-split grid + shared-mem state) =================
// One threadblock per (batch*head, row-group g in [0,GDN_G)); D=112 state rows are split
// across GDN_G blocks so the whole GPU is used, and the recurrent state + K/Q tiles live in
// dynamic shared memory (no global round-trips per step). Numerically identical to the naive
// scan (validated ~1e-7 vs the fp32 reference in gdn-kernel-test).
#define GDN_TS 64
// Gated-Delta-Net bidirectional linear-attention recurrence. Each frame updates a
// (D,D) key-value state and a length-D normalizer via the delta rule, decayed per
// frame; the state is read out by the current-frame query to produce num/den, which
// k_gdn_out later divides into the attention output. Runs a forward then a backward
// pass over the T frames (bidirectional). The state rows are split across GDN_G blocks
// per head so the whole GPU is busy; state + K/Q tiles are staged in shared memory.
__global__ void gdn_bidi(Bufs b)
{
  const int D = b.D, T = b.T, S = b.S, N = b.N;
  const int G = b.G, R = (D + G - 1) / G;
  const int bh = blockIdx.x / G, g = blockIdx.x % G;
  const int r0 = g * R;
  if(r0 >= D)
    return;
  const int Rloc = min(R, D - r0);
  const int tid = threadIdx.x, nth = blockDim.x, TS = GDN_TS;
  const float* q = b.q + (size_t)bh * D * N;
  const float* k = b.k + (size_t)bh * D * N;
  const float* v = b.v + (size_t)bh * D * N;
  const float* qrot = b.qrot + (size_t)bh * D * N;
  const float* krot = b.krot + (size_t)bh * D * N;
  const float* beta = b.beta + (size_t)bh * T * S;
  const float* decay = b.decay + (size_t)bh * T;
  float* num = b.num + (size_t)bh * D * N;
  float* den = b.den + (size_t)bh * N;
  extern __shared__ float sh[];
  float* skv = sh;
  float* sz = skv + R * D;
  float* dvl = sz + D;
  float* dz = dvl + R * S;
  float* tK = dz + S;
  // dir==0 forward pass (seeds state from init_kv/init_z if carried in), dir==1 backward.
  for(int dir = 0; dir < 2; ++dir)
  {
    // Load or zero the recurrent state (skv = KV state, sz = normalizer) for this pass.
    if(dir == 0 && b.init_kv)
    {
      for(int i = tid; i < Rloc * D; i += nth)
        skv[i] = b.init_kv[(size_t)bh * D * D + (size_t)r0 * D + i];
      for(int i = tid; i < D; i += nth)
        sz[i] = b.init_z[(size_t)bh * D + i];
    }
    else
    {
      for(int i = tid; i < Rloc * D; i += nth)
        skv[i] = 0.f;
      for(int i = tid; i < D; i += nth)
        sz[i] = 0.f;
    }
    __syncthreads();
    for(int step = 0; step < T; ++step)
    {
      int qf, kf, outf;
      float dval;
      bool du;
      if(dir == 0)
      {
        qf = step;
        kf = step;
        outf = step;
        dval = decay[step];
        du = true;
      }
      else
      {
        outf = T - 1 - step;
        qf = T - 1 - step;
        if(step == 0)
        {
          dval = 1.f;
          du = false;
          kf = 0;
        }
        else
        {
          dval = decay[T - step];
          kf = T - step;
          du = true;
        }
      }
      // Decay the carried state by this frame's gate before absorbing the new frame.
      for(int i = tid; i < Rloc * D; i += nth)
        skv[i] *= dval;
      for(int i = tid; i < D; i += nth)
        sz[i] *= dval;
      __syncthreads();
      // Delta-rule update: fold frame `kf`'s (k,v) into the KV state and normalizer.
      if(du)
      {
        for(int s0 = 0; s0 < S; s0 += TS)
        {
          int ts = min(TS, S - s0);
          for(int idx = tid; idx < D * ts; idx += nth)
          {
            int din = idx / ts, j = idx % ts;
            tK[din * TS + j] = krot[(size_t)din * N + kf * S + s0 + j];
          }
          __syncthreads();
          for(int idx = tid; idx < Rloc * ts; idx += nth)
          {
            int r = idx / ts, j = idx % ts;
            int s = s0 + j;
            float vp = 0.f;
            const float* row = &skv[r * D];
            for(int din = 0; din < D; ++din)
              vp += row[din] * tK[din * TS + j];
            dvl[r * S + s]
                = (v[(size_t)(r0 + r) * N + kf * S + s] - vp) * beta[kf * S + s];
          }
          __syncthreads();
        }
        for(int s = tid; s < S; s += nth)
        {
          int bk = kf * S + s;
          float zp = 0.f;
          for(int d = 0; d < D; ++d)
            zp += sz[d] * k[(size_t)d * N + bk];
          dz[s] = (1.f - zp) * beta[bk];
        }
        __syncthreads();
        for(int s0 = 0; s0 < S; s0 += TS)
        {
          int ts = min(TS, S - s0);
          for(int idx = tid; idx < D * ts; idx += nth)
          {
            int din = idx / ts, j = idx % ts;
            tK[din * TS + j] = krot[(size_t)din * N + kf * S + s0 + j];
          }
          __syncthreads();
          for(int idx = tid; idx < Rloc * D; idx += nth)
          {
            int r = idx / D, din = idx % D;
            float acc = 0.f;
            for(int j = 0; j < ts; ++j)
              acc += dvl[r * S + s0 + j] * tK[din * TS + j];
            skv[r * D + din] += acc;
          }
          __syncthreads();
        }
        for(int d = tid; d < D; d += nth)
        {
          float acc = 0.f;
          for(int s = 0; s < S; ++s)
            acc += k[(size_t)d * N + kf * S + s] * dz[s];
          sz[d] += acc;
        }
        __syncthreads();
      }
      // Read out: query frame `qf` against the current state -> num (per-token vector).
      for(int s0 = 0; s0 < S; s0 += TS)
      {
        int ts = min(TS, S - s0);
        for(int idx = tid; idx < D * ts; idx += nth)
        {
          int din = idx / ts, j = idx % ts;
          tK[din * TS + j] = qrot[(size_t)din * N + qf * S + s0 + j];
        }
        __syncthreads();
        for(int idx = tid; idx < Rloc * ts; idx += nth)
        {
          int r = idx / ts, j = idx % ts;
          float nv = 0.f;
          const float* row = &skv[r * D];
          for(int din = 0; din < D; ++din)
            nv += row[din] * tK[din * TS + j];
          num[(size_t)(r0 + r) * N + outf * S + s0 + j] += nv;
        }
        __syncthreads();
      }
      // Denominator readout (only row-group 0 owns the full normalizer): den = q . sz.
      if(g == 0)
      {
        for(int s = tid; s < S; s += nth)
        {
          int bq = qf * S + s, on = outf * S + s;
          float dv_ = 0.f;
          for(int d = 0; d < D; ++d)
            dv_ += sz[d] * q[(size_t)d * N + bq];
          den[on] += dv_;
        }
      }
      __syncthreads();
    }
    if(dir == 0)
    {
      for(int i = tid; i < Rloc * D; i += nth)
        b.out_kv[(size_t)bh * D * D + (size_t)r0 * D + i] = skv[i];
      if(g == 0)
        for(int i = tid; i < D; i += nth)
          b.out_z[(size_t)bh * D + i] = sz[i];
      __syncthreads();
    }
  }
}
// Grid size for gdn_bidi: one block per (head, row-group), i.e. nb heads * GDN_G groups.
static int gdn_grid(int nb)
{
  return nb * GDN_G;
}
// Dynamic shared-memory bytes gdn_bidi needs: KV state + normalizer + delta scratch + K/Q tile.
static size_t gdn_shbytes(int D, int Sv)
{
  int R = (D + GDN_G - 1) / GDN_G;
  return (size_t)(R * D + D + R * Sv + Sv + D * GDN_TS) * sizeof(float);
}
// GDN q/k/v preparation: de-interleave the fused qkv projection into head-major q/k/v,
// apply optional q/k RMS-normalization, ReLU-gate q and k, and scale k by k_scale.
__global__ void prep_qkv(Prep p)
{
  int H = p.H, D = p.D, N = p.N;
  long tot = (long)H * D * N;
  for(long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < tot;
      idx += (long)gridDim.x * blockDim.x)
  {
    int n = idx % N;
    int d = (idx / N) % D;
    int h = idx / ((long)N * D);
    long base = ((long)n * 3 * H + h) * D + d;
    float ql = p.qkv[base + 0L * H * D], kl = p.qkv[base + 1L * H * D],
          vl = p.qkv[base + 2L * H * D];
    if(p.qk_norm)
    {
      int c = h * D + d;
      ql *= p.q_inv[n] * p.q_nw[c];
      kl *= p.k_inv[n] * p.k_nw[c];
    }
    ql = ql > 0.f ? ql : 0.f;
    kl = (kl > 0.f ? kl : 0.f) * p.k_scale;
    long o = ((long)h * D + d) * N + n;
    p.q[o] = ql;
    p.k[o] = kl;
    p.v[o] = vl;
  }
}
// Apply rotary position embedding to the head-major q/k produced by prep_qkv,
// writing the rotated qrot/krot the GDN recurrence consumes.
__global__ void prep_rope(Prep p)
{
  int H = p.H, D = p.D, N = p.N;
  long tot = (long)H * D * N;
  for(long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < tot;
      idx += (long)gridDim.x * blockDim.x)
  {
    int n = idx % N;
    int d = (idx / N) % D;
    int h = idx / ((long)N * D);
    int dp = d ^ 1;
    float c = p.cos[(long)n * D + d], s = p.sin[(long)n * D + d];
    long o = ((long)h * D + d) * N + n, op = ((long)h * D + dp) * N + n;
    p.qrot[o] = p.q[o] * c + p.q[op] * s;
    p.krot[o] = p.k[o] * c + p.k[op] * s;
  }
}
// Apply the logistic sigmoid to each of the n elements of Y, in place.
__global__ void k_sigmoid(float* Y, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
  {
    Y[i] = 1.f / (1.f + expf(-Y[i]));
  }
}
// Repack the GDN beta gate from token-major (T*S, HEADS) to head-major (HEADS, T, S).
__global__ void k_beta_hts(float* B, const float* beta_ns, int T, int S)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)HEADS * T * S;
  if(i >= tot)
    return;
  int s = i % S;
  int t = (i / S) % T;
  int h = i / ((long)T * S);
  B[i] = beta_ns[(long)(t * S + s) * HEADS + h];
}
// Compute the per-(head,frame) GDN decay factor Dout = exp(-exp(A_log) * softplus(a+dt_bias)),
// the multiplicative state decay applied each frame in gdn_bidi.
__global__ void k_decay_ht(
    float* Dout, const float* a_out, const float* A_log, const float* dt_bias, int T)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)HEADS * T)
    return;
  int t = i % T;
  int h = i / T;
  float a = a_out[(long)t * HEADS + h] + dt_bias[h];
  float sp = (a > 20.f) ? a : log1pf(expf(a));
  Dout[i] = expf(-expf(A_log[h]) * sp);
}
// Final GDN readout: divide the numerator by the (per head,token) normalizer,
// Y(N,C) token-major = num(HEADS,HD,N) / (den(HEADS,N) + eps).
__global__ void k_gdn_out(float* Y, const float* num, const float* den, int N, float eps)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)N * C)
    return;
  int d = i % HD;
  int h = (i / HD) % HEADS;
  int n = i / C;
  Y[i] = num[((long)h * HD + d) * N + n] / (den[(long)h * N + n] + eps);
}
// Average the S spatial tokens within each of the T frames: X(T*S,C) -> Y(T,C).
// Feeds the GDN gate/decay projections, which are computed per frame not per token.
__global__ void k_mean_frames(float* Y, const float* X, int T, int S)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)T * C)
    return;
  int c = i % C;
  int t = i / C;
  float s = 0;
  for(int j = 0; j < S; j++)
    s += X[(long)(t * S + j) * C + c];
  Y[i] = s / S;
}
// Per-channel 3x3 spatial depthwise convolution (zero-padded) over each frame of the
// FFN activation X(T,Hs,Ws,Ch); Wt is the (Ch,3,3) kernel, Bs the per-channel bias.
__global__ void k_depthwise(
    float* Y, const float* X, const float* Wt, const float* Bs, int T, int Hs, int Ws,
    int Ch)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)T * Hs * Ws * Ch;
  if(i >= tot)
    return;
  int o = i % Ch;
  long sp = i / Ch;
  int w = sp % Ws;
  int h = (sp / Ws) % Hs;
  int t = sp / ((long)Hs * Ws);
  float acc = Bs[o];
  const float* wt = Wt + (long)o * 9;
  for(int dh = -1; dh <= 1; dh++)
    for(int dw = -1; dw <= 1; dw++)
    {
      int hh = h + dh, ww = w + dw;
      if(hh < 0 || hh >= Hs || ww < 0 || ww >= Ws)
        continue;
      long ni = ((long)(t * Hs + hh) * Ws + ww);
      acc += wt[(dh + 1) * 3 + (dw + 1)] * X[ni * Ch + o];
    }
  Y[i] = acc;
}
// Gated-Linear-Unit activation: split each row of X(M,2*half) into value/gate halves
// and emit Y = value * SiLU(gate) (M,half). The FFN's point-conv gating.
__global__ void k_glu(float* Y, const float* X, int M, int half)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)M * half)
    return;
  int c = i % half;
  int m = i / half;
  float a = X[(long)m * 2 * half + c], g = X[(long)m * 2 * half + half + c];
  Y[i] = a * (g / (1.f + expf(-g)));
}
// Build the sinusoidal timestep embedding for scalar t into Y (first `half` cos, next
// `half` sin), with log-spaced frequencies up to max_period. Feeds the t_embedder MLP.
__global__ void k_time_sinusoid(float* Y, float t, int half, float max_period)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= half)
    return;
  float freq = expf(-logf(max_period) * (float)i / (float)half);
  Y[i] = cosf(t * freq);
  Y[half + i] = sinf(t * freq);
}
// Apply the tanh-approximation GELU activation to each of the n elements of Y, in place.
__global__ void k_gelu_tanh(float* Y, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
  {
    float x = Y[i];
    Y[i] = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
}
// Split fused QKV (N,3C)->(q,k each N,C) and cross-attn KV (Lk,2C)->(ck,cv each Lk,C) on the GPU
// (replaces synchronous device->host->device round-trips).
__global__ void k_split_qkv(float* q, float* k, const float* qkv, int N, int Cc)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)N * Cc)
    return;
  int c = i % Cc, n = i / Cc;
  q[i] = qkv[(long)n * 3 * Cc + c];
  k[i] = qkv[(long)n * 3 * Cc + Cc + c];
}
// Split the fused cross-attn KV projection (Lk,2C) into separate ck and cv (each Lk,C).
__global__ void k_split_ckv(float* ck, float* cv, const float* ckv, int Lkn, int Cc)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)Lkn * Cc)
    return;
  int c = i % Cc, l = i / Cc;
  ck[i] = ckv[(long)l * 2 * Cc + c];
  cv[i] = ckv[(long)l * 2 * Cc + Cc + c];
}
// Cast a length-n fp32 buffer to bf16 (used to feed the bf16 tensor-core GEMMs in linear()).
__global__ void k_f2bf(__nv_bfloat16* o, const float* x, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
    o[i] = __float2bfloat16(x[i]);
}
// Scatter a head-major (H,Nsrc,HD) K/V block into a larger (H,Nfull,HD) buffer at token
// offset `off` — used to assemble the softmax cache history (sink + prev + current chunk).
__global__ void
k_place(float* dst, const float* src, int H, int Nfull, int Nsrc, int HD_, int off)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)H * Nsrc * HD_;
  if(i >= tot)
    return;
  int d = i % HD_;
  int n = (i / HD_) % Nsrc;
  int h = i / ((long)Nsrc * HD_);
  dst[((long)h * Nfull + (off + n)) * HD_ + d] = src[i];
}
// Concatenate the 128-ch noisy latent and 128-ch conditioning image along the feature
// axis into the 256-ch x_embedder input Y(Nn,256).
__global__ void k_concat_li(float* Y, const float* lat, const float* img, int Nn)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i >= (long)Nn * 256)
    return;
  int c = i % 256;
  int n = i / 256;
  Y[i] = c < 128 ? lat[(long)n * 128 + c] : img[(long)n * 128 + (c - 128)];
}
// FlowMatch-Euler step: lat += coef * pred (coef = sigma[s+1]-sigma[s]), in place over n.
__global__ void k_axpy(float* lat, const float* pred, float coef, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
    lat[i] += coef * pred[i];
}
// RGB8 (T,H,W,3) -> pixels (1,3,T,H,W) fp32 in [-1,1]  (mean=std=0.5 => x/127.5-1)
__global__ void k_preprocess(float* px, const unsigned char* rgb, int T, int H, int W)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)3 * T * H * W;
  if(i >= tot)
    return;
  int w = i % W;
  int h = (i / W) % H;
  int t = (i / ((long)W * H)) % T;
  int c = i / ((long)W * H * T);
  px[i] = (float)rgb[((long)(t * H + h) * W + w) * 3 + c] / 127.5f - 1.f;
}
// pixels (1,3,T,H,W) fp32 [-1,1] -> RGB8 (T,H,W,3)
__global__ void k_postprocess(unsigned char* rgb, const float* px, int T, int H, int W)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long tot = (long)3 * T * H * W;
  if(i >= tot)
    return;
  int w = i % W;
  int h = (i / W) % H;
  int t = (i / ((long)W * H)) % T;
  int c = i / ((long)W * H * T);
  float v = (px[i] + 1.f) * 127.5f;
  v = v < 0 ? 0 : (v > 255 ? 255 : v);
  rgb[((long)(t * H + h) * W + w) * 3 + c] = (unsigned char)(v + 0.5f);
}
// moments (1,256,F,15,26) -> z_dit (128,F,15,26): mode=first128, (x-mean)/std
__global__ void
k_affine_enc(float* z, const float* mom, const float* mean, const float* std, int F)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long HW = 15 * 26;
  long tot = (long)128 * F * HW;
  if(i >= tot)
    return;
  int c = i / (F * HW);
  z[i] = (mom[i] - mean[c]) / std[c];
}
// z_dit (128,F,15,26) -> z_vae: x*std+mean
__global__ void k_affine_dec(float* z, const float* mean, const float* std, int F)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long HW = 15 * 26;
  long tot = (long)128 * F * HW;
  if(i >= tot)
    return;
  int c = i / (F * HW);
  z[i] = z[i] * std[c] + mean[c];
}
// full (128,FULLF,15,26) chunk [start_f,start_f+T) -> chunk token-major (Nc,128)
__global__ void k_slice_chunk(float* chunk, const float* full, int start_f, int T)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  int Nc = T * 15 * 26;
  long tot = (long)Nc * 128;
  if(i >= tot)
    return;
  int c = i % 128;
  int n = i / 128;
  int w = n % 26, h = (n / 26) % 15, fl = n / (26 * 15);
  long fidx = ((long)c * 16 + (start_f + fl)) * (15 * 26) + (long)h * 26 + w;
  chunk[i] = full[fidx];
}
// Inverse of k_slice_chunk: write a chunk's token-major (Nc,128) denoised latent back
// into the full-clip (128,FULLF,15,26) tensor at frame offset start_f.
__global__ void k_scatter_chunk(float* full, const float* chunk, int start_f, int T)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  int Nc = T * 15 * 26;
  long tot = (long)Nc * 128;
  if(i >= tot)
    return;
  int c = i % 128;
  int n = i / 128;
  int w = n % 26, h = (n / 26) % 15, fl = n / (26 * 15);
  long fidx = ((long)c * 16 + (start_f + fl)) * (15 * 26) + (long)h * 26 + w;
  full[fidx] = chunk[i];
}
// Cast a length-n fp32 buffer to bf16 (for feeding an engine input that wants bf16).
__global__ void k_trt_f2bf(const float* in, __nv_bfloat16* out, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
    out[i] = __float2bfloat16(in[i]);
}
// Cast a length-n bf16 buffer back to fp32 (for converting a bf16 engine output).
__global__ void k_trt_bf2f(const __nv_bfloat16* in, float* out, long n)
{
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if(i < n)
    out[i] = __bfloat162float(in[i]);
}

// ================= host launchers (grid/block match the original inline launches) =================
static size_t GDN_SH = 0; // dynamic shared-mem size for gdn_bidi; set by sana_gdn_init.

void sana_gdn_init()
{
  if(!GDN_SH)
  {
    GDN_SH = gdn_shbytes(HD, S);
    cudaFuncSetAttribute(
        gdn_bidi, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GDN_SH);
  }
}

void launch_add_bias(float* Y, const float* b, int M, int No, void* stream)
{
  k_add_bias<<<gr((long)M * No), 256, 0, (cudaStream_t)stream>>>(Y, b, M, No);
}
void launch_layernorm(float* Y, const float* X, int M, int D, float eps, void* stream)
{
  k_layernorm<<<M, 128, 0, (cudaStream_t)stream>>>(Y, X, M, D, eps);
}
void launch_rmsnorm(
    float* Y, const float* X, const float* W, int M, int D, float eps, void* stream)
{
  k_rmsnorm<<<M, 128, 0, (cudaStream_t)stream>>>(Y, X, W, M, D, eps);
}
void launch_adaln(
    float* Y, const float* X, const float* shift, const float* scale, int M, int D,
    void* stream)
{
  k_adaln<<<gr((long)M * D), 256, 0, (cudaStream_t)stream>>>(Y, X, shift, scale, M, D);
}
void launch_add_gate(
    float* Y, const float* A, const float* gate, const float* Bm, int M, int D,
    void* stream)
{
  k_add_gate<<<gr((long)M * D), 256, 0, (cudaStream_t)stream>>>(Y, A, gate, Bm, M, D);
}
void launch_add(float* Y, const float* A, const float* Bm, long n, void* stream)
{
  k_add<<<gr(n), 256, 0, (cudaStream_t)stream>>>(Y, A, Bm, n);
}
void launch_silu_inplace(float* Y, long n, void* stream)
{
  k_silu_inplace<<<gr(n), 256, 0, (cudaStream_t)stream>>>(Y, n);
}
void launch_gate_silu(float* Y, const float* og, long n, void* stream)
{
  k_gate_silu<<<gr(n), 256, 0, (cudaStream_t)stream>>>(Y, og, n);
}
void launch_inv_rms(float* out, const float* X, int M, int D, float eps, void* stream)
{
  k_inv_rms<<<M, 128, 0, (cudaStream_t)stream>>>(out, X, M, D, eps);
}
void launch_softmax(float* Sm, int rows, int cols, int valid, void* stream)
{
  k_softmax<<<rows, 128, 0, (cudaStream_t)stream>>>(Sm, rows, cols, valid);
}
void launch_split_heads(float* O, const float* X, int M, void* stream)
{
  k_split_heads<<<gr((long)M * C), 256, 0, (cudaStream_t)stream>>>(O, X, M);
}
void launch_merge_heads(float* O, const float* X, int M, void* stream)
{
  k_merge_heads<<<gr((long)M * C), 256, 0, (cudaStream_t)stream>>>(O, X, M);
}
void launch_rope_bnhd(
    float* Y, const float* X, const float* cos, const float* sin, int M, void* stream)
{
  k_rope_bnhd<<<gr((long)M * C), 256, 0, (cudaStream_t)stream>>>(Y, X, cos, sin, M);
}
void launch_gdn_bidi(const Bufs& b, void* stream)
{
  gdn_bidi<<<gdn_grid(b.H), 256, GDN_SH, (cudaStream_t)stream>>>(b);
}
void launch_prep_qkv(const Prep& p, void* stream)
{
  int blk = gr((long)p.H * p.D * p.N);
  if(blk > 65535)
    blk = 65535;
  prep_qkv<<<blk, 256, 0, (cudaStream_t)stream>>>(p);
}
void launch_prep_rope(const Prep& p, void* stream)
{
  int blk = gr((long)p.H * p.D * p.N);
  if(blk > 65535)
    blk = 65535;
  prep_rope<<<blk, 256, 0, (cudaStream_t)stream>>>(p);
}
void launch_sigmoid(float* Y, long n, void* stream)
{
  k_sigmoid<<<gr(n), 256, 0, (cudaStream_t)stream>>>(Y, n);
}
void launch_beta_hts(float* B, const float* beta_ns, int T, int Sv, void* stream)
{
  k_beta_hts<<<gr((long)HEADS * T * Sv), 256, 0, (cudaStream_t)stream>>>(
      B, beta_ns, T, Sv);
}
void launch_decay_ht(
    float* Dout, const float* a_out, const float* A_log, const float* dt_bias, int T,
    void* stream)
{
  k_decay_ht<<<gr((long)HEADS * T), 256, 0, (cudaStream_t)stream>>>(
      Dout, a_out, A_log, dt_bias, T);
}
void launch_gdn_out(
    float* Y, const float* num, const float* den, int N, float eps, void* stream)
{
  k_gdn_out<<<gr((long)N * C), 256, 0, (cudaStream_t)stream>>>(Y, num, den, N, eps);
}
void launch_mean_frames(float* Y, const float* X, int T, int Sv, void* stream)
{
  k_mean_frames<<<gr((long)T * C), 256, 0, (cudaStream_t)stream>>>(Y, X, T, Sv);
}
void launch_depthwise(
    float* Y, const float* X, const float* Wt, const float* Bs, int T, int Hsz, int Wsz,
    int Ch, void* stream)
{
  k_depthwise<<<gr((long)T * Hsz * Wsz * Ch), 256, 0, (cudaStream_t)stream>>>(
      Y, X, Wt, Bs, T, Hsz, Wsz, Ch);
}
void launch_glu(float* Y, const float* X, int M, int half, void* stream)
{
  k_glu<<<gr((long)M * half), 256, 0, (cudaStream_t)stream>>>(Y, X, M, half);
}
void launch_time_sinusoid(float* Y, float t, int half, float max_period, void* stream)
{
  k_time_sinusoid<<<gr(half, 128), 128, 0, (cudaStream_t)stream>>>(
      Y, t, half, max_period);
}
void launch_gelu_tanh(float* Y, long n, void* stream)
{
  k_gelu_tanh<<<gr(n), 256, 0, (cudaStream_t)stream>>>(Y, n);
}
void launch_split_qkv(float* q, float* k, const float* qkv, int N, int Cc, void* stream)
{
  k_split_qkv<<<gr((long)N * Cc), 256, 0, (cudaStream_t)stream>>>(q, k, qkv, N, Cc);
}
void launch_split_ckv(
    float* ck, float* cv, const float* ckv, int Lkn, int Cc, void* stream)
{
  k_split_ckv<<<gr((long)Lkn * Cc), 256, 0, (cudaStream_t)stream>>>(ck, cv, ckv, Lkn, Cc);
}
void launch_f2bf(void* o, const float* x, long n, void* stream)
{
  k_f2bf<<<gr(n), 256, 0, (cudaStream_t)stream>>>((__nv_bfloat16*)o, x, n);
}
void launch_place(
    float* dst, const float* src, int H, int Nfull, int Nsrc, int HDv, int off,
    void* stream)
{
  k_place<<<gr((long)H * Nsrc * HDv), 256, 0, (cudaStream_t)stream>>>(
      dst, src, H, Nfull, Nsrc, HDv, off);
}
void launch_concat_li(float* Y, const float* lat, const float* img, int Nn, void* stream)
{
  k_concat_li<<<gr((long)Nn * 256), 256, 0, (cudaStream_t)stream>>>(Y, lat, img, Nn);
}
void launch_axpy(float* lat, const float* pred, float coef, long n, void* stream)
{
  k_axpy<<<gr(n), 256, 0, (cudaStream_t)stream>>>(lat, pred, coef, n);
}
void launch_preprocess(
    float* px, const unsigned char* rgb, int T, int H, int W, void* stream)
{
  k_preprocess<<<gr((long)3 * T * H * W), 256, 0, (cudaStream_t)stream>>>(
      px, rgb, T, H, W);
}
void launch_postprocess(
    unsigned char* rgb, const float* px, int T, int H, int W, void* stream)
{
  k_postprocess<<<gr((long)3 * T * H * W), 256, 0, (cudaStream_t)stream>>>(
      rgb, px, T, H, W);
}
void launch_affine_enc(
    float* z, const float* mom, const float* mean, const float* std, int F, void* stream)
{
  k_affine_enc<<<gr((long)128 * F * 15 * 26), 256, 0, (cudaStream_t)stream>>>(
      z, mom, mean, std, F);
}
void launch_affine_dec(
    float* z, const float* mean, const float* std, int F, void* stream)
{
  k_affine_dec<<<gr((long)128 * F * 15 * 26), 256, 0, (cudaStream_t)stream>>>(
      z, mean, std, F);
}
void launch_slice_chunk(float* chunk, const float* full, int start_f, int T, void* stream)
{
  k_slice_chunk<<<gr((long)T * 15 * 26 * 128), 256, 0, (cudaStream_t)stream>>>(
      chunk, full, start_f, T);
}
void launch_scatter_chunk(
    float* full, const float* chunk, int start_f, int T, void* stream)
{
  k_scatter_chunk<<<gr((long)T * 15 * 26 * 128), 256, 0, (cudaStream_t)stream>>>(
      full, chunk, start_f, T);
}
void launch_trt_f2bf(const float* in, void* out, long n, void* stream)
{
  k_trt_f2bf<<<(int)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
      in, (__nv_bfloat16*)out, n);
}
void launch_trt_bf2f(const void* in, float* out, long n, void* stream)
{
  k_trt_bf2f<<<(int)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)in, out, n);
}

} // namespace sana
} // namespace librediffusion
