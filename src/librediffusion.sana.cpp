/** SANA-Streaming DiT — C++/cuBLAS orchestration (pure host, compiled by cl.exe).
 *
 *  Holds ALL the SanaDiT orchestration: the DiT block forwards (dit_block / dit_block_s),
 *  the full-model + streaming rollouts (run_full / run_rollout / run_rollout_sc /
 *  run_v2v_core), the resident fp32 weight cache (WMAP/WKMAP + the bf16 GEMM copies),
 *  the FlowMatchEuler schedule, the causal WAN rope, preprocess/postprocess, the cuBLAS
 *  GEMMs (linear() via cublasGemmEx + the softmax-attention batched GEMMs), and the
 *  SanaDiT class. Every device kernel is invoked through a launch_* wrapper declared in
 *  librediffusion.sana_kernels.hpp and defined in the nvcc unit librediffusion.sana.cu —
 *  no device kernels or launch syntax live here. bf16 scratch is handled as opaque void*
 *  buffers so this unit needs no CUDA device headers beyond the host cuda_runtime API.
 */
#include "librediffusion.sana.hpp"
#include "librediffusion.sana_kernels.hpp"
#include "librediffusion.sana_trt.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The host-side float scalars this unit feeds to the kernels / cuBLAS (the GDN k_scale =
// powf(HD,-.5)*powf(S,-.5), the attention 1/sqrtf(HD) GEMM scales, and build_rope_dev's
// pow/cos/sin) were originally evaluated by nvcc's host pass with MSVC's default /fp:precise.
// This TU is compiled into the DLL with /fp:fast /arch:AVX2, which would reassociate/contract
// them and shift the results bit-for-bit. Pin precise semantics for the whole unit so the
// numerics stay identical to the pre-split build (host scalars only; the device kernels keep
// their own -use_fast_math in librediffusion.sana.cu).
#if defined(_MSC_VER)
#  pragma float_control(precise, on, push)
#endif

namespace librediffusion
{
namespace sana
{

static cublasHandle_t HBL = nullptr;
static int HBL_ref = 0; // reference count for the shared cuBLAS handle
static const float ONE = 1.f, ZERO = 0.f;
static const int BF16 = 2; // sizeof(__nv_bfloat16); bf16 scratch is opaque void* here.

// ================= host helpers =================
// Read exactly n fp32 values from the raw .bin at path p into v; false on missing/short file.
static bool rd(const std::string& p, size_t n, std::vector<float>& v)
{
  std::ifstream f(p, std::ios::binary);
  if(!f)
  {
    printf("[sana] miss %s\n", p.c_str());
    return false;
  }
  v.resize(n);
  f.read((char*)v.data(), n * 4);
  if((size_t)f.gcount() != n * 4)
  {
    printf("[sana] short %s\n", p.c_str());
    return false;
  }
  return true;
}
// Upload a host vector to a freshly allocated device buffer (returns the device pointer).
static float* up(const std::vector<float>& h)
{
  float* d = nullptr;
  if(cudaMallocAsync(&d, h.size() * 4, 0))
    return nullptr;
  cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  return d;
}
// Allocate an n-float device buffer from the stream-ordered pool (fast alloc/free reuse).
static float* dev(size_t n)
{
  float* d = nullptr;
  if(cudaMallocAsync(&d, n * 4, 0))
    return nullptr;
  return d;
} // stream-ordered pool (fast reuse)
// Download n floats from device buffer d back to the host vector h.
static void down(const float* d, size_t n, std::vector<float>& h)
{
  h.resize(n);
  cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
}

// --- resident weight cache: fp32 weights loaded once (never Arena-freed) ---
static std::unordered_set<float*> CACHED; // pointers owned by the weight cache
static std::unordered_map<std::string, float*> WMAP; // path -> resident fp32 weight
// --- bf16 GEMM: cached bf16 copy of each resident weight + reused activation/transient scratch ---
static std::unordered_map<const float*, void*> WBF;
static void *XBF = nullptr, *WTMP = nullptr;
static size_t XBF_CAP = 0, WTMP_CAP = 0;
// --- self-contained streaming (SC): the rollout PRODUCES its own cross-chunk caches ---
// When SC=1, dit_block_s reads GDN/softmax/FFN caches from these persistent per-block
// buffers (instead of LC() disk loads); when SC_Save=1 (the t=0 update pass) it captures
// this chunk's produced caches. softmax history = sink(chunk0) [+ prev(chunk c-1)].
static int SC = 0, SC_Save = 0, SC_Chunk = 0;
static float *g_gdn_kv[20] = {}, *g_gdn_z[20] = {}, *g_sink_k[20] = {},
             *g_sink_v[20] = {}, *g_prev_k[20] = {}, *g_prev_v[20] = {}, *g_ffn[20] = {};
static int g_sink_n[20] = {}, g_prev_n[20] = {};
// Free and clear all self-contained streaming caches (call at rollout start and end).
static void sc_reset()
{
  for(int i = 0; i < 20; i++)
  {
    for(float** p :
        {&g_gdn_kv[i], &g_gdn_z[i], &g_sink_k[i], &g_sink_v[i], &g_prev_k[i],
         &g_prev_v[i], &g_ffn[i]})
    {
      if(*p)
      {
        cudaFree(*p);
        *p = nullptr;
      }
    }
    g_sink_n[i] = g_prev_n[i] = 0;
  }
}
// Y[No,M] fp32 = W[No,K] @ X[M,K]^T via bf16 tensor cores (fp32 accumulate). Weights cast once
// (cached by pointer for resident weights); activations + transient weights cast into reused scratch.
static void linear(float* Y, const float* X, const float* W, int M, int K, int No)
{
  void* Wb;
  if(CACHED.count((float*)W))
  {
    auto it = WBF.find(W);
    if(it != WBF.end())
      Wb = it->second;
    else
    {
      cudaMalloc(&Wb, (size_t)No * K * BF16);
      launch_f2bf(Wb, W, (long)No * K, nullptr);
      WBF[W] = Wb;
    }
  }
  else
  {
    size_t nw = (size_t)No * K;
    if(nw > WTMP_CAP)
    {
      if(WTMP)
        cudaFree(WTMP);
      cudaMalloc(&WTMP, nw * BF16);
      WTMP_CAP = nw;
    }
    launch_f2bf(WTMP, W, (long)No * K, nullptr);
    Wb = WTMP;
  }
  size_t nx = (size_t)M * K;
  if(nx > XBF_CAP)
  {
    if(XBF)
      cudaFree(XBF);
    cudaMalloc(&XBF, nx * BF16);
    XBF_CAP = nx;
  }
  launch_f2bf(XBF, X, (long)M * K, nullptr);
  cublasGemmEx(
      HBL, CUBLAS_OP_T, CUBLAS_OP_N, No, M, K, &ONE, Wb, CUDA_R_16BF, K, XBF,
      CUDA_R_16BF, K, &ZERO, Y, CUDA_R_32F, No, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
// Relative L2 error ||a-g|| / ||g|| between a candidate and golden vector (the accuracy metric).
static float relL2(const std::vector<float>& a, const std::vector<float>& g)
{
  double nu = 0, de = 0;
  for(size_t i = 0; i < a.size(); ++i)
  {
    double d = a[i] - g[i];
    nu += d * d;
    de += (double)g[i] * g[i];
  }
  return (float)sqrt(nu / de);
}

// Scratch allocator for one block/step: hands out stream-ordered device buffers and frees
// them all at once (skipping any that belong to the resident weight cache) on free()/dtor.
struct Arena
{
  std::vector<float*> v;
  ~Arena() { free(); }
  float* a(size_t n)
  {
    float* d = dev(n);
    if(d)
      v.push_back(d);
    return d;
  }
  void free()
  {
    for(auto p : v)
      if(!CACHED.count(p))
        cudaFreeAsync(p, 0);
    v.clear();
  }
};

// Load a weight .bin into a host vector (non-resident, one-shot read).
static std::vector<float> LB(const std::string& d, const char* nm, size_t n)
{
  std::vector<float> v;
  rd(d + "/" + nm + ".bin", n, v);
  return v;
}
// Resident weight loader: reads each .bin at most once, keeps it on the device forever (in CACHED,
// so the per-call Arena never frees it). This is what makes a denoise step ~O(compute) instead of
// re-reading ~gigabytes of weights from disk every block, every step.
static float* LDB(const std::string& d, const char* nm, size_t n)
{
  std::string key = d + "/" + nm;
  auto it = WMAP.find(key);
  if(it != WMAP.end())
    return it->second;
  std::vector<float> h;
  if(!rd(key + ".bin", n, h))
    return nullptr;
  float* dp = up(h);
  if(dp)
  {
    WMAP[key] = dp;
    CACHED.insert(dp);
  }
  return dp;
}
// Resident temporal-conv weights: read + repack (C,C,3)->3x(C,C) once per block.
static std::unordered_map<std::string, float*> WKMAP;
// Load and cache one block's temporal-conv weights, repacked (C,C,3) -> 3 x (C,C).
static float* cached_wk(const std::string& sub)
{
  auto it = WKMAP.find(sub);
  if(it != WKMAP.end())
    return it->second;
  std::vector<float> h;
  if(!rd(sub + "/ffn_tc_w.bin", (size_t)C * C * 3, h))
    return nullptr;
  std::vector<float> packed((size_t)3 * C * C);
  for(int kk = 0; kk < 3; kk++)
    for(int co = 0; co < C; co++)
      for(int ci = 0; ci < C; ci++)
        packed[(size_t)kk * C * C + (size_t)co * C + ci]
            = h[((size_t)co * C + ci) * 3 + kk];
  float* dp = nullptr;
  if(cudaMalloc(&dp, (size_t)3 * C * C * 4))
    return nullptr;
  cudaMemcpy(dp, packed.data(), (size_t)3 * C * C * 4, cudaMemcpyHostToDevice);
  WKMAP[sub] = dp;
  CACHED.insert(dp);
  return dp;
}

// One DiT block, x_in(N,C) -> x_out(N,C). Block weights dir `sub`. Mirrors the validated forward.
static void dit_block(
    float* x_out, const float* x_in, const std::string& sub, int is_soft,
    const float* t0, const float* y, const float* rcos, const float* rsin, int valid)
{
  Arena A;
  float eps = 1e-8f;
  // adaLN conditioning: add the per-block scale_shift_table to the timestep vector t0 and
  // split into the 6 modulation vectors (shift/scale/gate for the self-attn and MLP paths).
  float* sst = LDB(sub, "scale_shift_table", 6 * C);
  A.v.push_back(sst);
  float* mods = A.a(6 * C);
  launch_add(mods, sst, t0, 6 * C, nullptr);
  float *sh_msa = mods, *sc_msa = mods + C, *g_msa = mods + 2 * C,
        *sh_mlp = mods + 3 * C, *sc_mlp = mods + 4 * C, *g_mlp = mods + 5 * C;
  // Self-attention input: LayerNorm then adaLN modulate (shift/scale) the normalized tokens.
  float* ln = A.a((size_t)Nt * C);
  launch_layernorm(ln, x_in, Nt, C, 1e-6f, nullptr);
  float* sa_in = A.a((size_t)Nt * C);
  launch_adaln(sa_in, ln, sh_msa, sc_msa, Nt, C, nullptr);
  float* x_sa = A.a((size_t)Nt * C);
  // GDN linear-attention branch: fused qkv -> prep/rope -> gate/decay -> bidirectional
  // recurrence -> normalized readout, then output-gate and projection.
  if(!is_soft)
  {
    float *qw = LDB(sub, "attn_q_w", (size_t)C * C),
          *kw = LDB(sub, "attn_k_w", (size_t)C * C),
          *vw = LDB(sub, "attn_v_w", (size_t)C * C);
    A.v.push_back(qw);
    A.v.push_back(kw);
    A.v.push_back(vw);
    float* qkvw = A.a((size_t)3 * C * C);
    cudaMemcpy(qkvw, qw, (size_t)C * C * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(qkvw + (size_t)C * C, kw, (size_t)C * C * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        qkvw + (size_t)2 * C * C, vw, (size_t)C * C * 4, cudaMemcpyDeviceToDevice);
    float* qkv = A.a((size_t)Nt * 3 * C);
    linear(qkv, sa_in, qkvw, Nt, C, 3 * C);
    float *qnw = LDB(sub, "attn_q_norm", C), *knw = LDB(sub, "attn_k_norm", C);
    A.v.push_back(qnw);
    A.v.push_back(knw);
    float *q_sec = A.a((size_t)Nt * C), *k_sec = A.a((size_t)Nt * C);
    launch_split_qkv(q_sec, k_sec, qkv, Nt, C, nullptr);
    float *qinv = A.a(Nt), *kinv = A.a(Nt);
    launch_inv_rms(qinv, q_sec, Nt, C, 1e-5f, nullptr);
    launch_inv_rms(kinv, k_sec, Nt, C, 1e-5f, nullptr);
    Prep p{};
    p.B = 1;
    p.H = HEADS;
    p.D = HD;
    p.N = Nt;
    p.k_scale = powf(HD, -0.5f) * powf(S, -0.5f);
    p.qk_norm = 1;
    p.qkv = qkv;
    p.q_inv = qinv;
    p.k_inv = kinv;
    p.q_nw = qnw;
    p.k_nw = knw;
    p.cos = rcos;
    p.sin = rsin;
    p.q = A.a((size_t)HEADS * HD * Nt);
    p.k = A.a((size_t)HEADS * HD * Nt);
    p.v = A.a((size_t)HEADS * HD * Nt);
    p.qrot = A.a((size_t)HEADS * HD * Nt);
    p.krot = A.a((size_t)HEADS * HD * Nt);
    launch_prep_qkv(p, nullptr);
    launch_prep_rope(p, nullptr);
    float *bpw = LDB(sub, "beta_proj_w", (size_t)HEADS * C),
          *bpb = LDB(sub, "beta_proj_b", HEADS);
    float *gpw = LDB(sub, "gate_proj_w", (size_t)HEADS * C),
          *gpb = LDB(sub, "gate_proj_b", HEADS);
    float *Alog = LDB(sub, "A_log", HEADS), *dtb = LDB(sub, "dt_bias", HEADS);
    A.v.push_back(bpw);
    A.v.push_back(bpb);
    A.v.push_back(gpw);
    A.v.push_back(gpb);
    A.v.push_back(Alog);
    A.v.push_back(dtb);
    float* beta_ns = A.a((size_t)Nt * HEADS);
    linear(beta_ns, sa_in, bpw, Nt, C, HEADS);
    launch_add_bias(beta_ns, bpb, Nt, HEADS, nullptr);
    launch_sigmoid(beta_ns, (long)Nt * HEADS, nullptr);
    float* beta_hts = A.a((size_t)HEADS * Tt * S);
    launch_beta_hts(beta_hts, beta_ns, Tt, S, nullptr);
    float* xf = A.a((size_t)Tt * C);
    launch_mean_frames(xf, sa_in, Tt, S, nullptr);
    float* a_out = A.a((size_t)Tt * HEADS);
    linear(a_out, xf, gpw, Tt, C, HEADS);
    launch_add_bias(a_out, gpb, Tt, HEADS, nullptr);
    float* decay_ht = A.a((size_t)HEADS * Tt);
    launch_decay_ht(decay_ht, a_out, Alog, dtb, Tt, nullptr);
    Bufs b{};
    b.H = HEADS;
    b.D = HD;
    b.T = Tt;
    b.S = S;
    b.N = Nt;
    b.eps = eps;
    b.G = GDN_G;
    b.q = p.q;
    b.k = p.k;
    b.v = p.v;
    b.qrot = p.qrot;
    b.krot = p.krot;
    b.beta = beta_hts;
    b.decay = decay_ht;
    b.num = A.a((size_t)HEADS * HD * Nt);
    b.den = A.a((size_t)HEADS * Nt);
    cudaMemset(b.num, 0, (size_t)HEADS * HD * Nt * 4);
    cudaMemset(b.den, 0, (size_t)HEADS * Nt * 4);
    b.state_kv = A.a((size_t)HEADS * HD * HD);
    b.state_z = A.a((size_t)HEADS * HD);
    b.out_kv = A.a((size_t)HEADS * HD * HD);
    b.out_z = A.a((size_t)HEADS * HD);
    b.dv = A.a((size_t)HEADS * HD * S);
    b.dz = A.a((size_t)HEADS * S);
    launch_gdn_bidi(b, nullptr);
    float* attn = A.a((size_t)Nt * C);
    launch_gdn_out(attn, b.num, b.den, Nt, eps, nullptr);
    float *ogw = LDB(sub, "attn_og_w", (size_t)C * C), *ogb = LDB(sub, "attn_og_b", C);
    A.v.push_back(ogw);
    A.v.push_back(ogb);
    float* og = A.a((size_t)Nt * C);
    linear(og, sa_in, ogw, Nt, C, C);
    launch_add_bias(og, ogb, Nt, C, nullptr);
    launch_gate_silu(attn, og, (long)Nt * C, nullptr);
    float *pw = LDB(sub, "attn_proj_w", (size_t)C * C), *pb = LDB(sub, "attn_proj_b", C);
    A.v.push_back(pw);
    A.v.push_back(pb);
    linear(x_sa, attn, pw, Nt, C, C);
    launch_add_bias(x_sa, pb, Nt, C, nullptr);
  }
  else
  {
    // Full softmax self-attention branch (the 5 "soft" blocks): q/k RMSNorm + rope,
    // scaled dot-product over all Nt tokens, output-gate and projection.
    float *qw = LDB(sub, "attn_q_w", (size_t)C * C),
          *kw = LDB(sub, "attn_k_w", (size_t)C * C),
          *vw = LDB(sub, "attn_v_w", (size_t)C * C);
    A.v.push_back(qw);
    A.v.push_back(kw);
    A.v.push_back(vw);
    float *q = A.a((size_t)Nt * C), *k = A.a((size_t)Nt * C), *v = A.a((size_t)Nt * C);
    linear(q, sa_in, qw, Nt, C, C);
    linear(k, sa_in, kw, Nt, C, C);
    linear(v, sa_in, vw, Nt, C, C);
    float *qnw = LDB(sub, "attn_q_norm", C), *knw = LDB(sub, "attn_k_norm", C);
    A.v.push_back(qnw);
    A.v.push_back(knw);
    float *qn = A.a((size_t)Nt * C), *kn = A.a((size_t)Nt * C);
    launch_rmsnorm(qn, q, qnw, Nt, C, 1e-6f, nullptr);
    launch_rmsnorm(kn, k, knw, Nt, C, 1e-6f, nullptr);
    float *qr = A.a((size_t)Nt * C), *kr = A.a((size_t)Nt * C);
    launch_rope_bnhd(qr, qn, rcos, rsin, Nt, nullptr);
    launch_rope_bnhd(kr, kn, rcos, rsin, Nt, nullptr);
    float *Q = A.a((size_t)Nt * C), *K = A.a((size_t)Nt * C), *V = A.a((size_t)Nt * C);
    launch_split_heads(Q, qr, Nt, nullptr);
    launch_split_heads(K, kr, Nt, nullptr);
    launch_split_heads(V, v, Nt, nullptr);
    float* Sc = A.a((size_t)HEADS * Nt * Nt);
    float sc = 1.f / sqrtf((float)HD);
    cublasSgemmStridedBatched(
        HBL, CUBLAS_OP_T, CUBLAS_OP_N, Nt, Nt, HD, &sc, K, HD, (long)Nt * HD, Q, HD,
        (long)Nt * HD, &ZERO, Sc, Nt, (long)Nt * Nt, HEADS);
    launch_softmax(Sc, HEADS * Nt, Nt, 0, nullptr);
    float* O = A.a((size_t)Nt * C);
    cublasSgemmStridedBatched(
        HBL, CUBLAS_OP_N, CUBLAS_OP_N, HD, Nt, Nt, &ONE, V, HD, (long)Nt * HD, Sc, Nt,
        (long)Nt * Nt, &ZERO, O, HD, (long)Nt * HD, HEADS);
    float* attn = A.a((size_t)Nt * C);
    launch_merge_heads(attn, O, Nt, nullptr);
    float *ogw = LDB(sub, "attn_og_w", (size_t)C * C), *ogb = LDB(sub, "attn_og_b", C);
    A.v.push_back(ogw);
    A.v.push_back(ogb);
    float* og = A.a((size_t)Nt * C);
    linear(og, sa_in, ogw, Nt, C, C);
    launch_add_bias(og, ogb, Nt, C, nullptr);
    launch_gate_silu(attn, og, (long)Nt * C, nullptr);
    float *pw = LDB(sub, "attn_proj_w", (size_t)C * C), *pb = LDB(sub, "attn_proj_b", C);
    A.v.push_back(pw);
    A.v.push_back(pb);
    linear(x_sa, attn, pw, Nt, C, C);
    launch_add_bias(x_sa, pb, Nt, C, nullptr);
  }
  // Residual after self-attention (gated by g_msa), then cross-attention to the text
  // embeddings y: q from the tokens, k/v from y, masked softmax over valid text tokens.
  float* x1 = A.a((size_t)Nt * C);
  launch_add_gate(x1, x_in, g_msa, x_sa, Nt, C, nullptr);
  float *cqw = LDB(sub, "cx_q_w", (size_t)C * C), *cqb = LDB(sub, "cx_q_b", C),
        *ckvw = LDB(sub, "cx_kv_w", (size_t)2 * C * C),
        *ckvb = LDB(sub, "cx_kv_b", 2 * C);
  A.v.push_back(cqw);
  A.v.push_back(cqb);
  A.v.push_back(ckvw);
  A.v.push_back(ckvb);
  float* cq = A.a((size_t)Nt * C);
  linear(cq, x1, cqw, Nt, C, C);
  launch_add_bias(cq, cqb, Nt, C, nullptr);
  float* ckv = A.a((size_t)Lk * 2 * C);
  linear(ckv, y, ckvw, Lk, C, 2 * C);
  launch_add_bias(ckv, ckvb, Lk, 2 * C, nullptr);
  float *ck = A.a((size_t)Lk * C), *cv = A.a((size_t)Lk * C);
  launch_split_ckv(ck, cv, ckv, Lk, C, nullptr);
  float *cqnw = LDB(sub, "cx_q_norm", C), *cknw = LDB(sub, "cx_k_norm", C);
  A.v.push_back(cqnw);
  A.v.push_back(cknw);
  float *cqn = A.a((size_t)Nt * C), *ckn = A.a((size_t)Lk * C);
  launch_rmsnorm(cqn, cq, cqnw, Nt, C, 1e-6f, nullptr);
  launch_rmsnorm(ckn, ck, cknw, Lk, C, 1e-6f, nullptr);
  float *CQ = A.a((size_t)Nt * C), *CKk = A.a((size_t)Lk * C), *CV = A.a((size_t)Lk * C);
  launch_split_heads(CQ, cqn, Nt, nullptr);
  launch_split_heads(CKk, ckn, Lk, nullptr);
  launch_split_heads(CV, cv, Lk, nullptr);
  float* CS = A.a((size_t)HEADS * Nt * Lk);
  float sc2 = 1.f / sqrtf((float)HD);
  cublasSgemmStridedBatched(
      HBL, CUBLAS_OP_T, CUBLAS_OP_N, Lk, Nt, HD, &sc2, CKk, HD, (long)Lk * HD, CQ, HD,
      (long)Nt * HD, &ZERO, CS, Lk, (long)Nt * Lk, HEADS);
  launch_softmax(CS, HEADS * Nt, Lk, valid, nullptr);
  float* CO = A.a((size_t)Nt * C);
  cublasSgemmStridedBatched(
      HBL, CUBLAS_OP_N, CUBLAS_OP_N, HD, Nt, Lk, &ONE, CV, HD, (long)Lk * HD, CS, Lk,
      (long)Nt * Lk, &ZERO, CO, HD, (long)Nt * HD, HEADS);
  float* cmerge = A.a((size_t)Nt * C);
  launch_merge_heads(cmerge, CO, Nt, nullptr);
  float *cpw = LDB(sub, "cx_proj_w", (size_t)C * C), *cpb = LDB(sub, "cx_proj_b", C);
  A.v.push_back(cpw);
  A.v.push_back(cpb);
  float* x_ca = A.a((size_t)Nt * C);
  linear(x_ca, cmerge, cpw, Nt, C, C);
  launch_add_bias(x_ca, cpb, Nt, C, nullptr);
  // Cross-attention residual, then the gated-conv FFN: LayerNorm + adaLN, inverted 1x1
  // conv (expand + SiLU), 3x3 spatial depthwise conv, GLU gate, point 1x1 conv, and a
  // 3-tap temporal conv across frames; result added back gated by g_mlp -> x_out.
  float* x2 = A.a((size_t)Nt * C);
  launch_add(x2, x1, x_ca, (long)Nt * C, nullptr);
  float* ln2 = A.a((size_t)Nt * C);
  launch_layernorm(ln2, x2, Nt, C, 1e-6f, nullptr);
  float* mlp_in = A.a((size_t)Nt * C);
  launch_adaln(mlp_in, ln2, sh_mlp, sc_mlp, Nt, C, nullptr);
  float *invw = LDB(sub, "ffn_inv_w", (size_t)CH * C), *invb = LDB(sub, "ffn_inv_b", CH);
  A.v.push_back(invw);
  A.v.push_back(invb);
  float* h1 = A.a((size_t)Nt * CH);
  linear(h1, mlp_in, invw, Nt, C, CH);
  launch_add_bias(h1, invb, Nt, CH, nullptr);
  launch_silu_inplace(h1, (long)Nt * CH, nullptr);
  float *dww = LDB(sub, "ffn_dw_w", (size_t)CH * 9), *dwb = LDB(sub, "ffn_dw_b", CH);
  A.v.push_back(dww);
  A.v.push_back(dwb);
  float* h2 = A.a((size_t)Nt * CH);
  launch_depthwise(h2, h1, dww, dwb, Tt, Hs, Ws, CH, nullptr);
  float* glu = A.a((size_t)Nt * HALF);
  launch_glu(glu, h2, Nt, HALF, nullptr);
  float* pww = LDB(sub, "ffn_pw_w", (size_t)C * HALF);
  A.v.push_back(pww);
  float* pc = A.a((size_t)Nt * C);
  linear(pc, glu, pww, Nt, HALF, C);
  float* Wk3 = cached_wk(sub);
  float* Wk[3] = {Wk3, Wk3 + (size_t)C * C, Wk3 + (size_t)2 * C * C};
  float* x_ffn = A.a((size_t)Nt * C);
  cudaMemcpy(x_ffn, pc, (size_t)Nt * C * 4, cudaMemcpyDeviceToDevice);
  int HW = Hs * Ws;
  for(int t = 0; t < Tt; t++)
    for(int kk = 0; kk < 3; kk++)
    {
      int ft = t + kk - 1;
      if(ft < 0 || ft >= Tt)
        continue;
      const float* Pin = pc + (size_t)ft * HW * C;
      float* Yout = x_ffn + (size_t)t * HW * C;
      cublasSgemm(
          HBL, CUBLAS_OP_T, CUBLAS_OP_N, C, HW, C, &ONE, Wk[kk], C, Pin, C, &ONE, Yout,
          C);
    }
  launch_add_gate(x_out, x2, g_mlp, x_ffn, Nt, C, nullptr);
  A.free(); // stream-ordered; no per-block device sync (kernels stay pipelined)
}

// Blocks 3,7,11,15,19 use full softmax attention; the rest use the GDN recurrence.
static const int SOFT[5] = {3, 7, 11, 15, 19};
// True if block i is a softmax-attention block (a member of SOFT).
static int is_soft(int i)
{
  for(int s : SOFT)
    if(s == i)
      return 1;
  return 0;
}

// Preload every fp32 weight run_v2v touches into the resident cache (WMAP/WKMAP), so the FIRST
// run_v2v pays no weight disk-I/O inside its timed region. Mirrors the LDB/cached_wk calls made by
// run_v2v_core + dit_block_s. Loading is idempotent (LDB/cached_wk return the cached pointer).
static void warm_weights(const std::string& d)
{
  auto L = [&](const std::string& dir, const char* nm, size_t n) { LDB(dir, nm, n); };
  // top-level (run_v2v_core LWT)
  L(d, "xemb_w", (size_t)C * 256);
  L(d, "xemb_b", C);
  L(d, "temb0_w", (size_t)C * 256);
  L(d, "temb0_b", C);
  L(d, "temb2_w", (size_t)C * C);
  L(d, "temb2_b", C);
  L(d, "tblock_w", (size_t)6 * C * C);
  L(d, "tblock_b", 6 * C);
  L(d, "final_sst", 2 * C);
  L(d, "final_lin_w", (size_t)128 * C);
  L(d, "final_lin_b", 128);
  L(d, "yfc1_w", (size_t)C * 2304);
  L(d, "yfc1_b", C);
  L(d, "yfc2_w", (size_t)C * C);
  L(d, "yfc2_b", C);
  L(d, "ynorm_w", C);
  // per-block (dit_block_s LW + cached_wk)
  for(int i = 0; i < 20; i++)
  {
    std::string W = d + "/block" + std::to_string(i);
    L(W, "scale_shift_table", 6 * C);
    L(W, "attn_q_w", (size_t)C * C);
    L(W, "attn_k_w", (size_t)C * C);
    L(W, "attn_v_w", (size_t)C * C);
    L(W, "attn_q_norm", C);
    L(W, "attn_k_norm", C);
    L(W, "attn_og_w", (size_t)C * C);
    L(W, "attn_og_b", C);
    L(W, "attn_proj_w", (size_t)C * C);
    L(W, "attn_proj_b", C);
    L(W, "cx_q_w", (size_t)C * C);
    L(W, "cx_q_b", C);
    L(W, "cx_kv_w", (size_t)2 * C * C);
    L(W, "cx_kv_b", 2 * C);
    L(W, "cx_q_norm", C);
    L(W, "cx_k_norm", C);
    L(W, "cx_proj_w", (size_t)C * C);
    L(W, "cx_proj_b", C);
    L(W, "ffn_inv_w", (size_t)CH * C);
    L(W, "ffn_inv_b", CH);
    L(W, "ffn_dw_w", (size_t)CH * 9);
    L(W, "ffn_dw_b", CH);
    L(W, "ffn_pw_w", (size_t)C * HALF);
    cached_wk(W);
    if(!is_soft(i))
    { // GDN-only weights (soft blocks use softmax attention instead)
      L(W, "beta_proj_w", (size_t)HEADS * C);
      L(W, "beta_proj_b", HEADS);
      L(W, "gate_proj_w", (size_t)HEADS * C);
      L(W, "gate_proj_b", HEADS);
      L(W, "A_log", HEADS);
      L(W, "dt_bias", HEADS);
    }
  }
  cudaDeviceSynchronize();
}

// Compute t0(6C), y(Lk*C), rcos/rsin(Nt*HD), and x(Nt*C from x256v) from the resident dir.
// Runs the 20-block loop + final layer on x -> out_dev (Nt*128). Caller owns out_dev.
static int run_full(
    const std::string& d, float* x_start /*Nt*C or nullptr to build from x256v*/,
    float* out_dev)
{
  auto LT = [&](const char* nm, size_t n) {
    std::vector<float> v;
    rd(d + "/" + nm + ".bin", n, v);
    return v;
  };
  auto LDT = [&](const char* nm, size_t n) {
    return LDB(d, nm, n);
  }; // resident (cached) — fixed weights/pos-embeds
  int valid = 18;
  float *rcos = LDT("rope_cos", (size_t)Nt * HD),
        *rsin = LDT("rope_sin", (size_t)Nt * HD);
  // x_embedder (if no x_start given)
  float* x = nullptr;
  if(x_start)
  {
    x = dev((size_t)Nt * C);
    cudaMemcpy(x, x_start, (size_t)Nt * C * 4, cudaMemcpyDeviceToDevice);
  }
  else
  {
    float* x256v = LDT("x256v", (size_t)Nt * 256);
    float *xemb_w = LDT("xemb_w", (size_t)C * 256), *xemb_b = LDT("xemb_b", C);
    x = dev((size_t)Nt * C);
    linear(x, x256v, xemb_w, Nt, 256, C);
    launch_add_bias(x, xemb_b, Nt, C, nullptr);
  }
  // t_embedder + t_block
  float ts;
  {
    auto h = LT("timestep", 1);
    ts = h.empty() ? 1000.f : h[0];
  }
  float* temb = dev(256);
  launch_time_sinusoid(temb, ts, 128, 10000.f, nullptr);
  float *t0w = LDT("temb0_w", (size_t)C * 256), *t0b = LDT("temb0_b", C);
  float* t1 = dev(C);
  linear(t1, temb, t0w, 1, 256, C);
  launch_add_bias(t1, t0b, 1, C, nullptr);
  launch_silu_inplace(t1, C, nullptr);
  float *t2w = LDT("temb2_w", (size_t)C * C), *t2b = LDT("temb2_b", C);
  float* tvec = dev(C);
  linear(tvec, t1, t2w, 1, C, C);
  launch_add_bias(tvec, t2b, 1, C, nullptr);
  float* tsil = dev(C);
  cudaMemcpy(tsil, tvec, C * 4, cudaMemcpyDeviceToDevice);
  launch_silu_inplace(tsil, C, nullptr);
  float *tbw = LDT("tblock_w", (size_t)6 * C * C), *tbb = LDT("tblock_b", 6 * C);
  float* t0 = dev(6 * C);
  linear(t0, tsil, tbw, 1, C, 6 * C);
  launch_add_bias(t0, tbb, 1, 6 * C, nullptr);
  // y_embedder — prompt-fixed, so compute once per dir and cache (resident).
  static std::unordered_map<std::string, float*> YCACHE;
  float* y;
  {
    auto yit = YCACHE.find(d);
    if(yit != YCACHE.end())
      y = yit->second;
    else
    {
      float* y_raw = LDT("y_raw", (size_t)Lk * 2304);
      float *yfc1w = LDT("yfc1_w", (size_t)C * 2304), *yfc1b = LDT("yfc1_b", C);
      float* y1 = dev((size_t)Lk * C);
      linear(y1, y_raw, yfc1w, Lk, 2304, C);
      launch_add_bias(y1, yfc1b, Lk, C, nullptr);
      launch_gelu_tanh(y1, (long)Lk * C, nullptr);
      float *yfc2w = LDT("yfc2_w", (size_t)C * C), *yfc2b = LDT("yfc2_b", C);
      float* y2 = dev((size_t)Lk * C);
      linear(y2, y1, yfc2w, Lk, C, C);
      launch_add_bias(y2, yfc2b, Lk, C, nullptr);
      float* ynw = LDT("ynorm_w", C);
      y = nullptr;
      cudaMalloc(&y, (size_t)Lk * C * 4);
      launch_rmsnorm(y, y2, ynw, Lk, C, 1e-6f, nullptr);
      cudaFreeAsync(y1, 0);
      cudaFreeAsync(y2, 0);
      YCACHE[d] = y;
      CACHED.insert(y);
    }
  }
  // 20 blocks
  float* cur = x;
  for(int i = 0; i < 20; i++)
  {
    std::string sub = d + "/block" + std::to_string(i);
    float* nxt = dev((size_t)Nt * C);
    dit_block(nxt, cur, sub, is_soft(i), t0, y, rcos, rsin, valid);
    if(cur != x)
      cudaFreeAsync(cur, 0);
    cur = nxt;
  }
  // final layer
  float* fsst = LDT("final_sst", 2 * C);
  float *fsh = dev(C), *fsc = dev(C);
  launch_add(fsh, fsst, tvec, C, nullptr);
  launch_add(fsc, fsst + C, tvec, C, nullptr);
  float* lnf = dev((size_t)Nt * C);
  launch_layernorm(lnf, cur, Nt, C, 1e-6f, nullptr);
  float* fmod = dev((size_t)Nt * C);
  launch_adaln(fmod, lnf, fsh, fsc, Nt, C, nullptr);
  float *flw = LDT("final_lin_w", (size_t)128 * C), *flb = LDT("final_lin_b", 128);
  linear(out_dev, fmod, flw, Nt, C, 128);
  launch_add_bias(out_dev, flb, Nt, 128, nullptr);
  cudaDeviceSynchronize();
  // free scratch only (weights/pos-embeds are resident in the cache, never freed here)
  for(float* pp : {x, temb, t1, tvec, tsil, t0, cur, fsh, fsc, lnf, fmod})
    cudaFreeAsync(pp, 0);
  return 0;
}

// ================= streaming 5-chunk rollout (variable T/N + cache carry) =================
// Streaming variant of dit_block: same DiT block but carries cross-chunk history so a
// short chunk of T frames attends to earlier frames. The GDN recurrence is seeded from
// the previous chunk's carried state, the softmax blocks prepend a cached K/V history
// (sink + prev), and the FFN temporal-conv reads a cached previous frame. In SC (self-
// contained) mode the caches live in the g_* device buffers this function fills when
// SC_Save is set; otherwise they are loaded from the cdir/block{i} disk caches (LC).
// Weights come from wdir/block{i}. Shapes use the chunk's T frames / N tokens.
static void dit_block_s(
    float* x_out, const float* x_in, const std::string& wdir, const std::string& cdir,
    int i, int is_soft, const float* t0, const float* y, const float* rcos,
    const float* rsin, int valid, int T, int N, int Ncache)
{
  Arena A;
  float eps = 1e-8f;
  std::string W = wdir + "/block" + std::to_string(i),
              CC = cdir + "/block" + std::to_string(i);
  auto LW = [&](const char* nm, size_t n) { return LDB(W, nm, n); };
  auto LWv = [&](const char* nm, size_t n) { return LB(W, nm, n); };
  auto LC = [&](const char* nm, size_t n) { return LDB(CC, nm, n); };
  float* sst = LW("scale_shift_table", 6 * C);
  A.v.push_back(sst);
  float* mods = A.a(6 * C);
  launch_add(mods, sst, t0, 6 * C, nullptr);
  float *sh_msa = mods, *sc_msa = mods + C, *g_msa = mods + 2 * C,
        *sh_mlp = mods + 3 * C, *sc_mlp = mods + 4 * C, *g_mlp = mods + 5 * C;
  float* ln = A.a((size_t)N * C);
  launch_layernorm(ln, x_in, N, C, 1e-6f, nullptr);
  float* sa_in = A.a((size_t)N * C);
  launch_adaln(sa_in, ln, sh_msa, sc_msa, N, C, nullptr);
  float* x_sa = A.a((size_t)N * C);
  // GDN linear-attention branch, seeded from the carried recurrent state (g_gdn_* in SC
  // mode, else the disk init_kv/init_z), and re-saving the carried-out state when SC_Save.
  if(!is_soft)
  {
    float *qw = LW("attn_q_w", (size_t)C * C), *kw = LW("attn_k_w", (size_t)C * C),
          *vw = LW("attn_v_w", (size_t)C * C);
    A.v.push_back(qw);
    A.v.push_back(kw);
    A.v.push_back(vw);
    float* qkvw = A.a((size_t)3 * C * C);
    cudaMemcpy(qkvw, qw, (size_t)C * C * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(qkvw + (size_t)C * C, kw, (size_t)C * C * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        qkvw + (size_t)2 * C * C, vw, (size_t)C * C * 4, cudaMemcpyDeviceToDevice);
    float* qkv = A.a((size_t)N * 3 * C);
    linear(qkv, sa_in, qkvw, N, C, 3 * C);
    float *qnw = LW("attn_q_norm", C), *knw = LW("attn_k_norm", C);
    A.v.push_back(qnw);
    A.v.push_back(knw);
    float *q_sec = A.a((size_t)N * C), *k_sec = A.a((size_t)N * C);
    launch_split_qkv(q_sec, k_sec, qkv, N, C, nullptr);
    float *qinv = A.a(N), *kinv = A.a(N);
    launch_inv_rms(qinv, q_sec, N, C, 1e-5f, nullptr);
    launch_inv_rms(kinv, k_sec, N, C, 1e-5f, nullptr);
    Prep p{};
    p.B = 1;
    p.H = HEADS;
    p.D = HD;
    p.N = N;
    p.k_scale = powf(HD, -0.5f) * powf(S, -0.5f);
    p.qk_norm = 1;
    p.qkv = qkv;
    p.q_inv = qinv;
    p.k_inv = kinv;
    p.q_nw = qnw;
    p.k_nw = knw;
    p.cos = rcos;
    p.sin = rsin;
    p.q = A.a((size_t)HEADS * HD * N);
    p.k = A.a((size_t)HEADS * HD * N);
    p.v = A.a((size_t)HEADS * HD * N);
    p.qrot = A.a((size_t)HEADS * HD * N);
    p.krot = A.a((size_t)HEADS * HD * N);
    launch_prep_qkv(p, nullptr);
    launch_prep_rope(p, nullptr);
    float *bpw = LW("beta_proj_w", (size_t)HEADS * C), *bpb = LW("beta_proj_b", HEADS),
          *gpw = LW("gate_proj_w", (size_t)HEADS * C), *gpb = LW("gate_proj_b", HEADS),
          *Alog = LW("A_log", HEADS), *dtb = LW("dt_bias", HEADS);
    A.v.push_back(bpw);
    A.v.push_back(bpb);
    A.v.push_back(gpw);
    A.v.push_back(gpb);
    A.v.push_back(Alog);
    A.v.push_back(dtb);
    float* beta_ns = A.a((size_t)N * HEADS);
    linear(beta_ns, sa_in, bpw, N, C, HEADS);
    launch_add_bias(beta_ns, bpb, N, HEADS, nullptr);
    launch_sigmoid(beta_ns, (long)N * HEADS, nullptr);
    float* beta_hts = A.a((size_t)HEADS * T * S);
    launch_beta_hts(beta_hts, beta_ns, T, S, nullptr);
    float* xf = A.a((size_t)T * C);
    launch_mean_frames(xf, sa_in, T, S, nullptr);
    float* a_out = A.a((size_t)T * HEADS);
    linear(a_out, xf, gpw, T, C, HEADS);
    launch_add_bias(a_out, gpb, T, HEADS, nullptr);
    float* decay_ht = A.a((size_t)HEADS * T);
    launch_decay_ht(decay_ht, a_out, Alog, dtb, T, nullptr);
    Bufs b{};
    b.H = HEADS;
    b.D = HD;
    b.T = T;
    b.S = S;
    b.N = N;
    b.eps = eps;
    b.G = GDN_G;
    b.q = p.q;
    b.k = p.k;
    b.v = p.v;
    b.qrot = p.qrot;
    b.krot = p.krot;
    b.beta = beta_hts;
    b.decay = decay_ht;
    if(SC)
    {
      b.init_kv = g_gdn_kv[i];
      b.init_z = g_gdn_z[i];
    } // consume from memory (null on chunk 0 => zero state)
    else
    {
      float *init_kv = LC("gdn_init_kv", (size_t)HEADS * HD * HD),
            *init_z = LC("gdn_init_z", (size_t)HEADS * HD);
      A.v.push_back(init_kv);
      A.v.push_back(init_z);
      b.init_kv = init_kv;
      b.init_z = init_z;
    }
    b.num = A.a((size_t)HEADS * HD * N);
    b.den = A.a((size_t)HEADS * N);
    cudaMemset(b.num, 0, (size_t)HEADS * HD * N * 4);
    cudaMemset(b.den, 0, (size_t)HEADS * N * 4);
    b.state_kv = A.a((size_t)HEADS * HD * HD);
    b.state_z = A.a((size_t)HEADS * HD);
    b.out_kv = A.a((size_t)HEADS * HD * HD);
    b.out_z = A.a((size_t)HEADS * HD);
    b.dv = A.a((size_t)HEADS * HD * S);
    b.dz = A.a((size_t)HEADS * S);
    launch_gdn_bidi(b, nullptr);
    if(SC && SC_Save)
    {
      if(!g_gdn_kv[i])
      {
        cudaMalloc(&g_gdn_kv[i], (size_t)HEADS * HD * HD * 4);
        cudaMalloc(&g_gdn_z[i], (size_t)HEADS * HD * 4);
      }
      cudaMemcpy(
          g_gdn_kv[i], b.out_kv, (size_t)HEADS * HD * HD * 4, cudaMemcpyDeviceToDevice);
      cudaMemcpy(g_gdn_z[i], b.out_z, (size_t)HEADS * HD * 4, cudaMemcpyDeviceToDevice);
    }
    float* attn = A.a((size_t)N * C);
    launch_gdn_out(attn, b.num, b.den, N, eps, nullptr);
    float *ogw = LW("attn_og_w", (size_t)C * C), *ogb = LW("attn_og_b", C);
    A.v.push_back(ogw);
    A.v.push_back(ogb);
    float* og = A.a((size_t)N * C);
    linear(og, sa_in, ogw, N, C, C);
    launch_add_bias(og, ogb, N, C, nullptr);
    launch_gate_silu(attn, og, (long)N * C, nullptr);
    float *pw = LW("attn_proj_w", (size_t)C * C), *pb = LW("attn_proj_b", C);
    A.v.push_back(pw);
    A.v.push_back(pb);
    linear(x_sa, attn, pw, N, C, C);
    launch_add_bias(x_sa, pb, N, C, nullptr);
  }
  else
  {
    // Full softmax self-attention over an extended key/value window: the current chunk's
    // N tokens plus Ncache cached history tokens (assembled below via k_place).
    float *qw = LW("attn_q_w", (size_t)C * C), *kw = LW("attn_k_w", (size_t)C * C),
          *vw = LW("attn_v_w", (size_t)C * C);
    A.v.push_back(qw);
    A.v.push_back(kw);
    A.v.push_back(vw);
    float *q = A.a((size_t)N * C), *k = A.a((size_t)N * C), *v = A.a((size_t)N * C);
    linear(q, sa_in, qw, N, C, C);
    linear(k, sa_in, kw, N, C, C);
    linear(v, sa_in, vw, N, C, C);
    float *qnw = LW("attn_q_norm", C), *knw = LW("attn_k_norm", C);
    A.v.push_back(qnw);
    A.v.push_back(knw);
    float *qn = A.a((size_t)N * C), *kn = A.a((size_t)N * C);
    launch_rmsnorm(qn, q, qnw, N, C, 1e-6f, nullptr);
    launch_rmsnorm(kn, k, knw, N, C, 1e-6f, nullptr);
    float *qr = A.a((size_t)N * C), *kr = A.a((size_t)N * C);
    launch_rope_bnhd(qr, qn, rcos, rsin, N, nullptr);
    launch_rope_bnhd(kr, kn, rcos, rsin, N, nullptr);
    float* Q = A.a((size_t)N * C);
    launch_split_heads(Q, qr, N, nullptr);
    float *Kc = A.a((size_t)N * C), *Vc = A.a((size_t)N * C);
    launch_split_heads(Kc, kr, N, nullptr);
    launch_split_heads(Vc, v, N, nullptr);
    // Build the full (H,Nf,HD) key/value buffers: prepend the cached history (sink frames
    // then previous-chunk frames, or the disk sm_ck/sm_cv), then append this chunk's Kc/Vc.
    int Nf = Ncache + N;
    float *Kf, *Vf;
    if(Ncache > 0)
    {
      Kf = A.a((size_t)HEADS * Nf * HD);
      Vf = A.a((size_t)HEADS * Nf * HD);
      if(SC)
      {
        launch_place(Kf, g_sink_k[i], HEADS, Nf, g_sink_n[i], HD, 0, nullptr);
        launch_place(Vf, g_sink_v[i], HEADS, Nf, g_sink_n[i], HD, 0, nullptr);
        if(Ncache > g_sink_n[i])
        {
          int off = g_sink_n[i];
          launch_place(Kf, g_prev_k[i], HEADS, Nf, g_prev_n[i], HD, off, nullptr);
          launch_place(Vf, g_prev_v[i], HEADS, Nf, g_prev_n[i], HD, off, nullptr);
        }
      }
      else
      {
        float *ck = LC("sm_ck", (size_t)HEADS * Ncache * HD),
              *cv = LC("sm_cv", (size_t)HEADS * Ncache * HD);
        A.v.push_back(ck);
        A.v.push_back(cv);
        launch_place(Kf, ck, HEADS, Nf, Ncache, HD, 0, nullptr);
        launch_place(Vf, cv, HEADS, Nf, Ncache, HD, 0, nullptr);
      }
      launch_place(Kf, Kc, HEADS, Nf, N, HD, Ncache, nullptr);
      launch_place(Vf, Vc, HEADS, Nf, N, HD, Ncache, nullptr);
    }
    else
    {
      Kf = Kc;
      Vf = Vc;
    }
    if(SC && SC_Save)
    { // Kf/Vf are independent copies now; overwrite prev (and sink on chunk 0)
      if(g_prev_k[i])
        cudaFree(g_prev_k[i]);
      if(g_prev_v[i])
        cudaFree(g_prev_v[i]);
      cudaMalloc(&g_prev_k[i], (size_t)HEADS * N * HD * 4);
      cudaMemcpy(g_prev_k[i], Kc, (size_t)HEADS * N * HD * 4, cudaMemcpyDeviceToDevice);
      cudaMalloc(&g_prev_v[i], (size_t)HEADS * N * HD * 4);
      cudaMemcpy(g_prev_v[i], Vc, (size_t)HEADS * N * HD * 4, cudaMemcpyDeviceToDevice);
      g_prev_n[i] = N;
      if(SC_Chunk == 0)
      {
        cudaMalloc(&g_sink_k[i], (size_t)HEADS * N * HD * 4);
        cudaMemcpy(
            g_sink_k[i], Kc, (size_t)HEADS * N * HD * 4, cudaMemcpyDeviceToDevice);
        cudaMalloc(&g_sink_v[i], (size_t)HEADS * N * HD * 4);
        cudaMemcpy(
            g_sink_v[i], Vc, (size_t)HEADS * N * HD * 4, cudaMemcpyDeviceToDevice);
        g_sink_n[i] = N;
      }
    }
    float* Sc = A.a((size_t)HEADS * N * Nf);
    float sc = 1.f / sqrtf((float)HD);
    cublasSgemmStridedBatched(
        HBL, CUBLAS_OP_T, CUBLAS_OP_N, Nf, N, HD, &sc, Kf, HD, (long)Nf * HD, Q, HD,
        (long)N * HD, &ZERO, Sc, Nf, (long)N * Nf, HEADS);
    launch_softmax(Sc, HEADS * N, Nf, 0, nullptr);
    float* O = A.a((size_t)N * C);
    cublasSgemmStridedBatched(
        HBL, CUBLAS_OP_N, CUBLAS_OP_N, HD, N, Nf, &ONE, Vf, HD, (long)Nf * HD, Sc, Nf,
        (long)N * Nf, &ZERO, O, HD, (long)N * HD, HEADS);
    float* attn = A.a((size_t)N * C);
    launch_merge_heads(attn, O, N, nullptr);
    float *ogw = LW("attn_og_w", (size_t)C * C), *ogb = LW("attn_og_b", C);
    A.v.push_back(ogw);
    A.v.push_back(ogb);
    float* og = A.a((size_t)N * C);
    linear(og, sa_in, ogw, N, C, C);
    launch_add_bias(og, ogb, N, C, nullptr);
    launch_gate_silu(attn, og, (long)N * C, nullptr);
    float *pw = LW("attn_proj_w", (size_t)C * C), *pb = LW("attn_proj_b", C);
    A.v.push_back(pw);
    A.v.push_back(pb);
    linear(x_sa, attn, pw, N, C, C);
    launch_add_bias(x_sa, pb, N, C, nullptr);
  }
  // Self-attention residual (gated by g_msa), then cross-attention to the text embeds y
  // (identical to dit_block; the streaming chunk only changes the token count N).
  float* x1 = A.a((size_t)N * C);
  launch_add_gate(x1, x_in, g_msa, x_sa, N, C, nullptr);
  float *cqw = LW("cx_q_w", (size_t)C * C), *cqb = LW("cx_q_b", C),
        *ckvw = LW("cx_kv_w", (size_t)2 * C * C), *ckvb = LW("cx_kv_b", 2 * C);
  A.v.push_back(cqw);
  A.v.push_back(cqb);
  A.v.push_back(ckvw);
  A.v.push_back(ckvb);
  float* cq = A.a((size_t)N * C);
  linear(cq, x1, cqw, N, C, C);
  launch_add_bias(cq, cqb, N, C, nullptr);
  float* ckv = A.a((size_t)Lk * 2 * C);
  linear(ckv, y, ckvw, Lk, C, 2 * C);
  launch_add_bias(ckv, ckvb, Lk, 2 * C, nullptr);
  float *ck2 = A.a((size_t)Lk * C), *cv2 = A.a((size_t)Lk * C);
  launch_split_ckv(ck2, cv2, ckv, Lk, C, nullptr);
  float *cqnw = LW("cx_q_norm", C), *cknw = LW("cx_k_norm", C);
  A.v.push_back(cqnw);
  A.v.push_back(cknw);
  float *cqn = A.a((size_t)N * C), *ckn = A.a((size_t)Lk * C);
  launch_rmsnorm(cqn, cq, cqnw, N, C, 1e-6f, nullptr);
  launch_rmsnorm(ckn, ck2, cknw, Lk, C, 1e-6f, nullptr);
  float *CQ = A.a((size_t)N * C), *CKk = A.a((size_t)Lk * C), *CV = A.a((size_t)Lk * C);
  launch_split_heads(CQ, cqn, N, nullptr);
  launch_split_heads(CKk, ckn, Lk, nullptr);
  launch_split_heads(CV, cv2, Lk, nullptr);
  float* CS = A.a((size_t)HEADS * N * Lk);
  float sc2 = 1.f / sqrtf((float)HD);
  cublasSgemmStridedBatched(
      HBL, CUBLAS_OP_T, CUBLAS_OP_N, Lk, N, HD, &sc2, CKk, HD, (long)Lk * HD, CQ, HD,
      (long)N * HD, &ZERO, CS, Lk, (long)N * Lk, HEADS);
  launch_softmax(CS, HEADS * N, Lk, valid, nullptr);
  float* CO = A.a((size_t)N * C);
  cublasSgemmStridedBatched(
      HBL, CUBLAS_OP_N, CUBLAS_OP_N, HD, N, Lk, &ONE, CV, HD, (long)Lk * HD, CS, Lk,
      (long)N * Lk, &ZERO, CO, HD, (long)N * HD, HEADS);
  float* cmerge = A.a((size_t)N * C);
  launch_merge_heads(cmerge, CO, N, nullptr);
  float *cpw = LW("cx_proj_w", (size_t)C * C), *cpb = LW("cx_proj_b", C);
  A.v.push_back(cpw);
  A.v.push_back(cpb);
  float* x_ca = A.a((size_t)N * C);
  linear(x_ca, cmerge, cpw, N, C, C);
  launch_add_bias(x_ca, cpb, N, C, nullptr);
  // Cross-attention residual, then the gated-conv FFN. The temporal conv's left neighbour
  // for the first frame comes from the carried fcache (g_ffn[i] in SC mode, else disk
  // ffn_cache); when SC_Save this chunk's last frame is stored as the next fcache.
  float* x2 = A.a((size_t)N * C);
  launch_add(x2, x1, x_ca, (long)N * C, nullptr);
  float* ln2 = A.a((size_t)N * C);
  launch_layernorm(ln2, x2, N, C, 1e-6f, nullptr);
  float* mlp_in = A.a((size_t)N * C);
  launch_adaln(mlp_in, ln2, sh_mlp, sc_mlp, N, C, nullptr);
  float *invw = LW("ffn_inv_w", (size_t)CH * C), *invb = LW("ffn_inv_b", CH);
  A.v.push_back(invw);
  A.v.push_back(invb);
  float* h1 = A.a((size_t)N * CH);
  linear(h1, mlp_in, invw, N, C, CH);
  launch_add_bias(h1, invb, N, CH, nullptr);
  launch_silu_inplace(h1, (long)N * CH, nullptr);
  float *dww = LW("ffn_dw_w", (size_t)CH * 9), *dwb = LW("ffn_dw_b", CH);
  A.v.push_back(dww);
  A.v.push_back(dwb);
  float* h2 = A.a((size_t)N * CH);
  launch_depthwise(h2, h1, dww, dwb, T, Hs, Ws, CH, nullptr);
  float* glu = A.a((size_t)N * HALF);
  launch_glu(glu, h2, N, HALF, nullptr);
  float* pww = LW("ffn_pw_w", (size_t)C * HALF);
  A.v.push_back(pww);
  float* pc = A.a((size_t)N * C);
  linear(pc, glu, pww, N, HALF, C);
  float* Wk3 = cached_wk(W);
  float* Wk[3] = {Wk3, Wk3 + (size_t)C * C, Wk3 + (size_t)2 * C * C};
  float* fcache = nullptr;
  if(SC)
    fcache = g_ffn[i]; // may be null on chunk 0 => zero left-pad
  else
  {
    fcache = LC("ffn_cache", (size_t)S * C);
    A.v.push_back(fcache);
  }
  float* x_ffn = A.a((size_t)N * C);
  cudaMemcpy(x_ffn, pc, (size_t)N * C * 4, cudaMemcpyDeviceToDevice);
  int HW = Hs * Ws;
  for(int t = 0; t < T; t++)
    for(int kk = 0; kk < 3; kk++)
    {
      int ft = t + kk - 1;
      const float* Pin;
      if(ft == -1)
      {
        if(!fcache)
          continue;
        Pin = fcache;
      }
      else if(ft < 0 || ft >= T)
        continue;
      else
        Pin = pc + (size_t)ft * HW * C;
      float* Yout = x_ffn + (size_t)t * HW * C;
      cublasSgemm(
          HBL, CUBLAS_OP_T, CUBLAS_OP_N, C, HW, C, &ONE, Wk[kk], C, Pin, C, &ONE, Yout,
          C);
    }
  if(SC && SC_Save)
  {
    if(!g_ffn[i])
      cudaMalloc(&g_ffn[i], (size_t)S * C * 4);
    cudaMemcpy(
        g_ffn[i], pc + (size_t)(T - 1) * S * C, (size_t)S * C * 4,
        cudaMemcpyDeviceToDevice);
  }
  launch_add_gate(x_out, x2, g_mlp, x_ffn, N, C, nullptr);
  A.free(); // stream-ordered; no per-block device sync
}

// Minimal reader for an integer field ("key": N) from the chunk's meta.json (T, ncache).
static int rd_meta(const std::string& dir, const char* key)
{
  std::ifstream f(dir + "/meta.json");
  std::string s((std::istreambuf_iterator<char>(f)), {});
  auto p = s.find(std::string("\"") + key + "\"");
  if(p == std::string::npos)
    return -1;
  p = s.find(':', p) + 1;
  return atoi(s.c_str() + p);
}

// Full 5-chunk streaming rollout. wdir = model weights (block{i}/ + top). datadir contains roll_c{0..4}.
// Returns the full 16-frame latent rel-L2 vs the Python sampler's gen in rel_l2.
static int
run_rollout(const std::string& wdir, const std::string& datadir, float& rel_l2)
{
  int valid = 18;
  float timesteps[4] = {1000.f, 961.f, 893.f, 743.f},
        sigmas[5] = {1.0f, 0.961f, 0.893f, 0.743f, 0.0f};
  auto LWT
      = [&](const char* nm, size_t n) { return LDB(wdir, nm, n); }; // resident (cached)
  float *xemb_w = LWT("xemb_w", (size_t)C * 256), *xemb_b = LWT("xemb_b", C);
  float *t0w = LWT("temb0_w", (size_t)C * 256), *t0b = LWT("temb0_b", C),
        *t2w = LWT("temb2_w", (size_t)C * C), *t2b = LWT("temb2_b", C),
        *tbw = LWT("tblock_w", (size_t)6 * C * C), *tbb = LWT("tblock_b", 6 * C);
  float *fsst = LWT("final_sst", 2 * C), *flw = LWT("final_lin_w", (size_t)128 * C),
        *flb = LWT("final_lin_b", 128);
  float *yfc1w = LWT("yfc1_w", (size_t)C * 2304), *yfc1b = LWT("yfc1_b", C),
        *yfc2w = LWT("yfc2_w", (size_t)C * C), *yfc2b = LWT("yfc2_b", C),
        *ynw = LWT("ynorm_w", C);
  int SOFTl[5] = {3, 7, 11, 15, 19};
  auto issoft = [&](int i) {
    for(int s : SOFTl)
      if(s == i)
        return 1;
    return 0;
  };
  double num = 0, den = 0;
  // Roll over the 5 chunks in order, carrying the DiT caches between them (via dit_block_s
  // reading the on-disk roll_c*/block* caches here) and accumulating the rel-L2 numerator.
  for(int c = 0; c < 5; c++)
  {
    std::string cd = datadir + "/roll_c" + std::to_string(c);
    int T = rd_meta(cd, "T"), Nc = T * S, Ncache = rd_meta(cd, "ncache");
    auto LT = [&](const char* nm, size_t n) {
      std::vector<float> v;
      rd(cd + "/" + nm + ".bin", n, v);
      return up(v);
    };
    // y_embedder for this chunk's prompt; rope + input latent/image loaded per chunk.
    float* y_raw = LT("y_raw", (size_t)Lk * 2304);
    float* y1 = dev((size_t)Lk * C);
    linear(y1, y_raw, yfc1w, Lk, 2304, C);
    launch_add_bias(y1, yfc1b, Lk, C, nullptr);
    launch_gelu_tanh(y1, (long)Lk * C, nullptr);
    float* y2 = dev((size_t)Lk * C);
    linear(y2, y1, yfc2w, Lk, C, C);
    launch_add_bias(y2, yfc2b, Lk, C, nullptr);
    float* y = dev((size_t)Lk * C);
    launch_rmsnorm(y, y2, ynw, Lk, C, 1e-6f, nullptr);
    float *rcos = LT("rope_cos", (size_t)Nc * HD),
          *rsin = LT("rope_sin", (size_t)Nc * HD);
    float *latent = LT("latent", (size_t)Nc * 128),
          *image = LT("image", (size_t)Nc * 128), *pred = dev((size_t)Nc * 128);
    // 4 FlowMatch-Euler denoise steps: x_embed (concat latent+image) + t_embed, run the
    // 20 blocks + final layer to get the velocity pred, then latent += (dsigma)*pred.
    for(int s = 0; s < 4; s++)
    {
      Arena A;
      float* x256 = A.a((size_t)Nc * 256);
      launch_concat_li(x256, latent, image, Nc, nullptr);
      float* x = A.a((size_t)Nc * C);
      linear(x, x256, xemb_w, Nc, 256, C);
      launch_add_bias(x, xemb_b, Nc, C, nullptr);
      float* temb = A.a(256);
      launch_time_sinusoid(temb, timesteps[s], 128, 10000.f, nullptr);
      float* t1 = A.a(C);
      linear(t1, temb, t0w, 1, 256, C);
      launch_add_bias(t1, t0b, 1, C, nullptr);
      launch_silu_inplace(t1, C, nullptr);
      float* tvec = A.a(C);
      linear(tvec, t1, t2w, 1, C, C);
      launch_add_bias(tvec, t2b, 1, C, nullptr);
      float* tsil = A.a(C);
      cudaMemcpy(tsil, tvec, C * 4, cudaMemcpyDeviceToDevice);
      launch_silu_inplace(tsil, C, nullptr);
      float* t0 = A.a(6 * C);
      linear(t0, tsil, tbw, 1, C, 6 * C);
      launch_add_bias(t0, tbb, 1, 6 * C, nullptr);
      float* cur = A.a((size_t)Nc * C);
      cudaMemcpy(cur, x, (size_t)Nc * C * 4, cudaMemcpyDeviceToDevice);
      for(int i = 0; i < 20; i++)
      {
        float* nxt = A.a((size_t)Nc * C);
        dit_block_s(
            nxt, cur, wdir, cd, i, issoft(i), t0, y, rcos, rsin, valid, T, Nc, Ncache);
        cur = nxt;
      }
      float *fsh = A.a(C), *fsc = A.a(C);
      launch_add(fsh, fsst, tvec, C, nullptr);
      launch_add(fsc, fsst + C, tvec, C, nullptr);
      float* lnf = A.a((size_t)Nc * C);
      launch_layernorm(lnf, cur, Nc, C, 1e-6f, nullptr);
      float* fmod = A.a((size_t)Nc * C);
      launch_adaln(fmod, lnf, fsh, fsc, Nc, C, nullptr);
      linear(pred, fmod, flw, Nc, C, 128);
      launch_add_bias(pred, flb, Nc, 128, nullptr);
      cudaDeviceSynchronize();
      launch_axpy(latent, pred, sigmas[s + 1] - sigmas[s], (long)Nc * 128, nullptr);
      cudaDeviceSynchronize();
      A.free();
    }
    // Compare this chunk's denoised latent to the Python sampler's gen.bin; accumulate rel-L2.
    std::vector<float> got;
    down(latent, (size_t)Nc * 128, got);
    std::vector<float> gold;
    rd(cd + "/gen.bin", (size_t)Nc * 128, gold);
    double nn = 0, dd = 0;
    for(size_t j = 0; j < got.size(); ++j)
    {
      double dv = got[j] - gold[j];
      nn += dv * dv;
      dd += (double)gold[j] * gold[j];
    }
    num += nn;
    den += dd;
    printf(
        "[sana] chunk %d (T=%d N=%d ncache=%d) rel-L2=%.4f%%\n", c, T, Nc, Ncache,
        100 * sqrt(nn / dd));
    for(float* pp : {y_raw, y1, y2, y, rcos, rsin, latent, image, pred})
      cudaFreeAsync(pp, 0);
  }
  rel_l2 = (float)sqrt(num / den);
  return 0;
}

// Self-contained 5-chunk rollout: PRODUCES its own cross-chunk caches (no roll_c*/block*/*cache
// files) via a t=0 update pass after each chunk. Non-cache per-chunk inputs (latent/image/y/rope)
// still read from datadir. Honest self-contained accuracy ~6.5% (bf16 error accumulates over chunks).
static int
run_rollout_sc(const std::string& wdir, const std::string& datadir, float& rel_l2)
{
  int valid = 18;
  float timesteps[4] = {1000.f, 961.f, 893.f, 743.f},
        sigmas[5] = {1.0f, 0.961f, 0.893f, 0.743f, 0.0f};
  auto LWT = [&](const char* nm, size_t n) { return LDB(wdir, nm, n); };
  float *xemb_w = LWT("xemb_w", (size_t)C * 256), *xemb_b = LWT("xemb_b", C);
  float *t0w = LWT("temb0_w", (size_t)C * 256), *t0b = LWT("temb0_b", C),
        *t2w = LWT("temb2_w", (size_t)C * C), *t2b = LWT("temb2_b", C),
        *tbw = LWT("tblock_w", (size_t)6 * C * C), *tbb = LWT("tblock_b", 6 * C);
  float *fsst = LWT("final_sst", 2 * C), *flw = LWT("final_lin_w", (size_t)128 * C),
        *flb = LWT("final_lin_b", 128);
  float *yfc1w = LWT("yfc1_w", (size_t)C * 2304), *yfc1b = LWT("yfc1_b", C),
        *yfc2w = LWT("yfc2_w", (size_t)C * C), *yfc2b = LWT("yfc2_b", C),
        *ynw = LWT("ynorm_w", C);
  int SOFTl[5] = {3, 7, 11, 15, 19};
  auto issoft = [&](int i) {
    for(int s : SOFTl)
      if(s == i)
        return 1;
    return 0;
  };
  // One full DiT forward for a chunk (x_embed + t_embed -> 20 blocks -> final -> pred).
  // Factored into a lambda so it can be reused for both the denoise steps and the extra
  // SC_Save t=0 update pass that populates the next chunk's caches.
  auto fwd
      = [&](float* pred, float* latent, float* image, float ts, float* y, float* rcos,
            float* rsin, int T, int Nc, int Ncache, const std::string& cd) {
    Arena A;
    float* x256 = A.a((size_t)Nc * 256);
    launch_concat_li(x256, latent, image, Nc, nullptr);
    float* x = A.a((size_t)Nc * C);
    linear(x, x256, xemb_w, Nc, 256, C);
    launch_add_bias(x, xemb_b, Nc, C, nullptr);
    float* temb = A.a(256);
    launch_time_sinusoid(temb, ts, 128, 10000.f, nullptr);
    float* t1 = A.a(C);
    linear(t1, temb, t0w, 1, 256, C);
    launch_add_bias(t1, t0b, 1, C, nullptr);
    launch_silu_inplace(t1, C, nullptr);
    float* tvec = A.a(C);
    linear(tvec, t1, t2w, 1, C, C);
    launch_add_bias(tvec, t2b, 1, C, nullptr);
    float* tsil = A.a(C);
    cudaMemcpy(tsil, tvec, C * 4, cudaMemcpyDeviceToDevice);
    launch_silu_inplace(tsil, C, nullptr);
    float* t0 = A.a(6 * C);
    linear(t0, tsil, tbw, 1, C, 6 * C);
    launch_add_bias(t0, tbb, 1, 6 * C, nullptr);
    float* cur = A.a((size_t)Nc * C);
    cudaMemcpy(cur, x, (size_t)Nc * C * 4, cudaMemcpyDeviceToDevice);
    for(int i = 0; i < 20; i++)
    {
      float* nxt = A.a((size_t)Nc * C);
      dit_block_s(
          nxt, cur, wdir, cd, i, issoft(i), t0, y, rcos, rsin, valid, T, Nc, Ncache);
      cur = nxt;
    }
    float *fsh = A.a(C), *fsc = A.a(C);
    launch_add(fsh, fsst, tvec, C, nullptr);
    launch_add(fsc, fsst + C, tvec, C, nullptr);
    float* lnf = A.a((size_t)Nc * C);
    launch_layernorm(lnf, cur, Nc, C, 1e-6f, nullptr);
    float* fmod = A.a((size_t)Nc * C);
    launch_adaln(fmod, lnf, fsh, fsc, Nc, C, nullptr);
    linear(pred, fmod, flw, Nc, C, 128);
    launch_add_bias(pred, flb, Nc, 128, nullptr);
    cudaDeviceSynchronize();
    A.free();
  };
  // Enable self-contained mode: dit_block_s now reads/writes the in-memory g_* caches.
  sc_reset();
  SC = 1;
  double num = 0, den = 0;
  for(int c = 0; c < 5; c++)
  {
    SC_Chunk = c;
    std::string cd = datadir + "/roll_c" + std::to_string(c);
    int T = rd_meta(cd, "T"), Nc = T * S;
    int Ncache = (c == 0) ? 0 : (c == 1) ? 1560 : 2730;
    auto LT = [&](const char* nm, size_t n) {
      std::vector<float> v;
      rd(cd + "/" + nm + ".bin", n, v);
      return up(v);
    };
    float* y_raw = LT("y_raw", (size_t)Lk * 2304);
    float* y1 = dev((size_t)Lk * C);
    linear(y1, y_raw, yfc1w, Lk, 2304, C);
    launch_add_bias(y1, yfc1b, Lk, C, nullptr);
    launch_gelu_tanh(y1, (long)Lk * C, nullptr);
    float* y2 = dev((size_t)Lk * C);
    linear(y2, y1, yfc2w, Lk, C, C);
    launch_add_bias(y2, yfc2b, Lk, C, nullptr);
    float* y = dev((size_t)Lk * C);
    launch_rmsnorm(y, y2, ynw, Lk, C, 1e-6f, nullptr);
    float *rcos = LT("rope_cos", (size_t)Nc * HD),
          *rsin = LT("rope_sin", (size_t)Nc * HD);
    float *latent = LT("latent", (size_t)Nc * 128),
          *image = LT("image", (size_t)Nc * 128), *pred = dev((size_t)Nc * 128);
    // 4 denoise steps (SC_Save off: consume caches, don't overwrite them).
    SC_Save = 0;
    for(int s = 0; s < 4; s++)
    {
      fwd(pred, latent, image, timesteps[s], y, rcos, rsin, T, Nc, Ncache, cd);
      launch_axpy(latent, pred, sigmas[s + 1] - sigmas[s], (long)Nc * 128, nullptr);
      cudaDeviceSynchronize();
    }
    std::vector<float> got;
    down(latent, (size_t)Nc * 128, got);
    std::vector<float> gold;
    rd(cd + "/gen.bin", (size_t)Nc * 128, gold);
    double nn = 0, dd = 0;
    for(size_t j = 0; j < got.size(); ++j)
    {
      double dv = got[j] - gold[j];
      nn += dv * dv;
      dd += (double)gold[j] * gold[j];
    }
    num += nn;
    den += dd;
    printf(
        "[sana] SC chunk %d (T=%d N=%d ncache=%d) rel-L2=%.4f%%\n", c, T, Nc, Ncache,
        100 * sqrt(nn / dd));
    // Extra t=0 forward with SC_Save on: captures this chunk's produced caches (GDN state,
    // softmax sink/prev K/V, FFN frame) for the next chunk to consume.
    if(c < 4)
    {
      SC_Save = 1;
      fwd(pred, latent, image, 0.f, y, rcos, rsin, T, Nc, Ncache, cd);
      SC_Save = 0;
    }
    for(float* pp : {y_raw, y1, y2, y, rcos, rsin, latent, image, pred})
      cudaFreeAsync(pp, 0);
  }
  SC = 0;
  sc_reset();
  rel_l2 = (float)sqrt(num / den);
  return 0;
}

// ================= run_v2v device helpers =================
static const int FULLF = 16; // latent frames for the whole clip

// Build the causal 3-axis (WAN-style) rotary tables for a chunk on the host and upload
// them: rcos/rsin are (Nc,HD), with the HD/2 frequency pairs split across the temporal
// axis (absolute frame start_f+fl) and the two spatial axes (h, w).
static void build_rope_dev(float** rcos, float** rsin, int start_f, int T)
{
  int Nc = T * S;
  std::vector<float> cosT((size_t)Nc * HD), sinT((size_t)Nc * HD);
  const double theta = 10000.0;
  for(int fl = 0; fl < T; fl++)
    for(int h = 0; h < Hs; h++)
      for(int w = 0; w < Ws; w++)
      {
        int n = (fl * Hs + h) * Ws + w;
        double ch[56], sh[56];
        auto axis = [&](int dim, int pos, int base, int cnt) {
          for(int k = 0; k < cnt; k++)
          {
            double fr = pow(theta, -(2.0 * k) / dim);
            double a = (double)pos * fr;
            ch[base + k] = cos(a);
            sh[base + k] = sin(a);
          }
        };
        axis(40, start_f + fl, 0, 20);
        axis(36, h, 20, 18);
        axis(36, w, 38, 18);
        for(int k = 0; k < 56; k++)
        {
          cosT[(size_t)n * HD + 2 * k] = (float)ch[k];
          cosT[(size_t)n * HD + 2 * k + 1] = (float)ch[k];
          sinT[(size_t)n * HD + 2 * k] = (float)-sh[k];
          sinT[(size_t)n * HD + 2 * k + 1] = (float)sh[k];
        }
      }
  *rcos = up(cosT);
  *rsin = up(sinT);
}

// In-memory self-contained rollout: image_full/noise_full (128,FULLF,15,26), text_embeds (Lk,2304 raw gemma)
// -> out_full (128,FULLF,15,26) denoised (DiT space). Mirrors run_rollout_sc with in-memory I/O.
static int run_v2v_core(
    const std::string& wdir, const float* image_full, const float* noise_full,
    const float* text_embeds, float* out_full)
{
  int valid = 18;
  float timesteps[4] = {1000.f, 961.f, 893.f, 743.f},
        sigmas[5] = {1.0f, 0.961f, 0.893f, 0.743f, 0.0f};
  auto LWT = [&](const char* nm, size_t n) { return LDB(wdir, nm, n); };
  float *xemb_w = LWT("xemb_w", (size_t)C * 256), *xemb_b = LWT("xemb_b", C);
  float *t0w = LWT("temb0_w", (size_t)C * 256), *t0b = LWT("temb0_b", C),
        *t2w = LWT("temb2_w", (size_t)C * C), *t2b = LWT("temb2_b", C),
        *tbw = LWT("tblock_w", (size_t)6 * C * C), *tbb = LWT("tblock_b", 6 * C);
  float *fsst = LWT("final_sst", 2 * C), *flw = LWT("final_lin_w", (size_t)128 * C),
        *flb = LWT("final_lin_b", 128);
  float *yfc1w = LWT("yfc1_w", (size_t)C * 2304), *yfc1b = LWT("yfc1_b", C),
        *yfc2w = LWT("yfc2_w", (size_t)C * C), *yfc2b = LWT("yfc2_b", C),
        *ynw = LWT("ynorm_w", C);
  // y (prompt-fixed)
  float* y1 = dev((size_t)Lk * C);
  linear(y1, text_embeds, yfc1w, Lk, 2304, C);
  launch_add_bias(y1, yfc1b, Lk, C, nullptr);
  launch_gelu_tanh(y1, (long)Lk * C, nullptr);
  float* y2 = dev((size_t)Lk * C);
  linear(y2, y1, yfc2w, Lk, C, C);
  launch_add_bias(y2, yfc2b, Lk, C, nullptr);
  float* y = dev((size_t)Lk * C);
  launch_rmsnorm(y, y2, ynw, Lk, C, 1e-6f, nullptr);
  // Chunk schedule: 5 chunks of Ts frames each, starting at frame startf (4 + 4*3 = 16).
  int Ts[5] = {4, 3, 3, 3, 3}, startf[5] = {0, 4, 7, 10, 13};
  // One DiT forward for a chunk (x_embed + t_embed -> 20 blocks -> final -> velocity pred);
  // reused for the denoise steps and the SC_Save t=0 cache-update pass.
  auto fwd = [&](float* pred, float* latent, float* image, float ts, float* rcos,
                 float* rsin, int T, int Nc, int Ncache) {
    Arena A;
    float* x256 = A.a((size_t)Nc * 256);
    launch_concat_li(x256, latent, image, Nc, nullptr);
    float* x = A.a((size_t)Nc * C);
    linear(x, x256, xemb_w, Nc, 256, C);
    launch_add_bias(x, xemb_b, Nc, C, nullptr);
    float* temb = A.a(256);
    launch_time_sinusoid(temb, ts, 128, 10000.f, nullptr);
    float* t1 = A.a(C);
    linear(t1, temb, t0w, 1, 256, C);
    launch_add_bias(t1, t0b, 1, C, nullptr);
    launch_silu_inplace(t1, C, nullptr);
    float* tvec = A.a(C);
    linear(tvec, t1, t2w, 1, C, C);
    launch_add_bias(tvec, t2b, 1, C, nullptr);
    float* tsil = A.a(C);
    cudaMemcpy(tsil, tvec, C * 4, cudaMemcpyDeviceToDevice);
    launch_silu_inplace(tsil, C, nullptr);
    float* t0 = A.a(6 * C);
    linear(t0, tsil, tbw, 1, C, 6 * C);
    launch_add_bias(t0, tbb, 1, 6 * C, nullptr);
    float* cur = A.a((size_t)Nc * C);
    cudaMemcpy(cur, x, (size_t)Nc * C * 4, cudaMemcpyDeviceToDevice);
    for(int i = 0; i < 20; i++)
    {
      float* nxt = A.a((size_t)Nc * C);
      dit_block_s(
          nxt, cur, wdir, "", i, is_soft(i), t0, y, rcos, rsin, valid, T, Nc, Ncache);
      cur = nxt;
    }
    float *fsh = A.a(C), *fsc = A.a(C);
    launch_add(fsh, fsst, tvec, C, nullptr);
    launch_add(fsc, fsst + C, tvec, C, nullptr);
    float* lnf = A.a((size_t)Nc * C);
    launch_layernorm(lnf, cur, Nc, C, 1e-6f, nullptr);
    float* fmod = A.a((size_t)Nc * C);
    launch_adaln(fmod, lnf, fsh, fsc, Nc, C, nullptr);
    linear(pred, fmod, flw, Nc, C, 128);
    launch_add_bias(pred, flb, Nc, 128, nullptr);
    cudaDeviceSynchronize();
    A.free();
  };
  // Self-contained streaming rollout over the 5 chunks (caches carried in the g_* buffers).
  sc_reset();
  SC = 1;
  for(int c = 0; c < 5; c++)
  {
    SC_Chunk = c;
    int T = Ts[c], Nc = T * S, Ncache = (c == 0) ? 0 : (c == 1) ? 1560 : 2730;
    // Slice this chunk's noise and conditioning-image frames out of the full-clip tensors.
    float *rcos, *rsin;
    build_rope_dev(&rcos, &rsin, startf[c], T);
    float* latent = dev((size_t)Nc * 128);
    launch_slice_chunk(latent, noise_full, startf[c], T, nullptr);
    float* image = dev((size_t)Nc * 128);
    launch_slice_chunk(image, image_full, startf[c], T, nullptr);
    float* pred = dev((size_t)Nc * 128);
    // 4 FlowMatch-Euler denoise steps for this chunk.
    SC_Save = 0;
    for(int s = 0; s < 4; s++)
    {
      fwd(pred, latent, image, timesteps[s], rcos, rsin, T, Nc, Ncache);
      launch_axpy(latent, pred, sigmas[s + 1] - sigmas[s], (long)Nc * 128, nullptr);
      cudaDeviceSynchronize();
    }
    // Write the denoised chunk back into out_full, then run the SC_Save t=0 pass to
    // populate the caches the next chunk consumes.
    launch_scatter_chunk(out_full, latent, startf[c], T, nullptr);
    if(c < 4)
    {
      SC_Save = 1;
      fwd(pred, latent, image, 0.f, rcos, rsin, T, Nc, Ncache);
      SC_Save = 0;
    }
    for(float* pp : {rcos, rsin, latent, image, pred})
      cudaFreeAsync(pp, 0);
  }
  SC = 0;
  sc_reset();
  for(float* pp : {y1, y2, y})
    cudaFreeAsync(pp, 0);
  return 0;
}

} // namespace sana

// ================= public class =================
// Private state of a SanaDiT instance: the DiT weights dir, the TRT engine + v2v data
// dirs, the resident gemma text embeds, and the (optionally resident) TRT wrapper.
struct SanaDiT::Impl
{
  std::string dir;
  std::string trt_dir, data_dir;
  float* text_embeds = nullptr;
  SanaTRT* trt = nullptr;
  ~Impl() { delete trt; }
};

// Construct: record the weights dir, lazily create the shared cuBLAS handle (ref-counted
// across instances), and configure the GDN kernel's dynamic shared-memory attribute.
SanaDiT::SanaDiT(const std::string& weights_dir)
{
  p_ = new Impl();
  p_->dir = weights_dir;
  if(sana::HBL_ref++ == 0)
    cublasCreate(&sana::HBL);
  sana::sana_gdn_init();
}
SanaDiT::~SanaDiT()
{
  delete p_;
  if(--sana::HBL_ref == 0 && sana::HBL)
  {
    cublasDestroy(sana::HBL);
    sana::HBL = nullptr;
  }
}
int SanaDiT::N() const
{
  return sana::Nt;
}
int SanaDiT::C() const
{
  return sana::C;
}

int SanaDiT::full_forward(std::vector<float>& out, float& rel_l2)
{
  using namespace sana;
  float* out_dev = dev((size_t)Nt * 128);
  if(!out_dev)
    return -1;
  if(run_full(p_->dir, nullptr, out_dev))
  {
    cudaFree(out_dev);
    return -1;
  }
  down(out_dev, (size_t)Nt * 128, out);
  cudaFreeAsync(out_dev, 0);
  std::vector<float> gold;
  rel_l2 = -1.f;
  if(rd(p_->dir + "/gold_model_out.bin", (size_t)Nt * 128, gold))
    rel_l2 = relL2(out, gold);
  return 0;
}

int SanaDiT::dit_forward(const float* x_embed_dev, float* noise_out_dev)
{
  using namespace sana;
  return run_full(p_->dir, const_cast<float*>(x_embed_dev), noise_out_dev);
}

int SanaDiT::run_rollout(const std::string& data_dir, float& rel_l2)
{
  return sana::run_rollout(p_->dir, data_dir, rel_l2);
}

int SanaDiT::run_rollout_sc(const std::string& data_dir, float& rel_l2)
{
  return sana::run_rollout_sc(p_->dir, data_dir, rel_l2);
}

int SanaDiT::set_trt(const std::string& trt_dir, const std::string& data_dir)
{
  p_->trt_dir = trt_dir;
  p_->data_dir = data_dir;
  // Create the resident TRT wrapper and deserialize the VAE encoder/decoder engines NOW, so the
  // ~2.4 GB plans are loaded once here (outside run_v2v) and reused across every run_v2v call.
  // (LRD_NO_RESIDENT=1 reproduces the old per-call-deserialize path for A/B profiling.)
  delete p_->trt;
  p_->trt = nullptr;
  if(!(getenv("LRD_NO_RESIDENT") && atoi(getenv("LRD_NO_RESIDENT"))))
  {
    p_->trt = new SanaTRT(trt_dir);
    p_->trt->warm_vae(); // deserialize the ~2.4 GB VAE engines once, resident
    sana::warm_weights(
        p_->dir); // preload the DiT fp32 weights once (the real first-call cost)
  }
  return 0;
}

int SanaDiT::encode_prompt_ids(const long long* ids, const long long* mask)
{
  using namespace sana;
  long long *ids_d, *mask_d;
  if(cudaMalloc(&ids_d, Lk * 8) || cudaMalloc(&mask_d, Lk * 8))
    return -1;
  cudaMemcpy(ids_d, ids, Lk * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(mask_d, mask, Lk * 8, cudaMemcpyHostToDevice);
  if(!p_->text_embeds)
    cudaMalloc(&p_->text_embeds, (size_t)Lk * 2304 * 4);
  SanaTRT trt(p_->trt_dir);
  int rc = trt.gemma_encode(ids_d, mask_d, p_->text_embeds);
  cudaFree(ids_d);
  cudaFree(mask_d);
  return rc;
}

int SanaDiT::set_text_embeds(const float* embeds)
{
  using namespace sana;
  if(!p_->text_embeds)
    cudaMalloc(&p_->text_embeds, (size_t)Lk * 2304 * 4);
  cudaMemcpy(p_->text_embeds, embeds, (size_t)Lk * 2304 * 4, cudaMemcpyHostToDevice);
  return 0;
}

int SanaDiT::run_v2v(
    const unsigned char* in_frames, int n_in, unsigned char* out_frames, int* n_out)
{
  using namespace sana;
  if(!p_->text_embeds)
  {
    printf("[sana] run_v2v: call encode_prompt first\n");
    return -1;
  }
  const int PT = 121, Hh = 480, Ww = 832, PIX = (size_t)3 * PT * Hh * Ww,
            LAT = (size_t)128 * FULLF * 15 * 26;
  // Resident TRT engines: created + warmed once in set_trt, reused across run_v2v calls so the
  // ~2.4 GB VAE encoder/decoder plans are NOT re-deserialized here. In the A/B baseline path
  // (LRD_NO_RESIDENT=1) p_->trt is null: use a fresh local wrapper so each engine is deserialized
  // inside the timed region, exactly like the original code.
  bool noResident = (getenv("LRD_NO_RESIDENT") && atoi(getenv("LRD_NO_RESIDENT")));
  SanaTRT localTrt(p_->trt_dir);
  if(!p_->trt && !noResident)
  {
    p_->trt = new SanaTRT(p_->trt_dir);
    p_->trt->warm_vae();
  }
  SanaTRT& trt = (p_->trt && !noResident) ? *p_->trt : localTrt;
  // Optional per-stage wall timing (LRD_PROF=1). Each timer syncs the device to attribute GPU work.
  bool prof = (getenv("LRD_PROF") && atoi(getenv("LRD_PROF")));
  auto clk = []() { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto tprev = clk();
  auto stage = [&](const char* nm) {
    if(prof)
    {
      cudaDeviceSynchronize();
      auto t = clk();
      printf("[prof] stage %-16s %.1f ms\n", nm, ms(tprev, t));
      tprev = t;
    }
  };
  auto LD = [&](const char* nm, size_t n) -> float* {
    std::vector<float> v;
    if(!rd(p_->data_dir + "/" + nm + ".bin", n, v))
      return nullptr;
    return up(v);
  };
  float *mean = LD("latents_mean", 128), *std_ = LD("latents_std", 128),
        *noise = LD("init_noise", LAT);
  if(!mean || !std_ || !noise)
  {
    printf("[sana] run_v2v: missing affine/noise in %s\n", p_->data_dir.c_str());
    return -1;
  }
  // preprocess input frames -> pixels
  unsigned char* rgb_d;
  cudaMalloc(&rgb_d, PIX);
  cudaMemcpy(rgb_d, in_frames, PIX, cudaMemcpyHostToDevice);
  float* pixels = dev(PIX);
  launch_preprocess(pixels, rgb_d, PT, Hh, Ww, nullptr);
  stage("preprocess");
  // VAE-encode -> moments -> affine -> image_full
  float* moments = dev((size_t)256 * FULLF * 15 * 26);
  trt.vae_encode_full(pixels, moments);
  float* image_full = dev(LAT);
  launch_affine_enc(image_full, moments, mean, std_, FULLF, nullptr);
  stage("vae_encode");
  // rollout (self-contained caches)
  float* out_full = dev(LAT);
  cudaMemset(out_full, 0, LAT * 4);
  run_v2v_core(p_->dir, image_full, noise, p_->text_embeds, out_full);
  stage("dit_rollout");
  // inverse affine -> VAE-decode -> postprocess
  launch_affine_dec(out_full, mean, std_, FULLF, nullptr);
  float* frames = dev(PIX);
  trt.vae_decode_16f(out_full, frames);
  stage("vae_decode");
  unsigned char* outrgb_d;
  cudaMalloc(&outrgb_d, PIX);
  launch_postprocess(outrgb_d, frames, PT, Hh, Ww, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(out_frames, outrgb_d, PIX, cudaMemcpyDeviceToHost);
  *n_out = PT;
  stage("postprocess");
  for(float* p : {mean, std_, noise, pixels, moments, image_full, out_full, frames})
    cudaFree(p);
  cudaFree(rgb_d);
  cudaFree(outrgb_d);
  return 0;
}

} // namespace librediffusion

#if defined(_MSC_VER)
#  pragma float_control(pop)
#endif
