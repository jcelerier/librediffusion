/** SANA-Streaming DiT — pure-CUDA kernel launchers (defined in librediffusion.sana.cu).
 *
 *  This header exposes ONLY POD types and thin `launch_*` host wrappers whose
 *  signatures avoid CUDA-only types (device buffers are `float*`/`void*`, streams
 *  are `void*`, null == the default stream). That keeps it includable from the
 *  pure-C++ orchestration units (librediffusion.sana.cpp / librediffusion.sana_trt.cpp),
 *  mirroring the src/kernels.hpp <-> src/kernels.cu split used elsewhere in the library.
 */
#pragma once

#include <cstddef>

namespace librediffusion
{
namespace sana
{

// ================= SANA architecture constants (shared by kernels + host) =================
static const int C = 2240, HEADS = 20, HD = 112;
// Reference-chunk shapes: Tt frames, Hs*Ws=S spatial tokens/frame, Nt total tokens,
// Lk text tokens, CH/HALF the FFN inverted-conv/point-conv widths.
static const int Tt = 4, Hs = 15, Ws = 26, S = Hs * Ws, Nt = Tt * S, Lk = 300,
                 CH = 13440, HALF = 6720;
// GDN row-split groups (the Bufs.G the host fills before launch_gdn_bidi).
static const int GDN_G = 16;

// ================= GDN argument bundles (POD; passed by value to the kernels) =================
// q/k/v projections (+ rope-rotated qrot/krot), the per-token beta gate and per-frame
// decay, optional carried-in recurrent state (init_kv/init_z), and the outputs (num/den
// readout, out_kv/out_z carried-out state). H/D/T/S/N are head count / head dim / frames /
// tokens-per-frame / total tokens.
struct Bufs
{
  const float *q, *k, *v, *qrot, *krot, *beta, *decay, *init_kv, *init_z;
  float *num, *den, *out_kv, *out_z, *state_kv, *state_z, *dv, *dz;
  int H, D, T, S, N;
  float eps;
  int G;
};
// The fused qkv projection, optional q/k RMS scale factors (q_inv/k_inv) and weights
// (q_nw/k_nw), rope cos/sin, and the split, normalized, rope-rotated q/k/v outputs
// (head-major). k_scale folds the GDN attn scaling.
struct Prep
{
  const float *qkv, *q_inv, *k_inv, *q_nw, *k_nw, *cos, *sin;
  float *q, *k, *v, *qrot, *krot;
  int B, H, D, N;
  float k_scale;
  int qk_norm;
};

// One-time GDN setup: computes the dynamic shared-mem size and opts gdn_bidi into it.
// Must be called once before the first launch_gdn_bidi.
void sana_gdn_init();

// ================= host launchers (grid/block derived from args, as the original inline
// launches were; stream == null binds the default stream) =================
void launch_add_bias(float* Y, const float* b, int M, int No, void* stream);
void launch_layernorm(float* Y, const float* X, int M, int D, float eps, void* stream);
void launch_rmsnorm(
    float* Y, const float* X, const float* W, int M, int D, float eps, void* stream);
void launch_adaln(
    float* Y, const float* X, const float* shift, const float* scale, int M, int D,
    void* stream);
void launch_add_gate(
    float* Y, const float* A, const float* gate, const float* Bm, int M, int D,
    void* stream);
void launch_add(float* Y, const float* A, const float* Bm, long n, void* stream);
void launch_silu_inplace(float* Y, long n, void* stream);
void launch_gate_silu(float* Y, const float* og, long n, void* stream);
void launch_inv_rms(float* out, const float* X, int M, int D, float eps, void* stream);
void launch_softmax(float* Sm, int rows, int cols, int valid, void* stream);
void launch_split_heads(float* O, const float* X, int M, void* stream);
void launch_merge_heads(float* O, const float* X, int M, void* stream);
void launch_rope_bnhd(
    float* Y, const float* X, const float* cos, const float* sin, int M, void* stream);
void launch_gdn_bidi(const Bufs& b, void* stream);
void launch_prep_qkv(const Prep& p, void* stream);
void launch_prep_rope(const Prep& p, void* stream);
void launch_sigmoid(float* Y, long n, void* stream);
void launch_beta_hts(float* B, const float* beta_ns, int T, int Sv, void* stream);
void launch_decay_ht(
    float* Dout, const float* a_out, const float* A_log, const float* dt_bias, int T,
    void* stream);
void launch_gdn_out(
    float* Y, const float* num, const float* den, int N, float eps, void* stream);
void launch_mean_frames(float* Y, const float* X, int T, int Sv, void* stream);
void launch_depthwise(
    float* Y, const float* X, const float* Wt, const float* Bs, int T, int Hsz, int Wsz,
    int Ch, void* stream);
void launch_glu(float* Y, const float* X, int M, int half, void* stream);
void launch_time_sinusoid(float* Y, float t, int half, float max_period, void* stream);
void launch_gelu_tanh(float* Y, long n, void* stream);
void launch_split_qkv(float* q, float* k, const float* qkv, int N, int Cc, void* stream);
void launch_split_ckv(
    float* ck, float* cv, const float* ckv, int Lkn, int Cc, void* stream);
// fp32 -> bf16 cast feeding the bf16 tensor-core GEMMs in linear() (o is a bf16 buffer).
void launch_f2bf(void* o, const float* x, long n, void* stream);
void launch_place(
    float* dst, const float* src, int H, int Nfull, int Nsrc, int HDv, int off,
    void* stream);
void launch_concat_li(float* Y, const float* lat, const float* img, int Nn, void* stream);
void launch_axpy(float* lat, const float* pred, float coef, long n, void* stream);
void launch_preprocess(
    float* px, const unsigned char* rgb, int T, int H, int W, void* stream);
void launch_postprocess(
    unsigned char* rgb, const float* px, int T, int H, int W, void* stream);
void launch_affine_enc(
    float* z, const float* mom, const float* mean, const float* std, int F, void* stream);
void launch_affine_dec(
    float* z, const float* mean, const float* std, int F, void* stream);
void launch_slice_chunk(float* chunk, const float* full, int start_f, int T, void* stream);
void launch_scatter_chunk(
    float* full, const float* chunk, int start_f, int T, void* stream);

// fp32<->bf16 casts at the TRT bf16-engine boundary (used by librediffusion.sana_trt.cpp;
// in/out bf16 buffers are void*).
void launch_trt_f2bf(const float* in, void* out, long n, void* stream);
void launch_trt_bf2f(const void* in, float* out, long n, void* stream);

} // namespace sana
} // namespace librediffusion
