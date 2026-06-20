
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <kernels.hpp>

#include <cstdint>
#include <cstdio>

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

// Scheduler step functor using Thrust for automatic kernel fusion
struct SchedulerStepFunctor {
    float alpha, beta, c_out, c_skip;

    __host__ __device__
    SchedulerStepFunctor(float a, float b, float co, float cs)
        : alpha(a), beta(b), c_out(co), c_skip(cs) {}

    __host__ __device__
    __half operator()(const thrust::tuple<__half, __half>& input) const {
        // Unpack tuple from zip_iterator (model_pred, x_t_latent)
        float mp = __half2float(thrust::get<0>(input));
        float xt = __half2float(thrust::get<1>(input));

        // F_theta = (x_t - beta * model_pred) / alpha
        float F_theta = (xt - beta * mp) / alpha;

        // denoised = c_out * F_theta + c_skip * x_t
        float result = c_out * F_theta + c_skip * xt;
        return __float2half(result);
    }
};

// Legacy kernel kept for reference but not used
__global__ void scheduler_step_kernel_fp16_legacy(
    const __half* model_pred,
    const __half* x_t_latent,
    __half* output,
    float alpha,
    float beta,
    float c_skip,
    float c_out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mp = __half2float(model_pred[idx]);
        float xt = __half2float(x_t_latent[idx]);
        float F_theta = (xt - beta * mp) / alpha;
        float result = c_out * F_theta + c_skip * xt;
        output[idx] = __float2half(result);
    }
}

// Scalar division functor
struct ScalarDivFunctor {
    float scalar;

    __host__ __device__
    ScalarDivFunctor(float s) : scalar(s) {}

    __host__ __device__
    __half operator()(__half x) const {
        return __float2half(__half2float(x) / scalar);
    }
};

// Scalar multiplication functor
struct ScalarMulFunctor {
    float scalar;

    __host__ __device__
    ScalarMulFunctor(float s) : scalar(s) {}

    __host__ __device__
    __half operator()(__half x) const {
        return __float2half(__half2float(x) * scalar);
    }
};

// Add noise functor: alpha * original + beta * noise
struct AddNoiseFunctor {
    float alpha, beta;

    __host__ __device__
    AddNoiseFunctor(float a, float b) : alpha(a), beta(b) {}

    __host__ __device__
    __half operator()(const thrust::tuple<__half, __half>& input) const {
        float orig = __half2float(thrust::get<0>(input));
        float noise = __half2float(thrust::get<1>(input));
        return __float2half(alpha * orig + beta * noise);
    }
};

// Tensor subtraction functor
struct TensorSubFunctor {
    __host__ __device__
    __half operator()(const thrust::tuple<__half, __half>& input) const {
        float a = __half2float(thrust::get<0>(input));
        float b = __half2float(thrust::get<1>(input));
        return __float2half(a - b);
    }
};

// CFG application functor: uncond + guidance_scale * (text - uncond)
struct ApplyCFGFunctor {
    float guidance_scale;

    __host__ __device__
    ApplyCFGFunctor(float gs) : guidance_scale(gs) {}

    __host__ __device__
    __half operator()(const thrust::tuple<__half, __half>& input) const {
        float uncond = __half2float(thrust::get<0>(input));
        float text = __half2float(thrust::get<1>(input));
        return __float2half(uncond + guidance_scale * (text - uncond));
    }
};

// Legacy kernels
__global__ void
scalar_div_kernel_fp16_legacy(const __half* data, __half* output, float scalar, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    float val = __half2float(data[idx]);
    output[idx] = __float2half(val / scalar);
  }
}

__global__ void scalar_mul_inplace_kernel_fp16_legacy(
    __half* data,
    float scalar,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(val * scalar);
    }
}

// Conversion kernel: FP32 -> FP16 (used after cuRAND generation)
__global__ void convert_fp32_to_fp16_kernel(
    const float* input,
    __half* output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __float2half(input[idx]);
    }
}

// Add noise kernel: noisy = alpha * original + beta * noise
__global__ void add_noise_kernel_fp16(
    const __half* original_samples,
    const __half* noise,
    __half* noisy_samples,
    float alpha,
    float beta,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float orig = __half2float(original_samples[idx]);
        float n = __half2float(noise[idx]);
        float result = alpha * orig + beta * n;
        noisy_samples[idx] = __float2half(result);
    }
}

__global__ void add_noise_direct_kernel_fp16(
    __half *original_samples, const __half *noise, float alpha, float beta, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float orig = __half2float(original_samples[idx]);
        float n = __half2float(noise[idx]);
        float result = alpha * orig + beta * n;
        original_samples[idx] = __float2half(result);
    }
}

// Apply CFG kernel: output = uncond + guidance_scale * (text - uncond)
__global__ void apply_cfg_kernel_fp16(
    const __half* noise_pred_uncond,
    const __half* noise_pred_text,
    __half* output,
    float guidance_scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float uncond = __half2float(noise_pred_uncond[idx]);
        float text = __half2float(noise_pred_text[idx]);
        float result = uncond + guidance_scale * (text - uncond);
        output[idx] = __float2half(result);
    }
}

// Ones-like kernel: fill with 1.0
__global__ void ones_like_kernel_fp16(
    __half* output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __float2half(1.0f);
    }
}

// Tensor subtraction kernel: output = a - b
__global__ void tensor_sub_kernel_fp16(
    const __half* a,
    const __half* b,
    __half* output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val_a = __half2float(a[idx]);
        float val_b = __half2float(b[idx]);
        output[idx] = __float2half(val_a - val_b);
    }
}

template <int BLOCK_SIZE = 256>
__global__ void cosine_similarity_kernel_optimized(
    const __half* x,              // [batch, seq_len, hidden_dim]
    const __half* y,              // [batch, cached_seq_len, hidden_dim]
    float* similarities,          // [batch, seq_len, cached_seq_len]
    int batch,
    int seq_len,
    int cached_seq_len,
    int hidden_dim)
{
    int pair_idx = blockIdx.x;
    int total_pairs = batch * seq_len * cached_seq_len;

    if(pair_idx >= total_pairs) return;

    // Decode pair indices
    int b = pair_idx / (seq_len * cached_seq_len);
    int remainder = pair_idx % (seq_len * cached_seq_len);
    int q_idx = remainder / cached_seq_len;
    int k_idx = remainder % cached_seq_len;

    // Calculate base offsets
    int x_offset = (b * seq_len + q_idx) * hidden_dim;
    int y_offset = (b * cached_seq_len + k_idx) * hidden_dim;

    // Use CUB block reduction for parallel summation
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_dot;
    __shared__ typename BlockReduce::TempStorage temp_storage_x_norm;
    __shared__ typename BlockReduce::TempStorage temp_storage_y_norm;

    // Thread-local accumulators
    float thread_dot = 0.0f;
    float thread_x_norm = 0.0f;
    float thread_y_norm = 0.0f;

    // Each thread processes hidden_dim / BLOCK_SIZE elements
    for(int d = threadIdx.x; d < hidden_dim; d += BLOCK_SIZE)
    {
        float x_val = __half2float(x[x_offset + d]);
        float y_val = __half2float(y[y_offset + d]);

        thread_dot += x_val * y_val;
        thread_x_norm += x_val * x_val;
        thread_y_norm += y_val * y_val;
    }

    // Block-wide reduction using CUB
    __syncthreads();
    float dot = BlockReduce(temp_storage_dot).Sum(thread_dot);
    __syncthreads();
    float x_norm = BlockReduce(temp_storage_x_norm).Sum(thread_x_norm);
    __syncthreads();
    float y_norm = BlockReduce(temp_storage_y_norm).Sum(thread_y_norm);

    // Thread 0 writes final result
    if(threadIdx.x == 0)
    {
        float sim = dot / (sqrtf(x_norm) * sqrtf(y_norm) + 1e-8f);
        similarities[pair_idx] = sim;
    }
}

void launch_scheduler_step_fp16(
    const void* model_pred,
    const void* x_t_latent,
    void* output,
    float alpha,
    float beta,
    float c_skip,
    float c_out,
    int batch,
    int channels,
    int height,
    int width,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int N = batch * channels * height * width;

    // Use Thrust with zip_iterator for kernel fusion
    // This eliminates intermediate memory writes and enables vectorization
    thrust::device_ptr<const __half> pred_ptr((const __half*)model_pred);
    thrust::device_ptr<const __half> xt_ptr((const __half*)x_t_latent);
    thrust::device_ptr<__half> out_ptr((__half*)output);

    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(pred_ptr, xt_ptr));
    auto input_end = input_begin + N;

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), input_begin, input_end, out_ptr,
        SchedulerStepFunctor(alpha, beta, c_out, c_skip));
}

void launch_scalar_div_fp16(
    const void* data,
    void* output,
    float scalar,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    thrust::device_ptr<const __half> in_ptr((const __half*)data);
    thrust::device_ptr<__half> out_ptr((__half*)output);

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), in_ptr, in_ptr + N, out_ptr,
        ScalarDivFunctor(scalar));
}

void launch_scalar_mul_inplace_fp16(
    void* data,
    float scalar,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    thrust::device_ptr<__half> data_ptr((__half*)data);

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), data_ptr, data_ptr + N, data_ptr,
        ScalarMulFunctor(scalar));
}

// Simple functor for conversion
struct Float2HalfFunctor
{
  __host__ __device__ __half operator()(float x) const noexcept
  {
    return __float2half(x);
  }
};

// ---------------------------------------------------------------------------
// Deterministic counter-based PCG32 Gaussian noise.
// librediffusion fork: torch.randn and cuRAND's curandGenerateNormal do NOT agree bit-for-bit
// (different uniform->Gaussian transform + memory layout), so python<->C++ generated noise
// diverged (txt2img). Replace cuRAND with ONE explicit counter-based spec shared with the
// Python side (src/streamdiffusion/deterministic_noise.py) and pcg.hpp's PCG32 constants.
// Per output element `i`: seed PCG32 with (s0=seed, s1=i), draw 2 u32 -> 2 uniforms in (0,1]
// -> Box-Muller z0 -> __float2half. Value depends ONLY on (seed, i): index-deterministic,
// independent of N/shape/threads. All math in fp32, cast to fp16 last. Also faster than the
// old cuRAND path (no malloc / temp fp32 buffer / Thrust pass).
__device__ __forceinline__ uint32_t pcg_rotr32(uint32_t x, uint32_t r)
{
  r &= 31u;
  return (x >> r) | (x << ((32u - r) & 31u));
}

struct PcgState
{
  unsigned long long state;
  unsigned long long inc;
};

// Mirrors pcg.hpp operator(): advance LCG, return previous-state output.
__device__ __forceinline__ uint32_t pcg_step(PcgState& s)
{
  unsigned long long old = s.state;
  s.state = old * 6364136223846793005ULL + s.inc;
  uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
  uint32_t rot = (uint32_t)(old >> 59u);
  return pcg_rotr32(xorshifted, rot);
}

// Mirrors pcg.hpp seed(s0, s1): m_inc=(s1<<1)|1; m_state=0; step(); m_state+=s0; step().
__device__ __forceinline__ void pcg_seed(PcgState& s, unsigned long long s0, unsigned long long s1)
{
  s.inc = (s1 << 1) | 1ULL;
  s.state = 0ULL;
  pcg_step(s);
  s.state += s0;
  pcg_step(s);
}

__global__ void pcg32_randn_kernel_fp16(__half* output, unsigned long long seed, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= N)
    return;
  PcgState s;
  pcg_seed(s, seed, (unsigned long long)i);
  // uniform in (0,1]: (u32 + 1) * 2^-32 ; all math fp32 to match the numpy replica
  const float two32 = 1.0f / 4294967296.0f; // 2^-32
  float u1 = ((float)pcg_step(s) + 1.0f) * two32;
  float u2 = ((float)pcg_step(s) + 1.0f) * two32;
  float r = sqrtf(-2.0f * logf(u1));
  float z0 = r * cosf(2.0f * 3.14159265358979323846f * u2);
  output[i] = __float2half(z0);
}

void launch_randn_fp16(
    void* output,
    unsigned long long seed,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    // Deterministic counter-based PCG32 Gaussian — bit-identical to the Python replica
    // (deterministic_noise.py). Writes __half directly; no temp buffer / malloc / Thrust.
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    pcg32_randn_kernel_fp16<<<blocks, threads, 0, stream>>>((__half*)output, seed, N);
}

void launch_add_noise_fp16(
    const void* original_samples,
    const void* noise,
    void* noisy_samples,
    float alpha,
    float beta,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Use Thrust with zip_iterator for fused operation
    thrust::device_ptr<const __half> orig_ptr((const __half*)original_samples);
    thrust::device_ptr<const __half> noise_ptr((const __half*)noise);
    thrust::device_ptr<__half> out_ptr((__half*)noisy_samples);

    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(orig_ptr, noise_ptr));
    auto input_end = input_begin + N;

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), input_begin, input_end, out_ptr,
        AddNoiseFunctor(alpha, beta));
}

void launch_add_noise_direct_fp16(
    void *original_samples, const void *noise, float alpha, float beta, int N, void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    add_noise_direct_kernel_fp16<<<blocks, threads, 0, stream>>>(
        (__half*)original_samples, (const __half*)noise, alpha, beta, N);
    /*
    // In-place version: output to same buffer as input
    thrust::device_ptr<__half> orig_ptr((__half*)original_samples);
    thrust::device_ptr<const __half> noise_ptr((const __half*)noise);

    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(orig_ptr, noise_ptr));
    auto input_end = input_begin + N;

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), input_begin, input_end,
        orig_ptr, // Write back to original
        AddNoiseFunctor(alpha, beta));
*/
}

void launch_apply_cfg_fp16(
    const void* noise_pred_uncond,
    const void* noise_pred_text,
    void* output,
    float guidance_scale,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Use Thrust for fused CFG application
    thrust::device_ptr<const __half> uncond_ptr((const __half*)noise_pred_uncond);
    thrust::device_ptr<const __half> text_ptr((const __half*)noise_pred_text);
    thrust::device_ptr<__half> out_ptr((__half*)output);

    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(uncond_ptr, text_ptr));
    auto input_end = input_begin + N;

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), input_begin, input_end, out_ptr,
        ApplyCFGFunctor(guidance_scale));
}

void launch_ones_like_fp16(
    void* output,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ones_like_kernel_fp16<<<blocks, threads, 0, stream>>>(
        (__half*)output,
        N
    );
}

void launch_tensor_clone(
    const void* src,
    void* dst,
    size_t num_bytes,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream);
}

// FP16 to FP32 conversion kernel
extern "C" __global__ void fp16_to_fp32_kernel(const __half* input, float* output, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    output[idx] = __half2float(input[idx]);
  }
}

// FP32 to FP16 conversion kernel
extern "C" __global__ void fp32_to_fp16_kernel(const float* input, __half* output, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
  {
    output[idx] = __float2half(input[idx]);
  }
}

void launch_fp16_to_fp32(
    const void* input,
    void* output,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)input,
        (float*)output,
        N
    );

    // Silently clear any errors
    cudaGetLastError();
}

void launch_fp32_to_fp16(
    const void* input,
    void* output,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (__half*)output,
        N
    );
}

void launch_tensor_sub_fp16(
    const void* a,
    const void* b,
    void* output,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Use Thrust for tensor subtraction
    thrust::device_ptr<const __half> a_ptr((const __half*)a);
    thrust::device_ptr<const __half> b_ptr((const __half*)b);
    thrust::device_ptr<__half> out_ptr((__half*)output);

    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(a_ptr, b_ptr));
    auto input_end = input_begin + N;

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), input_begin, input_end, out_ptr,
        TensorSubFunctor());
}

// V2V
void launch_cosine_similarity(
    const void* x,
    const void* y,
    void* similarities,
    int batch, int seq_len, int cached_seq_len, int hidden_dim,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_pairs = batch * seq_len * cached_seq_len;

    // One block per pair, threads collaborate on reduction
    constexpr int BLOCK_SIZE = 256;
    int blocks = total_pairs;

    cosine_similarity_kernel_optimized<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        (const __half*)x,
        (const __half*)y,
        (float*)similarities,
        batch, seq_len, cached_seq_len, hidden_dim
    );
}

// Nearest-neighbor selection: for each query, find best match in cache
// Uses cosine similarity to select features
__global__ void nearest_neighbor_kernel(
    const __half* current_features,   // [batch, seq_len, hidden_dim]
    const __half* cached_features,    // [batch, cached_seq_len, hidden_dim]
    const float* similarities,        // [batch, seq_len, cached_seq_len]
    __half* output,                   // [batch, seq_len, hidden_dim]
    float threshold,
    int batch,
    int seq_len,
    int cached_seq_len,
    int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = batch * seq_len;

    if(idx < total_queries)
    {
        int b = idx / seq_len;
        int q_idx = idx % seq_len;

        // Find max similarity for this query
        float max_sim = -1.0f;
        int best_idx = 0;

        int sim_offset = (b * seq_len + q_idx) * cached_seq_len;
        for(int k = 0; k < cached_seq_len; k++)
        {
            float sim = similarities[sim_offset + k];
            if(sim > max_sim)
            {
                max_sim = sim;
                best_idx = k;
            }
        }

        // Determine source: use current if similarity < threshold, else use neighbor
        int curr_offset = (b * seq_len + q_idx) * hidden_dim;
        int nn_offset = (b * cached_seq_len + best_idx) * hidden_dim;

        const __half* src = (max_sim < threshold) ? &current_features[curr_offset] : &cached_features[nn_offset];

        // Copy features
        int out_offset = (b * seq_len + q_idx) * hidden_dim;
        for(int d = 0; d < hidden_dim; d++)
        {
            output[out_offset + d] = src[d];
        }
    }
}

void launch_nearest_neighbor(
    const void* current_features,
    const void* cached_features,
    const void* similarities,
    void* output,
    float threshold,
    int batch, int seq_len, int cached_seq_len, int hidden_dim,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_queries = batch * seq_len;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    nearest_neighbor_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)current_features,
        (const __half*)cached_features,
        (const float*)similarities,
        (__half*)output,
        threshold,
        batch, seq_len, cached_seq_len, hidden_dim
    );
}

// Blend features: interpolate between current and nearest-neighbor features
__global__ void blend_features_kernel(
    const __half* current,
    const __half* neighbor,
    __half* output,
    float blend_strength,  // 0.8 = 80% neighbor, 20% current
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        float curr_val = __half2float(current[idx]);
        float nn_val = __half2float(neighbor[idx]);

        float blended = curr_val * (1.0f - blend_strength) + nn_val * blend_strength;

        output[idx] = __float2half(blended);
    }
}

void launch_blend_features(
    const void* current,
    const void* neighbor,
    void* output,
    float blend_strength,
    int N,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    blend_features_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)current,
        (const __half*)neighbor,
        (__half*)output,
        blend_strength,
        N
    );
}

// ============================ StreamV2V ToMe bank merge ============================
// Token Merging (random_bipartite_soft_matching) with a DETERMINISTIC even/odd split (replaces the
// original's torch.rand().argsort() so C++ and Python match bit-for-bit when both use this split).
// Input cat_{k,v,o}: [B, N, h] (N even) = concat(bank, new). Partition: src = even positions, dst = odd
// positions (r = N/2 each). For each src token, find its cosine-nearest dst token (metric = keys), then
// scatter-mean each src into its matched dst (include_self). Output dst_{k,v,o}: [B, r, h] = compacted bank.

// one thread per (batch, dst token) -> precompute dst (odd) token norms once; every src token's
// argmax below reuses them (otherwise nb/sqrtf is recomputed r times redundantly, once per src).
__global__ void tome_dstnorm_kernel(const __half* cat_k, int B, int N, int h, float* dstn)
{
    int r = N / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= B * r) return;
    int b = idx / r, j = idx % r;
    const __half* bj = cat_k + ((size_t)b * N + (size_t)(2 * j + 1)) * h;
    float nb = 0.f;
    for(int c = 0; c < h; c++) { float v = __half2float(bj[c]); nb += v * v; }
    dstn[idx] = sqrtf(nb) + 1e-12f;
}

// one thread per (batch, src token) -> argmax cosine over dst tokens (keys as metric)
__global__ void tome_match_kernel(const __half* cat_k, int B, int N, int h, const float* dstn, int* match)
{
    int r = N / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= B * r) return;
    int b = idx / r, i = idx % r;                       // src token i (even position 2i)
    const __half* a = cat_k + ((size_t)b * N + (size_t)2 * i) * h;
    float na = 0.f;
    for(int c = 0; c < h; c++) { float v = __half2float(a[c]); na += v * v; }
    na = sqrtf(na) + 1e-12f;
    float best = -1e30f; int bestj = 0;
    for(int j = 0; j < r; j++) {
        const __half* bj = cat_k + ((size_t)b * N + (size_t)(2 * j + 1)) * h;
        float dot = 0.f;
        for(int c = 0; c < h; c++) { dot += __half2float(a[c]) * __half2float(bj[c]); }
        float cos = dot / (na * dstn[(size_t)b * r + j]);   // dst norm precomputed
        if(cos > best) { best = cos; bestj = j; }
    }
    match[idx] = bestj;
}

// init dst accumulators from the dst (odd) tokens; count = 1 (include_self)
__global__ void tome_init_kernel(const __half* cat_k, const __half* cat_v, const __half* cat_o,
                                 int B, int N, int h, float* ak, float* av, float* ao, int* cnt)
{
    int r = N / 2;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)B * r * h;
    if(t >= tot) return;
    int c = t % h, j = (t / h) % r, b = t / (h * (long)r);
    size_t src = ((size_t)b * N + (size_t)(2 * j + 1)) * h + c;
    ak[t] = __half2float(cat_k[src]); av[t] = __half2float(cat_v[src]); ao[t] = __half2float(cat_o[src]);
    if(c == 0) cnt[b * r + j] = 1;
}

// scatter-add each src token into its matched dst accumulator (atomic)
__global__ void tome_scatter_kernel(const __half* cat_k, const __half* cat_v, const __half* cat_o,
                                    const int* match, int B, int N, int h,
                                    float* ak, float* av, float* ao, int* cnt)
{
    int r = N / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (batch, src token)
    if(idx >= B * r) return;
    int b = idx / r, i = idx % r;
    int j = match[idx];
    size_t s = ((size_t)b * N + (size_t)2 * i) * h;
    size_t d = ((size_t)b * r + j) * h;
    for(int c = 0; c < h; c++) {
        atomicAdd(&ak[d + c], __half2float(cat_k[s + c]));
        atomicAdd(&av[d + c], __half2float(cat_v[s + c]));
        atomicAdd(&ao[d + c], __half2float(cat_o[s + c]));
    }
    atomicAdd(&cnt[b * r + j], 1);
}

// finalize: out = accum / count  -> __half
__global__ void tome_finalize_kernel(const float* ak, const float* av, const float* ao, const int* cnt,
                                     int B, int N, int h, __half* ok, __half* ov, __half* oo)
{
    int r = N / 2;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)B * r * h;
    if(t >= tot) return;
    int j = (t / h) % r, b = t / (h * (long)r);
    float inv = 1.f / (float)cnt[b * r + j];
    ok[t] = __float2half(ak[t] * inv); ov[t] = __float2half(av[t] * inv); oo[t] = __float2half(ao[t] * inv);
}

void launch_tome_merge(const void* cat_k, const void* cat_v, const void* cat_o,
                       void* dst_k, void* dst_v, void* dst_o,
                       int B, int N, int h, void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int r = N / 2;
    size_t accN = (size_t)B * r * h;
    int *match = nullptr, *cnt = nullptr;
    float *ak = nullptr, *av = nullptr, *ao = nullptr, *dstn = nullptr;
    cudaMallocAsync(&match, (size_t)B * r * sizeof(int), stream);
    cudaMallocAsync(&cnt, (size_t)B * r * sizeof(int), stream);
    cudaMallocAsync(&dstn, (size_t)B * r * sizeof(float), stream);
    cudaMallocAsync(&ak, accN * sizeof(float), stream);
    cudaMallocAsync(&av, accN * sizeof(float), stream);
    cudaMallocAsync(&ao, accN * sizeof(float), stream);

    int th = 256;
    tome_dstnorm_kernel<<<(B * r + th - 1) / th, th, 0, stream>>>((const __half*)cat_k, B, N, h, dstn);
    tome_match_kernel<<<(B * r + th - 1) / th, th, 0, stream>>>((const __half*)cat_k, B, N, h, dstn, match);
    long tot = (long)B * r * h;
    tome_init_kernel<<<(tot + th - 1) / th, th, 0, stream>>>(
        (const __half*)cat_k, (const __half*)cat_v, (const __half*)cat_o, B, N, h, ak, av, ao, cnt);
    tome_scatter_kernel<<<(B * r + th - 1) / th, th, 0, stream>>>(
        (const __half*)cat_k, (const __half*)cat_v, (const __half*)cat_o, match, B, N, h, ak, av, ao, cnt);
    tome_finalize_kernel<<<(tot + th - 1) / th, th, 0, stream>>>(
        ak, av, ao, cnt, B, N, h, (__half*)dst_k, (__half*)dst_v, (__half*)dst_o);

    cudaFreeAsync(match, stream); cudaFreeAsync(cnt, stream); cudaFreeAsync(dstn, stream);
    cudaFreeAsync(ak, stream); cudaFreeAsync(av, stream); cudaFreeAsync(ao, stream);
}

// Weighted accumulate functor: acc = acc + weight * src
struct WeightedAccumulateFunctor {
    float weight;

    __host__ __device__
    WeightedAccumulateFunctor(float w) : weight(w) {}

    __host__ __device__
    __half operator()(const thrust::tuple<__half, __half>& input) const {
        float acc = __half2float(thrust::get<0>(input));
        float src = __half2float(thrust::get<1>(input));
        return __float2half(acc + weight * src);
    }
};

void launch_weighted_accumulate_fp16(
    void* accumulator,
    const void* source,
    float weight,
    int N,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    thrust::device_ptr<__half> acc_ptr((__half*)accumulator);
    thrust::device_ptr<const __half> src_ptr((const __half*)source);

    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(acc_ptr, src_ptr));
    auto input_end = input_begin + N;

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), input_begin, input_end, acc_ptr,
        WeightedAccumulateFunctor(weight));
}

void launch_zero_fill_fp16(void* output, int N, void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    cudaMemsetAsync(output, 0, N * sizeof(__half), stream);
}

// RGBA to RGB normalization kernel (float32)
__global__ void rgba_to_rgb_normalized_fp32_kernel(
    const uint8_t* rgba_in,  // NHWC format [N, H, W, 4]
    float* rgb_out,          // NHWC format [N, H, W, 3]
    int n, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = n * h * w;

    if (idx < total_pixels) {
        int pixel_idx = idx;
        int rgba_idx = pixel_idx * 4;
        int rgb_idx = pixel_idx * 3;

        // Read RGBA values and convert to normalized RGB [-1, 1]
        float r = float(rgba_in[rgba_idx + 0]) / 127.5f - 1.0f;
        float g = float(rgba_in[rgba_idx + 1]) / 127.5f - 1.0f;
        float b = float(rgba_in[rgba_idx + 2]) / 127.5f - 1.0f;
        // Alpha channel (rgba_in[rgba_idx + 3]) is ignored

        rgb_out[rgb_idx + 0] = r;
        rgb_out[rgb_idx + 1] = g;
        rgb_out[rgb_idx + 2] = b;
    }
}

// RGBA to RGB normalization kernel (float16)
__global__ void rgba_to_rgb_normalized_fp16_kernel(
    const uint8_t* rgba_in,  // NHWC format [N, H, W, 4]
    __half* rgb_out,         // NHWC format [N, H, W, 3]
    int n, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = n * h * w;

    if (idx < total_pixels) {
        int pixel_idx = idx;
        int rgba_idx = pixel_idx * 4;
        int rgb_idx = pixel_idx * 3;

        // Read RGBA values and convert to normalized RGB [-1, 1]
        float r = float(rgba_in[rgba_idx + 0]) / 127.5f - 1.0f;
        float g = float(rgba_in[rgba_idx + 1]) / 127.5f - 1.0f;
        float b = float(rgba_in[rgba_idx + 2]) / 127.5f - 1.0f;
        // Alpha channel (rgba_in[rgba_idx + 3]) is ignored

        rgb_out[rgb_idx + 0] = __float2half(r);
        rgb_out[rgb_idx + 1] = __float2half(g);
        rgb_out[rgb_idx + 2] = __float2half(b);
    }
}

// RGB to RGBA denormalization kernel (float32)
__global__ void rgb_to_rgba_denormalized_fp32_kernel(
    const float* rgb_in,     // NHWC format [N, H, W, 3]
    uint8_t* rgba_out,       // NHWC format [N, H, W, 4]
    int n, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = n * h * w;

    if (idx < total_pixels) {
        int pixel_idx = idx;
        int rgb_idx = pixel_idx * 3;
        int rgba_idx = pixel_idx * 4;

        // Read RGB values and denormalize from [-1, 1] to [0, 255]
        float r = (rgb_in[rgb_idx + 0] * 0.5f + 0.5f) * 255.0f;
        float g = (rgb_in[rgb_idx + 1] * 0.5f + 0.5f) * 255.0f;
        float b = (rgb_in[rgb_idx + 2] * 0.5f + 0.5f) * 255.0f;

        // Clamp to [0, 255]
        r = fmaxf(0.0f, fminf(255.0f, r));
        g = fmaxf(0.0f, fminf(255.0f, g));
        b = fmaxf(0.0f, fminf(255.0f, b));

        rgba_out[rgba_idx + 0] = uint8_t(r);
        rgba_out[rgba_idx + 1] = uint8_t(g);
        rgba_out[rgba_idx + 2] = uint8_t(b);
        rgba_out[rgba_idx + 3] = 255; // Set alpha to fully opaque
    }
}

// RGB to RGBA denormalization kernel (float16)
__global__ void rgb_to_rgba_denormalized_fp16_kernel(
    const __half* rgb_in,    // NHWC format [N, H, W, 3]
    uint8_t* rgba_out,       // NHWC format [N, H, W, 4]
    int n, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = n * h * w;

    if (idx < total_pixels) {
        int pixel_idx = idx;
        int rgb_idx = pixel_idx * 3;
        int rgba_idx = pixel_idx * 4;

        // Read RGB values and denormalize from [-1, 1] to [0, 255]
        float r = (__half2float(rgb_in[rgb_idx + 0]) * 0.5f + 0.5f) * 255.0f;
        float g = (__half2float(rgb_in[rgb_idx + 1]) * 0.5f + 0.5f) * 255.0f;
        float b = (__half2float(rgb_in[rgb_idx + 2]) * 0.5f + 0.5f) * 255.0f;

        // Clamp to [0, 255]
        r = fmaxf(0.0f, fminf(255.0f, r));
        g = fmaxf(0.0f, fminf(255.0f, g));
        b = fmaxf(0.0f, fminf(255.0f, b));

        rgba_out[rgba_idx + 0] = uint8_t(r);
        rgba_out[rgba_idx + 1] = uint8_t(g);
        rgba_out[rgba_idx + 2] = uint8_t(b);
        rgba_out[rgba_idx + 3] = 255; // Set alpha to fully opaque
    }
}

// Launch functions
void launch_rgba_to_rgb_normalized_fp32(
    const void* rgba_in,
    void* rgb_out,
    int n, int h, int w,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_pixels = n * h * w;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    rgba_to_rgb_normalized_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
        (const uint8_t*)rgba_in, (float*)rgb_out, n, h, w);
}

void launch_rgba_to_rgb_normalized_fp16(
    const void* rgba_in,
    void* rgb_out,
    int n, int h, int w,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_pixels = n * h * w;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    rgba_to_rgb_normalized_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        (const uint8_t*)rgba_in, (__half*)rgb_out, n, h, w);
}

// ControlNet control image: RGBA uint8 NHWC -> RGB fp16 NCHW [N,3,H,W] in [0,1].
__global__ void rgba_to_rgb_chw_01_fp16_kernel(
    const uint8_t* rgba_in, __half* rgb_out, int n, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = n * h * w;
    if(idx < total_pixels)
    {
        int hw = h * w;
        int img = idx / hw;        // batch index
        int pix = idx % hw;        // pixel within image
        int rgba_idx = idx * 4;    // NHWC source
        // NCHW destination: channel plane offset = img*3*hw + c*hw + pix
        int base = img * 3 * hw + pix;
        rgb_out[base + 0 * hw] = __float2half(float(rgba_in[rgba_idx + 0]) / 255.0f);
        rgb_out[base + 1 * hw] = __float2half(float(rgba_in[rgba_idx + 1]) / 255.0f);
        rgb_out[base + 2 * hw] = __float2half(float(rgba_in[rgba_idx + 2]) / 255.0f);
    }
}

void launch_rgba_to_rgb_chw_01_fp16(
    const void* rgba_in, void* rgb_out, int n, int h, int w, void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_pixels = n * h * w;
    int block = 256, grid = (total_pixels + block - 1) / block;
    rgba_to_rgb_chw_01_fp16_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)rgba_in, (__half*)rgb_out, n, h, w);
}

// ---- IP-Adapter CLIP image preprocess: separable Lanczos-3 resize RGBA->224 + CLIP normalize ----
// Matches PIL LANCZOS (a=3) downscale + HF CLIPImageProcessor (rescale 1/255, mean/std). One thread
// per output (c, y, x) of the [1,3,224,224] result; computes both separable passes inline (cheap at
// 224x224). For a square input the shortest-edge-224 resize + center crop reduces to a direct resize.
#define IPCLIP_OUT 224
__device__ __forceinline__ float lanczos3(float x)
{
    // a = 3 windowed sinc. lanczos(0)=1.
    if(x < 0.0f) x = -x;
    if(x >= 3.0f) return 0.0f;
    if(x < 1e-7f) return 1.0f;
    const float pix = 3.14159265358979323846f * x;
    return 3.0f * sinf(pix) * sinf(pix / 3.0f) / (pix * pix);
}

__global__ void clip_image_preprocess_kernel(
    const uint8_t* __restrict__ rgba_in, __half* __restrict__ chw_out, int in_h, int in_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = IPCLIP_OUT * IPCLIP_OUT;
    if(idx >= 3 * HW) return;
    int c = idx / HW;
    int pix = idx % HW;
    int oy = pix / IPCLIP_OUT;
    int ox = pix % IPCLIP_OUT;

    // PIL-style separable resampling: per-axis scale = in/out, filterscale = max(scale,1).
    const float scale_x = (float)in_w / (float)IPCLIP_OUT;
    const float scale_y = (float)in_h / (float)IPCLIP_OUT;
    const float fsx = scale_x > 1.0f ? scale_x : 1.0f;
    const float fsy = scale_y > 1.0f ? scale_y : 1.0f;
    const float supx = 3.0f * fsx;
    const float supy = 3.0f * fsy;

    const float cx = (ox + 0.5f) * scale_x;
    const float cy = (oy + 0.5f) * scale_y;
    int lx = (int)floorf(cx - supx), rx = (int)ceilf(cx + supx);
    int ly = (int)floorf(cy - supy), ry = (int)ceilf(cy + supy);

    // Precompute and normalize the y weights into the accumulation, x weights inline.
    float wsum_y = 0.0f;
    for(int sy = ly; sy < ry; ++sy)
        wsum_y += lanczos3((sy + 0.5f - cy) / fsy);
    float wsum_x = 0.0f;
    for(int sx = lx; sx < rx; ++sx)
        wsum_x += lanczos3((sx + 0.5f - cx) / fsx);
    if(wsum_y == 0.0f) wsum_y = 1.0f;
    if(wsum_x == 0.0f) wsum_x = 1.0f;

    float acc = 0.0f;
    for(int sy = ly; sy < ry; ++sy)
    {
        float wy = lanczos3((sy + 0.5f - cy) / fsy) / wsum_y;
        int cyi = sy < 0 ? 0 : (sy >= in_h ? in_h - 1 : sy);
        float row = 0.0f;
        for(int sx = lx; sx < rx; ++sx)
        {
            float wx = lanczos3((sx + 0.5f - cx) / fsx) / wsum_x;
            int cxi = sx < 0 ? 0 : (sx >= in_w ? in_w - 1 : sx);
            uint8_t v = rgba_in[(cyi * in_w + cxi) * 4 + c];
            row += wx * (float)v;
        }
        acc += wy * row;
    }
    // acc is in [0,255]; rescale + CLIP normalize.
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float istd[3] = {1.0f / 0.26862954f, 1.0f / 0.26130258f, 1.0f / 0.27577711f};
    float val = (acc / 255.0f - mean[c]) * istd[c];
    chw_out[idx] = __float2half(val);
}

void launch_clip_image_preprocess_fp16(
    const void* rgba_in, void* chw_out, int in_h, int in_w, void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total = 3 * IPCLIP_OUT * IPCLIP_OUT;
    int block = 256, grid = (total + block - 1) / block;
    clip_image_preprocess_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)rgba_in, (__half*)chw_out, in_h, in_w);
}
#undef IPCLIP_OUT

// Multi-ControlNet residual sum: acc[i] += src[i].
__global__ void tensor_add_fp16_kernel(__half* acc, const __half* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        acc[i] = __float2half(__half2float(acc[i]) + __half2float(src[i]));
}

void launch_tensor_add_fp16(void* acc, const void* src, int n, void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int block = 256, grid = (n + block - 1) / block;
    tensor_add_fp16_kernel<<<grid, block, 0, stream>>>((__half*)acc, (const __half*)src, n);
}

void launch_rgb_to_rgba_denormalized_fp32(
    const void* rgb_in,
    void* rgba_out,
    int n, int h, int w,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_pixels = n * h * w;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    rgb_to_rgba_denormalized_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
        (const float*)rgb_in, (uint8_t*)rgba_out, n, h, w);
}

void launch_rgb_to_rgba_denormalized_fp16(
    const void* rgb_in,
    void* rgba_out,
    int n, int h, int w,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total_pixels = n * h * w;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    rgb_to_rgba_denormalized_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        (const __half*)rgb_in, (uint8_t*)rgba_out, n, h, w);
}
