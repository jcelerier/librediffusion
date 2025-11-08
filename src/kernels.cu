
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

void launch_randn_fp16(
    void* output,
    unsigned long long seed,
    int N,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    thread_local curandGenerator_t gen = nullptr;
    thread_local bool gen_initialized = false;

    if (!gen_initialized) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        gen_initialized = true;
    }

    // Set seed and stream
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetStream(gen, stream);

    // Allocate temporary FP32 buffer (cuRAND doesn't support FP16 directly)
    float* temp_fp32;
    cudaMallocAsync(&temp_fp32, N * sizeof(float), stream);

    // Generate normal distribution directly to FP32
    curandGenerateNormal(gen, temp_fp32, N, 0.0f, 1.0f);

    // Convert FP32 -> FP16 using Thrust
    thrust::device_ptr<float> fp32_ptr(temp_fp32);
    thrust::device_ptr<__half> fp16_ptr((__half*)output);

    thrust::transform(
        thrust::cuda::par_nosync.on(stream), fp32_ptr, fp32_ptr + N, fp16_ptr,
        Float2HalfFunctor());

    // Cleanup
    cudaFreeAsync(temp_fp32, stream);
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
