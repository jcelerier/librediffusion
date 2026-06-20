/*
 * Proof of Concept: Pure CUDA Kernel for Scheduler Step
 *
 * This demonstrates replacing PyTorch operations with raw CUDA.
 * Compile: nvcc -O3 -arch=sm_86 --compiler-options '-fPIC' -shared poc_cuda_kernel.cu -o libpoc_cuda.so
 * Test: python test_poc_cuda.py
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

//==============================================================================
// Kernel: Fused Scheduler Step (Replaces lines 438-441 from pipeline.py)
//==============================================================================

__global__ void scheduler_step_fused_fp16(
    const __half* __restrict__ model_pred,
    const __half* __restrict__ x_t_latent,
    __half* __restrict__ output,
    const __half alpha_prod_t_sqrt,
    const __half beta_prod_t_sqrt,
    const __half c_skip,
    const __half c_out,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        __half x_t = x_t_latent[idx];
        __half pred = model_pred[idx];

        // F_theta = (x_t - beta * model_pred) / alpha
        __half F_theta = __hdiv(
            __hsub(x_t, __hmul(beta_prod_t_sqrt, pred)),
            alpha_prod_t_sqrt
        );

        // output = c_out * F_theta + c_skip * x_t
        output[idx] = __hadd(
            __hmul(c_out, F_theta),
            __hmul(c_skip, x_t)
        );
    }
}

// FP32 version for comparison
__global__ void scheduler_step_fused_fp32(
    const float* __restrict__ model_pred,
    const float* __restrict__ x_t_latent,
    float* __restrict__ output,
    const float alpha_prod_t_sqrt,
    const float beta_prod_t_sqrt,
    const float c_skip,
    const float c_out,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x_t = x_t_latent[idx];
        float pred = model_pred[idx];

        // F_theta = (x_t - beta * model_pred) / alpha
        float F_theta = (x_t - beta_prod_t_sqrt * pred) / alpha_prod_t_sqrt;

        // output = c_out * F_theta + c_skip * x_t
        output[idx] = c_out * F_theta + c_skip * x_t;
    }
}

//==============================================================================
// Kernel: Add Noise (Replaces lines 424-428 from pipeline.py)
//==============================================================================

__global__ void add_noise_fp16(
    const __half* __restrict__ original_samples,
    const __half* __restrict__ noise,
    __half* __restrict__ noisy_samples,
    const __half alpha_prod_t_sqrt,
    const __half beta_prod_t_sqrt,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // noisy = alpha * original + beta * noise
        noisy_samples[idx] = __hadd(
            __hmul(alpha_prod_t_sqrt, original_samples[idx]),
            __hmul(beta_prod_t_sqrt, noise[idx])
        );
    }
}

//==============================================================================
// Kernel: Classifier-Free Guidance (Replaces lines 490-491 from pipeline.py)
//==============================================================================

__global__ void apply_cfg_fp16(
    const __half* __restrict__ noise_pred_uncond,
    const __half* __restrict__ noise_pred_text,
    __half* __restrict__ output,
    const __half guidance_scale,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // output = uncond + guidance_scale * (text - uncond)
        __half uncond = noise_pred_uncond[idx];
        __half text = noise_pred_text[idx];
        __half delta = __hsub(text, uncond);

        output[idx] = __hadd(
            uncond,
            __hmul(guidance_scale, delta)
        );
    }
}

//==============================================================================
// In-place scalar operations
//==============================================================================

// In-place scalar multiplication (FP16)
__global__ void scalar_mul_inplace_fp16(
    __half* __restrict__ data,
    const __half scalar,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = __hmul(data[idx], scalar);
    }
}

// In-place scalar division (FP16)
__global__ void scalar_div_inplace_fp16(
    __half* __restrict__ data,
    const __half scalar,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = __hdiv(data[idx], scalar);
    }
}

// Non-in-place scalar division (FP16)
__global__ void scalar_div_fp16(
    const __half* __restrict__ data,
    __half* __restrict__ output,
    const __half scalar,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __hdiv(data[idx], scalar);
    }
}

// Element-wise subtraction (FP16)
__global__ void tensor_sub_fp16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ output,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __hsub(a[idx], b[idx]);
    }
}

//==============================================================================
// Kernel: Random Normal Generation (FP16)
//==============================================================================

// Kernel to generate random normal values (mean=0, std=1) using curand
__global__ void randn_kernel_fp16(
    __half* __restrict__ output,
    unsigned long long seed,
    unsigned long long offset,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Initialize curand state for this thread
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, offset, &state);

        // Generate random normal value (returns float)
        float val = curand_normal(&state);

        // Convert to half precision
        output[idx] = __float2half(val);
    }
}

//==============================================================================
// C API for Python binding
//==============================================================================

extern "C" {

// Launch scheduler step (FP16)
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
    int N = batch * channels * height * width;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    scheduler_step_fused_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)model_pred,
        (const __half*)x_t_latent,
        (__half*)output,
        __float2half(alpha),
        __float2half(beta),
        __float2half(c_skip),
        __float2half(c_out),
        N
    );
}

// Launch scheduler step (FP32)
void launch_scheduler_step_fp32(
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
    int N = batch * channels * height * width;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    scheduler_step_fused_fp32<<<blocks, threads, 0, stream>>>(
        (const float*)model_pred,
        (const float*)x_t_latent,
        (float*)output,
        alpha,
        beta,
        c_skip,
        c_out,
        N
    );
}

// Launch add noise
void launch_add_noise_fp16(
    const void* original_samples,
    const void* noise,
    void* noisy_samples,
    float alpha,
    float beta,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    add_noise_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)original_samples,
        (const __half*)noise,
        (__half*)noisy_samples,
        __float2half(alpha),
        __float2half(beta),
        N
    );
}

// Launch CFG
void launch_apply_cfg_fp16(
    const void* noise_pred_uncond,
    const void* noise_pred_text,
    void* output,
    float guidance_scale,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    apply_cfg_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)noise_pred_uncond,
        (const __half*)noise_pred_text,
        (__half*)output,
        __float2half(guidance_scale),
        N
    );
}

// Launch in-place scalar multiplication
void launch_scalar_mul_inplace_fp16(
    void* data,
    float scalar,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    scalar_mul_inplace_fp16<<<blocks, threads, 0, stream>>>(
        (__half*)data,
        __float2half(scalar),
        N
    );
}

// Launch in-place scalar division
void launch_scalar_div_inplace_fp16(
    void* data,
    float scalar,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    scalar_div_inplace_fp16<<<blocks, threads, 0, stream>>>(
        (__half*)data,
        __float2half(scalar),
        N
    );
}

// Launch non-in-place scalar division
void launch_scalar_div_fp16(
    const void* data,
    void* output,
    float scalar,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    scalar_div_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)data,
        (__half*)output,
        __float2half(scalar),
        N
    );
}

// Launch tensor subtraction
void launch_tensor_sub_fp16(
    const void* a,
    const void* b,
    void* output,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    tensor_sub_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)a,
        (const __half*)b,
        (__half*)output,
        N
    );
}

// Clone tensor (device-to-device memory copy)
void launch_tensor_clone(
    const void* src,
    void* dst,
    size_t num_bytes,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    if (stream == 0) {
        // Use synchronous copy for default stream
        cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToDevice);
    } else {
        // Use async copy for non-default stream
        cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream);
    }
}

// Launch random normal generation
void launch_randn_fp16(
    void* output,
    unsigned long long seed,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Use a unique offset for each call to ensure different random sequences
    static unsigned long long call_counter = 0;
    unsigned long long offset = call_counter++;

    randn_kernel_fp16<<<blocks, threads, 0, stream>>>(
        (__half*)output,
        seed,
        offset,
        N
    );
}

// ============================================================================
// Concat operation (concatenate tensors along dimension 0)
// ============================================================================

// Optimized: Use synchronous copy for small tensors to avoid async overhead
// For tensors < 1MB, the cudaMemcpyAsync launch overhead dominates actual copy time
void launch_concat(
    void** input_ptrs,
    int num_inputs,
    size_t* input_byte_sizes,  // Size in bytes for each input
    void* output,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Calculate total size
    size_t total_size = 0;
    for (int i = 0; i < num_inputs; i++) {
        total_size += input_byte_sizes[i];
    }

    // For small tensors, use synchronous copy to avoid async launch overhead
    // Threshold: 256KB (empirically determined - launch overhead ~5-10μs dominates below this)
    const size_t SYNC_THRESHOLD = 256 * 1024;
    bool use_sync = (total_size < SYNC_THRESHOLD);

    // Calculate offsets and copy each input to output
    size_t offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        if (use_sync) {
            // Synchronous copy for small tensors - avoids launch overhead
            cudaMemcpy(
                (char*)output + offset,
                input_ptrs[i],
                input_byte_sizes[i],
                cudaMemcpyDeviceToDevice
            );
        } else {
            // Async copy for large tensors - allows overlap
            cudaMemcpyAsync(
                (char*)output + offset,
                input_ptrs[i],
                input_byte_sizes[i],
                cudaMemcpyDeviceToDevice,
                stream
            );
        }
        offset += input_byte_sizes[i];
    }
}

// Legacy FP16 version for backwards compatibility
void launch_concat_fp16(
    void** input_ptrs,
    int num_inputs,
    int* input_sizes,  // Number of elements for each input
    void* output,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Calculate offsets and copy each input to output
    size_t offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        size_t bytes = input_sizes[i] * sizeof(__half);
        cudaMemcpyAsync(
            (char*)output + offset,
            input_ptrs[i],
            bytes,
            cudaMemcpyDeviceToDevice,
            stream
        );
        offset += bytes;
    }
}

// ============================================================================
// ones_like operation (fill tensor with 1.0)
// ============================================================================

__global__ void ones_like_kernel_fp16(
    __half* __restrict__ output,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = __float2half(1.0f);
    }
}

void launch_ones_like_fp16(
    void* output,
    int N,
    void* stream_ptr
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    ones_like_kernel_fp16<<<blocks, threads, 0, stream>>>(
        (__half*)output,
        N
    );
}

// ============================================================================
// randn_like operation (fill tensor with random normal values, same shape as input)
// ============================================================================

void launch_randn_like_fp16(
    void* output,
    unsigned long long seed,
    int N,
    void* stream_ptr
) {
    // This is just randn with the shape from the output tensor
    launch_randn_fp16(output, seed, N, stream_ptr);
}

// Get CUDA error string
const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}

} // extern "C"
