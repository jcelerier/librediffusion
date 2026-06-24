/**
 * TensorRT-RTX Model Wrappers Implementation
 */

#include "tensorrt_wrappers.hpp"

#include "clip_tokenizer_c.h"
#include "kernels.hpp"
#include "model_cache.hpp"
#include "librediffusion.hpp" // For CUDATensor

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace librediffusion
{

// ============================================================================
// TensorRTLogger Implementation
// ============================================================================

std::string TensorRTLogger::severityToString(Severity severity)
{
  switch(severity)
  {
    case Severity::kVERBOSE:
      return "VERBOSE";
    case Severity::kINFO:
      return "INFO";
    case Severity::kWARNING:
      return "WARNING";
    case Severity::kERROR:
      return "ERROR";
    case Severity::kINTERNAL_ERROR:
      return "INTERNAL_ERROR";
    default:
      return "UNKNOWN";
  }
}

void TensorRTLogger::log(Severity severity, const char* msg) noexcept
{
  // Only log warnings and errors to avoid spam
  if(severity < Severity::kWARNING)
  {
    std::cout << "[TensorRT-RTX " << severityToString(severity) << "] " << msg
              << std::endl;
  }
}

// ============================================================================
// UNetWrapper Implementation
// ============================================================================

UNetWrapper::UNetWrapper(const std::string& engine_path, bool use_v2v)
    : use_v2v_(use_v2v)
{
  loadEngine(engine_path);

  // Initialize attention buffers for V2V mode
  if(use_v2v_)
  {
    if(has_v2v_kvo_)
      std::cout << "Note: TEMPORAL_V2V using the live kvo_cache extended-attention path (forward_v2v)."
                << std::endl;
    else if(!has_v2v_outputs_)
      std::cerr << "Warning: TEMPORAL_V2V requested but engine '" << engine_path
                << "' is neither a kvo (kvo_cache_in_*) nor a legacy attention_* StreamV2V UNet; "
                   "temporal coherence is DISABLED (plain img2img is used). Export a v2v UNet to enable it."
                << std::endl;
    attention_output_buffers_.reserve(NUM_ATTENTION_OUTPUTS);
    // Buffers will be allocated on first forward pass when we know the sizes
  }
}

UNetWrapper::~UNetWrapper()
{
  // Unique_ptrs will handle cleanup automatically
  if(ipadapter_scale_pinned_)
    cudaFreeHost(ipadapter_scale_pinned_);
  if(lora_scale_pinned_)
    cudaFreeHost(lora_scale_pinned_);
}

void UNetWrapper::loadEngine(const std::string& engine_path)
{
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      cached_engine_->createExecutionContext());

  if(!context_)
  {
    throw std::runtime_error("Failed to create execution context");
  }

  // Detect a statically-shaped `timestep` input. The prebuilt SDXL UNet ONNX from
  // stabilityai/sdxl-turbo-tensorrt declares timestep as a fixed [1] (its optimization
  // profile is min=opt=max=(1,)), whereas SD1.5/SD-Turbo and our from-scratch SDXL trace
  // use a dynamic batch-sized timestep. When the engine fixes the extent we must bind that
  // exact size; binding [batch] (batch != 1) against a [1] engine fails shape validation.
  nvinfer1::ICudaEngine* eng = cached_engine_->getEngine();
  if(eng)
  {
    const char* kTimestep = "timestep";
    nvinfer1::Dims ts = eng->getTensorShape(kTimestep);
    if(ts.nbDims == 1 && ts.d[0] > 0)
    {
      // A positive (non -1) declared dim means a fixed extent baked into the engine.
      timestep_fixed_extent_ = static_cast<int>(ts.d[0]);
      std::cout << "Note: engine declares a fixed timestep extent of "
                << timestep_fixed_extent_ << " for " << engine_path << std::endl;
    }

    // Detect a control-aware UNet engine: it declares input_control_00 (+ ..NN + _middle). When
    // present, forward_controlnet() binds the ControlNet residuals; plain forward() leaves them unset
    // (which would fail allInputDimensionsSpecified — callers must use forward_controlnet for these).
    //
    // CRUCIAL: the residual COUNT and per-index geometry are read from the engine, NOT hardcoded to
    // the SD1.5 12-residual table. SD1.5 declares 12 input_control_NN (dynamic spatial), the pruned
    // SDXS UNet declares 6 (static spatial). We enumerate them here so forward_controlnet() iterates
    // 0..count-1 with the engine's actual channels and the correct downsample factor.
    for(int i = 0; i < eng->getNbIOTensors(); i++)
    {
      const char* tn = eng->getIOTensorName(i);
      if(!tn) continue;
      std::string nm = tn;
      if(nm.rfind("input_control_", 0) == 0 && nm != "input_control_middle"
         && eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kINPUT)
        num_control_down_++;
      if(nm == "input_control_00")
      {
        has_control_inputs_ = true;
        std::cout << "Note: engine is control-aware (input_control_* inputs) for " << engine_path
                  << std::endl;
      }
      else if(nm == "attention_0" && eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kOUTPUT)
      {
        // StreamV2V-capable UNet: it exposes the per-block attention OUTPUTS (attention_0..N) that the
        // temporal feature bank reads. Only engines exported via the tensorrt_orig UNetV2V path have
        // these. Standard SD1.5/SDXL/turbo engines do NOT -> forward() must NOT try to bind them
        // (getTensorShape/setTensorAddress on a missing tensor throws). Gate the v2v binding on this.
        has_v2v_outputs_ = true;
        std::cout << "Note: engine is StreamV2V-capable (attention_* outputs) for " << engine_path
                  << std::endl;
      }
      else if(nm.rfind("kvo_cache_in_", 0) == 0
              && eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kINPUT)
      {
        // LIVE StreamV2V UNet: declares kvo_cache_in_0..15 (rolling K/V bank) + kvo_cache_out_*.
        // forward_v2v binds the bank; plain forward() must NOT (allInputDimensionsSpecified would fail).
        num_kvo_layers_++;
        if(nm == "kvo_cache_in_0")
          has_v2v_kvo_ = true;
      }
      else if(nm == "v2v_inject_params"
              && eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kINPUT)
      {
        // StreamV2V dynamic feature injection: [fi_strength, threshold] fp32 [2] bound at runtime.
        has_v2v_inject_params_ = true;
      }
      else if(nm == "ipadapter_scale" && eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kINPUT)
      {
        has_ipadapter_ = true;
        // The scale vector length = num IP layers (fixed per engine: SD1.5 = 16, SDXL = 70). A static
        // engine exposes it directly; a DYNAMIC ipadapter_scale (-1) must be read from the optimization
        // profile MAX (the SDXL IP engine is built dynamic [1..70] -> 70). Previously the dynamic case
        // silently kept the default (16), so SDXL IP under-fed 16 of 70 scales -> OOB read of the other
        // 54 layers' scales -> wrong conditioning.
        nvinfer1::Dims d = eng->getTensorShape(tn);
        if(d.nbDims == 1 && d.d[0] > 0)
          num_ip_layers_ = (int)d.d[0];
        else if(d.nbDims == 1)
        {
          nvinfer1::Dims dmax = eng->getProfileShape(tn, 0, nvinfer1::OptProfileSelector::kMAX);
          if(dmax.nbDims == 1 && dmax.d[0] > 0)
            num_ip_layers_ = (int)dmax.d[0];
        }
        std::cout << "Note: engine is IP-Adapter (ipadapter_scale input, "
                  << (num_ip_layers_ ? std::to_string(num_ip_layers_) : std::string("dynamic"))
                  << " layers) for " << engine_path << std::endl;
      }
      else if(nm == "lora_scale" && eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kINPUT)
      {
        // Runtime LoRA: lora_scale[N] HALF vector (N = #runtime LoRAs). Static engine declares N>0;
        // a dynamic build reads N from the profile MAX (mirrors ipadapter_scale).
        has_lora_scale_ = true;
        nvinfer1::Dims d = eng->getTensorShape(tn);
        if(d.nbDims == 1 && d.d[0] > 0)
          num_runtime_loras_ = (int)d.d[0];
        else if(d.nbDims == 1)
        {
          nvinfer1::Dims dmax = eng->getProfileShape(tn, 0, nvinfer1::OptProfileSelector::kMAX);
          if(dmax.nbDims == 1 && dmax.d[0] > 0)
            num_runtime_loras_ = (int)dmax.d[0];
        }
        if(num_runtime_loras_ > 0)
          lora_scale_host_.assign(num_runtime_loras_, 1.0f);  // default: every runtime LoRA fully on
        std::cout << "Note: engine has runtime LoRA (lora_scale input, "
                  << num_runtime_loras_ << " slot(s)) for " << engine_path << std::endl;
      }
    }

    // Derive the per-residual ControlNet geometry from the engine's input_control_NN bindings, so the
    // wrapper is model-agnostic (SD1.5 = 12 down, pruned SDXS = 6 down, ...). For each input_control_NN
    // we read the channel count, and the spatial downsample factor relative to the base latent.
    //  - The channel dim (d[1]) is fixed on every control-aware engine -> read directly.
    //  - The factor = base_latent_dim / control_dim. On a STATIC engine (e.g. SDXS) the control spatial
    //    dim and `sample`'s spatial dim are both declared (>0), so factor = sampleH / controlH exactly.
    //    On a DYNAMIC-spatial engine (SD1.5/SDXL trace, declares -1) we fall back to the canonical
    //    diffusers factor table selected by the residual count.
    if(has_control_inputs_ && num_control_down_ > 0)
    {
      control_down_ch_.assign(num_control_down_, 0);
      control_down_fac_.assign(num_control_down_, 1);

      // sample's spatial extent (>0 only on a fully-static engine) lets us recover factors directly.
      nvinfer1::Dims sd = eng->getTensorShape("sample");
      const int sampleH = (sd.nbDims >= 3 && sd.d[2] > 0) ? (int)sd.d[2] : -1;

      bool static_factors_ok = (sampleH > 0);
      for(int i = 0; i < num_control_down_; i++)
      {
        char name[24];
        std::snprintf(name, sizeof(name), "input_control_%02d", i);
        nvinfer1::Dims d = eng->getTensorShape(name);
        control_down_ch_[i] = (d.nbDims >= 2 && d.d[1] > 0) ? (int)d.d[1] : 0;
        if(static_factors_ok && d.nbDims >= 3 && d.d[2] > 0)
          control_down_fac_[i] = sampleH / (int)d.d[2];
        else
          static_factors_ok = false;  // any dynamic dim -> use the table fallback below
      }
      {
        nvinfer1::Dims dm = eng->getTensorShape("input_control_middle");
        control_mid_ch_ = (dm.nbDims >= 2 && dm.d[1] > 0) ? (int)dm.d[1] : 1280;
        if(static_factors_ok && dm.nbDims >= 3 && dm.d[2] > 0)
          control_mid_fac_ = sampleH / (int)dm.d[2];
      }

      if(!static_factors_ok)
      {
        // Dynamic-spatial engine: spatial dims are -1, so they can't be read here. Use the canonical
        // diffusers factor sequence keyed by the residual count (preserves the proven SD1.5/SDXL path).
        static const int kFac12[12] = {1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8};  // SD1.5 (12 down)
        static const int kFac9[9] = {1, 1, 1, 2, 2, 2, 4, 4, 4};             // SDXL trace (9 down)
        static const int kFac6[6] = {1, 1, 2, 2, 4, 4};                      // SDXS (6 down)
        const int* fac = (num_control_down_ == 9) ? kFac9
                         : (num_control_down_ == 6) ? kFac6
                                                    : kFac12;
        for(int i = 0; i < num_control_down_ && i < 12; i++)
          control_down_fac_[i] = fac[i];
        control_mid_fac_ = control_down_fac_[num_control_down_ - 1];  // mid = deepest down factor
      }

      std::cout << "Note: control-aware UNet geometry = " << num_control_down_
                << " down residuals, factors {";
      for(int i = 0; i < num_control_down_; i++)
        std::cout << control_down_fac_[i] << (i + 1 < num_control_down_ ? "," : "");
      std::cout << "}, mid factor " << control_mid_fac_ << " (" << engine_path << ")" << std::endl;
    }

    // StreamV2V kvo: enumerate kvo_cache_in_0.. and read each layer's baked (seq, hidden) =
    // [2, maxframes, batch, seq, hidden]. maxframes/batch are dynamic; seq/hidden are static per layer.
    if(has_v2v_kvo_)
    {
      // num_kvo_layers_ was counted in the IO loop above, so iterate exactly the existing bindings
      // (probing kvo_cache_in_<N> past the end would log a TRT "invalid tensor name" error).
      for(int i = 0; i < num_kvo_layers_; i++)
      {
        char nm[24];
        std::snprintf(nm, sizeof(nm), "kvo_cache_in_%d", i);
        nvinfer1::Dims d = eng->getTensorShape(nm);
        kvo_seq_.push_back((d.nbDims == 5) ? (int)d.d[3] : 0);
        kvo_hidden_.push_back((d.nbDims == 5) ? (int)d.d[4] : 0);
        if(i == 0 && d.nbDims == 5 && d.d[0] > 0)
          kvo_components_ = (int)d.d[0];  // 2 = [K,V] extended-attn; 3 = [K,V,O] feature injection
      }
      nvinfer1::Dims dmax
          = eng->getProfileShape("kvo_cache_in_0", 0, nvinfer1::OptProfileSelector::kMAX);
      if(dmax.nbDims == 5)
      {
        kvo_profile_max_frames_ = (int)dmax.d[1];
        kvo_profile_max_batch_ = (int)dmax.d[2];
      }
      std::cout << "Note: engine is StreamV2V kvo (" << num_kvo_layers_ << " layers, "
                << kvo_components_ << " components " << (kvo_components_ >= 3 ? "[K,V,O]+inject" : "[K,V]")
                << ", max frames " << kvo_profile_max_frames_ << ", max batch " << kvo_profile_max_batch_
                << ") for " << engine_path << std::endl;
    }
  }
}

bool UNetWrapper::needsReallocation(
    int batch, int height, int width, int seq_len, int hidden_dim, int pooled_dim)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch
         || shape_cache_.height != height || shape_cache_.width != width
         || shape_cache_.seq_len != seq_len || shape_cache_.hidden_dim != hidden_dim
         || shape_cache_.pooled_dim != pooled_dim;
}

void UNetWrapper::setLoraScale(int idx, float v)
{
  if(idx < 0 || idx >= num_runtime_loras_)
    return;
  if((int)lora_scale_host_.size() < num_runtime_loras_)
    lora_scale_host_.assign(num_runtime_loras_, 1.0f);
  if(lora_scale_host_[idx] != v)
  {
    lora_scale_host_[idx] = v;
    lora_scale_dirty_ = true;
  }
}

const void* UNetWrapper::loraScaleBufferAddr() const
{
  return lora_scale_buffer_ ? (const void*)lora_scale_buffer_->data() : nullptr;
}

// Bind the runtime lora_scale[N] input (called from forward()/forward_sdxl() before enqueueV3 when the
// engine declares it). Grow-only device buffer (stable address for CUDA-graph) + PINNED host staging,
// change-gated H2D (a steady-state graph replay does no copy). The engine's lora_scale is HALF.
void UNetWrapper::bindLoraScale(cudaStream_t stream)
{
  if(!has_lora_scale_ || num_runtime_loras_ <= 0)
    return;
  const int N = num_runtime_loras_;
  if((int)lora_scale_host_.size() < N)
    lora_scale_host_.resize(N, 1.0f);
  if(!lora_scale_buffer_ || (int)lora_scale_buffer_->size() < N)
  {
    lora_scale_buffer_ = std::make_unique<CUDATensor<__half>>(N);
    lora_scale_dirty_ = true;
  }
  if(lora_scale_pinned_cap_ < N)
  {
    if(lora_scale_pinned_)
      cudaFreeHost(lora_scale_pinned_);
    cudaMallocHost((void**)&lora_scale_pinned_, N * sizeof(__half));
    lora_scale_pinned_cap_ = N;
    lora_scale_dirty_ = true;
  }
  if(lora_scale_dirty_)
  {
    for(int i = 0; i < N; i++)
      lora_scale_pinned_[i] = __float2half(lora_scale_host_[i]);
    cudaMemcpyAsync(lora_scale_buffer_->data(), lora_scale_pinned_, N * sizeof(__half),
                    cudaMemcpyHostToDevice, stream);
    lora_scale_dirty_ = false;
  }
  nvinfer1::Dims d;
  d.nbDims = 1;
  d.d[0] = N;
  context_->setInputShape("lora_scale", d);
  context_->setTensorAddress("lora_scale", lora_scale_buffer_->data());
}

void UNetWrapper::forward(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    cudaStream_t stream)
{
  // Calculate buffer sizes
  int sample_size = batch * 4 * height * width;
  int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  int timestep_size = timestep_extent;
  int encoder_size = batch * seq_len * hidden_dim;
  int output_size = batch * 4 * height * width;

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim);

  // Allocate persistent buffers on first call or when shapes change (matches Python)
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_size);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);

    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.initialized = true;
  }

  // Copy input data into persistent buffers
  cudaMemcpyAsync(
      sample_buffer_->data(), sample, sample_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      timestep_buffer_->data(), timestep, timestep_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      encoder_hidden_states_buffer_->data(), encoder_hidden_states,
      encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

  // CRITICAL: Synchronize stream after copies to ensure TensorRT reads complete data
  // Even though Python doesn't explicitly sync, tensor.copy_() likely handles this internally
  // Without this, TensorRT may read stale/incomplete data from persistent buffers
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  // CRITICAL: Set input shapes BEFORE setting tensor addresses (matches Python's exact order)
  // Python calls set_input_shape() in infer(), right before set_tensor_address()
  // This happens even if shapes haven't changed (though Python caches to skip redundant calls)
  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims;
  timestep_dims.nbDims = 1;
  timestep_dims.d[0] = timestep_extent;  // [batch] for SD; honors a fixed engine extent if present
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};

  if(!context_->setInputShape("sample", sample_dims))
  {
    throw std::runtime_error("Failed to set sample shape");
  }
  if(!context_->setInputShape("timestep", timestep_dims))
  {
    throw std::runtime_error("Failed to set timestep shape");
  }
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  }

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses using persistent buffer addresses (stays constant across calls)
  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
  {
    throw std::runtime_error("Failed to set sample tensor address");
  }
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
  {
    throw std::runtime_error("Failed to set timestep tensor address");
  }
  if(!context_->setTensorAddress(
         "encoder_hidden_states", encoder_hidden_states_buffer_->data()))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  }
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // StreamV2V: Set attention output tensor addresses. Gated on has_v2v_outputs_ — a standard engine
  // (no attention_* outputs) would throw here (getTensorShape/setTensorAddress on a missing tensor).
  if(use_v2v_ && has_v2v_outputs_)
  {
    // Allocate attention buffers on first call
    if(attention_output_buffers_.empty())
    {
      int seq_len_spatial = height * width;  // Spatial sequence length
      for(int i = 0; i < NUM_ATTENTION_OUTPUTS; i++)
      {
        // Get the actual shape from the engine for this attention output
        std::string attn_name = "attention_" + std::to_string(i);
        nvinfer1::Dims attn_dims = context_->getTensorShape(attn_name.c_str());

        // attn_dims should be [batch, seq_len, hidden_dim]
        int attn_hidden_dim = attn_dims.d[2];  // Hidden dimension varies per layer
        int attn_size = batch * seq_len_spatial * attn_hidden_dim;

        attention_output_buffers_.push_back(
            std::make_unique<CUDATensor<__half>>(attn_size));
      }
    }

    // Set tensor addresses for attention outputs
    for(int i = 0; i < NUM_ATTENTION_OUTPUTS; i++)
    {
      std::string attn_name = "attention_" + std::to_string(i);
      if(!context_->setTensorAddress(attn_name.c_str(), attention_output_buffers_[i]->data()))
      {
        throw std::runtime_error("Failed to set " + attn_name + " tensor address");
      }
    }
  }

  bindLoraScale(stream);  // runtime-LoRA engines: bind lora_scale[N] (no-op otherwise)

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }

  // Copy output from persistent buffer back to caller's buffer
  cudaMemcpyAsync(
      output, output_buffer_->data(), output_size * sizeof(__half),
      cudaMemcpyDeviceToDevice, stream);

  // TODO: StreamV2V attention outputs are now in attention_output_buffers_
  // These will be used for feature injection in LibreDiffusionPipeline
}

void UNetWrapper::forward_v2v(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    int cache_maxframes, bool use_tome, float fi_strength, float fi_threshold,
    cudaStream_t stream)
{
  if(!has_v2v_kvo_)
    throw std::runtime_error("forward_v2v called on an engine without kvo_cache_in_* inputs");
  // ToMe (token-merging bank compaction) needs the 3-component [K,V,O] cache (the inject engine) and a
  // single compacted bank slot (maxframes=1): each frame merges concat(bank, new) -> bank.
  use_tome = use_tome && (kvo_components_ >= 3);
  if(use_tome)
    cache_maxframes = 1;
  if(cache_maxframes < 1)
    cache_maxframes = 1;
  if(kvo_profile_max_frames_ > 0 && cache_maxframes > kvo_profile_max_frames_)
    cache_maxframes = kvo_profile_max_frames_;

  const int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  const int sample_size = batch * 4 * height * width;
  const int encoder_size = batch * seq_len * hidden_dim;
  const int output_size = batch * 4 * height * width;

  // Reuse the shared sample/timestep/encoder/output buffers (same shapes as forward()).
  if(needsReallocation(batch, height, width, seq_len, hidden_dim))
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_extent);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.initialized = true;
  }

  // (Re)allocate the rolling bank + per-frame out buffers when (batch, maxframes) change. The bank is
  // zeroed so frame 0 sees an empty temporal context (== plain attention, matching create_kvo_cache).
  if((int)kvo_bank_.size() != num_kvo_layers_ || bank_batch_ != batch
     || bank_maxframes_ != cache_maxframes || bank_tome_ != use_tome)
  {
    kvo_bank_.clear();
    kvo_out_.clear();
    tome_cat_k_.clear();
    tome_cat_v_.clear();
    tome_cat_o_.clear();
    kvo_bank_.reserve(num_kvo_layers_);
    kvo_out_.reserve(num_kvo_layers_);
    for(int i = 0; i < num_kvo_layers_; i++)
    {
      const size_t slot = (size_t)batch * kvo_seq_[i] * kvo_hidden_[i];  // one component's frame slot
      auto bank = std::make_unique<CUDATensor<__half>>((size_t)kvo_components_ * cache_maxframes * slot);
      cudaMemsetAsync(bank->data(), 0, bank->size() * sizeof(__half), stream);
      kvo_bank_.push_back(std::move(bank));
      kvo_out_.push_back(std::make_unique<CUDATensor<__half>>((size_t)kvo_components_ * slot));  // [C,1,B,seq,h]
      if(use_tome)
      {
        // concat(bank_slot, new) = [B, 2*seq, hidden] per component, reused each frame.
        tome_cat_k_.push_back(std::make_unique<CUDATensor<__half>>(2u * slot));
        tome_cat_v_.push_back(std::make_unique<CUDATensor<__half>>(2u * slot));
        tome_cat_o_.push_back(std::make_unique<CUDATensor<__half>>(2u * slot));
      }
    }
    bank_batch_ = batch;
    bank_maxframes_ = cache_maxframes;
    bank_tome_ = use_tome;
  }

  cudaMemcpyAsync(sample_buffer_->data(), sample, sample_size * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(timestep_buffer_->data(), timestep, timestep_extent * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(encoder_hidden_states_buffer_->data(), encoder_hidden_states,
                  encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims;
  timestep_dims.nbDims = 1;
  timestep_dims.d[0] = timestep_extent;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  if(!context_->setInputShape("sample", sample_dims))
    throw std::runtime_error("forward_v2v: failed to set sample shape");
  if(!context_->setInputShape("timestep", timestep_dims))
    throw std::runtime_error("forward_v2v: failed to set timestep shape");
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
    throw std::runtime_error("forward_v2v: failed to set encoder_hidden_states shape");

  // Bind the rolling bank to kvo_cache_in_* and the per-frame outputs to kvo_cache_out_*.
  for(int i = 0; i < num_kvo_layers_; i++)
  {
    char nin[24], nout[24];
    std::snprintf(nin, sizeof(nin), "kvo_cache_in_%d", i);
    std::snprintf(nout, sizeof(nout), "kvo_cache_out_%d", i);
    nvinfer1::Dims din;
    din.nbDims = 5;
    din.d[0] = kvo_components_;
    din.d[1] = cache_maxframes;
    din.d[2] = batch;
    din.d[3] = kvo_seq_[i];
    din.d[4] = kvo_hidden_[i];
    if(!context_->setInputShape(nin, din))
      throw std::runtime_error(std::string("forward_v2v: failed to set ") + nin + " shape");
    if(!context_->setTensorAddress(nin, kvo_bank_[i]->data()))
      throw std::runtime_error(std::string("forward_v2v: failed to set ") + nin + " address");
    if(!context_->setTensorAddress(nout, kvo_out_[i]->data()))
      throw std::runtime_error(std::string("forward_v2v: failed to set ") + nout + " address");
  }

  // Dynamic feature-injection params: upload [fi_strength, threshold] to the engine input each frame.
  if(has_v2v_inject_params_)
  {
    if(!v2v_inject_params_buffer_)
      v2v_inject_params_buffer_ = std::make_unique<CUDATensor<float>>(2);
    const float params[2] = {fi_strength, fi_threshold};
    cudaMemcpyAsync(v2v_inject_params_buffer_->data(), params, 2 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims pd;
    pd.nbDims = 1;
    pd.d[0] = 2;
    if(!context_->setInputShape("v2v_inject_params", pd))
      throw std::runtime_error("forward_v2v: failed to set v2v_inject_params shape");
    if(!context_->setTensorAddress("v2v_inject_params", v2v_inject_params_buffer_->data()))
      throw std::runtime_error("forward_v2v: failed to set v2v_inject_params address");
  }

  if(!context_->allInputDimensionsSpecified())
    throw std::runtime_error("forward_v2v: not all input dimensions specified");
  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
    throw std::runtime_error("forward_v2v: failed to set sample address");
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
    throw std::runtime_error("forward_v2v: failed to set timestep address");
  if(!context_->setTensorAddress("encoder_hidden_states", encoder_hidden_states_buffer_->data()))
    throw std::runtime_error("forward_v2v: failed to set encoder_hidden_states address");
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
    throw std::runtime_error("forward_v2v: failed to set latent address");

  if(!context_->enqueueV3(stream))
    throw std::runtime_error("forward_v2v: failed to enqueue inference");

  cudaMemcpyAsync(output, output_buffer_->data(), output_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);

  // Roll the bank: shift each K/V group down one frame slot, write this frame's K/V at the last slot.
  // (Attention is order-invariant over keys, so this exactly reproduces torch.roll(-1)+[:,−1]=new.)
  for(int i = 0; i < num_kvo_layers_; i++)
  {
    const int seq = kvo_seq_[i], h = kvo_hidden_[i];
    const size_t slot = (size_t)batch * seq * h;   // one component's frame slot [B,seq,h]
    const size_t sh = (size_t)seq * h;             // one batch row [seq,h]
    __half* bank = kvo_bank_[i]->data();
    const __half* out = kvo_out_[i]->data();

    if(use_tome)
    {
      // ToMe: per component, build cat = [B, 2*seq, h] = per-batch concat(bank_slot, new), token-merge
      // it back down to [B, seq, h], and write the compacted result into the (single) bank slot.
      __half* catk = tome_cat_k_[i]->data();
      __half* catv = tome_cat_v_[i]->data();
      __half* cato = tome_cat_o_[i]->data();
      for(int b = 0; b < batch; b++)
      {
        for(int c = 0; c < 3; c++)
        {
          __half* cat = (c == 0) ? catk : (c == 1) ? catv : cato;
          const __half* bk = bank + (size_t)c * slot;  // bank component c (maxframes==1)
          const __half* nw = out + (size_t)c * slot;   // this frame's component c
          cudaMemcpyAsync(cat + (size_t)b * 2 * sh, bk + (size_t)b * sh, sh * sizeof(__half),
                          cudaMemcpyDeviceToDevice, stream);
          cudaMemcpyAsync(cat + (size_t)b * 2 * sh + sh, nw + (size_t)b * sh, sh * sizeof(__half),
                          cudaMemcpyDeviceToDevice, stream);
        }
      }
      launch_tome_merge(catk, catv, cato, bank + 0 * slot, bank + 1 * slot, bank + 2 * slot,
                        batch, 2 * seq, h, stream);
    }
    else
    {
      for(int c = 0; c < kvo_components_; c++)  // K, V (, O) — roll each independently
      {
        __half* base = bank + (size_t)c * cache_maxframes * slot;
        for(int m = 0; m < cache_maxframes - 1; m++)
          cudaMemcpyAsync(base + (size_t)m * slot, base + (size_t)(m + 1) * slot,
                          slot * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(base + (size_t)(cache_maxframes - 1) * slot, out + (size_t)c * slot,
                        slot * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
      }
    }
  }
}

void UNetWrapper::resetV2VBank()
{
  // Clear the rolling bank so the next frame starts an empty temporal context. Reset is between
  // clips (not in the hot loop), so a default-stream memset is fine.
  for(auto& b : kvo_bank_)
    if(b)
      cudaMemset(b->data(), 0, b->size() * sizeof(__half));
}

void UNetWrapper::forward_controlnet(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    const __half* const* down_residuals, const __half* mid_residual,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    cudaStream_t stream)
{
  // ControlNet residual geometry, DERIVED from this UNet engine's input_control_NN bindings at init
  // (see ctor). This is model-agnostic: SD1.5 yields 12 down with factors {1,1,1,2,2,2,4,4,4,8,8,8},
  // the pruned SDXS UNet yields 6 down with factors {1,1,2,2,4,4}. Do NOT hardcode the SD1.5 table:
  // the caller (run_controlnets) hands us exactly num_control_down_ residual pointers from the
  // matching ControlNet engine, and the UNet engine expects exactly these shapes.
  const int kNumDown = num_control_down_;
  const int* kDownCh = control_down_ch_.data();
  const int* kDownFac = control_down_fac_.data();
  const int kMidCh = control_mid_ch_, kMidFac = control_mid_fac_;
  if(kNumDown <= 0)
    throw std::runtime_error("forward_controlnet called on a non-control-aware UNet engine");

  int sample_size = batch * 4 * height * width;
  int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  int encoder_size = batch * seq_len * hidden_dim;
  int output_size = batch * 4 * height * width;

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim);
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_extent);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);
    control_down_buffers_.clear();
    control_down_buffers_.resize(kNumDown);
    for(int i = 0; i < kNumDown; i++)
    {
      int h = height / kDownFac[i], w = width / kDownFac[i];
      control_down_buffers_[i]
          = std::make_unique<CUDATensor<__half>>(batch * kDownCh[i] * h * w);
    }
    control_mid_buffer_ = std::make_unique<CUDATensor<__half>>(
        batch * kMidCh * (height / kMidFac) * (width / kMidFac));
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.initialized = true;
  }

  // Copy core inputs into persistent buffers.
  cudaMemcpyAsync(sample_buffer_->data(), sample, sample_size * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(timestep_buffer_->data(), timestep, timestep_extent * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(encoder_hidden_states_buffer_->data(), encoder_hidden_states,
                  encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  // Copy the control residuals into persistent buffers (stable addresses for setTensorAddress).
  for(int i = 0; i < kNumDown; i++)
  {
    int h = height / kDownFac[i], w = width / kDownFac[i];
    cudaMemcpyAsync(control_down_buffers_[i]->data(), down_residuals[i],
                    (size_t)batch * kDownCh[i] * h * w * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
  }
  cudaMemcpyAsync(control_mid_buffer_->data(), mid_residual,
                  (size_t)batch * kMidCh * (height / kMidFac) * (width / kMidFac) * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  // Set core shapes.
  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims;
  timestep_dims.nbDims = 1;
  timestep_dims.d[0] = timestep_extent;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  if(!context_->setInputShape("sample", sample_dims))
    throw std::runtime_error("Failed to set sample shape");
  if(!context_->setInputShape("timestep", timestep_dims))
    throw std::runtime_error("Failed to set timestep shape");
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
    throw std::runtime_error("Failed to set encoder_hidden_states shape");

  // Set control residual shapes + addresses.
  for(int i = 0; i < kNumDown; i++)
  {
    char name[24];
    std::snprintf(name, sizeof(name), "input_control_%02d", i);
    int h = height / kDownFac[i], w = width / kDownFac[i];
    nvinfer1::Dims4 d{batch, kDownCh[i], h, w};
    if(!context_->setInputShape(name, d))
      throw std::runtime_error(std::string("Failed to set ") + name + " shape");
    if(!context_->setTensorAddress(name, control_down_buffers_[i]->data()))
      throw std::runtime_error(std::string("Failed to set ") + name + " address");
  }
  {
    nvinfer1::Dims4 dmid{batch, kMidCh, height / kMidFac, width / kMidFac};
    if(!context_->setInputShape("input_control_middle", dmid))
      throw std::runtime_error("Failed to set input_control_middle shape");
    if(!context_->setTensorAddress("input_control_middle", control_mid_buffer_->data()))
      throw std::runtime_error("Failed to set input_control_middle address");
  }

  if(!context_->allInputDimensionsSpecified())
    throw std::runtime_error("Not all input dimensions specified (controlnet UNet)");

  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
    throw std::runtime_error("Failed to set sample tensor address");
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
    throw std::runtime_error("Failed to set timestep tensor address");
  if(!context_->setTensorAddress("encoder_hidden_states", encoder_hidden_states_buffer_->data()))
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
    throw std::runtime_error("Failed to set latent tensor address");

  if(!context_->enqueueV3(stream))
    throw std::runtime_error("Failed to enqueue inference (controlnet UNet)");

  cudaMemcpyAsync(output, output_buffer_->data(), output_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
}

void UNetWrapper::forward_ipadapter(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    const float* ipadapter_scale, int num_ip_layers,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    cudaStream_t stream)
{
  int sample_size = batch * 4 * height * width;
  int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  int encoder_size = batch * seq_len * hidden_dim;  // seq_len = 77 + num_image_tokens (e.g. 81)
  int output_size = batch * 4 * height * width;

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim);
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_extent);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.initialized = true;
  }
  if(!ipadapter_scale_buffer_ || (int)ipadapter_scale_buffer_->size() < num_ip_layers)
    ipadapter_scale_buffer_ = std::make_unique<CUDATensor<float>>(num_ip_layers);

  cudaMemcpyAsync(sample_buffer_->data(), sample, sample_size * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(timestep_buffer_->data(), timestep, timestep_extent * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(encoder_hidden_states_buffer_->data(), encoder_hidden_states,
                  encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  // ipadapter_scale is host-supplied (a small per-layer vector) -> H2D. Stage through PINNED host memory so
  // this copy is legal inside CUDA-graph capture (pageable H2D is not) and skip the copy entirely when the
  // values are unchanged (so a captured graph replays with NO H2D, and the warm pre-capture enqueue baked
  // the correct device value). On change while a graph exists, the caller-side signature/grow path recaptures.
  if(ipadapter_scale_pinned_cap_ < num_ip_layers)
  {
    if(ipadapter_scale_pinned_) cudaFreeHost(ipadapter_scale_pinned_);
    cudaMallocHost((void**)&ipadapter_scale_pinned_, num_ip_layers * sizeof(float));
    ipadapter_scale_pinned_cap_ = num_ip_layers;
    ipadapter_scale_last_.clear();
  }
  bool scale_changed = (int)ipadapter_scale_last_.size() != num_ip_layers
      || std::memcmp(ipadapter_scale_last_.data(), ipadapter_scale, num_ip_layers * sizeof(float)) != 0;
  if(scale_changed)
  {
    std::memcpy(ipadapter_scale_pinned_, ipadapter_scale, num_ip_layers * sizeof(float));
    cudaMemcpyAsync(ipadapter_scale_buffer_->data(), ipadapter_scale_pinned_,
                    num_ip_layers * sizeof(float), cudaMemcpyHostToDevice, stream);
    ipadapter_scale_last_.assign(ipadapter_scale, ipadapter_scale + num_ip_layers);
  }
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims; timestep_dims.nbDims = 1; timestep_dims.d[0] = timestep_extent;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  nvinfer1::Dims scale_dims; scale_dims.nbDims = 1; scale_dims.d[0] = num_ip_layers;
  if(!context_->setInputShape("sample", sample_dims))
    throw std::runtime_error("Failed to set sample shape");
  if(!context_->setInputShape("timestep", timestep_dims))
    throw std::runtime_error("Failed to set timestep shape");
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  if(!context_->setInputShape("ipadapter_scale", scale_dims))
    throw std::runtime_error("Failed to set ipadapter_scale shape");

  if(!context_->allInputDimensionsSpecified())
    throw std::runtime_error("Not all input dimensions specified (IP-Adapter UNet)");

  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
    throw std::runtime_error("Failed to set sample tensor address");
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
    throw std::runtime_error("Failed to set timestep tensor address");
  if(!context_->setTensorAddress("encoder_hidden_states", encoder_hidden_states_buffer_->data()))
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  if(!context_->setTensorAddress("ipadapter_scale", ipadapter_scale_buffer_->data()))
    throw std::runtime_error("Failed to set ipadapter_scale tensor address");
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
    throw std::runtime_error("Failed to set latent tensor address");

  if(!context_->enqueueV3(stream))
    throw std::runtime_error("Failed to enqueue inference (IP-Adapter UNet)");

  cudaMemcpyAsync(output, output_buffer_->data(), output_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
}

void UNetWrapper::forward_ipadapter_sdxl(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    const __half* text_embeds, const __half* time_ids,
    const float* ipadapter_scale, int num_ip_layers,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    int pooled_dim, cudaStream_t stream)
{
  // = forward_sdxl (sample/timestep/ehs/text_embeds/time_ids) + forward_ipadapter (ipadapter_scale).
  int sample_size = batch * 4 * height * width;
  int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  int encoder_size = batch * seq_len * hidden_dim;  // seq_len = 77 + num_image_tokens
  int output_size = batch * 4 * height * width;
  int text_embeds_size = batch * pooled_dim;
  int time_ids_size = batch * 6;

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim, pooled_dim);
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_extent);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    text_embeds_buffer_ = std::make_unique<CUDATensor<__half>>(text_embeds_size);
    time_ids_buffer_ = std::make_unique<CUDATensor<__half>>(time_ids_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.pooled_dim = pooled_dim;
    shape_cache_.initialized = true;
  }
  if(!ipadapter_scale_buffer_ || (int)ipadapter_scale_buffer_->size() < num_ip_layers)
    ipadapter_scale_buffer_ = std::make_unique<CUDATensor<float>>(num_ip_layers);

  cudaMemcpyAsync(sample_buffer_->data(), sample, sample_size * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(timestep_buffer_->data(), timestep, timestep_extent * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(encoder_hidden_states_buffer_->data(), encoder_hidden_states,
                  encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(text_embeds_buffer_->data(), text_embeds,
                  text_embeds_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(time_ids_buffer_->data(), time_ids,
                  time_ids_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

  // ipadapter_scale H2D via PINNED host + skip-when-unchanged (CUDA-graph-capture-safe), as forward_ipadapter.
  if(ipadapter_scale_pinned_cap_ < num_ip_layers)
  {
    if(ipadapter_scale_pinned_) cudaFreeHost(ipadapter_scale_pinned_);
    cudaMallocHost((void**)&ipadapter_scale_pinned_, num_ip_layers * sizeof(float));
    ipadapter_scale_pinned_cap_ = num_ip_layers;
    ipadapter_scale_last_.clear();
  }
  bool scale_changed = (int)ipadapter_scale_last_.size() != num_ip_layers
      || std::memcmp(ipadapter_scale_last_.data(), ipadapter_scale, num_ip_layers * sizeof(float)) != 0;
  if(scale_changed)
  {
    std::memcpy(ipadapter_scale_pinned_, ipadapter_scale, num_ip_layers * sizeof(float));
    cudaMemcpyAsync(ipadapter_scale_buffer_->data(), ipadapter_scale_pinned_,
                    num_ip_layers * sizeof(float), cudaMemcpyHostToDevice, stream);
    ipadapter_scale_last_.assign(ipadapter_scale, ipadapter_scale + num_ip_layers);
  }

  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims; timestep_dims.nbDims = 1; timestep_dims.d[0] = timestep_extent;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  nvinfer1::Dims text_embeds_dims; text_embeds_dims.nbDims = 2;
  text_embeds_dims.d[0] = batch; text_embeds_dims.d[1] = pooled_dim;
  nvinfer1::Dims time_ids_dims; time_ids_dims.nbDims = 2;
  time_ids_dims.d[0] = batch; time_ids_dims.d[1] = 6;
  nvinfer1::Dims scale_dims; scale_dims.nbDims = 1; scale_dims.d[0] = num_ip_layers;

  if(!context_->setInputShape("sample", sample_dims))
    throw std::runtime_error("Failed to set sample shape");
  if(!context_->setInputShape("timestep", timestep_dims))
    throw std::runtime_error("Failed to set timestep shape");
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  if(!context_->setInputShape("text_embeds", text_embeds_dims))
    throw std::runtime_error("Failed to set text_embeds shape");
  if(!context_->setInputShape("time_ids", time_ids_dims))
    throw std::runtime_error("Failed to set time_ids shape");
  if(!context_->setInputShape("ipadapter_scale", scale_dims))
    throw std::runtime_error("Failed to set ipadapter_scale shape");

  if(!context_->allInputDimensionsSpecified())
    throw std::runtime_error("Not all input dimensions specified (SDXL IP-Adapter UNet)");

  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
    throw std::runtime_error("Failed to set sample tensor address");
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
    throw std::runtime_error("Failed to set timestep tensor address");
  if(!context_->setTensorAddress("encoder_hidden_states", encoder_hidden_states_buffer_->data()))
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  if(!context_->setTensorAddress("text_embeds", text_embeds_buffer_->data()))
    throw std::runtime_error("Failed to set text_embeds tensor address");
  if(!context_->setTensorAddress("time_ids", time_ids_buffer_->data()))
    throw std::runtime_error("Failed to set time_ids tensor address");
  if(!context_->setTensorAddress("ipadapter_scale", ipadapter_scale_buffer_->data()))
    throw std::runtime_error("Failed to set ipadapter_scale tensor address");
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
    throw std::runtime_error("Failed to set latent tensor address");

  if(!context_->enqueueV3(stream))
    throw std::runtime_error("Failed to enqueue inference (SDXL IP-Adapter UNet)");

  cudaMemcpyAsync(output, output_buffer_->data(), output_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
}

void UNetWrapper::forward_controlnet_sdxl(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    const __half* text_embeds, const __half* time_ids,
    const __half* const* down_residuals, const __half* mid_residual,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    int pooled_dim, cudaStream_t stream)
{
  // SDXL ControlNet residual geometry: 9 down + mid, factors [1,1,1,2,2,2,4,4,4] + mid 4.
  static const int kDownCh[9] = {320, 320, 320, 320, 640, 640, 640, 1280, 1280};
  static const int kDownFac[9] = {1, 1, 1, 2, 2, 2, 4, 4, 4};
  const int kNumDown = 9, kMidCh = 1280, kMidFac = 4;

  int sample_size = batch * 4 * height * width;
  int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  int encoder_size = batch * seq_len * hidden_dim;
  int output_size = batch * 4 * height * width;

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim, pooled_dim);
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_extent);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    text_embeds_buffer_ = std::make_unique<CUDATensor<__half>>(batch * pooled_dim);
    time_ids_buffer_ = std::make_unique<CUDATensor<__half>>(batch * 6);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);
    control_down_buffers_.clear();
    control_down_buffers_.resize(kNumDown);
    for(int i = 0; i < kNumDown; i++)
    {
      int h = height / kDownFac[i], w = width / kDownFac[i];
      control_down_buffers_[i] = std::make_unique<CUDATensor<__half>>(batch * kDownCh[i] * h * w);
    }
    control_mid_buffer_ = std::make_unique<CUDATensor<__half>>(
        batch * kMidCh * (height / kMidFac) * (width / kMidFac));
    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.pooled_dim = pooled_dim;
    shape_cache_.initialized = true;
  }

  cudaMemcpyAsync(sample_buffer_->data(), sample, sample_size * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(timestep_buffer_->data(), timestep, timestep_extent * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(encoder_hidden_states_buffer_->data(), encoder_hidden_states,
                  encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(text_embeds_buffer_->data(), text_embeds, batch * pooled_dim * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(time_ids_buffer_->data(), time_ids, batch * 6 * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  for(int i = 0; i < kNumDown; i++)
  {
    int h = height / kDownFac[i], w = width / kDownFac[i];
    cudaMemcpyAsync(control_down_buffers_[i]->data(), down_residuals[i],
                    (size_t)batch * kDownCh[i] * h * w * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
  }
  cudaMemcpyAsync(control_mid_buffer_->data(), mid_residual,
                  (size_t)batch * kMidCh * (height / kMidFac) * (width / kMidFac) * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims; timestep_dims.nbDims = 1; timestep_dims.d[0] = timestep_extent;
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  nvinfer1::Dims2 te_dims{batch, pooled_dim};
  nvinfer1::Dims2 ti_dims{batch, 6};
  if(!context_->setInputShape("sample", sample_dims))
    throw std::runtime_error("Failed to set sample shape");
  if(!context_->setInputShape("timestep", timestep_dims))
    throw std::runtime_error("Failed to set timestep shape");
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  if(!context_->setInputShape("text_embeds", te_dims))
    throw std::runtime_error("Failed to set text_embeds shape");
  if(!context_->setInputShape("time_ids", ti_dims))
    throw std::runtime_error("Failed to set time_ids shape");
  for(int i = 0; i < kNumDown; i++)
  {
    char name[24];
    std::snprintf(name, sizeof(name), "input_control_%02d", i);
    int h = height / kDownFac[i], w = width / kDownFac[i];
    nvinfer1::Dims4 d{batch, kDownCh[i], h, w};
    if(!context_->setInputShape(name, d))
      throw std::runtime_error(std::string("Failed to set ") + name + " shape");
    if(!context_->setTensorAddress(name, control_down_buffers_[i]->data()))
      throw std::runtime_error(std::string("Failed to set ") + name + " address");
  }
  {
    nvinfer1::Dims4 dmid{batch, kMidCh, height / kMidFac, width / kMidFac};
    if(!context_->setInputShape("input_control_middle", dmid))
      throw std::runtime_error("Failed to set input_control_middle shape");
    if(!context_->setTensorAddress("input_control_middle", control_mid_buffer_->data()))
      throw std::runtime_error("Failed to set input_control_middle address");
  }

  if(!context_->allInputDimensionsSpecified())
    throw std::runtime_error("Not all input dimensions specified (controlnet SDXL UNet)");

  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
    throw std::runtime_error("Failed to set sample tensor address");
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
    throw std::runtime_error("Failed to set timestep tensor address");
  if(!context_->setTensorAddress("encoder_hidden_states", encoder_hidden_states_buffer_->data()))
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  if(!context_->setTensorAddress("text_embeds", text_embeds_buffer_->data()))
    throw std::runtime_error("Failed to set text_embeds tensor address");
  if(!context_->setTensorAddress("time_ids", time_ids_buffer_->data()))
    throw std::runtime_error("Failed to set time_ids tensor address");
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
    throw std::runtime_error("Failed to set latent tensor address");

  if(!context_->enqueueV3(stream))
    throw std::runtime_error("Failed to enqueue inference (controlnet SDXL UNet)");

  cudaMemcpyAsync(output, output_buffer_->data(), output_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
}

void UNetWrapper::forward_sdxl(
    const float* sample, const float* timestep, const __half* encoder_hidden_states,
    const __half* text_embeds, const __half* time_ids,
    __half* output, int batch, int height, int width, int seq_len, int hidden_dim,
    int pooled_dim, cudaStream_t stream)
{
  // Calculate buffer sizes
  int sample_size = batch * 4 * height * width;
  // The prebuilt SDXL UNet uses a static timestep [1]; honor the engine's declared extent.
  int timestep_extent = (timestep_fixed_extent_ > 0) ? timestep_fixed_extent_ : batch;
  int timestep_size = timestep_extent;
  int encoder_size = batch * seq_len * hidden_dim;
  int output_size = batch * 4 * height * width;
  int text_embeds_size = batch * pooled_dim;
  int time_ids_size = batch * 6;  // SDXL uses 6 time IDs

  bool needs_realloc = needsReallocation(batch, height, width, seq_len, hidden_dim, pooled_dim);

  // Allocate persistent buffers on first call or when shapes change (matches Python)
  if(needs_realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<float>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(timestep_size);
    encoder_hidden_states_buffer_ = std::make_unique<CUDATensor<__half>>(encoder_size);
    text_embeds_buffer_ = std::make_unique<CUDATensor<__half>>(text_embeds_size);
    time_ids_buffer_ = std::make_unique<CUDATensor<__half>>(time_ids_size);
    output_buffer_ = std::make_unique<CUDATensor<__half>>(output_size);

    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.seq_len = seq_len;
    shape_cache_.hidden_dim = hidden_dim;
    shape_cache_.pooled_dim = pooled_dim;
    shape_cache_.initialized = true;
  }

  // Copy input data into persistent buffers
  cudaMemcpyAsync(
      sample_buffer_->data(), sample, sample_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      timestep_buffer_->data(), timestep, timestep_size * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      encoder_hidden_states_buffer_->data(), encoder_hidden_states,
      encoder_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      text_embeds_buffer_->data(), text_embeds,
      text_embeds_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      time_ids_buffer_->data(), time_ids,
      time_ids_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

  // CRITICAL: Synchronize stream after copies to ensure TensorRT reads complete data
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  // CRITICAL: Set input shapes BEFORE setting tensor addresses (matches Python's exact order)
  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims timestep_dims;
  timestep_dims.nbDims = 1;
  timestep_dims.d[0] = timestep_extent;  // [1] for the prebuilt SDXL UNet, else [batch]
  nvinfer1::Dims3 hidden_states_dims{batch, seq_len, hidden_dim};
  nvinfer1::Dims text_embeds_dims;
  text_embeds_dims.nbDims = 2;
  text_embeds_dims.d[0] = batch;
  text_embeds_dims.d[1] = pooled_dim;
  nvinfer1::Dims time_ids_dims;
  time_ids_dims.nbDims = 2;
  time_ids_dims.d[0] = batch;
  time_ids_dims.d[1] = 6;

  if(!context_->setInputShape("sample", sample_dims))
  {
    throw std::runtime_error("Failed to set sample shape");
  }
  if(!context_->setInputShape("timestep", timestep_dims))
  {
    throw std::runtime_error("Failed to set timestep shape");
  }
  if(!context_->setInputShape("encoder_hidden_states", hidden_states_dims))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states shape");
  }
  if(!context_->setInputShape("text_embeds", text_embeds_dims))
  {
    throw std::runtime_error("Failed to set text_embeds shape");
  }
  if(!context_->setInputShape("time_ids", time_ids_dims))
  {
    throw std::runtime_error("Failed to set time_ids shape");
  }

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses using persistent buffer addresses (stays constant across calls)
  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
  {
    throw std::runtime_error("Failed to set sample tensor address");
  }
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
  {
    throw std::runtime_error("Failed to set timestep tensor address");
  }
  if(!context_->setTensorAddress(
         "encoder_hidden_states", encoder_hidden_states_buffer_->data()))
  {
    throw std::runtime_error("Failed to set encoder_hidden_states tensor address");
  }
  if(!context_->setTensorAddress("text_embeds", text_embeds_buffer_->data()))
  {
    throw std::runtime_error("Failed to set text_embeds tensor address");
  }
  if(!context_->setTensorAddress("time_ids", time_ids_buffer_->data()))
  {
    throw std::runtime_error("Failed to set time_ids tensor address");
  }
  if(!context_->setTensorAddress("latent", output_buffer_->data()))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  bindLoraScale(stream);  // runtime-LoRA SDXL engines: bind lora_scale[N] (no-op otherwise)

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }

  // Copy output from persistent buffer back to caller's buffer
  cudaMemcpyAsync(
      output, output_buffer_->data(), output_size * sizeof(__half),
      cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// ControlNetWrapper Implementation
// ============================================================================

ControlNetWrapper::ControlNetWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

ControlNetWrapper::~ControlNetWrapper()
{
  if(scale_pinned_)
    cudaFreeHost(scale_pinned_);
}

void ControlNetWrapper::loadEngine(const std::string& engine_path)
{
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
    throw std::runtime_error("Failed to load ControlNet engine from cache: " + engine_path);
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(cached_engine_->createExecutionContext());
  if(!context_)
    throw std::runtime_error("Failed to create ControlNet execution context");

  // Detect at load: (1) the conditioning_scale scalar input (name varies — SD1.5 trace = "onnx::Cast_4",
  // SDXL = "conditioning_scale"; find the lone 0-dim FP32 input), (2) the down_block_* output COUNT
  // (12 SD1.5 / 9 SDXL), (3) whether the engine is SDXL (has text_embeds + time_ids inputs).
  nvinfer1::ICudaEngine* eng = cached_engine_->getEngine();
  num_down_ = 0;
  bool has_text_embeds = false, has_time_ids = false;
  for(int i = 0; eng && i < eng->getNbIOTensors(); i++)
  {
    const char* tn = eng->getIOTensorName(i);
    if(!tn) continue;
    std::string nm = tn;
    if(eng->getTensorIOMode(tn) == nvinfer1::TensorIOMode::kINPUT)
    {
      nvinfer1::Dims d = eng->getTensorShape(tn);
      if(d.nbDims == 0 && eng->getTensorDataType(tn) == nvinfer1::DataType::kFLOAT)
        scale_input_name_ = tn;
      if(nm == "text_embeds") has_text_embeds = true;
      if(nm == "time_ids") has_time_ids = true;
    }
    else if(nm.rfind("down_block_", 0) == 0)
    {
      num_down_++;
    }
  }
  is_sdxl_ = has_text_embeds && has_time_ids;
  if(num_down_ <= 0 || num_down_ > MAX_DOWN) num_down_ = 12;  // sane fallback
  std::cout << "Note: ControlNet engine " << (is_sdxl_ ? "SDXL" : "SD1.5") << ", " << num_down_
            << " down residuals, scale input = '"
            << (scale_input_name_.empty() ? "(none)" : scale_input_name_) << "'" << std::endl;
}

void ControlNetWrapper::forward(
    const __half* sample, const float* timestep, const __half* encoder_hidden_states,
    const __half* controlnet_cond, float conditioning_scale,
    const __half* text_embeds, const __half* time_ids,
    int batch, int height, int width, int img_height, int img_width, int seq_len, int hidden_dim,
    int pooled_dim, const __half** down_out, const __half** mid_out, cudaStream_t stream)
{
  // Per-arch residual geometry, selected by the engine's detected down_block_* COUNT (num_down_) — NOT
  // by is_sdxl_ alone — so pruned SD-family UNets (e.g. SDXS = 6 down) get the right channels/factors
  // for their OUTPUT buffers. SD1.5 = 12 (factors up to 8), SDXL = 9 (up to 4), SDXS = 6 (up to 4).
  static const int kSD15Ch[12] = {320,320,320,320,640,640,640,1280,1280,1280,1280,1280};
  static const int kSD15Fac[12] = {1,1,1,2,2,2,4,4,4,8,8,8};
  static const int kSDXLCh[9] = {320,320,320,320,640,640,640,1280,1280};
  static const int kSDXLFac[9] = {1,1,1,2,2,2,4,4,4};
  static const int kSDXSCh[6] = {320,320,320,640,640,1280};
  static const int kSDXSFac[6] = {1,1,2,2,4,4};
  const int nd = num_down_;
  const int* kDownCh = (nd == 6) ? kSDXSCh : is_sdxl_ ? kSDXLCh : kSD15Ch;
  const int* kDownFac = (nd == 6) ? kSDXSFac : is_sdxl_ ? kSDXLFac : kSD15Fac;
  // mid factor = the deepest down factor: SD1.5 8, SDXL/SDXS 4.
  const int kMidCh = 1280, kMidFac = (nd == 6 || is_sdxl_) ? 4 : 8;

  const int sample_size = batch * 4 * height * width;
  const int ehs_size = batch * seq_len * hidden_dim;
  const int cond_size = batch * 3 * img_height * img_width;

  const bool realloc = !shape_cache_.init || shape_cache_.batch != batch
                       || shape_cache_.height != height || shape_cache_.width != width
                       || shape_cache_.img_h != img_height || shape_cache_.img_w != img_width
                       || shape_cache_.seq_len != seq_len || shape_cache_.hidden_dim != hidden_dim
                       || shape_cache_.pooled_dim != pooled_dim;
  if(realloc)
  {
    sample_buffer_ = std::make_unique<CUDATensor<__half>>(sample_size);
    timestep_buffer_ = std::make_unique<CUDATensor<float>>(batch);
    ehs_buffer_ = std::make_unique<CUDATensor<__half>>(ehs_size);
    cond_buffer_ = std::make_unique<CUDATensor<__half>>(cond_size);
    scale_buffer_ = std::make_unique<CUDATensor<float>>(1);
    if(is_sdxl_)
    {
      text_embeds_buffer_ = std::make_unique<CUDATensor<__half>>(batch * pooled_dim);
      time_ids_buffer_ = std::make_unique<CUDATensor<__half>>(batch * 6);
    }
    down_buffers_.clear();
    down_buffers_.resize(nd);
    for(int i = 0; i < nd; i++)
    {
      int h = height / kDownFac[i], w = width / kDownFac[i];
      down_buffers_[i] = std::make_unique<CUDATensor<__half>>(batch * kDownCh[i] * h * w);
    }
    mid_buffer_ = std::make_unique<CUDATensor<__half>>(
        batch * kMidCh * (height / kMidFac) * (width / kMidFac));
    shape_cache_ = {batch, height, width, img_height, img_width, seq_len, hidden_dim, pooled_dim, true};
  }

  cudaMemcpyAsync(sample_buffer_->data(), sample, sample_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(timestep_buffer_->data(), timestep, batch * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(ehs_buffer_->data(), encoder_hidden_states, ehs_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(cond_buffer_->data(), controlnet_cond, cond_size * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
  // conditioning_scale: stage through pinned host memory (capturable) + skip when unchanged.
  if(!scale_pinned_)
    cudaMallocHost((void**)&scale_pinned_, sizeof(float));
  if(conditioning_scale != scale_last_)
  {
    *scale_pinned_ = conditioning_scale;
    cudaMemcpyAsync(scale_buffer_->data(), scale_pinned_, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    scale_last_ = conditioning_scale;
  }
  if(is_sdxl_)
  {
    cudaMemcpyAsync(text_embeds_buffer_->data(), text_embeds, batch * pooled_dim * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(time_ids_buffer_->data(), time_ids, batch * 6 * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
  }
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  nvinfer1::Dims4 sample_dims{batch, 4, height, width};
  nvinfer1::Dims ts_dims; ts_dims.nbDims = 1; ts_dims.d[0] = batch;
  nvinfer1::Dims3 ehs_dims{batch, seq_len, hidden_dim};
  nvinfer1::Dims4 cond_dims{batch, 3, img_height, img_width};
  if(!context_->setInputShape("sample", sample_dims))
    throw std::runtime_error("CN: failed to set sample shape");
  if(!context_->setInputShape("timestep", ts_dims))
    throw std::runtime_error("CN: failed to set timestep shape");
  if(!context_->setInputShape("encoder_hidden_states", ehs_dims))
    throw std::runtime_error("CN: failed to set encoder_hidden_states shape");
  if(!context_->setInputShape("controlnet_cond", cond_dims))
    throw std::runtime_error("CN: failed to set controlnet_cond shape");
  if(!scale_input_name_.empty())
  {
    nvinfer1::Dims scalar_dims; scalar_dims.nbDims = 0;
    if(!context_->setInputShape(scale_input_name_.c_str(), scalar_dims))
      throw std::runtime_error("CN: failed to set scale scalar shape");
  }
  if(is_sdxl_)
  {
    nvinfer1::Dims2 te{batch, pooled_dim};
    nvinfer1::Dims2 ti{batch, 6};
    if(!context_->setInputShape("text_embeds", te))
      throw std::runtime_error("CN: failed to set text_embeds shape");
    if(!context_->setInputShape("time_ids", ti))
      throw std::runtime_error("CN: failed to set time_ids shape");
  }

  if(!context_->allInputDimensionsSpecified())
    throw std::runtime_error("CN: not all input dimensions specified");

  if(!context_->setTensorAddress("sample", sample_buffer_->data()))
    throw std::runtime_error("CN: failed to set sample address");
  if(!context_->setTensorAddress("timestep", timestep_buffer_->data()))
    throw std::runtime_error("CN: failed to set timestep address");
  if(!context_->setTensorAddress("encoder_hidden_states", ehs_buffer_->data()))
    throw std::runtime_error("CN: failed to set encoder_hidden_states address");
  if(!context_->setTensorAddress("controlnet_cond", cond_buffer_->data()))
    throw std::runtime_error("CN: failed to set controlnet_cond address");
  if(!scale_input_name_.empty())
  {
    if(!context_->setTensorAddress(scale_input_name_.c_str(), scale_buffer_->data()))
      throw std::runtime_error("CN: failed to set scale address");
  }
  if(is_sdxl_)
  {
    if(!context_->setTensorAddress("text_embeds", text_embeds_buffer_->data()))
      throw std::runtime_error("CN: failed to set text_embeds address");
    if(!context_->setTensorAddress("time_ids", time_ids_buffer_->data()))
      throw std::runtime_error("CN: failed to set time_ids address");
  }

  // Bind the down + mid outputs to our persistent buffers and report them back.
  for(int i = 0; i < nd; i++)
  {
    char name[24];
    std::snprintf(name, sizeof(name), "down_block_%02d", i);
    if(!context_->setTensorAddress(name, down_buffers_[i]->data()))
      throw std::runtime_error(std::string("CN: failed to set ") + name + " address");
    down_out[i] = down_buffers_[i]->data();
  }
  if(!context_->setTensorAddress("mid_block", mid_buffer_->data()))
    throw std::runtime_error("CN: failed to set mid_block address");
  *mid_out = mid_buffer_->data();

  if(!context_->enqueueV3(stream))
    throw std::runtime_error("CN: failed to enqueue inference");
}

// ============================================================================
// VAEEncoderWrapper Implementation
// ============================================================================

VAEEncoderWrapper::VAEEncoderWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

VAEEncoderWrapper::~VAEEncoderWrapper()
{
  // Unique_ptrs will handle cleanup
}

void VAEEncoderWrapper::loadEngine(const std::string& engine_path)
{
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      cached_engine_->createExecutionContext());

  if(!context_)
  {
    throw std::runtime_error("Failed to create execution context");
  }
}

bool VAEEncoderWrapper::needsReallocation(int batch, int height, int width)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch
         || shape_cache_.height != height || shape_cache_.width != width;
}

void VAEEncoderWrapper::encode(
    const __half* images, __half* latent, int batch, int height, int width,
    cudaStream_t stream)
{
  // Allocate persistent FP32 buffer on first call or if shape changed
  // This matches Python's behavior of allocating buffers once
  int images_elements = batch * 3 * height * width;

  if(!images_fp32_buffer_ || needsReallocation(batch, height, width))
  {
    images_fp32_buffer_ = std::make_unique<CUDATensor<float>>(images_elements);

    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.initialized = true;
  }

  // ALWAYS set input shape to match Python (don't cache)
  nvinfer1::Dims4 images_dims{batch, 3, height, width};
  if(!context_->setInputShape("images", images_dims))
  {
    throw std::runtime_error("Failed to set images shape");
  }

  // Convert FP16→FP32 into persistent buffer (reuse same buffer each time)
  // This ensures the buffer address stays constant across all calls
  launch_fp16_to_fp32(images, images_fp32_buffer_->data(), images_elements, stream);

  // CRITICAL: Synchronize stream to ensure conversion is complete before TensorRT uses the buffer
  // Without this, TensorRT might read incomplete/wrong data from the FP32 buffer
  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses (use persistent buffer address)
  if(!context_->setTensorAddress("images", images_fp32_buffer_->data()))
  {
    throw std::runtime_error("Failed to set images tensor address");
  }
  if(!context_->setTensorAddress("latent", latent))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }
}

void VAEEncoderWrapper::encode(
    const float* images, __half* latent, int batch, int height, int width,
    cudaStream_t stream)
{
  // Allocate persistent FP32 buffer on first call or if shape changed
  // This matches Python's behavior of allocating buffers once
  int images_elements = batch * 3 * height * width;

  if(!images_fp32_buffer_ || needsReallocation(batch, height, width))
  {
    images_fp32_buffer_ = std::make_unique<CUDATensor<float>>(images_elements);

    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.initialized = true;
  }

  // ALWAYS set input shape to match Python (don't cache)
  nvinfer1::Dims4 images_dims{batch, 3, height, width};
  if(!context_->setInputShape("images", images_dims))
  {
    throw std::runtime_error("Failed to set images shape");
  }

  // `images` is DEVICE memory: this GPU encode path mirrors the __half overload (which
  // converts device->device). Copy the fp32 device input into the persistent device
  // buffer with a device-to-device copy on the stream. (Previously this used
  // cudaMemcpyHostToDevice, which is wrong for a device source — it only "worked" via
  // UVA inference and corrupted the input otherwise; the path is exercised by the
  // validation harness, not the high-level CPU-RGBA img2img.)
  cudaMemcpyAsync(
      images_fp32_buffer_->data(), (const void*)images, images_elements * sizeof(float),
      cudaMemcpyDeviceToDevice, stream);

  /* LRD-PERF: redundant pre-enqueue sync removed (same-stream async copies already ordered) */

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses (use persistent buffer address)
  if(!context_->setTensorAddress("images", images_fp32_buffer_->data()))
  {
    throw std::runtime_error("Failed to set images tensor address");
  }
  if(!context_->setTensorAddress("latent", latent))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }
}

// ============================================================================
// VAEDecoderWrapper Implementation
// ============================================================================

VAEDecoderWrapper::VAEDecoderWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

VAEDecoderWrapper::~VAEDecoderWrapper()
{
  // Unique_ptrs will handle cleanup
}

void VAEDecoderWrapper::loadEngine(const std::string& engine_path)
{
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  std::cout << "Note: Using cached TensorRT engine for " << engine_path << std::endl;
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      cached_engine_->createExecutionContext());

  if(!context_)
  {
    throw std::runtime_error("Failed to create execution context");
  }
}

bool VAEDecoderWrapper::needsReallocation(int batch, int height, int width)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch
         || shape_cache_.height != height || shape_cache_.width != width;
}

void VAEDecoderWrapper::decode(
    const __half* latent, __half* images, int batch, int height, int width,
    cudaStream_t stream)
{
  // Allocate persistent FP32 buffer on first call or if shape changed
  // This matches Python's behavior of allocating buffers once
  int latent_elements = batch * 4 * height * width;
  if(!latent_fp32_buffer_ || needsReallocation(batch, height, width))
  {
    latent_fp32_buffer_ = std::make_unique<CUDATensor<float>>(latent_elements);

    // Set input shape (only needs to be done once or when shape changes)
    nvinfer1::Dims4 latent_dims{batch, 4, height, width};
    if(!context_->setInputShape("latent", latent_dims))
    {
      throw std::runtime_error("Failed to set latent shape");
    }

    shape_cache_.batch = batch;
    shape_cache_.height = height;
    shape_cache_.width = width;
    shape_cache_.initialized = true;
  }

  // Convert FP16→FP32 into persistent buffer (reuse same buffer each time)
  // This ensures the buffer address stays constant across all calls
  launch_fp16_to_fp32(latent, latent_fp32_buffer_->data(), latent_elements, stream);

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Set tensor addresses (use persistent buffer address)
  if(!context_->setTensorAddress("latent", latent_fp32_buffer_->data()))
  {
    throw std::runtime_error("Failed to set latent tensor address");
  }
  if(!context_->setTensorAddress("images", images))
  {
    throw std::runtime_error("Failed to set images tensor address");
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    throw std::runtime_error("Failed to enqueue inference");
  }
}

// ============================================================================
// CLIPWrapper Implementation
// ============================================================================

CLIPWrapper::CLIPWrapper(const std::string& engine_path)
{
  loadEngine(engine_path);
}

CLIPWrapper::~CLIPWrapper()
{
  context_.reset();
}

void CLIPWrapper::loadEngine(const std::string& engine_path)
{
  context_.reset();
  // Get engine from cache (loads from disk if not cached)
  cached_engine_ = getCachedEngine(engine_path);
  if(!cached_engine_ || !cached_engine_->isValid())
  {
    throw std::runtime_error("Failed to load CLIP engine from cache: " + engine_path);
  }

  // Create our own execution context from the cached engine
  context_.reset(cached_engine_->createExecutionContext());
  if(!context_)
  {
    throw std::runtime_error("Failed to create CLIP execution context");
  }
}

bool CLIPWrapper::needsReallocation(int batch)
{
  return !shape_cache_.initialized || shape_cache_.batch != batch;
}

void CLIPWrapper::encode(
    const int32_t* input_ids, __half* text_embeddings, int batch, cudaStream_t stream,
    __half* pooler_output)
{
  assert(cached_engine_ && cached_engine_->isValid());
  // Set input shape [batch, 77]
  nvinfer1::Dims input_dims;
  input_dims.nbDims = 2;
  input_dims.d[0] = batch;
  input_dims.d[1] = 77; // CLIP sequence length

  if(needsReallocation(batch))
  {
    if(!context_->setInputShape("input_ids", input_dims))
    {
      throw std::runtime_error("Failed to set CLIP input_ids shape");
    }

    shape_cache_.batch = batch;
    shape_cache_.initialized = true;
  }

  if(!context_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Not all CLIP input dimensions specified");
  }

  // Query output dimensions from engine to support both SD1.5 (768) and SD-Turbo (1024)
  nvinfer1::Dims output_dims = context_->getTensorShape("text_embeddings");
  if(output_dims.nbDims != 3)
  {
    throw std::runtime_error("Unexpected CLIP output dimensions");
  }

  int hidden_dim = output_dims.d[2];  // Extract hidden dimension from engine

  // librediffusion fix: honor the engine's declared output dtype. Prebuilt SDXL / penultimate
  // CLIP engines emit text_embeddings in FP16; the legacy code assumed FP32 and ran an extra
  // FP32->FP16 conversion, reinterpreting the FP16 bytes as FP32 and producing inf/garbage.
  // When the engine output is already HALF, bind the caller's FP16 buffer directly.
  auto* eng = cached_engine_->getEngine();
  const bool text_is_half
      = (eng->getTensorDataType("text_embeddings") == nvinfer1::DataType::kHALF);

  // Allocate temporary FP32 buffer for engine output only when the engine emits FP32.
  size_t output_size = batch * 77 * hidden_dim;
  float* d_output_fp32 = nullptr;
  if(!text_is_half)
    cudaMalloc(&d_output_fp32, output_size * sizeof(float));

  // Check if engine has pooler_output (SDXL CLIP encoder 2)
  float* d_pooler_fp32 = nullptr;
  bool has_pooler = false;
  bool pooler_is_half = false;

  if(pooler_output)
  {
    has_pooler = true;
    pooler_is_half
        = (eng->getTensorDataType("pooler_output") == nvinfer1::DataType::kHALF);
    if(!pooler_is_half)
    {
      nvinfer1::Dims pooler_dims = context_->getTensorShape("pooler_output");
      size_t pooler_size = batch * pooler_dims.d[1];
      cudaMalloc(&d_pooler_fp32, pooler_size * sizeof(float));
    }
  }

  auto cleanup = [&]() {
    if(d_output_fp32) cudaFree(d_output_fp32);
    if(d_pooler_fp32) cudaFree(d_pooler_fp32);
  };

  // Set tensor addresses
  // Note: TensorRT API requires non-const void*, so we cast away const
  if(!context_->setTensorAddress("input_ids", const_cast<int32_t*>(input_ids)))
  {
    cleanup();
    throw std::runtime_error("Failed to set CLIP input_ids tensor address");
  }
  void* text_addr = text_is_half ? (void*)text_embeddings : (void*)d_output_fp32;
  if(!context_->setTensorAddress("text_embeddings", text_addr))
  {
    cleanup();
    throw std::runtime_error("Failed to set CLIP text_embeddings tensor address");
  }

  // Set pooler_output address if present
  if(has_pooler)
  {
    void* pooler_addr = pooler_is_half ? (void*)pooler_output : (void*)d_pooler_fp32;
    if(!context_->setTensorAddress("pooler_output", pooler_addr))
    {
      cleanup();
      throw std::runtime_error("Failed to set CLIP pooler_output tensor address");
    }
  }

  // Enqueue inference
  if(!context_->enqueueV3(stream))
  {
    cleanup();
    throw std::runtime_error("Failed to enqueue CLIP inference");
  }

  // Convert FP32 output to FP16 only when the engine emitted FP32.
  if(!text_is_half)
    launch_fp32_to_fp16(d_output_fp32, text_embeddings, output_size, stream);

  if(pooler_output && has_pooler && !pooler_is_half && d_pooler_fp32)
  {
    nvinfer1::Dims pooler_dims = context_->getTensorShape("pooler_output");
    size_t pooler_size = batch * pooler_dims.d[1];
    launch_fp32_to_fp16(d_pooler_fp32, pooler_output, pooler_size, stream);
  }

  // Wait for conversion to complete and cleanup
  cudaStreamSynchronize(stream);
  cleanup();
}

__half* CLIPWrapper::computeEmbeddings(const std::string& prompt, cudaStream_t stream, int pad_token)
{
  // Tokenize the prompt using Rust tokenizer with configurable padding
  std::vector<int32_t> token_ids_host(77);
  int num_tokens = clip_tokenizer_encode_with_padding(prompt.c_str(), token_ids_host.data(), pad_token);

  if(num_tokens < 0)
  {
    throw std::runtime_error("Failed to tokenize prompt: " + prompt);
  }

  // Allocate device memory for token IDs
  int32_t* d_token_ids;
  cudaMalloc(&d_token_ids, num_tokens * sizeof(int32_t));
  cudaMemcpy(
      d_token_ids, token_ids_host.data(), num_tokens * sizeof(int32_t),
      cudaMemcpyHostToDevice);

  // Query output dimensions from engine to support both SD1.5 (768) and SD-Turbo (1024)
  // Need to set input shape first to get valid output shape
  nvinfer1::Dims input_dims;
  input_dims.nbDims = 2;
  input_dims.d[0] = 1;  // batch = 1
  input_dims.d[1] = 77; // sequence length
  context_->setInputShape("input_ids", input_dims);

  nvinfer1::Dims output_dims = context_->getTensorShape("text_embeddings");
  int hidden_dim = output_dims.d[2];  // Extract hidden dimension from engine

  // Allocate device memory for embeddings: [1, 77, hidden_dim]
  __half* d_embeddings;
  size_t embedding_size = 1 * 77 * hidden_dim * sizeof(__half);
  cudaMalloc(&d_embeddings, embedding_size);

  // Run CLIP inference
  this->encode(d_token_ids, d_embeddings, 1, stream);
  cudaStreamSynchronize(stream);

  // Cleanup token IDs
  cudaFree(d_token_ids);

  return d_embeddings;
}

__half* CLIPWrapper::computeEmbeddingsWithPooled(
    const std::string& prompt, cudaStream_t stream, int pad_token,
    __half** pooled_output)
{
  // Tokenize the prompt using Rust tokenizer with configurable padding
  std::vector<int32_t> token_ids_host(77);
  int num_tokens = clip_tokenizer_encode_with_padding(prompt.c_str(), token_ids_host.data(), pad_token);

  if(num_tokens < 0)
  {
    throw std::runtime_error("Failed to tokenize prompt: " + prompt);
  }

  // Allocate device memory for token IDs
  int32_t* d_token_ids;
  cudaMalloc(&d_token_ids, num_tokens * sizeof(int32_t));
  cudaMemcpy(
      d_token_ids, token_ids_host.data(), num_tokens * sizeof(int32_t),
      cudaMemcpyHostToDevice);

  // Set input dimensions first (required before querying output shapes)
  nvinfer1::Dims input_dims;
  input_dims.nbDims = 2;
  input_dims.d[0] = 1;  // batch = 1
  input_dims.d[1] = 77; // sequence length
  if(!context_->setInputShape("input_ids", input_dims))
  {
    cudaFree(d_token_ids);
    throw std::runtime_error("Failed to set input_ids shape");
  }

  if(!context_->allInputDimensionsSpecified())
  {
    cudaFree(d_token_ids);
    throw std::runtime_error("Not all input dimensions specified");
  }

  // Check what kind of CLIP engine this is by checking the text_embeddings output shape
  // CLIP1 (encoder 1): text_embeddings is [batch, 77, 768] - sequence embeddings
  // CLIP2 (encoder 2): has hidden_states [batch, 77, 1280] + text_embeddings [batch, 1280]
  //
  // We detect CLIP2 by checking if text_embeddings is 2D (pooled) instead of 3D (sequence)

  __half* d_sequence_embeddings = nullptr;
  __half* d_pooled_embeddings = nullptr;

  nvinfer1::Dims text_embeddings_dims = context_->getTensorShape("text_embeddings");

  if(text_embeddings_dims.nbDims == 2)
  {
    // CLIP2: text_embeddings is [batch, 1280] (pooled), also has hidden_states [batch, 77, 1280]
    nvinfer1::Dims hidden_states_dims = context_->getTensorShape("hidden_states");

    if(hidden_states_dims.nbDims != 3)
    {
      cudaFree(d_token_ids);
      throw std::runtime_error("CLIP2 hidden_states has unexpected dimensions");
    }

    int hidden_dim = hidden_states_dims.d[2];  // Should be 1280 for CLIP2
    int pooled_dim = text_embeddings_dims.d[1]; // Should be 1280 for CLIP2

    // librediffusion fix: honor the engine's declared output dtype for hidden_states.
    // Prebuilt SDXL CLIP2 emits hidden_states in FP16; the legacy code hard-assumed FP32
    // and ran an extra FP32->FP16 conversion, reinterpreting the FP16 bytes and producing
    // inf/garbage. (pooled text_embeddings was already bound as FP16, which is why it stayed
    // correct.) When hidden_states is HALF, bind the FP16 buffer directly; else keep FP32 path.
    auto* eng = cached_engine_->getEngine();
    const bool hidden_is_half
        = (eng->getTensorDataType("hidden_states") == nvinfer1::DataType::kHALF);

    // Allocate output buffers
    size_t sequence_size = 1 * 77 * hidden_dim;
    size_t pooled_size = 1 * pooled_dim;

    float* d_hidden_fp32 = nullptr;  // only used if engine emits FP32 hidden_states
    if(!hidden_is_half)
      cudaMalloc(&d_hidden_fp32, sequence_size * sizeof(float));
    cudaMalloc(&d_pooled_embeddings, pooled_size * sizeof(__half));  // text_embeddings is FP16
    cudaMalloc(&d_sequence_embeddings, sequence_size * sizeof(__half));  // final FP16 output

    auto clip2_cleanup = [&]() {
      cudaFree(d_token_ids);
      if(d_hidden_fp32) cudaFree(d_hidden_fp32);
      cudaFree(d_sequence_embeddings);
      cudaFree(d_pooled_embeddings);
    };

    // Set tensor addresses - mixed precision
    if(!context_->setTensorAddress("input_ids", d_token_ids))
    {
      clip2_cleanup();
      throw std::runtime_error("Failed to set CLIP2 input_ids tensor address");
    }
    void* hidden_addr = hidden_is_half ? (void*)d_sequence_embeddings : (void*)d_hidden_fp32;
    if(!context_->setTensorAddress("hidden_states", hidden_addr))
    {
      clip2_cleanup();
      throw std::runtime_error("Failed to set CLIP2 hidden_states tensor address");
    }
    if(!context_->setTensorAddress("text_embeddings", d_pooled_embeddings))  // Engine writes FP16 here
    {
      clip2_cleanup();
      throw std::runtime_error("Failed to set CLIP2 text_embeddings tensor address");
    }

    // Run inference
    if(!context_->enqueueV3(stream))
    {
      clip2_cleanup();
      throw std::runtime_error("Failed to enqueue CLIP2 inference");
    }

    // Convert FP32 hidden_states to FP16 only when the engine emitted FP32.
    if(!hidden_is_half)
      launch_fp32_to_fp16(d_hidden_fp32, d_sequence_embeddings, sequence_size, stream);

    cudaStreamSynchronize(stream);

    // Cleanup
    cudaFree(d_token_ids);
    if(d_hidden_fp32) cudaFree(d_hidden_fp32);  // Free the FP32 intermediate buffer (if any)

    // Return both outputs
    if(pooled_output)
    {
      *pooled_output = d_pooled_embeddings;
    }
    else
    {
      cudaFree(d_pooled_embeddings);
    }

    return d_sequence_embeddings;
  }
  else
  {
    // CLIP1: Only has text_embeddings output [batch, 77, 768]
    nvinfer1::Dims output_dims = context_->getTensorShape("text_embeddings");

    if(output_dims.nbDims != 3)
    {
      cudaFree(d_token_ids);
      throw std::runtime_error("CLIP1 text_embeddings has unexpected dimensions");
    }

    int hidden_dim = output_dims.d[2];  // Should be 768 for CLIP1

    // Allocate output buffer
    size_t embedding_size = 1 * 77 * hidden_dim;
    cudaMalloc(&d_sequence_embeddings, embedding_size * sizeof(__half));

    // Run inference using existing encode method
    this->encode(d_token_ids, d_sequence_embeddings, 1, stream, nullptr);

    cudaStreamSynchronize(stream);
    cudaFree(d_token_ids);

    // CLIP1 has no pooled output
    if(pooled_output)
    {
      *pooled_output = nullptr;
    }

    return d_sequence_embeddings;
  }
}

SDXLPromptEmbeddings computeClipEmbeddings_SDXL(
    CLIPWrapper& clip, CLIPWrapper& clip2, const std::string& prompt, int batch_size,
    int height, int width, cudaStream_t clip_stream)
{
  SDXLPromptEmbeddings ret;

  // CLIP encoder 1: 768-dim with EOS padding (49407)
  __half* d_embeds1 = clip.computeEmbeddings(prompt, clip_stream, 49407);

  // CLIP encoder 2: 1280-dim with pooled output, PAD padding (0)
  __half* d_embeds2
      = clip2.computeEmbeddingsWithPooled(prompt, clip_stream, 0, &ret.pooled_embeds);

  // Concatenate embeddings: [batch, 77, 768+1280] = [batch, 77, 2048]
  size_t total_size = batch_size * 77 * 2048;

  cudaMalloc(&ret.embeddings, total_size * sizeof(__half));

  // Copy embeddings sequentially for each batch element and sequence position
  for(int b = 0; b < batch_size; ++b)
  {
    for(int s = 0; s < 77; ++s)
    {
      // Copy 768-dim from encoder 1
      cudaMemcpy(
          ret.embeddings + (b * 77 + s) * 2048, d_embeds1 + (b * 77 + s) * 768,
          768 * sizeof(__half), cudaMemcpyDeviceToDevice);
      // Copy 1280-dim from encoder 2
      cudaMemcpy(
          ret.embeddings + (b * 77 + s) * 2048 + 768, d_embeds2 + (b * 77 + s) * 1280,
          1280 * sizeof(__half), cudaMemcpyDeviceToDevice);
    }
  }

  cudaFree(d_embeds1);
  cudaFree(d_embeds2);

  // Prepare time_ids: [original_height, original_width, crop_top, crop_left, target_height, target_width]
  std::vector<__half> time_ids_host(batch_size * 6);
  for(int i = 0; i < batch_size; ++i)
  {
    time_ids_host[i * 6 + 0] = __half(static_cast<float>(height)); // original_height
    time_ids_host[i * 6 + 1] = __half(static_cast<float>(width));  // original_width
    time_ids_host[i * 6 + 2] = __half(0.0f);                       // crop_top
    time_ids_host[i * 6 + 3] = __half(0.0f);                       // crop_left
    time_ids_host[i * 6 + 4] = __half(static_cast<float>(height)); // target_height
    time_ids_host[i * 6 + 5] = __half(static_cast<float>(width));  // target_width
  }

  cudaMalloc(&ret.time_ids, batch_size * 6 * sizeof(__half));
  cudaMemcpy(
      ret.time_ids, time_ids_host.data(), batch_size * 6 * sizeof(__half),
      cudaMemcpyHostToDevice);

  return ret;
}

// ============================================================================
// CLIPImageEncoderWrapper (IP-Adapter on-device image encode + projection)
// ============================================================================

CLIPImageEncoderWrapper::CLIPImageEncoderWrapper(
    const std::string& encoder_engine_path, const std::string& proj_engine_path)
{
  loadEngines(encoder_engine_path, proj_engine_path);
}

CLIPImageEncoderWrapper::~CLIPImageEncoderWrapper()
{
  proj_context_.reset();
  enc_context_.reset();
}

void CLIPImageEncoderWrapper::loadEngines(
    const std::string& encoder_engine_path, const std::string& proj_engine_path)
{
  enc_engine_ = getCachedEngine(encoder_engine_path);
  if(!enc_engine_ || !enc_engine_->isValid())
    throw std::runtime_error(
        "Failed to load IP-Adapter image encoder engine: " + encoder_engine_path);
  enc_context_.reset(enc_engine_->createExecutionContext());
  if(!enc_context_)
    throw std::runtime_error("Failed to create IP-Adapter image encoder context");

  proj_engine_ = getCachedEngine(proj_engine_path);
  if(!proj_engine_ || !proj_engine_->isValid())
    throw std::runtime_error(
        "Failed to load IP-Adapter image projection engine: " + proj_engine_path);
  proj_context_.reset(proj_engine_->createExecutionContext());
  if(!proj_context_)
    throw std::runtime_error("Failed to create IP-Adapter image projection context");

  // Discover num_tokens / token_dim from the proj engine's ip_tokens output [B, num_tokens, dim].
  nvinfer1::Dims in_dims;
  in_dims.nbDims = 2;
  in_dims.d[0] = 1;
  in_dims.d[1] = 1024; // CLIP projection_dim
  proj_context_->setInputShape("image_embeds", in_dims);
  nvinfer1::Dims out = proj_context_->getTensorShape("ip_tokens");
  if(out.nbDims == 3)
  {
    num_tokens_ = out.d[1];
    token_dim_ = out.d[2];
  }
  else
  {
    num_tokens_ = 4;
    token_dim_ = 768;
  }
}

void CLIPImageEncoderWrapper::encodeImage(
    const uint8_t* cpu_rgba, int in_h, int in_w, __half* pos_out, __half* neg_out,
    cudaStream_t stream)
{
  // 1. Upload host RGBA and preprocess (Lanczos-3 resize 224 + CLIP normalize) -> [1,3,224,224] fp16.
  size_t rgba_n = (size_t)in_h * in_w * 4;
  if(!d_rgba_ || d_rgba_->size() < rgba_n)
    d_rgba_ = std::make_unique<CUDATensor<uint8_t>>(rgba_n);
  cudaMemcpyAsync(
      d_rgba_->data(), cpu_rgba, rgba_n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);

  const size_t pixel_n = (size_t)1 * 3 * 224 * 224;
  if(!d_pixel_)
    d_pixel_ = std::make_unique<CUDATensor<__half>>(pixel_n);
  launch_clip_image_preprocess_fp16(
      d_rgba_->data(), d_pixel_->data(), in_h, in_w, stream);

  // 2. CLIP image encode: pixel_values [1,3,224,224] -> image_embeds [1,1024].
  if(!d_image_embeds_)
    d_image_embeds_ = std::make_unique<CUDATensor<__half>>(1024);
  {
    nvinfer1::Dims d;
    d.nbDims = 4;
    d.d[0] = 1; d.d[1] = 3; d.d[2] = 224; d.d[3] = 224;
    if(!enc_context_->setInputShape("pixel_values", d))
      throw std::runtime_error("Failed to set CLIP image encoder pixel_values shape");
    if(!enc_context_->allInputDimensionsSpecified())
      throw std::runtime_error("CLIP image encoder: not all input dims specified");
    enc_context_->setTensorAddress("pixel_values", d_pixel_->data());
    enc_context_->setTensorAddress("image_embeds", d_image_embeds_->data());
    if(!enc_context_->enqueueV3(stream))
      throw std::runtime_error("Failed to enqueue CLIP image encoder");
  }

  // 3a. Project the image_embeds -> positive tokens [num_tokens, dim].
  auto run_proj = [&](const __half* embeds, __half* out) {
    nvinfer1::Dims d;
    d.nbDims = 2;
    d.d[0] = 1; d.d[1] = 1024;
    if(!proj_context_->setInputShape("image_embeds", d))
      throw std::runtime_error("Failed to set ip_image_proj image_embeds shape");
    if(!proj_context_->allInputDimensionsSpecified())
      throw std::runtime_error("ip_image_proj: not all input dims specified");
    proj_context_->setTensorAddress("image_embeds", const_cast<__half*>(embeds));
    proj_context_->setTensorAddress("ip_tokens", out);
    if(!proj_context_->enqueueV3(stream))
      throw std::runtime_error("Failed to enqueue ip_image_proj");
  };
  run_proj(d_image_embeds_->data(), pos_out);

  // 3b. NEGATIVE tokens = projection of ZEROS image_embeds (encoder NOT re-run; base IP-Adapter).
  if(neg_out)
  {
    if(!d_zero_embeds_)
    {
      d_zero_embeds_ = std::make_unique<CUDATensor<__half>>(1024);
      cudaMemsetAsync(d_zero_embeds_->data(), 0, 1024 * sizeof(__half), stream);
    }
    run_proj(d_zero_embeds_->data(), neg_out);
  }

  cudaStreamSynchronize(stream);
}

} // namespace librediffusion
