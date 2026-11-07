/** FLUX.2-klein-4B TensorRT wrappers implementation. */
#include "tensorrt_wrappers_klein.hpp"

#include "model_cache.hpp"

#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace librediffusion
{

namespace
{
std::unique_ptr<nvinfer1::IExecutionContext>
makeContext(std::shared_ptr<CachedTensorRTEngine>& eng, const std::string& path)
{
  eng = getCachedEngine(path);
  if(!eng || !eng->isValid())
    throw std::runtime_error("klein: failed to load engine: " + path);
  auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(eng->createExecutionContext());
  if(!ctx)
    throw std::runtime_error("klein: failed to create context: " + path);
  return ctx;
}

const char* dtypeName(nvinfer1::DataType t)
{
  switch(t)
  {
    case nvinfer1::DataType::kFLOAT: return "fp32";
    case nvinfer1::DataType::kHALF: return "fp16";
    case nvinfer1::DataType::kBF16: return "bf16";
    case nvinfer1::DataType::kINT64: return "int64";
    case nvinfer1::DataType::kINT32: return "int32";
    default: return "another type";
  }
}

// The forward()s bind raw device buffers sized for one element type, so an engine whose I/O differs
// (an older exporter wrote the qwen output as fp32) would be written past the end of its buffer: an
// illegal address that loses the CUDA context. Refuse it at load instead.
using TensorTypes = std::initializer_list<std::pair<const char*, nvinfer1::DataType>>;
void requireTypes(const CachedTensorRTEngine& eng, const std::string& path, TensorTypes tensors)
{
  const nvinfer1::ICudaEngine* e = eng.getEngine();
  for(const auto& [name, type] : tensors)
  {
    bool found = false;
    for(int i = 0; i < e->getNbIOTensors() && !found; ++i)
      found = std::string(e->getIOTensorName(i)) == name;
    if(!found)
      throw std::runtime_error(
          "klein: " + path + " has no '" + name + "' tensor; rebuild it with train-lora.py");
    if(const auto actual = e->getTensorDataType(name); actual != type)
      throw std::runtime_error(
          "klein: " + path + ": '" + name + "' is " + dtypeName(actual) + ", expected "
          + dtypeName(type) + "; rebuild it with train-lora.py");
  }
}

void bindIn(nvinfer1::IExecutionContext* ctx, const char* name, const void* ptr, nvinfer1::Dims d)
{
  if(!ctx->setInputShape(name, d))
    throw std::runtime_error(std::string("klein: setInputShape failed: ") + name);
  if(!ctx->setTensorAddress(name, const_cast<void*>(ptr)))
    throw std::runtime_error(std::string("klein: setTensorAddress(in) failed: ") + name);
}

void bindOut(nvinfer1::IExecutionContext* ctx, const char* name, void* ptr)
{
  if(!ctx->setTensorAddress(name, ptr))
    throw std::runtime_error(std::string("klein: setTensorAddress(out) failed: ") + name);
}

nvinfer1::Dims dims2(int a, int b)
{
  nvinfer1::Dims d; d.nbDims = 2; d.d[0] = a; d.d[1] = b; return d;
}
nvinfer1::Dims dims3(int a, int b, int c)
{
  nvinfer1::Dims d; d.nbDims = 3; d.d[0] = a; d.d[1] = b; d.d[2] = c; return d;
}
nvinfer1::Dims dims1(int a)
{
  nvinfer1::Dims d; d.nbDims = 1; d.d[0] = a; return d;
}
nvinfer1::Dims dims4(int a, int b, int c, int e)
{
  nvinfer1::Dims d; d.nbDims = 4; d.d[0] = a; d.d[1] = b; d.d[2] = c; d.d[3] = e; return d;
}
// [1, L, 64, 2, 2] RoPE rotation table
nvinfer1::Dims dims_rope(int L)
{
  nvinfer1::Dims d; d.nbDims = 5; d.d[0] = 1; d.d[1] = L; d.d[2] = 64; d.d[3] = 2; d.d[4] = 2; return d;
}
} // namespace

// ---------------- Flux2Transformer ----------------
Flux2TransformerWrapper::Flux2TransformerWrapper(const std::string& p) { loadEngine(p); }
Flux2TransformerWrapper::~Flux2TransformerWrapper() = default;
void Flux2TransformerWrapper::loadEngine(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
  requireTypes(
      *cached_engine_, p,
      {{"hidden_states", nvinfer1::DataType::kBF16},
       {"encoder_hidden_states", nvinfer1::DataType::kBF16},
       {"timestep", nvinfer1::DataType::kFLOAT},
       {"img_ids", nvinfer1::DataType::kFLOAT},
       {"txt_ids", nvinfer1::DataType::kFLOAT},
       {"velocity", nvinfer1::DataType::kBF16}});
}

void Flux2TransformerWrapper::forward(
    const __nv_bfloat16* hidden_states, const __nv_bfloat16* encoder_hidden_states,
    const float* timestep, const float* img_ids, const float* txt_ids,
    __nv_bfloat16* velocity, int batch, int Lp, int Lt, int D, cudaStream_t stream)
{
  auto* c = context_.get();
  bindIn(c, "hidden_states", hidden_states, dims3(batch, Lp, 128));
  bindIn(c, "encoder_hidden_states", encoder_hidden_states, dims3(batch, Lt, D));
  bindIn(c, "timestep", timestep, dims1(batch));
  bindIn(c, "img_ids", img_ids, dims3(batch, Lp, 4));
  bindIn(c, "txt_ids", txt_ids, dims3(batch, Lt, 4));
  bindOut(c, "velocity", velocity);
  if(!c->enqueueV3(stream))
    throw std::runtime_error("klein: transformer enqueueV3 failed");
}

// ---------------- Flux3Transformer ----------------
Flux3TransformerWrapper::Flux3TransformerWrapper(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
  requireTypes(
      *cached_engine_, p,
      {{"img", nvinfer1::DataType::kBF16},
       {"cond", nvinfer1::DataType::kBF16},
       {"txt", nvinfer1::DataType::kBF16},
       {"rope_img", nvinfer1::DataType::kFLOAT},
       {"rope_cond", nvinfer1::DataType::kFLOAT},
       {"rope_txt", nvinfer1::DataType::kFLOAT},
       {"t", nvinfer1::DataType::kFLOAT},
       {"r", nvinfer1::DataType::kFLOAT},
       {"velocity", nvinfer1::DataType::kBF16}});
}
Flux3TransformerWrapper::~Flux3TransformerWrapper() = default;

int Flux3TransformerWrapper::max_cond_tokens() const
{
  const auto d = cached_engine_->getEngine()->getProfileShape(
      "cond", 0, nvinfer1::OptProfileSelector::kMAX);
  return d.nbDims == 3 ? (int)d.d[1] : 0;
}

void Flux3TransformerWrapper::forward(
    const __nv_bfloat16* img, const __nv_bfloat16* cond, const __nv_bfloat16* txt,
    const float* rope_img, const float* rope_cond, const float* rope_txt, const float* t,
    const float* r, __nv_bfloat16* velocity, int Li, int Lc, int Lt, cudaStream_t stream)
{
  auto* c = context_.get();
  bindIn(c, "img", img, dims3(1, Li, 128));
  bindIn(c, "cond", cond, dims3(1, Lc, 128));
  bindIn(c, "txt", txt, dims3(1, Lt, 20480));
  bindIn(c, "rope_img", rope_img, dims_rope(Li));
  bindIn(c, "rope_cond", rope_cond, dims_rope(Lc));
  bindIn(c, "rope_txt", rope_txt, dims_rope(Lt));
  bindIn(c, "t", t, dims1(1));
  bindIn(c, "r", r, dims1(1));
  bindOut(c, "velocity", velocity);
  if(!c->enqueueV3(stream))
    throw std::runtime_error("flux3: transformer enqueueV3 failed");
}

// ---------------- Qwen3Encoder ----------------
Qwen3EncoderWrapper::Qwen3EncoderWrapper(const std::string& p) { loadEngine(p); }
Qwen3EncoderWrapper::~Qwen3EncoderWrapper() = default;
void Qwen3EncoderWrapper::loadEngine(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
  requireTypes(
      *cached_engine_, p,
      {{"input_ids", nvinfer1::DataType::kINT64},
       {"attention_mask", nvinfer1::DataType::kINT64},
       {"encoder_hidden_states", nvinfer1::DataType::kBF16}});
}

void Qwen3EncoderWrapper::forward(
    const int64_t* input_ids, const int64_t* attention_mask, __nv_bfloat16* encoder_hidden_states,
    int batch, int Lt, int D, cudaStream_t stream)
{
  auto* c = context_.get();
  bindIn(c, "input_ids", input_ids, dims2(batch, Lt));
  bindIn(c, "attention_mask", attention_mask, dims2(batch, Lt));
  bindOut(c, "encoder_hidden_states", encoder_hidden_states);
  if(!c->enqueueV3(stream))
    throw std::runtime_error("klein: qwen enqueueV3 failed");
}

// ---------------- KleinVAEEncoder ----------------
KleinVAEEncoderWrapper::KleinVAEEncoderWrapper(const std::string& p) { loadEngine(p); }
KleinVAEEncoderWrapper::~KleinVAEEncoderWrapper() = default;
void KleinVAEEncoderWrapper::loadEngine(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
  requireTypes(*cached_engine_, p, {{"image", nvinfer1::DataType::kBF16}, {"latent", nvinfer1::DataType::kBF16}});
}
void KleinVAEEncoderWrapper::encode(
    const __nv_bfloat16* image, __nv_bfloat16* latent, int batch, int H, int W, cudaStream_t stream)
{
  auto* c = context_.get();
  bindIn(c, "image", image, dims4(batch, 3, H, W));
  bindOut(c, "latent", latent);
  if(!c->enqueueV3(stream))
    throw std::runtime_error("klein: vae encode enqueueV3 failed");
}

// ---------------- KleinVAEDecoder ----------------
KleinVAEDecoderWrapper::KleinVAEDecoderWrapper(const std::string& p) { loadEngine(p); }
KleinVAEDecoderWrapper::~KleinVAEDecoderWrapper() = default;
void KleinVAEDecoderWrapper::loadEngine(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
  requireTypes(*cached_engine_, p, {{"latent", nvinfer1::DataType::kBF16}, {"image", nvinfer1::DataType::kBF16}});
}
void KleinVAEDecoderWrapper::decode(
    const __nv_bfloat16* latent, __nv_bfloat16* image, int batch, int h, int w, cudaStream_t stream)
{
  auto* c = context_.get();
  bindIn(c, "latent", latent, dims4(batch, 32, h, w));
  bindOut(c, "image", image);
  if(!c->enqueueV3(stream))
    throw std::runtime_error("klein: vae decode enqueueV3 failed");
}

} // namespace librediffusion
