/** FLUX.2-klein-4B TensorRT wrappers implementation. */
#include "tensorrt_wrappers_klein.hpp"

#include "model_cache.hpp"

#include <stdexcept>

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
} // namespace

// ---------------- Flux2Transformer ----------------
Flux2TransformerWrapper::Flux2TransformerWrapper(const std::string& p) { loadEngine(p); }
Flux2TransformerWrapper::~Flux2TransformerWrapper() = default;
void Flux2TransformerWrapper::loadEngine(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
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

// ---------------- Qwen3Encoder ----------------
Qwen3EncoderWrapper::Qwen3EncoderWrapper(const std::string& p) { loadEngine(p); }
Qwen3EncoderWrapper::~Qwen3EncoderWrapper() = default;
void Qwen3EncoderWrapper::loadEngine(const std::string& p)
{
  context_ = makeContext(cached_engine_, p);
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
