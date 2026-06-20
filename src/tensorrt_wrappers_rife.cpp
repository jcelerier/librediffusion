/** RIFE (IFNet) frame-interpolation TensorRT wrapper implementation. */
#include "tensorrt_wrappers_rife.hpp"

#include "model_cache.hpp"

#include <stdexcept>

namespace librediffusion
{

namespace
{
nvinfer1::Dims dims4(int a, int b, int c, int e)
{
  nvinfer1::Dims d; d.nbDims = 4; d.d[0] = a; d.d[1] = b; d.d[2] = c; d.d[3] = e; return d;
}
} // namespace

RifeWrapper::RifeWrapper(const std::string& p) { loadEngine(p); }
RifeWrapper::~RifeWrapper() = default;

void RifeWrapper::loadEngine(const std::string& p)
{
  cached_engine_ = getCachedEngine(p);
  if(!cached_engine_ || !cached_engine_->isValid())
    throw std::runtime_error("rife: failed to load engine: " + p);
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(cached_engine_->createExecutionContext());
  if(!context_)
    throw std::runtime_error("rife: failed to create context: " + p);
}

void RifeWrapper::interpolate(
    const __half* frames, __half* mid, int batch, int H, int W, cudaStream_t stream)
{
  auto* c = context_.get();
  if(!c->setInputShape("frames", dims4(batch, 6, H, W)))
    throw std::runtime_error("rife: setInputShape(frames) failed");
  if(!c->setTensorAddress("frames", const_cast<__half*>(frames)))
    throw std::runtime_error("rife: setTensorAddress(frames) failed");
  if(!c->setTensorAddress("mid", mid))
    throw std::runtime_error("rife: setTensorAddress(mid) failed");
  if(!c->enqueueV3(stream))
    throw std::runtime_error("rife: enqueueV3 failed");
}

} // namespace librediffusion
