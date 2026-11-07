/** SANA-Streaming TRT wrapper implementation (see .hpp).
 *  Pure C++ (compiled by cl.exe in the DLL, and by nvcc in the standalone in-process test):
 *  hosts the NvInfer.h C++ runtime. The fp32<->bf16 casts at the bf16-engine boundary are
 *  done through the launch_trt_{f2bf,bf2f} wrappers defined in the nvcc unit
 *  librediffusion.sana.cu (no device kernels live here). Links nvinfer_11 (already a
 *  dependency of the librediffusion DLL). Engine deserialize -> enqueueV3, name-based.
 */
#include "NvInfer.h"
#include "cuda_tensor.hpp"
#include "librediffusion.sana_kernels.hpp"
#include "librediffusion.sana_trt.hpp"
#include "model_cache.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace librediffusion
{
namespace
{

#define TCK(x)                                                                  \
  do                                                                            \
  {                                                                             \
    cudaError_t e = (x);                                                        \
    if(e)                                                                       \
    {                                                                           \
      printf("[sana_trt] CUDA err %s @ %d\n", cudaGetErrorString(e), __LINE__); \
      return -1;                                                                \
    }                                                                           \
  } while(0)

// Product of all dimensions of a TensorRT Dims (element count of a tensor).
static size_t vol(const Dims& d)
{
  size_t v = 1;
  for(int i = 0; i < d.nbDims; i++)
    v *= (size_t)d.d[i];
  return v;
}

// Profiling toggle: set env LRD_PROF=1 to print per-stage deserialize/run timings.
static bool profOn()
{
  static int v = (getenv("LRD_PROF") && atoi(getenv("LRD_PROF"))) ? 1 : 0;
  return v;
}
// Monotonic wall-clock timestamp in milliseconds (for the LRD_PROF stage timings).
static double nowMs()
{
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// One input the caller supplies as a device buffer (fp32 or int64); `key` matches a
// substring of the engine input tensor name ("" = bind to the next unbound input).
struct EIn
{
  std::string key;
  const void* dev;
  DataType src;
  std::vector<int> shape;
};

// A cached engine (shared, owned by GlobalEngineCache) kept resident with its own execution
// context + stream and reused across calls. Binding tensor addresses + enqueueV3 is all that
// happens per call. The ICudaEngine/IRuntime lifetime is the shared cache's, not ours.
struct EngineRec
{
  std::shared_ptr<CachedTensorRTEngine> cached;
  std::unique_ptr<IExecutionContext> ctx;
  cudaStream_t stream = nullptr;
  ~EngineRec()
  {
    // ctx released by unique_ptr; the engine/runtime remain owned by the shared cache.
    if(stream)
      cudaStreamDestroy(stream);
  }
};

// Fetch `plan` from the shared GlobalEngineCache (deserializing once, LRU-managed), then create
// a resident execution context + stream for it. Returns null on failure.
static EngineRec* loadEngine(const std::string& plan)
{
  double t0 = nowMs();
  auto rec = new EngineRec();
  rec->cached = getCachedEngine(plan);
  if(!rec->cached || !rec->cached->isValid())
  {
    printf("[sana_trt] deserialize failed: %s\n", plan.c_str());
    delete rec;
    return nullptr;
  }
  rec->ctx.reset(rec->cached->createExecutionContext());
  if(!rec->ctx)
  {
    printf("[sana_trt] context create failed: %s\n", plan.c_str());
    delete rec;
    return nullptr;
  }
  if(cudaStreamCreate(&rec->stream))
  {
    printf("[sana_trt] stream create failed\n");
    delete rec;
    return nullptr;
  }
  if(profOn())
    printf("[prof] deserialize %-38s %.1f ms\n", plan.c_str(), nowMs() - t0);
  return rec;
}

// Bind inputs (casting fp32->bf16 where the engine wants bf16), run, and convert the single
// output tensor to fp32 into out_dev, on the resident engine `rec`. Does NOT free the engine.
static int runEngineRec(
    EngineRec* rec, const std::string& plan, std::vector<EIn>& ins, float* out_dev)
{
  double tr = nowMs();
  ICudaEngine* eng = rec->cached->getEngine();
  IExecutionContext* ctx = rec->ctx.get();
  cudaStream_t stream = rec->stream;

  // RAII bf16 scratch buffers we allocate for fp32<->bf16 cast copies (freed on return).
  // reserve() so emplace_back never reallocates (keeps already-bound device pointers stable).
  std::vector<CUDATensor<__nv_bfloat16>> owned;
  owned.reserve(ins.size() + 1);
  std::vector<bool> used(ins.size(), false);
  int rc = -1;
  const char* outName = nullptr;
  DataType outDt = DataType::kFLOAT;
  void* outBuf = nullptr;
  size_t outVol = 0;

  // Match each engine input tensor to a caller EIn (by name substring), set its shape,
  // and bind either the caller buffer directly or an fp32->bf16 converted copy.
  int nIO = eng->getNbIOTensors();
  for(int i = 0; i < nIO; i++)
  {
    const char* name = eng->getIOTensorName(i);
    if(eng->getTensorIOMode(name) != TensorIOMode::kINPUT)
      continue;
    int si = -1;
    for(size_t s = 0; s < ins.size(); s++)
    {
      if(used[s])
        continue;
      if(ins[s].key.empty() || std::string(name).find(ins[s].key) != std::string::npos)
      {
        si = (int)s;
        break;
      }
    }
    if(si < 0)
    {
      printf("[sana_trt] no input for tensor '%s'\n", name);
      goto cleanup;
    }
    used[si] = true;
    EIn& in = ins[si];
    Dims d;
    d.nbDims = (int)in.shape.size();
    for(size_t k = 0; k < in.shape.size(); k++)
      d.d[k] = in.shape[k];
    ctx->setInputShape(name, d);
    size_t v = vol(d);
    DataType edt = eng->getTensorDataType(name);
    void* bind;
    if(edt == in.src)
    {
      bind = const_cast<void*>(in.dev);
    } // same dtype: bind caller buffer
    else if(in.src == DataType::kFLOAT && edt == DataType::kBF16)
    { // fp32 -> bf16
      owned.emplace_back(v); // bf16 scratch (v elements)
      bind = owned.back().data();
      sana::launch_trt_f2bf((const float*)in.dev, bind, (long)v, (void*)stream);
    }
    else
    {
      printf(
          "[sana_trt] unhandled input cast %d->%d for '%s'\n", (int)in.src, (int)edt,
          name);
      goto cleanup;
    }
    ctx->setTensorAddress(name, bind);
  }
  // Bind the single output tensor: fp32 goes straight to out_dev, bf16 to a temp buffer
  // that launch_trt_bf2f converts into out_dev after the run.
  for(int i = 0; i < nIO; i++)
  {
    const char* name = eng->getIOTensorName(i);
    if(eng->getTensorIOMode(name) != TensorIOMode::kOUTPUT)
      continue;
    Dims d = ctx->getTensorShape(name);
    outDt = eng->getTensorDataType(name);
    outVol = vol(d);
    outName = name;
    if(outDt == DataType::kFLOAT)
    {
      outBuf = out_dev;
    } // write straight into caller fp32
    else
    {
      owned.emplace_back(outVol); // bf16 scratch for the engine output
      outBuf = owned.back().data();
    }
    ctx->setTensorAddress(name, outBuf);
  }
  if(!outName)
  {
    printf("[sana_trt] no output tensor\n");
    goto cleanup;
  }
  if(!ctx->enqueueV3(stream))
  {
    printf("[sana_trt] enqueueV3 failed: %s\n", plan.c_str());
    goto cleanup;
  }
  if(outDt == DataType::kBF16)
    sana::launch_trt_bf2f(outBuf, out_dev, (long)outVol, (void*)stream);
  else if(outDt != DataType::kFLOAT)
  {
    printf("[sana_trt] unhandled output dtype %d\n", (int)outDt);
    goto cleanup;
  }
  TCK(cudaStreamSynchronize(stream));
  rc = 0;
cleanup:
  // owned CUDATensor scratch buffers free themselves on return (RAII).
  if(profOn())
    printf("[prof] run         %-38s %.1f ms\n", plan.c_str(), nowMs() - tr);
  return rc;
}

} // namespace

// Impl holds the engine dir + a cache of resident (deserialized-once) engines keyed by
// plan path. The ~2.4 GB VAE engines used by run_v2v stay resident across calls; only
// tensor addresses are rebound + enqueueV3 per call.
struct SanaTRT::Impl
{
  std::string dir;
  std::map<std::string, std::unique_ptr<EngineRec>> cache;

  // Run a resident engine: deserialize once (lazily on first use), reuse thereafter.
  int runResident(const std::string& plan, std::vector<EIn>& ins, float* out_dev)
  {
    auto it = cache.find(plan);
    if(it == cache.end())
    {
      EngineRec* rec = loadEngine(plan);
      if(!rec)
        return -1;
      it = cache.emplace(plan, std::unique_ptr<EngineRec>(rec)).first;
    }
    return runEngineRec(it->second.get(), plan, ins, out_dev);
  }
};
SanaTRT::SanaTRT(const std::string& trt_dir)
{
  p_ = new Impl();
  p_->dir = trt_dir;
}
SanaTRT::~SanaTRT()
{
  delete p_;
}

int SanaTRT::warm_vae()
{
  // Deserialize the two engines used by run_v2v into the resident cache (idempotent: runResident
  // reuses an already-cached engine). loadEngine is called at most once per plan.
  for(const char* rel :
      {"/vae/ltx2_vae_encoder_full_bf16.plan", "/vae/ltx2_vae_decoder_16f_bf16.plan"})
  {
    std::string plan = p_->dir + rel;
    if(p_->cache.find(plan) == p_->cache.end())
    {
      EngineRec* rec = loadEngine(plan);
      if(!rec)
        return -1;
      p_->cache.emplace(plan, std::unique_ptr<EngineRec>(rec));
    }
  }
  return 0;
}

// 25-frame VAE encode: pixels -> moments (the single-window encoder engine).
int SanaTRT::vae_encode(const float* pixels_dev, float* moments_dev)
{
  std::vector<EIn> ins = {{"", pixels_dev, DataType::kFLOAT, {1, 3, 25, 480, 832}}};
  return p_->runResident(p_->dir + "/vae/ltx2_vae_encoder_bf16.plan", ins, moments_dev);
}
// 3-frame VAE decode: latent -> frames (the single-window decoder engine).
int SanaTRT::vae_decode(const float* latent_dev, float* frames_dev)
{
  std::vector<EIn> ins = {{"", latent_dev, DataType::kFLOAT, {1, 128, 3, 15, 26}}};
  return p_->runResident(p_->dir + "/vae/ltx2_vae_decoder_bf16.plan", ins, frames_dev);
}
// Full-clip (121-frame) VAE encode used by run_v2v: pixels -> moments (resident engine).
int SanaTRT::vae_encode_full(const float* pixels_dev, float* moments_dev)
{
  std::vector<EIn> ins = {{"", pixels_dev, DataType::kFLOAT, {1, 3, 121, 480, 832}}};
  return p_->runResident(
      p_->dir + "/vae/ltx2_vae_encoder_full_bf16.plan", ins, moments_dev);
}
// Full-clip (16-latent-frame -> 121-frame) VAE decode used by run_v2v (resident engine).
int SanaTRT::vae_decode_16f(const float* latent_dev, float* frames_dev)
{
  std::vector<EIn> ins = {{"", latent_dev, DataType::kFLOAT, {1, 128, 16, 15, 26}}};
  return p_->runResident(
      p_->dir + "/vae/ltx2_vae_decoder_16f_bf16.plan", ins, frames_dev);
}
int SanaTRT::gemma_encode(
    const long long* ids_dev, const long long* mask_dev, float* hidden_dev)
{
  // gemma is ~10.5 GB and only used by encode_prompt (never run_v2v): load then free to bound
  // VRAM, rather than keeping it resident like the VAE engines.
  std::vector<EIn> ins
      = {{"input_ids", ids_dev, DataType::kINT64, {1, 300}},
         {"attention_mask", mask_dev, DataType::kINT64, {1, 300}}};
  std::string plan = p_->dir + "/gemma/gemma_encoder.plan";
  EngineRec* rec = loadEngine(plan);
  if(!rec)
    return -1;
  int rc = runEngineRec(rec, plan, ins, hidden_dev);
  delete rec; // release our context + shared_ptr ref to the engine
  // Drop the ~10.5 GB gemma engine from the shared cache so it does not stay resident
  // (the VAE engines are the only ones that must persist across calls).
  GlobalEngineCache::instance().engines().erase(plan);
  return rc;
}

} // namespace librediffusion
