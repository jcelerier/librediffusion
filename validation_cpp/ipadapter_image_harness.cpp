// On-device IP-Adapter image-encoder validation harness.
//
// Drives CLIPImageEncoderWrapper (the new on-device CLIP image encode + ImageProjModel projection)
// on the portrait fixture (raw RGBA bin) and compares the resulting pos/neg tokens to the Python
// golden (900/901_static__ipadapter_tokens_{pos,neg}.npy). Gate: cosine >= 0.99 for both.
//
// Usage:
//   ipadapter_image_harness <encoder.engine> <proj.engine> <rgba.bin> <H> <W> <golden_dir>

#include "tensorrt_wrappers.hpp"
#include "librediffusion.hpp"  // CUDATensor

#include "npy.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace librediffusion;

static double cosine(const std::vector<float>& a, const std::vector<float>& b)
{
  double dot = 0, na = 0, nb = 0;
  size_t n = std::min(a.size(), b.size());
  for(size_t i = 0; i < n; ++i)
  {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

int main(int argc, char** argv)
{
  if(argc < 7)
  {
    std::fprintf(stderr, "usage: %s <enc.engine> <proj.engine> <rgba.bin> <H> <W> <golden_dir>\n",
                 argv[0]);
    return 2;
  }
  std::string enc_path = argv[1], proj_path = argv[2], rgba_path = argv[3];
  int H = std::stoi(argv[4]), W = std::stoi(argv[5]);
  std::string gold = argv[6];

  // Load raw RGBA.
  std::ifstream f(rgba_path, std::ios::binary);
  std::vector<uint8_t> rgba((size_t)H * W * 4);
  f.read((char*)rgba.data(), rgba.size());
  if(!f)
  {
    std::fprintf(stderr, "failed to read %s\n", rgba_path.c_str());
    return 2;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CLIPImageEncoderWrapper enc(enc_path, proj_path);
  int N = enc.numTokens(), dim = enc.tokenDim();
  std::printf("encoder loaded: %d tokens x %d dim\n", N, dim);

  CUDATensor<__half> d_pos(N * dim), d_neg(N * dim);
  enc.encodeImage(rgba.data(), H, W, d_pos.data(), d_neg.data(), stream);
  cudaStreamSynchronize(stream);

  // Download tokens -> float.
  auto download = [&](CUDATensor<__half>& d) {
    std::vector<__half> h(N * dim);
    cudaMemcpy(h.data(), d.data(), h.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> out(h.size());
    for(size_t i = 0; i < h.size(); ++i)
      out[i] = __half2float(h[i]);
    return out;
  };
  auto pos = download(d_pos), neg = download(d_neg);

  npy::Array gpos = npy::load(gold + "/900_static__ipadapter_tokens_pos.npy");
  npy::Array gneg = npy::load(gold + "/901_static__ipadapter_tokens_neg.npy");

  double cpos = cosine(pos, gpos.data);
  double cneg = cosine(neg, gneg.data);
  std::printf("POS cos vs golden = %.5f\n", cpos);
  std::printf("NEG cos vs golden = %.5f\n", cneg);

  bool ok = cpos >= 0.99 && cneg >= 0.99;
  std::printf("GATE: %s\n", ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  cudaStreamDestroy(stream);
  // Avoid the GlobalEngineCache atexit teardown crashing on TRT shutdown ordering — the validation
  // result is already computed + flushed; hard-exit with the gate status.
  _exit(ok ? 0 : 1);
}
