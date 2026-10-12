// Bit-exactness probe for the shared PCG32 randn: calls launch_randn_fp16 and writes the raw
// fp16 (uint16) buffer to a file, so the Python replica (deterministic_noise.py) can compare
// byte-for-byte. Usage: randn_dump <seed> <N> <out.bin>
#include "kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
// note: no cuda_fp16.h needed — the fp16 buffer is treated as raw uint16_t here.

int main(int argc, char** argv)
{
  if(argc != 4)
  {
    std::fprintf(stderr, "usage: %s <seed> <N> <out.bin>\n", argv[0]);
    return 2;
  }
  unsigned long long seed = std::strtoull(argv[1], nullptr, 10);
  int N = std::atoi(argv[2]);
  const char* out_path = argv[3];

  void* d_out = nullptr;
  cudaMalloc(&d_out, (size_t)N * sizeof(uint16_t));
  launch_randn_fp16(d_out, seed, N, nullptr);
  cudaDeviceSynchronize();

  std::vector<uint16_t> host(N);
  cudaMemcpy(host.data(), d_out, (size_t)N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  FILE* f = std::fopen(out_path, "wb");
  if(!f)
  {
    std::fprintf(stderr, "cannot write %s\n", out_path);
    return 1;
  }
  std::fwrite(host.data(), sizeof(uint16_t), N, f);
  std::fclose(f);
  std::printf("wrote %d fp16 samples (seed=%llu) to %s\n", N, seed, out_path);
  return 0;
}
