#include "nchw.hpp"

#include <cutlass/util/device_nchw_to_nhwc.h>
#include <cutlass/util/device_nhwc_to_nchw.h>

template <typename T>
void launch_nhwc_to_nchw_rgb_t(
    int n, int h, int w, const T* in_rgb, T* out_rgb, void* stream)
{
  static constexpr int c = 3;

  dim3 grid((c + 31) / 32, (h * w + 31) / 32, n);
  dim3 block(32, 8);
  cutlass::nhwc_to_nchw_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      out_rgb, (T*)in_rgb, n, h, w, c);
}

template <typename T>
void launch_nchw_to_nhwc_rgb_t(
    int n, int h, int w, const T* in_rgb, T* out_rgb, void* stream)
{
  static constexpr int c = 3;
  dim3 grid((h * w + 31) / 32, (c + 31) / 32, n);
  dim3 block(32, 8);
  cutlass::nchw_to_nhwc_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      out_rgb, (T*)in_rgb, n, h, w, c);
}

void launch_nhwc_to_nchw_rgb(
    int n, int h, int w, const float* in_rgb, float* out_rgb, void* stream)
{
  //<<<<<<< Updated upstream
  //  return launch_nhwc_to_nchw_rgb_t(n, h, w, in_rgb, out_rgb, stream);
  //=======
  using namespace cutlass;
  int c = 3;

  dim3 grid((c + 31) / 32, (h * w + 31) / 32, n);
  dim3 block(32, 8);
  nhwc_to_nchw_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      out_rgb, in_rgb, n, h, w, c);
  //>>>>>>> Stashed changes
}

void launch_nhwc_to_nchw_rgb(
    int n, int h, int w, const __half* in_rgb, __half* out_rgb, void* stream)
{
  // <<<<<<< Updated upstream
  //   return launch_nhwc_to_nchw_rgb_t(n, h, w, in_rgb, out_rgb, stream);
  // =======
  using namespace cutlass;
  int c = 3;

  dim3 grid((c + 31) / 32, (h * w + 31) / 32, n);
  dim3 block(32, 8);
  nhwc_to_nchw_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      out_rgb, in_rgb, n, h, w, c);
  // >>>>>>> Stashed changes
}

void launch_nchw_to_nhwc_rgb(
    int n, int h, int w, const float* in_rgb, float* out_rgb, void* stream)
{
  // <<<<<<< Updated upstream
  //   return launch_nchw_to_nhwc_rgb_t(n, h, w, in_rgb, out_rgb, stream);
  // =======
  using namespace cutlass;
  int c = 3;

  dim3 grid((h * w + 31) / 32, (c + 31) / 32, n);
  dim3 block(32, 8);
  nchw_to_nhwc_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      out_rgb, in_rgb, n, h, w, c);
  // >>>>>>> Stashed changes
}

void launch_nchw_to_nhwc_rgb(
    int n, int h, int w, const __half* in_rgb, __half* out_rgb, void* stream)
{
  // <<<<<<< Updated upstream
  //   return launch_nchw_to_nhwc_rgb_t(n, h, w, in_rgb, out_rgb, stream);
  // =======
  using namespace cutlass;
  int c = 3;

  dim3 grid((h * w + 31) / 32, (c + 31) / 32, n);
  dim3 block(32, 8);
  nchw_to_nhwc_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      out_rgb, in_rgb, n, h, w, c);
  // >>>>>>> Stashed changes
}
