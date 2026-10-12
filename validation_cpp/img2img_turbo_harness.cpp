// img2img-turbo skip-VAE validation: run the C-API pipeline on the golden input + golden ehs,
// compare the decoded image to the golden output_image (PSNR), and save a PNG.
//
//   img2img_turbo_harness <engine_dir> <golden_dir>
//
// engine_dir: holds unet.engine / vae_encoder.engine / vae_decoder.engine
// golden_dir: holds input_c_t.npy, caption_enc.npy, output_image.npy
#include "librediffusion_c.h"
#include "npy.hpp"
#include "png_min.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static float* upload(const npy::Array& a)
{
  float* d = nullptr;
  cudaMalloc(&d, a.data.size() * sizeof(float));
  cudaMemcpy(d, a.data.data(), a.data.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}

int main(int argc, char** argv)
{
  std::string eng = argc > 1 ? argv[1] : "/media/data1/img2img-turbo/engines_edge2image";
  std::string gold = argc > 2 ? argv[2] : "/media/data1/img2img-turbo/goldens_edge2image";

  npy::Array image = npy::load(gold + "/input_c_t.npy");     // [1,3,512,512]
  npy::Array ehs = npy::load(gold + "/caption_enc.npy");     // [1,77,1024]
  npy::Array ref = npy::load(gold + "/output_image.npy");    // [1,3,512,512]
  printf("loaded goldens: image %zu, ehs %zu, ref %zu floats\n",
         image.data.size(), ehs.data.size(), ref.data.size());

  auto h = librediffusion_img2img_turbo_create(
      (eng + "/unet.engine").c_str(), (eng + "/vae_encoder.engine").c_str(),
      (eng + "/vae_decoder.engine").c_str(), 0);
  if(!h) { fprintf(stderr, "create failed\n"); return 1; }

  float* d_image = upload(image);
  float* d_ehs = upload(ehs);
  float* d_out = nullptr;
  cudaMalloc(&d_out, ref.data.size() * sizeof(float));

  auto err = librediffusion_img2img_turbo_forward(h, d_image, d_ehs, d_out, 0);
  if(err != LIBREDIFFUSION_SUCCESS) { fprintf(stderr, "forward error %d\n", (int)err); return 1; }

  std::vector<float> out(ref.data.size());
  cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // PSNR over [-1,1] (range 2 -> peak^2 = 4)
  double mse = 0.0;
  for(size_t i = 0; i < out.size(); i++)
  {
    double d = (double)out[i] - (double)ref.data[i];
    mse += d * d;
  }
  mse /= out.size();
  double psnr = 10.0 * std::log10(4.0 / mse);
  printf("PSNR vs golden output_image: %.2f dB  (mse=%.3e)\n", psnr, mse);

  pngmin::write_nchw_norm(eng + "/cpp_output.png", out.data(), 512, 512);
  printf("saved %s/cpp_output.png\n", eng.c_str());

  librediffusion_img2img_turbo_destroy(h);
  cudaFree(d_image); cudaFree(d_ehs); cudaFree(d_out);
  return psnr >= 40.0 ? 0 : 2;
}
