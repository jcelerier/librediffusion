/**
 * PyBind11 bindings for StreamDiffusion C++ pipeline
 */

#include "stream_diffusion.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace streamdiffusion;

// Helper to convert PyTorch tensor to raw pointer
template <typename T>
T* get_cuda_ptr(py::object tensor)
{
  // Get the data_ptr() from PyTorch tensor
  auto data_ptr_method = tensor.attr("data_ptr");
  uint64_t ptr_value = data_ptr_method().cast<uint64_t>();
  return reinterpret_cast<T*>(ptr_value);
}

// Helper to ensure tensor is contiguous (returns the tensor itself or a contiguous copy)
py::object ensure_contiguous(py::object tensor)
{
  auto is_contiguous = tensor.attr("is_contiguous")().cast<bool>();
  if(!is_contiguous)
  {
    // Make contiguous - returns a new tensor with proper memory layout
    return tensor.attr("contiguous")();
  }
  return tensor;
}

// Wrapper class to handle PyTorch tensor conversions
class StreamDiffusionPipelinePy
{
public:
  StreamDiffusionPipelinePy(const StreamDiffusionConfig& config)
      : pipeline_(config)
  {
  }

  void prepare(
      py::object prompt_embeds_tensor, py::object timesteps_tensor,
      py::object alpha_prod_t_sqrt_tensor, py::object beta_prod_t_sqrt_tensor,
      py::object c_skip_tensor, py::object c_out_tensor, int num_timesteps, int seq_len,
      int hidden_dim)
  {
    // Ensure all tensors are contiguous
    py::object prompt_embeds_c = ensure_contiguous(prompt_embeds_tensor);
    py::object timesteps_c = ensure_contiguous(timesteps_tensor);
    py::object alpha_prod_t_sqrt_c = ensure_contiguous(alpha_prod_t_sqrt_tensor);
    py::object beta_prod_t_sqrt_c = ensure_contiguous(beta_prod_t_sqrt_tensor);
    py::object c_skip_c = ensure_contiguous(c_skip_tensor);
    py::object c_out_c = ensure_contiguous(c_out_tensor);

    __half* prompt_embeds = get_cuda_ptr<__half>(prompt_embeds_c);
    float* timesteps = get_cuda_ptr<float>(timesteps_c);
    float* alpha_prod_t_sqrt = get_cuda_ptr<float>(alpha_prod_t_sqrt_c);
    float* beta_prod_t_sqrt = get_cuda_ptr<float>(beta_prod_t_sqrt_c);
    float* c_skip = get_cuda_ptr<float>(c_skip_c);
    float* c_out = get_cuda_ptr<float>(c_out_c);

    pipeline_.prepare(
        prompt_embeds, timesteps, alpha_prod_t_sqrt, beta_prod_t_sqrt, c_skip, c_out,
        num_timesteps, seq_len, hidden_dim);
  }

  void set_init_noise(py::object noise_tensor)
  {
    // Allow setting the init_noise from Python for testing/validation
    py::object noise_c = ensure_contiguous(noise_tensor);
    __half* noise = get_cuda_ptr<__half>(noise_c);
    pipeline_.set_init_noise(noise);
  }

  void predict_x0_batch(py::object x_t_latent_tensor, py::object x_0_pred_tensor)
  {
    // x_t_latent is input, x_0_pred is output
    py::object x_t_latent_c = ensure_contiguous(x_t_latent_tensor);

    __half* x_t_latent = get_cuda_ptr<__half>(x_t_latent_c);
    __half* x_0_pred = get_cuda_ptr<__half>(x_0_pred_tensor); // Output: use original

    // Pass 0 for stream - will use internal stream_ due to fix in stream_diffusion.cpp
    pipeline_.predict_x0_batch(x_t_latent, x_0_pred, 0);
  }

  void encode_image(py::object image_tensor, py::object latent_tensor)
  {
    // Only ensure INPUT tensors are contiguous (image is input, latent is output)
    py::object image_contiguous = ensure_contiguous(image_tensor);

    __half* image = get_cuda_ptr<__half>(image_contiguous);
    __half* latent = get_cuda_ptr<__half>(latent_tensor); // Output: use original

    pipeline_.encode_image(image, latent, 0);
  }

  void decode_latent(py::object latent_tensor, py::object image_tensor)
  {
    // latent is input, image is output
    py::object latent_c = ensure_contiguous(latent_tensor);

    __half* latent = get_cuda_ptr<__half>(latent_c);
    __half* image = get_cuda_ptr<__half>(image_tensor); // Output: use original

    pipeline_.decode_latent(latent, image, 0);
  }

  void inference(py::object image_in_tensor, py::object image_out_tensor)
  {
    // image_in is input, image_out is output
    py::object image_in_c = ensure_contiguous(image_in_tensor);

    __half* image_in = get_cuda_ptr<__half>(image_in_c);
    __half* image_out = get_cuda_ptr<__half>(image_out_tensor); // Output: use original

    pipeline_.inference(image_in, image_out, 0);
  }

  void txt2img(py::object image_out_tensor)
  {
    // No inputs, only output
    __half* image_out = get_cuda_ptr<__half>(image_out_tensor); // Output: use original
    pipeline_.txt2img(image_out, 0);
  }

  const StreamDiffusionConfig& config() const { return pipeline_.config(); }

private:
  StreamDiffusionPipeline pipeline_;
};

PYBIND11_MODULE(streamdiffusion_cpp, m)
{
  m.doc() = "StreamDiffusion C++ acceleration module";

  // Bind StreamDiffusionConfig
  py::class_<StreamDiffusionConfig>(m, "StreamDiffusionConfig")
      .def(py::init<>())
      .def_readwrite("width", &StreamDiffusionConfig::width)
      .def_readwrite("height", &StreamDiffusionConfig::height)
      .def_readwrite("latent_width", &StreamDiffusionConfig::latent_width)
      .def_readwrite("latent_height", &StreamDiffusionConfig::latent_height)
      .def_readwrite("batch_size", &StreamDiffusionConfig::batch_size)
      .def_readwrite("denoising_steps", &StreamDiffusionConfig::denoising_steps)
      .def_readwrite("frame_buffer_size", &StreamDiffusionConfig::frame_buffer_size)
      .def_readwrite("guidance_scale", &StreamDiffusionConfig::guidance_scale)
      .def_readwrite("do_add_noise", &StreamDiffusionConfig::do_add_noise)
      .def_readwrite("use_denoising_batch", &StreamDiffusionConfig::use_denoising_batch)
      .def_readwrite("seed", &StreamDiffusionConfig::seed)
      .def_readwrite("cfg_type", &StreamDiffusionConfig::cfg_type)
      .def_readwrite("unet_engine_path", &StreamDiffusionConfig::unet_engine_path)
      .def_readwrite("vae_encoder_path", &StreamDiffusionConfig::vae_encoder_path)
      .def_readwrite("vae_decoder_path", &StreamDiffusionConfig::vae_decoder_path);

  // Bind StreamDiffusionPipeline
  py::class_<StreamDiffusionPipelinePy>(m, "StreamDiffusionPipeline")
      .def(py::init<const StreamDiffusionConfig&>())
      .def(
          "prepare", &StreamDiffusionPipelinePy::prepare,
          "Prepare the pipeline with prompt embeddings and scheduler parameters",
          py::arg("prompt_embeds"), py::arg("timesteps"), py::arg("alpha_prod_t_sqrt"),
          py::arg("beta_prod_t_sqrt"), py::arg("c_skip"), py::arg("c_out"),
          py::arg("num_timesteps"), py::arg("seq_len"), py::arg("hidden_dim"))
      .def(
          "set_init_noise", &StreamDiffusionPipelinePy::set_init_noise,
          "Set initial noise from Python (for testing/validation)", py::arg("noise"))
      .def(
          "predict_x0_batch", &StreamDiffusionPipelinePy::predict_x0_batch,
          "Run denoising on latent", py::arg("x_t_latent"), py::arg("x_0_pred"))
      .def(
          "encode_image", &StreamDiffusionPipelinePy::encode_image,
          "Encode image to latent", py::arg("image"), py::arg("latent"))
      .def(
          "decode_latent", &StreamDiffusionPipelinePy::decode_latent,
          "Decode latent to image", py::arg("latent"), py::arg("image"))
      .def(
          "inference", &StreamDiffusionPipelinePy::inference, "Full inference pipeline",
          py::arg("image_in"), py::arg("image_out"))
      .def(
          "txt2img", &StreamDiffusionPipelinePy::txt2img, "Text to image generation",
          py::arg("image_out"))
      .def_property_readonly("config", &StreamDiffusionPipelinePy::config);
}
