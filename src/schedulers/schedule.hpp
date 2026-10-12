#pragma once
// Turns timestep indices into the arrays librediffusion_prepare_scheduler (and, for shifted-timestep
// models, librediffusion_prepare_noise_schedule) take, from one of the precomputed 50-entry tables
// next to this file. Shared by the score node and the harnesses so both drive the pipeline with
// exactly the same numbers.

#include "timestep_params.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace streamdiffusion
{

struct ScheduleArrays
{
  // librediffusion_prepare_scheduler: what the UNet is fed, and the x0-prediction coefficients
  std::vector<float> timesteps;
  std::vector<float> alpha_prod_t_sqrt;
  std::vector<float> beta_prod_t_sqrt;
  std::vector<float> c_skip;
  std::vector<float> c_out;
  // librediffusion_prepare_noise_schedule: the noise-injection coefficients. Empty unless they
  // differ from the ones above (unet_timestep_scale != 1).
  std::vector<float> noise_alpha_prod_t_sqrt;
  std::vector<float> noise_beta_prod_t_sqrt;

  void clear()
  {
    timesteps.clear();
    alpha_prod_t_sqrt.clear();
    beta_prod_t_sqrt.clear();
    c_skip.clear();
    c_out.clear();
    noise_alpha_prod_t_sqrt.clear();
    noise_beta_prod_t_sqrt.clear();
  }
  std::size_t size() const noexcept { return timesteps.size(); }
  bool has_noise_schedule() const noexcept { return !noise_alpha_prod_t_sqrt.empty(); }
};

// LCMScheduler coefficients at any integer timestep in [0, 999], for the schedule every table here
// was generated from (tools/generate_scheduler_tables.py): scaled_linear betas 0.00085..0.012 over
// 1000 steps, LCM boundary condition with sigma_data = 0.5 and timestep_scaling = 10. Computed in
// float like the scheduler's torch tensors, so it reproduces the tables' entries.
inline TimestepParams lcm_params_at(int t)
{
  const float lo = std::sqrt(0.00085f), hi = std::sqrt(0.012f);
  float alphas_cumprod = 1.f;
  for(int i = 0; i <= t; i++)
  {
    const float b = lo + (hi - lo) * (float)i / 999.f;
    alphas_cumprod *= 1.f - b * b;
  }
  const float sigma_data = 0.5f, scaled_t = 10.f * (float)t;
  const float d = scaled_t * scaled_t + sigma_data * sigma_data;
  return TimestepParams{
      .c_skip = sigma_data * sigma_data / d,
      .c_out = scaled_t / std::sqrt(d),
      .alpha_prod_t_sqrt = std::sqrt(alphas_cumprod),
      .beta_prod_t_sqrt = std::sqrt(1.f - alphas_cumprod)};
}

// Fills `out` (cleared first; its capacity is reused) with one entry per index of `indices`; indices
// outside the table are skipped.
//
// `unet_timestep_scale` (bundle.json unet_timestep_scale) is for models distilled with a shifted
// timestep, e.g. NitroFusion's TimestepShiftLCMScheduler: the UNet is fed t_s = floor(t * scale), the
// x0 prediction uses the coefficients of t_s (LCMScheduler.step is handed the shifted timestep), and
// the input is noised at the original t (re-noising reads the unshifted timestep list). At scale 1
// the table entries pass through unchanged and no noise schedule is produced.
inline void build_schedule(
    std::span<const int> values, std::span<const TimestepParams> params,
    std::span<const int> indices, float unet_timestep_scale, ScheduleArrays& out)
{
  out.clear();
  const bool shifted = unet_timestep_scale != 1.f;
  for(int idx : indices)
  {
    if(idx < 0 || idx >= (int)values.size() || idx >= (int)params.size())
      continue;
    const auto& noise = params[idx];
    const int t = values[idx];
    const int t_unet = shifted ? (int)std::floor((float)t * unet_timestep_scale) : t;
    const TimestepParams p = shifted ? lcm_params_at(t_unet) : noise;

    out.timesteps.push_back((float)t_unet);
    out.alpha_prod_t_sqrt.push_back(p.alpha_prod_t_sqrt);
    out.beta_prod_t_sqrt.push_back(p.beta_prod_t_sqrt);
    out.c_skip.push_back(p.c_skip);
    out.c_out.push_back(p.c_out);
    if(shifted)
    {
      out.noise_alpha_prod_t_sqrt.push_back(noise.alpha_prod_t_sqrt);
      out.noise_beta_prod_t_sqrt.push_back(noise.beta_prod_t_sqrt);
    }
  }
}

} // namespace streamdiffusion
