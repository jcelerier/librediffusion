#pragma once
namespace streamdiffusion
{

struct TimestepParams
{
  float c_skip;
  float c_out;
  float alpha_prod_t_sqrt;
  float beta_prod_t_sqrt;
};

} // namespace streamdiffusion
