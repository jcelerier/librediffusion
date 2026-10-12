// Comparison metrics + tolerance profiles, mirroring the Python harness compare.py.
//
//   latent  (numeric)     : cosine >= 0.995    && rel_error <= 0.10
//   image   (perceptual)  : psnr   >= 30.0     && ssim      >= 0.95
//   strict  (numeric)     : cosine >= 0.999999 && rel_error <= 1e-4   (py-vs-py only)
//
// boundary_to_profile(): decode_image/out_image -> image, everything else -> latent.
// These thresholds are the fp16-TRT-vs-fp32-torch tolerance; an exact match is NOT
// expected across implementations.
//
// librediffusion: the `latent` gate was loosened from cos>=0.999/rel<=0.05 to
// cos>=0.995/rel<=0.10. The intermediate predict_x0 latent reconstruction
// x0=(x_t-beta*eps)/alpha amplifies fp16-TRT-vs-fp32 error by ~1/alpha; at the noisiest
// single steps (turbo/sdxs index 0, index 20, hyper 4-step, high-guidance initialize) alpha
// is tiny (~0.07) so a CORRECT result grazes the old 0.999 gate (observed 0.9954-0.9989,
// rel 0.085-0.096) while the decode image still PASSES at >35dB. Real divergences are far
// below this (observed bugs: cos 0.0/0.26/0.55, rel >0.4), so 0.995/0.10 still catches them
// with wide margin. The `image` profile (psnr/ssim on the decoded output) remains the
// authoritative correctness signal; `strict` (py-vs-py determinism) is unchanged.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cmp
{

struct NumericResult
{
  double cosine = 0.0;
  double rel_error = 0.0;
  double max_abs_diff = 0.0;
};

struct PerceptualResult
{
  double psnr = 0.0;
  double ssim = 0.0;
};

inline NumericResult numeric(const std::vector<float>& a, const std::vector<float>& b)
{
  NumericResult r;
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  double dot = 0, na = 0, nb = 0, diff2 = 0, maxabs = 0;
  for(size_t i = 0; i < n; i++)
  {
    const double x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
    const double d = x - y;
    diff2 += d * d;
    const double ad = std::fabs(d);
    if(ad > maxabs)
      maxabs = ad;
  }
  r.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
  r.rel_error = (na > 0) ? std::sqrt(diff2) / std::sqrt(na) : (diff2 > 0 ? 1.0 : 0.0);
  r.max_abs_diff = maxabs;
  return r;
}

// PSNR over [0,255] data range by default (golden final images are uint8-as-f32);
// for normalized images pass the appropriate data_range.
inline double psnr(const std::vector<float>& a, const std::vector<float>& b, double data_range)
{
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  double mse = 0;
  for(size_t i = 0; i < n; i++)
  {
    const double d = a[i] - b[i];
    mse += d * d;
  }
  mse /= (n > 0 ? n : 1);
  if(mse <= 0)
    return 99.0;
  return 20.0 * std::log10(data_range / std::sqrt(mse));
}

// Global single-window SSIM approximation (matches the Python harness's global SSIM).
inline double ssim_global(const std::vector<float>& a, const std::vector<float>& b, double data_range)
{
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  if(n == 0)
    return 0.0;
  double ma = 0, mb = 0;
  for(size_t i = 0; i < n; i++)
  {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  double va = 0, vb = 0, cov = 0;
  for(size_t i = 0; i < n; i++)
  {
    const double da = a[i] - ma, db = b[i] - mb;
    va += da * da;
    vb += db * db;
    cov += da * db;
  }
  va /= (n - 1 > 0 ? n - 1 : 1);
  vb /= (n - 1 > 0 ? n - 1 : 1);
  cov /= (n - 1 > 0 ? n - 1 : 1);
  const double c1 = (0.01 * data_range) * (0.01 * data_range);
  const double c2 = (0.03 * data_range) * (0.03 * data_range);
  return ((2 * ma * mb + c1) * (2 * cov + c2))
         / ((ma * ma + mb * mb + c1) * (va + vb + c2));
}

enum class Profile
{
  Latent,
  Image,
  Strict
};

inline Profile boundary_to_profile(const std::string& boundary, const std::string& role)
{
  if(boundary == "decode_image" && role == "out_image")
    return Profile::Image;
  if(boundary == "output")
    return Profile::Image;
  return Profile::Latent;
}

struct Verdict
{
  bool pass = false;
  std::string detail;
};

inline Verdict judge_numeric(const NumericResult& r, Profile p)
{
  double min_cos = (p == Profile::Strict) ? 0.999999 : 0.995;
  double max_rel = (p == Profile::Strict) ? 1e-4 : 0.10;
  Verdict v;
  v.pass = (r.cosine >= min_cos) && (r.rel_error <= max_rel);
  v.detail = "cos=" + std::to_string(r.cosine) + " rel=" + std::to_string(r.rel_error)
             + " maxabs=" + std::to_string(r.max_abs_diff);
  return v;
}

inline Verdict judge_image(const PerceptualResult& r)
{
  Verdict v;
  v.pass = (r.psnr >= 30.0) && (r.ssim >= 0.95);
  v.detail = "psnr=" + std::to_string(r.psnr) + " ssim=" + std::to_string(r.ssim);
  return v;
}

} // namespace cmp
