#pragma once
// Readers for the few scalars a host needs out of a bundle's manifest (<bundle>/bundle.json, written
// by train-lora.py's write_manifest). Header-only and dependency-free on purpose: the score node, the
// SDK / JIT / standalone builds and the validation harnesses all include it, and a few numbers out of
// a file we generate ourselves do not justify a JSON library every one of them would have to carry.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>

namespace librediffusion
{

// A top-level number from <bundle_dir>/bundle.json, or nullopt when there is no manifest (the
// pre-manifest hash bundles), no such key, or the value is not a number (e.g. null). This is a scan,
// not a parse: every key read through it is unique in the file, so the first match is the
// top-level one.
inline std::optional<double> bundle_number(const std::string& bundle_dir, std::string_view key)
{
  std::ifstream f{bundle_dir + "/bundle.json", std::ios::binary};
  if(!f)
    return std::nullopt;
  const std::string text{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};

  const auto k = text.find("\"" + std::string{key} + "\"");
  if(k == std::string::npos)
    return std::nullopt;
  const auto colon = text.find(':', k);
  if(colon == std::string::npos)
    return std::nullopt;
  const char* begin = text.c_str() + colon + 1;
  char* end = nullptr;
  const double v = std::strtod(begin, &end); // skips leading whitespace itself
  if(end == begin || !std::isfinite(v))
    return std::nullopt;
  return v;
}

// Cross-attention width declared by the manifest, or 0 when there is no usable one. The SD1.5 family
// is not one width (SD1.5 is 768, SD2.1 is 1024) and nothing else on the host side tells them apart.
inline int bundle_embedding_dim(const std::string& bundle_dir)
{
  const auto dim = bundle_number(bundle_dir, "embedding_dim");
  // nonsense manifest: ignore it rather than propagate a huge allocation
  if(!dim || *dim < 1 || *dim > 65536)
    return 0;
  return static_cast<int>(*dim);
}

// Factor applied to the timestep the UNet is fed, while the noise schedule stays at the unscaled t.
// Models distilled with a shifted timestep record it (NitroFusion: 0.25 Realism, 0.5 Vibrant); for
// everything else, and for anything outside (0, 1], it is 1.
inline float bundle_unet_timestep_scale(const std::string& bundle_dir)
{
  const auto s = bundle_number(bundle_dir, "unet_timestep_scale");
  if(!s || *s <= 0. || *s > 1.)
    return 1.f;
  return static_cast<float>(*s);
}

}
