// Minimal PNG writer (no deps) for harness previews. Writes 8-bit RGB. Uses zlib's
// stored (uncompressed) deflate blocks so we need no compression library — files are
// larger but valid PNGs, fine for small preview images.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pngmin
{

namespace detail
{
inline uint32_t crc_table[256];
inline bool crc_init = false;
inline void make_crc()
{
  for(uint32_t n = 0; n < 256; n++)
  {
    uint32_t c = n;
    for(int k = 0; k < 8; k++)
      c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
    crc_table[n] = c;
  }
  crc_init = true;
}
inline uint32_t crc(const uint8_t* buf, size_t len)
{
  if(!crc_init)
    make_crc();
  uint32_t c = 0xffffffffu;
  for(size_t i = 0; i < len; i++)
    c = crc_table[(c ^ buf[i]) & 0xff] ^ (c >> 8);
  return c ^ 0xffffffffu;
}
inline uint32_t adler32(const uint8_t* d, size_t n)
{
  uint32_t a = 1, b = 0;
  for(size_t i = 0; i < n; i++)
  {
    a = (a + d[i]) % 65521;
    b = (b + a) % 65521;
  }
  return (b << 16) | a;
}
inline void put32(std::vector<uint8_t>& v, uint32_t x)
{
  v.push_back((x >> 24) & 0xff);
  v.push_back((x >> 16) & 0xff);
  v.push_back((x >> 8) & 0xff);
  v.push_back(x & 0xff);
}
inline void chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data)
{
  put32(out, (uint32_t)data.size());
  std::vector<uint8_t> td;
  td.insert(td.end(), type, type + 4);
  td.insert(td.end(), data.begin(), data.end());
  out.insert(out.end(), td.begin(), td.end());
  put32(out, crc(td.data(), td.size()));
}
} // namespace detail

// rgb: H*W*3 bytes, row-major.
inline bool write_rgb(const std::string& path, const uint8_t* rgb, int w, int h)
{
  using namespace detail;
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};

  std::vector<uint8_t> ihdr;
  put32(ihdr, (uint32_t)w);
  put32(ihdr, (uint32_t)h);
  ihdr.push_back(8);  // bit depth
  ihdr.push_back(2);  // color type RGB
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  chunk(out, "IHDR", ihdr);

  // raw image data with filter byte 0 per row
  std::vector<uint8_t> raw;
  raw.reserve((size_t)h * (w * 3 + 1));
  for(int y = 0; y < h; y++)
  {
    raw.push_back(0);
    raw.insert(raw.end(), rgb + (size_t)y * w * 3, rgb + (size_t)(y + 1) * w * 3);
  }

  // zlib stream: 2-byte header + stored deflate blocks + adler32
  std::vector<uint8_t> z = {0x78, 0x01};
  size_t off = 0;
  while(off < raw.size())
  {
    size_t block = raw.size() - off;
    if(block > 65535)
      block = 65535;
    bool last = (off + block) >= raw.size();
    z.push_back(last ? 1 : 0);
    z.push_back(block & 0xff);
    z.push_back((block >> 8) & 0xff);
    z.push_back(~block & 0xff);
    z.push_back((~block >> 8) & 0xff);
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + block);
    off += block;
  }
  put32(z, adler32(raw.data(), raw.size()));
  chunk(out, "IDAT", z);
  chunk(out, "IEND", {});

  FILE* f = std::fopen(path.c_str(), "wb");
  if(!f)
    return false;
  std::fwrite(out.data(), 1, out.size(), f);
  std::fclose(f);
  return true;
}

// Convert NCHW fp32 in [-1,1] (first image) to RGB8 and write.
inline bool write_nchw_norm(const std::string& path, const float* nchw, int h, int w)
{
  std::vector<uint8_t> rgb((size_t)h * w * 3);
  const int plane = h * w;
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
      for(int ch = 0; ch < 3; ch++)
      {
        float v = nchw[ch * plane + y * w + x];
        v = (v + 1.0f) * 0.5f * 255.0f;
        if(v < 0)
          v = 0;
        if(v > 255)
          v = 255;
        rgb[(y * w + x) * 3 + ch] = (uint8_t)(v + 0.5f);
      }
  return write_rgb(path, rgb.data(), w, h);
}

} // namespace pngmin
