// Minimal NumPy .npy reader/writer (v1.0) for the C++ validation harness.
// Reads the fp32 C-order tensors the Python golden harness dumps; writes the
// C++ side's dumps in the same format so both can be diffed by either side.
//
// Supports: little-endian, C-contiguous, dtypes '<f4' (float32), '<f8' (float64),
// '<u1'/'|u1' (uint8). Everything is returned as float (converted on load).
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy
{

struct Array
{
  std::vector<size_t> shape;
  std::vector<float> data; // always stored as float32 in host memory

  size_t size() const
  {
    size_t n = 1;
    for(auto d : shape)
      n *= d;
    return n;
  }
};

namespace detail
{
inline std::string parse_header_field(const std::string& hdr, const std::string& key)
{
  auto pos = hdr.find(key);
  if(pos == std::string::npos)
    throw std::runtime_error("npy: missing header key " + key);
  pos += key.size();
  // skip ': ' and optional quote
  while(pos < hdr.size() && (hdr[pos] == ' ' || hdr[pos] == ':'))
    pos++;
  return hdr.substr(pos);
}
} // namespace detail

inline Array load(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if(!f)
    throw std::runtime_error("npy: cannot open " + path);

  char magic[6];
  f.read(magic, 6);
  if(std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("npy: bad magic in " + path);

  uint8_t ver_major = 0, ver_minor = 0;
  f.read(reinterpret_cast<char*>(&ver_major), 1);
  f.read(reinterpret_cast<char*>(&ver_minor), 1);

  uint32_t header_len = 0;
  if(ver_major == 1)
  {
    uint16_t hl16 = 0;
    f.read(reinterpret_cast<char*>(&hl16), 2);
    header_len = hl16;
  }
  else
  {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }

  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  // descr
  std::string descr;
  {
    auto pos = header.find("'descr'");
    if(pos == std::string::npos)
      throw std::runtime_error("npy: no descr in " + path);
    pos = header.find('\'', pos + 7);
    auto end = header.find('\'', pos + 1);
    descr = header.substr(pos + 1, end - pos - 1);
  }

  // fortran_order
  if(header.find("'fortran_order': True") != std::string::npos)
    throw std::runtime_error("npy: fortran_order unsupported in " + path);

  // shape
  Array arr;
  {
    auto pos = header.find("'shape'");
    pos = header.find('(', pos);
    auto end = header.find(')', pos);
    std::string s = header.substr(pos + 1, end - pos - 1);
    size_t i = 0;
    while(i < s.size())
    {
      while(i < s.size() && (s[i] == ' ' || s[i] == ','))
        i++;
      if(i >= s.size())
        break;
      size_t j = i;
      while(j < s.size() && s[j] >= '0' && s[j] <= '9')
        j++;
      if(j > i)
        arr.shape.push_back(std::stoul(s.substr(i, j - i)));
      i = j;
    }
  }

  const size_t n = arr.size();
  arr.data.resize(n);

  if(descr == "<f4" || descr == "f4")
  {
    f.read(reinterpret_cast<char*>(arr.data.data()), n * sizeof(float));
  }
  else if(descr == "<f8" || descr == "f8")
  {
    std::vector<double> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), n * sizeof(double));
    for(size_t i = 0; i < n; i++)
      arr.data[i] = static_cast<float>(tmp[i]);
  }
  else if(descr == "<u1" || descr == "|u1" || descr == "u1")
  {
    std::vector<uint8_t> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), n);
    for(size_t i = 0; i < n; i++)
      arr.data[i] = static_cast<float>(tmp[i]);
  }
  else
  {
    throw std::runtime_error("npy: unsupported dtype '" + descr + "' in " + path);
  }
  return arr;
}

inline void save(const std::string& path, const float* data, const std::vector<size_t>& shape)
{
  std::ofstream f(path, std::ios::binary);
  if(!f)
    throw std::runtime_error("npy: cannot write " + path);

  std::string shape_str = "(";
  for(size_t i = 0; i < shape.size(); i++)
  {
    shape_str += std::to_string(shape[i]);
    if(shape.size() == 1 || i + 1 < shape.size())
      shape_str += ", ";
  }
  shape_str += ")";

  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
  // pad so that 10 + header.size() is a multiple of 64, ending in '\n'
  size_t total = 10 + header.size() + 1;
  size_t pad = (64 - (total % 64)) % 64;
  header.append(pad, ' ');
  header += '\n';

  f.write("\x93NUMPY", 6);
  uint8_t ver[2] = {1, 0};
  f.write(reinterpret_cast<char*>(ver), 2);
  uint16_t hl = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<char*>(&hl), 2);
  f.write(header.data(), header.size());

  size_t n = 1;
  for(auto d : shape)
    n *= d;
  f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

} // namespace npy
