#pragma once

#include <new>
#include <string>

namespace librediffusion
{
// The device lacks the memory an allocation needs. The C API reports it as
// LIBREDIFFUSION_ERROR_OUT_OF_MEMORY.
class device_out_of_memory : public std::bad_alloc
{
public:
  explicit device_out_of_memory(std::string what)
      : m_what{std::move(what)}
  {
  }
  const char* what() const noexcept override { return m_what.c_str(); }

private:
  std::string m_what;
};
}
