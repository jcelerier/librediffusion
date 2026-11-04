#pragma once

#include "librediffusion_c.h"

#include <exception>
#include <new>

namespace librediffusion
{
inline thread_local librediffusion_error_t g_last_create_error = LIBREDIFFUSION_SUCCESS;

inline librediffusion_error_t set_create_error(librediffusion_error_t e) noexcept
{
  g_last_create_error = e;
  return e;
}

inline void set_create_error(const std::exception& e) noexcept
{
  g_last_create_error = dynamic_cast<const std::bad_alloc*>(&e) ? LIBREDIFFUSION_ERROR_OUT_OF_MEMORY
                                                                : LIBREDIFFUSION_ERROR_INTERNAL;
}
}
