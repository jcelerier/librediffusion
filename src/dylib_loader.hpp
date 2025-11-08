#pragma once

/**
 * Cross-platform dynamic library loader
 * Supports both POSIX (dlopen) and Windows (LoadLibrary)
 */

#if defined(_WIN32) || defined(_WIN64)
#define LIBREDIFFUSION_LOADER_WINDOWS 1
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h>
#elif __has_include(<dlfcn.h>)
#define LIBREDIFFUSION_LOADER_POSIX 1
#include <dlfcn.h>
#else
  #error "No dynamic library loading support available"
#endif

#include <cassert>

namespace sd
{

/**
 * @brief Cross-platform dynamic library loader
 * 
 * Wraps dlopen/LoadLibrary for portable dynamic symbol loading.
 */
class dylib_loader
{
public:
  /**
   * @brief Load a dynamic library
   * @param name Library name (e.g., "librediffusion.dll" or "liblibrediffusion.so")
   */
  explicit dylib_loader(const char* const name)
  {
#if defined(LIBREDIFFUSION_LOADER_WINDOWS)
    m_handle = LoadLibraryA(name);
#elif defined(LIBREDIFFUSION_LOADER_POSIX)
    m_handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
  }

  /**
   * @brief Try loading from multiple possible library names
   * @param names Null-terminated array of library names to try
   */
  explicit dylib_loader(const char* const* names)
  {
    for (const char* const* p = names; *p != nullptr; ++p)
    {
#if defined(LIBREDIFFUSION_LOADER_WINDOWS)
      m_handle = LoadLibraryA(*p);
#elif defined(LIBREDIFFUSION_LOADER_POSIX)
      m_handle = dlopen(*p, RTLD_LAZY | RTLD_LOCAL);
#endif
      if (m_handle)
        break;
    }
  }

  dylib_loader(const dylib_loader&) = delete;
  dylib_loader& operator=(const dylib_loader&) = delete;

  dylib_loader(dylib_loader&& other) noexcept
      : m_handle{other.m_handle}
  {
    other.m_handle = nullptr;
  }

  dylib_loader& operator=(dylib_loader&& other) noexcept
  {
    if (this != &other)
    {
      close();
      m_handle = other.m_handle;
      other.m_handle = nullptr;
    }
    return *this;
  }

  ~dylib_loader() { close(); }

  /**
   * @brief Load a symbol from the library
   * @tparam T Function pointer type
   * @param sym Symbol name
   * @return Function pointer, or nullptr if not found
   */
  template <typename T>
  T symbol(const char* const sym) const noexcept
  {
    if (!m_handle)
      return nullptr;

#if defined(LIBREDIFFUSION_LOADER_WINDOWS)
    return reinterpret_cast<T>(GetProcAddress(static_cast<HMODULE>(m_handle), sym));
#elif defined(LIBREDIFFUSION_LOADER_POSIX)
    return reinterpret_cast<T>(dlsym(m_handle, sym));
#endif
  }

  /**
   * @brief Check if the library was loaded successfully
   */
  explicit operator bool() const noexcept { return m_handle != nullptr; }

  /**
   * @brief Get the last error message (platform-specific)
   */
  static const char* last_error() noexcept
  {
#if defined(LIBREDIFFUSION_LOADER_WINDOWS)
    thread_local static char buffer[256];
    FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buffer, sizeof(buffer), nullptr);
    return buffer;
#elif defined(LIBREDIFFUSION_LOADER_POSIX)
    return dlerror();
#endif
  }

private:
  void close() noexcept
  {
    if (m_handle)
    {
#if defined(LIBREDIFFUSION_LOADER_WINDOWS)
      FreeLibrary(static_cast<HMODULE>(m_handle));
#elif defined(LIBREDIFFUSION_LOADER_POSIX)
      dlclose(m_handle);
#endif
      m_handle = nullptr;
    }
  }

  void* m_handle{nullptr};
};

} // namespace sd

/*===========================================================================*/
/* Symbol Loading Macros                                                     */
/*===========================================================================*/

/**
 * Helper macros for declaring and initializing function pointer members.
 * 
 * Usage:
 *   // In struct definition:
 *   LIBREDIFFUSION_SYMBOL_DEF(sd, config_create);  // declares: decltype(&::sd_config_create) config_create{};
 *   
 *   // In constructor:
 *   LIBREDIFFUSION_SYMBOL_INIT(sd, config_create); // loads symbol "sd_config_create" into config_create
 */

// String concatenation helpers
#define LIBREDIFFUSION_SYMBOL_STR_IMPL(x) #x
#define LIBREDIFFUSION_SYMBOL_STR(x) LIBREDIFFUSION_SYMBOL_STR_IMPL(x)
#define LIBREDIFFUSION_SYMBOL_CAT_IMPL(a, b) a##_##b
#define LIBREDIFFUSION_SYMBOL_CAT(a, b) LIBREDIFFUSION_SYMBOL_CAT_IMPL(a, b)

// Main symbol macros
#define LIBREDIFFUSION_SYMBOL_NAME_STR(prefix, name) \
  LIBREDIFFUSION_SYMBOL_STR(LIBREDIFFUSION_SYMBOL_CAT(prefix, name))
#define LIBREDIFFUSION_SYMBOL_FULL_NAME(prefix, name) \
  LIBREDIFFUSION_SYMBOL_CAT(prefix, name)

/**
 * @brief Declare a function pointer member
 * @param prefix Symbol prefix (e.g., sd)
 * @param name Symbol name without prefix (e.g., config_create)
 * 
 * Expands to: decltype(&::sd_config_create) config_create{};
 */
#define LIBREDIFFUSION_SYMBOL_DEF(prefix, name) \
  decltype(&::LIBREDIFFUSION_SYMBOL_FULL_NAME(prefix, name)) name { }

/**
 * @brief Initialize a function pointer from the library
 * @param prefix Symbol prefix (e.g., sd)
 * @param name Symbol name without prefix (e.g., config_create)
 * 
 * Sets available = false and returns if symbol not found.
 */
#define LIBREDIFFUSION_SYMBOL_INIT(prefix, name)                                        \
  do                                                                                    \
  {                                                                                     \
    name                                                                                \
        = m_library.symbol<decltype(&::LIBREDIFFUSION_SYMBOL_FULL_NAME(prefix, name))>( \
            LIBREDIFFUSION_SYMBOL_NAME_STR(prefix, name));                              \
    if(!name)                                                                           \
    {                                                                                   \
      fprintf(                                                                          \
          stderr,                                                                       \
          "missing: '" #name "' ; " LIBREDIFFUSION_SYMBOL_NAME_STR(prefix, name) "\n"); \
      available = false;                                                                \
      return;                                                                           \
    }                                                                                   \
  } while(0)

/**
 * @brief Initialize a function pointer (optional - doesn't fail if missing)
 */
#define LIBREDIFFUSION_SYMBOL_INIT_OPT(prefix, name)                                    \
  do                                                                                    \
  {                                                                                     \
    name                                                                                \
        = m_library.symbol<decltype(&::LIBREDIFFUSION_SYMBOL_FULL_NAME(prefix, name))>( \
            LIBREDIFFUSION_SYMBOL_NAME_STR(prefix, name));                              \
  } while(0)

/**
 * @brief Declare with custom member name (for C++ keywords)
 * @param prefix Symbol prefix
 * @param name Actual symbol name suffix  
 * @param member Member variable name
 */
#define LIBREDIFFUSION_SYMBOL_DEF2(prefix, name, member) \
  decltype(&::LIBREDIFFUSION_SYMBOL_FULL_NAME(prefix, name)) member { }

/**
 * @brief Initialize with custom member name
 */
#define LIBREDIFFUSION_SYMBOL_INIT2(prefix, name, member)                               \
  do                                                                                    \
  {                                                                                     \
    member                                                                              \
        = m_library.symbol<decltype(&::LIBREDIFFUSION_SYMBOL_FULL_NAME(prefix, name))>( \
            LIBREDIFFUSION_SYMBOL_NAME_STR(prefix, name));                              \
    if(!member)                                                                         \
    {                                                                                   \
      available = false;                                                                \
      return;                                                                           \
    }                                                                                   \
  } while(0)
