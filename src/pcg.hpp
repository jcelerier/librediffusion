#pragma once
#include <cstdint>
#include <random>
class pcg
{
public:
  using result_type = uint32_t;
  static constexpr result_type min() noexcept { return 0; }
  static constexpr result_type max() noexcept { return UINT32_MAX; }

  constexpr pcg() noexcept
      : m_state(UINT64_C(0x853c49e6748fea9b))
      , m_inc(UINT64_C(0xda3e39cb94b95bdb))
  {
  }
  explicit pcg(std::random_device& rd) noexcept { seed(rd); }

  constexpr void seed(uint64_t s0, uint64_t s1) noexcept
  {
    m_state = 0;
    m_inc = (s1 << 1) | 1;
    (void)operator()();
    m_state += s0;
    (void)operator()();
  }

  void seed(std::random_device& rd) noexcept
  {
    uint64_t s0 = uint64_t(rd()) << 31 | uint64_t(rd());
    uint64_t s1 = uint64_t(rd()) << 31 | uint64_t(rd());
    seed(s0, s1);
  }

  constexpr result_type operator()() noexcept
  {
    uint64_t oldstate = m_state;
    m_state = oldstate * UINT64_C(6364136223846793005) + m_inc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    int rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  constexpr void discard(unsigned long long n) noexcept
  {
    for(unsigned long long i = 0; i < n; ++i)
      operator()();
  }

  friend constexpr bool operator==(pcg lhs, pcg rhs) noexcept
  {
    return lhs.m_state == rhs.m_state && lhs.m_inc == rhs.m_inc;
  }
  friend constexpr bool operator!=(pcg lhs, pcg rhs) noexcept
  {
    return lhs.m_state != rhs.m_state || lhs.m_inc != rhs.m_inc;
  }

private:
  uint64_t m_state;
  uint64_t m_inc;
};
