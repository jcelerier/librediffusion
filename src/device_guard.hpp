#pragma once

/* Per-handle CUDA device.
 *
 * cudaSetDevice sets THREAD state, not process state, so a handle built on one thread and driven
 * from another (a host's render/worker thread) would otherwise run on whatever device that thread
 * defaults to — device 0. Each handle records the device it was created on and every entry point
 * scopes the current device to it, so a call is correct from any thread. */

#include <cuda_runtime.h>

namespace librediffusion
{

// Sets the current device for the enclosing scope and restores the previous one. A no-op when the
// thread is already on the right device, which is the common single-GPU case.
class DeviceGuard
{
public:
  explicit DeviceGuard(int device) noexcept
  {
    if(device < 0)
      return;
    if(cudaGetDevice(&prev_) != cudaSuccess)
      return;
    if(prev_ == device)
      return;
    if(cudaSetDevice(device) == cudaSuccess)
      restore_ = true;
  }
  ~DeviceGuard() noexcept
  {
    if(restore_)
      cudaSetDevice(prev_);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
  int prev_{-1};
  bool restore_{false};
};

// The device a *_create was asked for, or -1 when it is not a usable ordinal. Callers turn -1 into
// their own "invalid argument" result; validating here keeps the message identical everywhere.
inline int resolve_device(int requested) noexcept
{
  int count = 0;
  if(cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
    return -1;
  if(requested < 0 || requested >= count)
    return -1;
  return requested;
}

} // namespace librediffusion
