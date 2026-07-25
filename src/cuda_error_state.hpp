// Pending-CUDA-error bookkeeping, shared by every C-API translation unit.
//
// CUDA errors come in two kinds and F-03 turned on telling them apart.
//
// A PER-CALL failure (cudaErrorMemoryAllocation and friends) is left pending until somebody reads
// it, and reading it clears it. Left unread it is reported by the NEXT entry point — which did
// nothing wrong — so a rejected create made the following, entirely valid, create return -4 and the
// host saw a working bundle refuse to load.
//
// A CONTEXT-LEVEL failure (an illegal address, a launch failure, ECC) cannot be cleared at all: the
// context is dead and every subsequent CUDA call in the process returns it forever. Nothing here can
// recover from that, so it is reported for what it is rather than retried.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>

namespace librediffusion
{

inline thread_local cudaError_t g_last_cuda_error = cudaSuccess;
// Context loss is a property of the CUDA context, not of the calling thread.
inline std::atomic<bool> g_cuda_context_lost{false};

inline void set_cuda_error(cudaError_t err)
{
  g_last_cuda_error = err;
}

inline bool cuda_error_is_context_fatal(cudaError_t err)
{
  switch(err)
  {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorECCUncorrectable:
    case cudaErrorContextIsDestroyed:
    case cudaErrorDeviceUninitialized:
      return true;
    default:
      return false;
  }
}

inline bool cuda_context_lost()
{
  return g_cuda_context_lost.load(std::memory_order_relaxed);
}

// Consume whatever the runtime has pending, so it cannot be attributed to the next call.
inline cudaError_t drain_cuda_error()
{
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    set_cuda_error(err);
    if(cuda_error_is_context_fatal(err))
    {
      g_cuda_context_lost.store(true, std::memory_order_relaxed);
      std::fprintf(
          stderr, "[librediffusion] CUDA CONTEXT LOST: %s - this process cannot render again\n",
          cudaGetErrorString(err));
    }
  }
  return err;
}

} // namespace librediffusion
