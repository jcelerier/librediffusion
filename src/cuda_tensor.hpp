#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nppi.h>
#include <stdexcept>

namespace librediffusion
{
template <typename T>
class CUDATensor
{
public:
  CUDATensor()
      : data_(nullptr)
      , size_(0)
      , owns_memory_(false)
  {
  }

  CUDATensor(size_t size)
      : size_(size)
      , owns_memory_(true)
  {
    cudaMalloc(&data_, size * sizeof(T));
  }

  CUDATensor(T* data, size_t size)
      : data_(data)
      , size_(size)
      , owns_memory_(false)
  {
  }

  ~CUDATensor()
  {
    if(owns_memory_ && data_)
    {
      cudaFree(data_);
    }
  }

  // No copy, only move
  CUDATensor(const CUDATensor&) = delete;
  CUDATensor& operator=(const CUDATensor&) = delete;

  CUDATensor(CUDATensor&& other) noexcept
      : data_(other.data_)
      , size_(other.size_)
      , owns_memory_(other.owns_memory_)
  {
    other.data_ = nullptr;
    other.owns_memory_ = false;
  }

  CUDATensor& operator=(CUDATensor&& other) noexcept
  {
    if(this != &other)
    {
      // Free existing memory if we own it
      if(owns_memory_ && data_)
      {
        cudaFree(data_);
      }
      // Move from other
      data_ = other.data_;
      size_ = other.size_;
      owns_memory_ = other.owns_memory_;
      // Reset other
      other.data_ = nullptr;
      other.size_ = 0;
      other.owns_memory_ = false;
    }
    return *this;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }

  void fill(T value, cudaStream_t stream = 0);
  void load_d2d(const T* src, size_t count, cudaStream_t stream = 0);
  void store_d2d(T* dst, size_t count, cudaStream_t stream = 0) const;

private:
  T* data_;
  size_t size_;
  bool owns_memory_;
};



template <typename T>
void CUDATensor<T>::fill(T value, cudaStream_t stream)
{
  // TODO: Implement optimized fill kernel
  if(stream == 0)
  {
    cudaMemset(data_, 0, size_ * sizeof(T));
  }
  else
  {
    cudaMemsetAsync(data_, 0, size_ * sizeof(T), stream);
  }
}

template <typename T>
void CUDATensor<T>::load_d2d(const T* src, size_t count, cudaStream_t stream)
{
  if(count > size_)
  {
    throw std::runtime_error("Copy count exceeds tensor size");
  }

  if(stream == 0)
  {
    cudaMemcpy(data_, src, count * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  else
  {
    cudaMemcpyAsync(data_, src, count * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  }
}

template <typename T>
void CUDATensor<T>::store_d2d(T* dst, size_t count, cudaStream_t stream) const
{
  if(count > size_)
  {
    throw std::runtime_error("Copy count exceeds tensor size");
  }

  if(stream == 0)
  {
    cudaMemcpy(dst, data_, count * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  else
  {
    cudaMemcpyAsync(dst, data_, count * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  }
}

}
