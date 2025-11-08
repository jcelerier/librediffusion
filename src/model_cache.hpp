/**
 * Thread-Safe LRU Cache for TensorRT Engines
 *
 * Uses boost::concurrent_flat_map for lock-free concurrent access.
 * Caches the low-level TensorRT objects (IRuntime, ICudaEngine) which are
 * expensive to load from disk. Wrappers create their own IExecutionContext
 * from the cached engine.
 */

#pragma once

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <cuda_runtime.h>

#include <boost/unordered/concurrent_flat_map.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace librediffusion
{

/**
 * Logger for TensorRT
 */
class TensorRTLogger : public nvinfer1::ILogger
{
public:
  TensorRTLogger() = default;
  ~TensorRTLogger() override = default;

  void log(Severity severity, const char* msg) noexcept override;

private:
  std::string severityToString(Severity severity);
};

/**
 * Low-level file stream reader for TensorRT engine deserialization.
 */
class FileStreamReader : public nvinfer1::IStreamReaderV2
{
public:
#if defined(_WIN32) || defined(_WIN64)
  explicit FileStreamReader(const std::string& path)
      : file_handle_(INVALID_HANDLE_VALUE)
  {
    file_handle_ = CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  }

  ~FileStreamReader() override
  {
    if(file_handle_ != INVALID_HANDLE_VALUE)
    {
      CloseHandle(file_handle_);
    }
  }

  int64_t fileSize() const noexcept
  {
    if(!isValid())
      return 0;

    LARGE_INTEGER file_size;
    if(GetFileSizeEx(file_handle_, &file_size))
    {
      return file_size.QuadPart;
    }
    else
    {
      return 0;
    }
  }

  bool isValid() const { return file_handle_ != INVALID_HANDLE_VALUE; }

  int64_t read(void* destination, int64_t nbBytes, cudaStream_t stream) noexcept override
  {
    if(file_handle_ == INVALID_HANDLE_VALUE || nbBytes <= 0)
      return -1;

    // Check if destination is device memory
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, destination);

    bool is_device_memory = (err == cudaSuccess) &&
                            (attrs.type == cudaMemoryTypeDevice ||
                             attrs.type == cudaMemoryTypeManaged);

    if(is_device_memory)
    {
      // Read to host buffer first, then copy to device
      thread_local std::vector<char> host_buffer;
      if(host_buffer.size() < static_cast<size_t>(nbBytes))
      {
        host_buffer.resize(static_cast<size_t>(nbBytes));
      }

      // Read in chunks to support 64-bit sizes (ReadFile uses 32-bit DWORD)
      int64_t total_read = 0;
      while(total_read < nbBytes)
      {
        int64_t remaining = nbBytes - total_read;
        DWORD chunk_size = (remaining > MAXDWORD) ? MAXDWORD : static_cast<DWORD>(remaining);
        DWORD bytes_read = 0;
        if(!ReadFile(file_handle_, host_buffer.data() + total_read, chunk_size,
                     &bytes_read, nullptr))
        {
          return -1;
        }
        if(bytes_read == 0)
          break; // EOF
        total_read += bytes_read;
      }

      if(total_read > 0)
      {
        cudaMemcpyAsync(destination, host_buffer.data(), total_read,
                        cudaMemcpyHostToDevice, stream);
      }
      return total_read;
    }
    else
    {
      // Direct read to host memory in chunks to support 64-bit sizes
      int64_t total_read = 0;
      while(total_read < nbBytes)
      {
        int64_t remaining = nbBytes - total_read;
        DWORD chunk_size = (remaining > MAXDWORD) ? MAXDWORD : static_cast<DWORD>(remaining);
        DWORD bytes_read = 0;
        if(!ReadFile(file_handle_, static_cast<char*>(destination) + total_read,
                     chunk_size, &bytes_read, nullptr))
        {
          return -1;
        }
        if(bytes_read == 0)
          break; // EOF
        total_read += bytes_read;
      }
      return total_read;
    }
  }

  bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override
  {
    if(file_handle_ == INVALID_HANDLE_VALUE)
      return false;

    DWORD move_method;
    switch(where)
    {
      case nvinfer1::SeekPosition::kSET:
        move_method = FILE_BEGIN;
        break;
      case nvinfer1::SeekPosition::kCUR:
        move_method = FILE_CURRENT;
        break;
      case nvinfer1::SeekPosition::kEND:
        move_method = FILE_END;
        break;
      default:
        return false;
    }

    LARGE_INTEGER li;
    li.QuadPart = offset;
    return SetFilePointerEx(file_handle_, li, nullptr, move_method) != 0;
  }

private:
  HANDLE file_handle_;

#else // Linux/Unix

  explicit FileStreamReader(const std::string& path)
      : fd_(-1)
  {
    fd_ = ::open(path.c_str(), O_RDONLY);
  }

  ~FileStreamReader() override
  {
    if(fd_ >= 0)
    {
      ::close(fd_);
    }
  }

  int64_t fileSize() const noexcept
  {
    if(!isValid())
      return 0;

    struct stat file_status{};
    if(fstat(fd_, &file_status) == -1)
      return 0;

    return file_status.st_size;
  }

  bool isValid() const { return fd_ >= 0; }

  int64_t read(void* destination, int64_t nbBytes, cudaStream_t stream) noexcept override
  {
    if(fd_ < 0 || nbBytes <= 0)
      return -1;

    // Check if destination is device memory
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, destination);

    bool is_device_memory
        = (err == cudaSuccess)
          && (attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged);

    if(is_device_memory)
    {
      // Read to host buffer first, then copy to device
      // FIXME mmap ??
      thread_local std::vector<char> host_buffer;
      if(host_buffer.size() < static_cast<size_t>(nbBytes))
      {
        host_buffer.resize(static_cast<size_t>(nbBytes));
      }

      ssize_t total_read = 0;
      while(total_read < nbBytes)
      {
        ssize_t bytes_read
            = ::read(fd_, host_buffer.data() + total_read, nbBytes - total_read);
        if(bytes_read < 0)
          return -1;
        if(bytes_read == 0)
          break; // EOF
        total_read += bytes_read;
      }

      if(total_read > 0)
      {
        cudaMemcpyAsync(
            destination, host_buffer.data(), total_read, cudaMemcpyHostToDevice, stream);
      }
      return total_read;
    }
    else
    {
      // Direct read to host memory
      ssize_t total_read = 0;
      while(total_read < nbBytes)
      {
        ssize_t bytes_read = ::read(
            fd_, static_cast<char*>(destination) + total_read, nbBytes - total_read);
        if(bytes_read < 0)
          return -1;
        if(bytes_read == 0)
          break; // EOF
        total_read += bytes_read;
      }
      return total_read;
    }
  }

  bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override
  {
    if(fd_ < 0)
      return false;

    int whence;
    switch(where)
    {
      case nvinfer1::SeekPosition::kSET:
        whence = SEEK_SET;
        break;
      case nvinfer1::SeekPosition::kCUR:
        whence = SEEK_CUR;
        break;
      case nvinfer1::SeekPosition::kEND:
        whence = SEEK_END;
        break;
      default:
        return false;
    }

    return ::lseek(fd_, offset, whence) != -1;
  }

private:
  int fd_;

#endif

  // Required by IStreamReaderV2
  FileStreamReader(const FileStreamReader&) = delete;
  FileStreamReader& operator=(const FileStreamReader&) = delete;
};

/**
 * Cached TensorRT engine components.
 *
 * Holds the runtime and deserialized engine which are expensive to create.
 * Wrappers obtain a shared_ptr to this and create their own IExecutionContext.
 */
class CachedTensorRTEngine
{
public:
  CachedTensorRTEngine() = default;

  CachedTensorRTEngine(nvinfer1::IRuntime* rt, nvinfer1::ICudaEngine* eng)
      : runtime_(rt)
      , engine_(eng)
  {
    touch();
  }

  ~CachedTensorRTEngine()
  {
    // Order matters: engine must be destroyed before runtime
    if(engine_)
      delete engine_;
    if(runtime_)
      delete runtime_;
  }

  // Non-copyable, non-movable due to raw TensorRT pointers
  CachedTensorRTEngine(const CachedTensorRTEngine&) = delete;
  CachedTensorRTEngine& operator=(const CachedTensorRTEngine&) = delete;
  CachedTensorRTEngine(CachedTensorRTEngine&&) = delete;
  CachedTensorRTEngine& operator=(CachedTensorRTEngine&&) = delete;

  void touch() const
  {
    last_access_time_.store(
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()),
        std::memory_order_relaxed);
    access_count_.fetch_add(1, std::memory_order_relaxed);
  }

  uint64_t lastAccessTime() const
  {
    return last_access_time_.load(std::memory_order_relaxed);
  }

  uint64_t accessCount() const
  {
    return access_count_.load(std::memory_order_relaxed);
  }

  /**
   * Create an execution context from this cached engine.
   * Each wrapper should call this to get its own context.
   */
  nvinfer1::IExecutionContext* createExecutionContext() const
  {
    touch();
    if(!engine_)
      return nullptr;
    return engine_->createExecutionContext();
  }

  nvinfer1::ICudaEngine* getEngine() const
  {
    touch();
    return engine_;
  }

  nvinfer1::IRuntime* getRuntime() const
  {
    return runtime_;
  }

  bool isValid() const
  {
    return engine_ != nullptr;
  }

private:
  nvinfer1::IRuntime* runtime_{nullptr};
  nvinfer1::ICudaEngine* engine_{nullptr};

  mutable std::atomic<uint64_t> last_access_time_{0};
  mutable std::atomic<uint64_t> access_count_{0};
};

/**
 * Thread-safe LRU cache for TensorRT engines using boost::concurrent_flat_map.
 *
 * The cache stores shared_ptr to CachedTensorRTEngine, allowing multiple
 * wrappers to share the same underlying engine while having their own
 * execution contexts.
 */
class TensorRTEngineCache
{
public:
  using EnginePtr = std::shared_ptr<CachedTensorRTEngine>;
  using MapType = boost::unordered::concurrent_flat_map<std::string, EnginePtr>;

  explicit TensorRTEngineCache(size_t max_entries = 16)
      : max_entries_(max_entries)
  {
  }

  /**
   * Get a cached engine or nullptr if not found.
   * Updates access time for LRU tracking.
   */
  EnginePtr get(const std::string& engine_path)
  {
    EnginePtr result;

    cache_.visit(engine_path, [&result](const auto& entry) {
      entry.second->touch();
      result = entry.second;
    });

    return result;
  }

  /**
   * Insert an engine into the cache.
   * Returns the cached engine (may be different if another thread inserted first).
   */
  EnginePtr insert(const std::string& engine_path, EnginePtr engine)
  {
    EnginePtr result = engine;

    bool inserted = cache_.emplace_or_cvisit(
        engine_path, engine,
        [&result](const auto& entry) {
          // Key already exists, use existing engine
          entry.second->touch();
          result = entry.second;
        });

    if(inserted)
    {
      evict_if_needed();
    }

    return result;
  }

  /**
   * Get or load an engine using the provided loader function.
   * Thread-safe: only one thread will load if key doesn't exist.
   *
   * @param engine_path Path to the engine file (used as cache key)
   * @param loader Function that loads the engine, returns EnginePtr
   */
  template <typename Loader>
  EnginePtr get_or_load(const std::string& engine_path, Loader&& loader)
  {
    // Fast path: check if already in cache
    EnginePtr result = get(engine_path);
    if(result)
    {
      return result;
    }

    // Slow path: need to load from disk
    std::unique_lock<std::mutex> lock(load_mutex_);

    // Double-check after acquiring lock
    result = get(engine_path);
    if(result)
    {
      return result;
    }

    // Load the engine
    EnginePtr engine = loader();

    // Insert and return
    return insert(engine_path, std::move(engine));
  }

  /**
   * Clear all cached entries
   */
  void clear()
  {
    cache_.clear();
  }

  /**
   * Get current cache size
   */
  size_t size() const
  {
    return cache_.size();
  }

  /**
   * Remove a specific entry from cache
   */
  bool erase(const std::string& engine_path)
  {
    return cache_.erase(engine_path) > 0;
  }

  /**
   * Set maximum cache size
   */
  void set_max_entries(size_t max_entries)
  {
    max_entries_ = max_entries;
    evict_if_needed();
  }

  void evict_one()
  {
    if(cache_.size() == 0)
      return;

    if(auto lru_key = get_lru())
    {
      cache_.erase(*lru_key);
    }
  }

private:
  MapType cache_;
  size_t max_entries_;
  std::mutex load_mutex_;

  std::optional<std::string> get_lru()
  {
    if(cache_.size() == 0)
      return std::nullopt;

    std::string lru_key;
    uint64_t oldest_time = std::numeric_limits<uint64_t>::max();
    bool found = false;

    cache_.visit_all([&](const auto& entry) {
      uint64_t access_time = entry.second->lastAccessTime();
      if(access_time < oldest_time)
      {
        oldest_time = access_time;
        lru_key = entry.first;
        found = true;
      }
    });

    if(found)
      return lru_key;

    return std::nullopt;
  }

  void evict_if_needed()
  {
    while(cache_.size() > max_entries_)
    {
      if(auto lru_key = get_lru())
      {
        cache_.erase(*lru_key);
      }
      else
      {
        break;
      }
    }
  }
};

/**
 * Global TensorRT engine cache singleton.
 */
class GlobalEngineCache
{
public:
  static GlobalEngineCache& instance()
  {
    static GlobalEngineCache cache;
    return cache;
  }

  TensorRTEngineCache& engines()
  {
    return engine_cache_;
  }

  /**
   * Clear all caches
   */
  void clear()
  {
    engine_cache_.clear();
  }

  /**
   * Set maximum cache size
   */
  void set_max_entries(size_t max_entries)
  {
    engine_cache_.set_max_entries(max_entries);
  }

  TensorRTLogger& logger() noexcept { return logger_; }

private:
  GlobalEngineCache() = default;

  TensorRTEngineCache engine_cache_{16};
  TensorRTLogger logger_;
};

/**
 * Load a TensorRT engine from file using streaming I/O.
 * Uses low-level OS file operations to avoid loading entire engine into memory.
 * Returns a shared_ptr to the cached engine.
 */
inline std::shared_ptr<CachedTensorRTEngine>
loadEngineFromFile(const std::string& engine_path, nvinfer1::ILogger& logger)
{
  // Create the file stream reader
  FileStreamReader stream_reader(engine_path);
  if(!stream_reader.isValid())
  {
    fprintf(stderr, " !! Failed to open engine file: %s\n", engine_path.c_str());
    return nullptr;
  }

  // Mem info
  {
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    auto sz = stream_reader.fileSize();

    int n = GlobalEngineCache::instance().engines().size();
    while(f < sz && n > 0)
    {
      cudaMemGetInfo(&f, &t);
      if(f < sz)
      {
        GlobalEngineCache::instance().engines().evict_one();
        n--;
      }
      else
      {
        break;
      }
    }

    cudaMemGetInfo(&f, &t);
    // Not enough memory to load the model
    if(f < sz)
      return nullptr;
  }

  // Create runtime
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  if(!runtime)
  {
    fprintf(stderr, " !! Failed to create TensorRT runtime\n");
    return nullptr;
  }

  runtime->setMaxThreads(16);
  runtime->setEngineHostCodeAllowed(true);

  // Deserialize engine using streaming API
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(stream_reader);
  if(!engine)
  {
    fprintf(
        stderr, " !! Failed to deserialize engine from stream: %s\n",
        engine_path.c_str());

    delete runtime;
    return nullptr;
  }

  return std::make_shared<CachedTensorRTEngine>(runtime, engine);
}

/**
 * Get a cached engine, loading from disk if necessary.
 * This is the main entry point for cached engine access.
 */
inline std::shared_ptr<CachedTensorRTEngine>
getCachedEngine(const std::string& engine_path)
{
  return GlobalEngineCache::instance().engines().get_or_load(
      engine_path, [&engine_path]() {
    return loadEngineFromFile(engine_path, GlobalEngineCache::instance().logger());
  });
}

} // namespace librediffusion
