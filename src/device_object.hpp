#pragma once

#include <memory>
#include <utility>

#include "util.hpp"

template <typename ALLOCATOR, typename T, size_t SIZE_IN = 1>
class DeviceObject {
 public:
  DeviceObject() = default;

  explicit DeviceObject(int dev_id) : _dev_id(dev_id) {
    T* temp{nullptr};
    //_allocator.allocate(reinterpret_cast<void**>(&temp), _SIZE_BYTES, _dev_id);
    _allocator.allocate(reinterpret_cast<void**>(&temp), _SIZE_BYTES);
    assert(temp);
    memset(static_cast<void*>(temp), 0x0, _SIZE_BYTES);
    std::unique_ptr<T, Deleter> up(temp, Deleter(_dev_id));
    _up = std::move(up);
    _ptr = _up.get();
  }

  DeviceObject(DeviceObject&& other) noexcept
    : _up(std::move(other._up)), _dev_id(other._dev_id) {
    other._dev_id = -1;
    _ptr = _up.get();
  }

  DeviceObject& operator=(DeviceObject&& other) noexcept {
    if (this != &other) {
      _up = std::move(other._up);
      _dev_id = other._dev_id;
      other._dev_id = -1;
      _ptr = _up.get();
    }
    return *this;
  }

  DeviceObject(const DeviceObject& other) {
    _dev_id = other._dev_id;

    T* temp{nullptr};
    //_allocator.allocate(reinterpret_cast<void**>(&temp), _SIZE_BYTES, _dev_id);
    _allocator.allocate(reinterpret_cast<void**>(&temp), _SIZE_BYTES);
    assert(temp);

    //int prev_id{-1};
    //CHECK_HIP(hipGetDevice(&prev_id));
    //CHECK_HIP(hipSetDevice(_dev_id));
    CHECK_HIP(hipMemcpy(temp, other._ptr->get(), other._SIZE_BYTES, hipMemcpyDeviceToDevice));
    //CHECK_HIP(hipSetDevice(prev_id));

    std::unique_ptr<T, Deleter> up(temp, Deleter(_dev_id));
    _up = std::move(up);
    _ptr = _up.get();
  }

  DeviceObject& operator=(const DeviceObject& other) {
    if (this != &other) {
      _dev_id = other._dev_id;

      T* temp{nullptr};
      //_allocator.allocate(reinterpret_cast<void**>(&temp), _SIZE_BYTES, _dev_id);
      _allocator.allocate(reinterpret_cast<void**>(&temp), _SIZE_BYTES);
      assert(temp);

      //int prev_id{-1};
      //CHECK_HIP(hipGetDevice(&prev_id));
      //CHECK_HIP(hipSetDevice(_dev_id));
      CHECK_HIP(hipMemcpy(temp, other._ptr, _SIZE_BYTES, hipMemcpyDeviceToDevice));
      //CHECK_HIP(hipSetDevice(prev_id));

      std::unique_ptr<T, Deleter> up(temp, Deleter(_dev_id));
      _up = std::move(up);
      _ptr = _up.get();
    }
    return *this;
  }

  __host__ __device__ T* get() { return _ptr; }

 private:
  class Deleter {
   public:
    Deleter() {}
    Deleter(int dev_id) : _dev_id(dev_id) {}
    //void operator()(void* x) { _a.deallocate(x, _SIZE_BYTES, _dev_id); }
    void operator()(void* x) { _a.deallocate(x); }

   private:
    ALLOCATOR _a;
    int _dev_id{};
  };

  int _dev_id{-1};
  ALLOCATOR _allocator{};
  std::unique_ptr<T, Deleter> _up{nullptr};
  T* _ptr{nullptr};
  static constexpr size_t _SIZE_BYTES{sizeof(T) * SIZE_IN};
};
