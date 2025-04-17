#pragma once

#include "device_object.hpp"
#include "util.hpp"

namespace rocshmem {

class TicketMutex {
  using TicketT = uint64_t;
  struct Turn {
    volatile TicketT vol_{0};
    int8_t padding[248];
  };
  static_assert(sizeof(Turn) == 256);

 public:
  TicketMutex() = default;

  __device__ TicketT lock() {
    TicketT my_ticket{grab_ticket_()};
    GPU_DPRINTF("mutex lock with ticket %lu\n", my_ticket);
    wait_turn_(my_ticket);
    GPU_DPRINTF("proceeding with ticket %lu\n", my_ticket);
    return my_ticket;
  }

  __device__ void unlock(TicketT my_ticket) {
    TicketT next_ticket{my_ticket + 1};
    GPU_DPRINTF("mutex unlock next ticket %lu\n", my_ticket);
    signal_next_(next_ticket);
  }

  __device__ TicketT grab_ticket_() {
    TicketT ticket{__hip_atomic_fetch_add(&ticket_, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM)};
    return ticket;
  }

  __device__ bool is_turn_(TicketT ticket) {
    size_t index{ticket % memory_channels_};
    return turns_[index].vol_ == ticket;
  }

  __device__ void wait_turn_(TicketT ticket) {
    size_t index{ticket % memory_channels_};
    while (turns_[index].vol_ != ticket) {
      GPU_DPRINTF("spinning on ticket %lu\n", ticket);
    }
  }

  __device__ void signal_next_(TicketT ticket) {
    size_t index{ticket % memory_channels_};
    turns_[index].vol_ = ticket;
    __threadfence();
  }

 private:
  TicketT ticket_{0};
  static constexpr unsigned memory_channels_{32};
  Turn turns_[memory_channels_]{};
};

template <typename ALLOCATOR>
class TicketMutexDevObj {
  using DevObjT = DeviceObject<ALLOCATOR, TicketMutex>;

 public:
  TicketMutexDevObj() = default;

  TicketMutexDevObj(int dev_id) {
    obj_ = DevObjT(dev_id);
    new (obj_.get()) TicketMutex();
  }

  ~TicketMutexDevObj() {
    obj_.get()->~TicketMutex();
  }

  __host__ __device__ TicketMutex* get() { return obj_.get(); }

 private:
  DevObjT obj_{};
};

}  // namespace rocshmem

