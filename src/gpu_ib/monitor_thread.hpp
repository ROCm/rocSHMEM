#pragma once

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <string>
#include <iomanip>

using namespace std::chrono_literals;

namespace rocshmem {

struct DVPair {
  struct mlx5dv_qp* qp;
  struct mlx5dv_cq* cq;
};

struct QueueMonitor {
  uint8_t* sq_buf;
  uint32_t sq_stride;
  uint32_t sq_cnt;
  std::vector<uint8_t> sq_shadow;
  uint32_t sq_last = 0;

  uint8_t* cq_buf;
  uint32_t cq_stride;
  uint32_t cq_cnt;
  std::vector<uint8_t> cq_shadow;
  uint32_t cq_last = 0;

  volatile uint32_t* qp_dbrec = nullptr;
  uint32_t qp_dbrec_shadow = 0;

  volatile uint64_t* bf_reg = nullptr;
  uint64_t bf_reg_shadow = 0;

  volatile uint32_t* cq_dbrec = nullptr;
  uint32_t cq_dbrec_shadow = 0;

  std::ofstream out_stream;
  std::string filename;

  QueueMonitor(mlx5dv_qp* qp, mlx5dv_cq* cq, const std::string& file)
    : sq_buf(reinterpret_cast<uint8_t*>(qp->sq.buf)),
      sq_stride(qp->sq.stride),
      sq_cnt(qp->sq.wqe_cnt),
      sq_shadow(qp->sq.stride * qp->sq.wqe_cnt),
      cq_buf(reinterpret_cast<uint8_t*>(cq->buf)),
      cq_stride(cq->cqe_size),
      cq_cnt(cq->cqe_cnt),
      cq_shadow(cq->cqe_size * cq->cqe_cnt),
      filename(file) {
    std::memcpy(sq_shadow.data(), sq_buf, sq_stride * sq_cnt);
    std::memcpy(cq_shadow.data(), cq_buf, cq_stride * cq_cnt);
    out_stream.open(filename, std::ios::out | std::ios::trunc);
    // Capture pointers for observed registers
    qp_dbrec = reinterpret_cast<volatile uint32_t*>(qp->dbrec);
    bf_reg   = reinterpret_cast<volatile uint64_t*>(qp->bf.reg);
    cq_dbrec = reinterpret_cast<volatile uint32_t*>(cq->dbrec);

    if (qp_dbrec) qp_dbrec_shadow = *qp_dbrec;
    if (bf_reg)   bf_reg_shadow   = *bf_reg;
    if (cq_dbrec) cq_dbrec_shadow = *cq_dbrec;
  }

  ~QueueMonitor() {
    if (out_stream.is_open()) {
      out_stream.close();
    }
  }

  // Delete copy constructor/assignment
  QueueMonitor(const QueueMonitor&) = delete;
  QueueMonitor& operator=(const QueueMonitor&) = delete;

  // Move constructor
  QueueMonitor(QueueMonitor&& other) noexcept
    : sq_buf(other.sq_buf),
      sq_stride(other.sq_stride),
      sq_cnt(other.sq_cnt),
      sq_shadow(std::move(other.sq_shadow)),
      sq_last(other.sq_last),
      cq_buf(other.cq_buf),
      cq_stride(other.cq_stride),
      cq_cnt(other.cq_cnt),
      cq_shadow(std::move(other.cq_shadow)),
      cq_last(other.cq_last),
      out_stream(std::move(other.out_stream)),
      filename(std::move(other.filename)) {}

  // Move assignment
  QueueMonitor& operator=(QueueMonitor&& other) noexcept {
    if (this != &other) {
      sq_buf = other.sq_buf;
      sq_stride = other.sq_stride;
      sq_cnt = other.sq_cnt;
      sq_shadow = std::move(other.sq_shadow);
      sq_last = other.sq_last;

      cq_buf = other.cq_buf;
      cq_stride = other.cq_stride;
      cq_cnt = other.cq_cnt;
      cq_shadow = std::move(other.cq_shadow);
      cq_last = other.cq_last;

      out_stream = std::move(other.out_stream);
      filename = std::move(other.filename);
    }
    return *this;
  }

  void print_bytes(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      out_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]) << " ";
    }
    out_stream << "\n";
  }

  void monitor() {
    for (uint32_t n = 0; n < sq_cnt; ++n) {
      uint32_t i = (sq_last + n) % sq_cnt;
      uint8_t* curr = sq_buf + i * sq_stride;
      uint8_t* shadow = sq_shadow.data() + i * sq_stride;
      if (std::memcmp(curr, shadow, sq_stride) != 0) {
        out_stream << "SQ[" << i << "]: ";
        print_bytes(curr, sq_stride);
        std::memcpy(shadow, curr, sq_stride);
        sq_last = (i + 1) % sq_cnt;
        break;  // Observe only one new entry per cycle
      }
    }
    for (uint32_t n = 0; n < cq_cnt; ++n) {
      uint32_t i = (cq_last + n) % cq_cnt;
      uint8_t* curr = cq_buf + i * cq_stride;
      uint8_t* shadow = cq_shadow.data() + i * cq_stride;
      if (std::memcmp(curr, shadow, cq_stride) != 0) {
        out_stream << "CQ[" << i << "]: ";
        print_bytes(curr, cq_stride);
        std::memcpy(shadow, curr, cq_stride);
        cq_last = (i + 1) % cq_cnt;
        break;  // Observe only one new entry per cycle
      }
    }
    if (qp_dbrec) {
      uint32_t val = *qp_dbrec;
      if (val != qp_dbrec_shadow) {
        out_stream << "QP.dbrec: " << std::hex << std::setw(8) << std::setfill('0') << val << "\n";
        qp_dbrec_shadow = val;
      }
    }
    if (bf_reg) {
      uint64_t val = *bf_reg;
      if (val != bf_reg_shadow) {
        out_stream << "QP.bf.reg: " << std::hex << std::setw(16) << std::setfill('0') << val << "\n";
        bf_reg_shadow = val;
      }
    }
    if (cq_dbrec) {
      uint32_t val = *cq_dbrec;
      if (val != cq_dbrec_shadow) {
        out_stream << "CQ.dbrec: " << std::hex << std::setw(8) << std::setfill('0') << val << "\n";
        cq_dbrec_shadow = val;
      }
    }
    out_stream.flush();
  }
};

class Monitor {
 public:
  void register_queue(mlx5dv_qp* sq, mlx5dv_cq* cq, const std::string& file) {
    queues.emplace_back(sq, cq, file);
  }

  void start() {
    monitor_thread = std::thread(&Monitor::monitor_loop, this);
  }

  void stop() {
    if (monitor_thread.joinable()) {
      monitor_thread.join();
    }
  }

  ~Monitor() {
    stop();
  }

 private:
  std::vector<QueueMonitor> queues;
  std::thread monitor_thread;

  void monitor_loop() {
    size_t idx = 0;
    const size_t num_queues = queues.size();

    while (1) {
      if (num_queues > 0) {
        queues[idx].monitor();  // Call updated monitor() for this queue
        idx = (idx + 1) % num_queues;
      }

      std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
  }
};

}  // namespace rocshmem
