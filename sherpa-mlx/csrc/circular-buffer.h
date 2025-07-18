// sherpa-mlx/csrc/circular-buffer.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation

#ifndef SHERPA_MLX_CSRC_CIRCULAR_BUFFER_H_
#define SHERPA_MLX_CSRC_CIRCULAR_BUFFER_H_

#include <cstdint>
#include <vector>

namespace sherpa_mlx {

class CircularBuffer {
 public:
  // Capacity of this buffer. Should be large enough.
  // If it is full, it resizes the buffer, copying the old data.
  // No data loss on buffer overflow.
  explicit CircularBuffer(int32_t capacity);

  // Push an array
  //
  // @param p Pointer to the start address of the array
  // @param n Number of elements in the array
  //
  // Note: If n + Size() > capacity, it resizes the buffer automagically.
  // No data loss on buffer overflow.
  void Push(const float *p, int32_t n);

  // @param start_index Should in the range [head_, tail_)
  // @param n Number of elements to get
  // @return Return a vector of size n containing the requested elements
  std::vector<float> Get(int32_t start_index, int32_t n) const;

  // Remove n elements from the buffer
  //
  // @param n Should be in the range [0, size_]
  void Pop(int32_t n);

  // Number of elements in the buffer.
  int32_t Size() const { return tail_ - head_; }

  // Current position of the head
  int32_t Head() const { return head_; }

  // Current position of the tail
  int32_t Tail() const { return tail_; }

  void Reset() {
    head_ = 0;
    tail_ = 0;
  }

  void Resize(int32_t new_capacity);

 private:
  std::vector<float> buffer_;

  int32_t head_ = 0;  // linear index; always increasing; never wraps around
  int32_t tail_ = 0;  // linear index, always increasing; never wraps around.
};

}  // namespace sherpa_mlx

#endif  // SHERPA_MLX_CSRC_CIRCULAR_BUFFER_H_
