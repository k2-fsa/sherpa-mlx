// sherpa-mlx/csrc/silero-vad-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_MLX_CSRC_SILERO_VAD_MODEL_H_
#define SHERPA_MLX_CSRC_SILERO_VAD_MODEL_H_

#include <memory>

#include "sherpa-mlx/csrc/vad-model.h"

namespace sherpa_mlx {

class SileroVadModel : public VadModel {
 public:
  explicit SileroVadModel(const VadModelConfig &config);

  template <typename Manager>
  SileroVadModel(Manager *mgr, const VadModelConfig &config);

  ~SileroVadModel() override;

  // reset the internal model states
  void Reset() override;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples.
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  bool IsSpeech(const float *samples, int32_t n) override;

  // For silero vad V4, it is WindowShift().
  // For silero vad V5, it is WindowShift()+64 for 16kHz and
  //                          WindowShift()+32 for 8kHz
  int32_t WindowSize() const override;

  // 512
  int32_t WindowShift() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

  void SetMinSilenceDuration(float s) override;
  void SetThreshold(float threshold) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mlx

#endif  // SHERPA_MLX_CSRC_SILERO_VAD_MODEL_H_
