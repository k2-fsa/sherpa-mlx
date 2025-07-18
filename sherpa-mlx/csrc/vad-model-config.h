// sherpa-mlx/csrc/vad-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_MLX_CSRC_VAD_MODEL_CONFIG_H_
#define SHERPA_MLX_CSRC_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mlx/csrc/parse-options.h"
#include "sherpa-mlx/csrc/silero-vad-model-config.h"

namespace sherpa_mlx {

struct VadModelConfig {
  SileroVadModelConfig silero_vad;

  int32_t sample_rate = 16000;

  // true to show debug information when loading models
  bool debug = false;

  VadModelConfig() = default;

  VadModelConfig(const SileroVadModelConfig &silero_vad, int32_t sample_rate,
                 bool debug)
      : silero_vad(silero_vad), sample_rate(sample_rate), debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mlx

#endif  // SHERPA_MLX_CSRC_VAD_MODEL_CONFIG_H_
