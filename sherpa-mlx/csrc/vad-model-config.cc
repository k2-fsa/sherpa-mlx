// sherpa-mlx/csrc/vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mlx/csrc/vad-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-mlx/csrc/macros.h"
#include "sherpa-mlx/csrc/text-utils.h"

namespace sherpa_mlx {

void VadModelConfig::Register(ParseOptions *po) {
  silero_vad.Register(po);

  po->Register("vad-sample-rate", &sample_rate,
               "Sample rate expected by the VAD model");

  po->Register("vad-debug", &debug,
               "true to display debug information when loading vad models");
}

bool VadModelConfig::Validate() const {
  if (!silero_vad.model.empty()) {
    return silero_vad.Validate();
  }

  SHERPA_MLX_LOGE("Please provide one VAD model.");

  return false;
}

std::string VadModelConfig::ToString() const {
  std::ostringstream os;

  os << "VadModelConfig(";
  os << "silero_vad=" << silero_vad.ToString() << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_mlx
