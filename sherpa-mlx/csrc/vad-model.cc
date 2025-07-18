// sherpa-mlx/csrc/vad-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mlx/csrc/vad-model.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mlx/csrc/macros.h"
#include "sherpa-mlx/csrc/silero-vad-model.h"

namespace sherpa_mlx {

std::unique_ptr<VadModel> VadModel::Create(const VadModelConfig &config) {
  if (!config.silero_vad.model.empty()) {
    return std::make_unique<SileroVadModel>(config);
  }

  SHERPA_MLX_LOGE("Please provide a vad model");
  return nullptr;
}

template <typename Manager>
std::unique_ptr<VadModel> VadModel::Create(Manager *mgr,
                                           const VadModelConfig &config) {
  if (!config.silero_vad.model.empty()) {
    return std::make_unique<SileroVadModel>(mgr, config);
  }

  SHERPA_MLX_LOGE("Please provide a vad model");
  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<VadModel> VadModel::Create(
    AAssetManager *mgr, const VadModelConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<VadModel> VadModel::Create(
    NativeResourceManager *mgr, const VadModelConfig &config);
#endif
}  // namespace sherpa_mlx
