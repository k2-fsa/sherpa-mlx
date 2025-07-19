// sherpa-mlx/csrc/silero-vad-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mlx/csrc/silero-vad-model.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "mlx/mlx.h"
#include "sherpa-mlx/csrc/file-utils.h"
#include "sherpa-mlx/csrc/macros.h"

namespace mx = mlx::core;

namespace sherpa_mlx {

class SileroVadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config) : config_(config) { Init(); }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config) : config_(config) {
    Init();

    // We need to read the model from asset
    SHERPA_MLX_LOGE("Not implemented yet");
    SHERPA_MLX_EXIT(-1);
  }

  void Reset() {
    ResetV4();

    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_MLX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_MLX_EXIT(-1);
    }

    float prob = Run(samples, n);

    float threshold = config_.silero_vad.threshold;

    current_sample_ += config_.silero_vad.window_size;

    if (prob > threshold && temp_end_ != 0) {
      temp_end_ = 0;
    }

    if (prob > threshold && temp_start_ == 0) {
      // start speaking, but we require that it must satisfy
      // min_speech_duration
      temp_start_ = current_sample_;
      return false;
    }

    if (prob > threshold && temp_start_ != 0 && !triggered_) {
      if (current_sample_ - temp_start_ < min_speech_samples_) {
        return false;
      }

      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && !triggered_) {
      // silence
      temp_start_ = 0;
      temp_end_ = 0;
      return false;
    }

    if ((prob > threshold - 0.15) && triggered_) {
      // speaking
      return true;
    }

    if ((prob > threshold) && !triggered_) {
      // start speaking
      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && triggered_) {
      // stop to speak
      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        // continue speaking
        return true;
      }
      // stopped speaking
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    return false;
  }

  int32_t WindowShift() const { return config_.silero_vad.window_size; }

  int32_t WindowSize() const { return config_.silero_vad.window_size; }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = config_.sample_rate * s;
  }

  void SetThreshold(float threshold) {
    config_.silero_vad.threshold = threshold;
  }

 private:
  void Init() {
    if (config_.sample_rate != 16000) {
      SHERPA_MLX_LOGE("Expected sample rate 16000. Given: %d",
                      config_.sample_rate);
      SHERPA_MLX_EXIT(-1);
    }

    min_silence_samples_ =
        config_.sample_rate * config_.silero_vad.min_silence_duration;

    min_speech_samples_ =
        config_.sample_rate * config_.silero_vad.min_speech_duration;

    SHERPA_MLX_LOGE("init model");
    model_ = std::make_unique<mx::ImportedFunction>(
        mx::import_function(config_.silero_vad.model));
    SHERPA_MLX_LOGE("init model done");

    Reset();
  }

  void ResetV4() {
    SHERPA_MLX_LOGE("init states");
    if (states_.empty()) {
      states_.reserve(4);
      for (int32_t i = 0; i != 4; ++i) {
        auto s = mx::zeros({1, 64}, mx::float32);
        mx::eval(s);
        states_.push_back(std::move(s));
      }
      SHERPA_MLX_LOGE("init done");
      return;
    }

    for (auto &s : states_) {
      std::fill_n(s.data<float>(), s.size(), 0);
    }
  }

  float Run(const float *samples, int32_t n) { return RunV4(samples, n); }

  float RunV4(const float *samples, int32_t n) {
    // SHERPA_MLX_LOGE("run with n %d", n);
    auto x = mx::array(samples, {1, n});
    mx::eval(x);

    // SHERPA_MLX_LOGE("here");
    std::vector<mx::array> inputs;
    // SHERPA_MLX_LOGE("here");
    inputs.reserve(1 + states_.size());
    SHERPA_MLX_LOGE("here");

    inputs.push_back(std::move(x));
    SHERPA_MLX_LOGE("here");

    for (auto &s : states_) {
      inputs.push_back(std::move(s));
    }
    SHERPA_MLX_LOGE("here");
    mx::eval(inputs);

    SHERPA_MLX_LOGE("iii %d", (int)states_.size());

    SHERPA_MLX_LOGE("herek0 %d", (int)inputs[0].size());
    SHERPA_MLX_LOGE("herek1 %d", (int)inputs[1].size());
    SHERPA_MLX_LOGE("herek2 %d", (int)inputs[2].size());
    SHERPA_MLX_LOGE("herek3 %d", (int)inputs[3].size());

    std::vector<mx::array> outputs = (*model_)(inputs);
    SHERPA_MLX_LOGE("here");
    mx::eval(outputs);

    SHERPA_MLX_LOGE("here %d", (int)outputs.size());
    SHERPA_MLX_LOGE("here0 %d", (int)outputs[0].size());
    SHERPA_MLX_LOGE("here1 %d", (int)outputs[1].size());
    SHERPA_MLX_LOGE("here2 %d", (int)outputs[2].size());
    SHERPA_MLX_LOGE("here3 %d", (int)outputs[3].size());

    // SHERPA_MLX_LOGE("kkk %d", (int)outputs.size());
    // SHERPA_MLX_LOGE("outputs0 %d", (int)outputs[0].size());
    SHERPA_MLX_LOGE("outputs %p", outputs[0].data<float>());
    SHERPA_MLX_LOGE("outputs %.3f", outputs[0].item<float>());
    SHERPA_MLX_LOGE("outputs %p", outputs[0].data<float>());
    float prob = outputs[0].item<float>();
    SHERPA_MLX_LOGE("here: prob: %.3f", prob);
    for (int32_t i = 0; i != states_.size(); ++i) {
      // SHERPA_MLX_LOGE("here i %d", i);
      states_[i] = std::move(outputs[i + 1]);
    }
    // SHERPA_MLX_LOGE("here");

    return prob;
  }

 private:
  VadModelConfig config_;
  int32_t min_silence_samples_ = 0;
  int32_t min_speech_samples_ = 0;

  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;

  std::unique_ptr<mx::ImportedFunction> model_;
  std::vector<mx::array> states_;
};

SileroVadModel::SileroVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SileroVadModel::SileroVadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SileroVadModel::~SileroVadModel() = default;

void SileroVadModel::Reset() { return impl_->Reset(); }

bool SileroVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t SileroVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t SileroVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t SileroVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t SileroVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void SileroVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void SileroVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

#if __ANDROID_API__ >= 9
template SileroVadModel::SileroVadModel(AAssetManager *mgr,
                                        const VadModelConfig &config);
#endif

#if __OHOS__
template SileroVadModel::SileroVadModel(NativeResourceManager *mgr,
                                        const VadModelConfig &config);
#endif

}  // namespace sherpa_mlx
