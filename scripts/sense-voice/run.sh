#!/bin/zsh
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

function install_deps() {
  python -m pip install mlx mlx_lm parakeet-mlx kaldi-native-fbank soundfile librosa
}

function download_model_files() {
  curl -SL -O https://huggingface.co/csukuangfj/mlx-sense-voice-small-safe-tensors/resolve/main/sense-voice-small.safetensors
  curl -SL -O https://huggingface.co/csukuangfj/SenseVoiceSmall/resolve/main/am.mvn
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/zh.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/en.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/ja.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/ko.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/yue.wav
}

install_deps
download_model_files

python ./export.py --dtype float32 --use-quant 0
python ./export.py --dtype float16 --use-quant 0
python ./export.py --dtype bfloat16 --use-quant 0

python ./export.py --dtype float32 --use-quant 1
python ./export.py --dtype float16 --use-quant 1
python ./export.py --dtype bfloat16 --use-quant 1
ls -lh *.mlxfn

for w in zh.wav en.wav ja.wav ko.wav yue.wav; do
  for dtype in float32 float16 bfloat16; do
    for use_quant in 0 1; do
      python ./test.py --dtype $dtype --use-quant $use_quant --wav $w
    done
  done
done
