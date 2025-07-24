#!/usr/bin/env bash

set -ex

function install_deps() {
  python3 -m pip install mlx mlx_lm parakeet-mlx kaldi-native-fbank
}

function download_model_files() {
  # see https://huggingface.co/nvidia/parakeet-tdt_ctc-110m
  curl -SL -O https://huggingface.co/mlx-community/parakeet-tdt_ctc-110m/resolve/main/config.json
  curl -SL -O https://huggingface.co/mlx-community/parakeet-tdt_ctc-110m/resolve/main/model.safetensors
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2/resolve/main/test_wavs/0.wav
}

install_deps

if [ ! -f config.json ]; then
  download_model_files
fi

python3 ./generate_tokens.py

echo "export transducer"
for dtype in float32 float16 bfloat16; do
  for use_quant in 0 1; do
    python3 ./export.py --dtype $dtype --use-quant $use_quant
  done
done

echo "export ctc"
for dtype in float32 float16 bfloat16; do
  for use_quant in 0 1; do
    python3 ./export_ctc.py --dtype $dtype --use-quant $use_quant
  done
done

ls -lh

echo "test transducer"
for dtype in float32 float16 bfloat16; do
  for use_quant in 0 1; do
    python3 ./test.py --dtype $dtype --use-quant $use_quant
  done
done

echo "test ctc"
for dtype in float32 float16 bfloat16; do
  for use_quant in 0 1; do
    python3 ./test_ctc.py --dtype $dtype --use-quant $use_quant
  done
done

ls -lh
