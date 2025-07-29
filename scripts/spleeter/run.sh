#!/usr/bin/env bash

set -ex

if [ ! -f vocals.pt ]; then
  curl -SL -O https://huggingface.co/csukuangfj/spleeter-torch/resolve/main/2stems/vocals.pt
fi

if [ ! -f accompaniment.pt ]; then
  curl -SL -O https://huggingface.co/csukuangfj/spleeter-torch/resolve/main/2stems/accompaniment.pt
fi

if [ ! -f ./accompaniment.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-mlx/releases/download/source-separation-models/audio_example.wav
fi

python3 ./export.py --dtype float32
python3 ./export.py --dtype float16
python3 ./export.py --dtype bfloat16

ls -lh *.mlxfn
