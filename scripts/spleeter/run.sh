#!/usr/bin/env bash

set -ex

if [ ! -f vocals.pt ]; then
  curl -SL -O https://huggingface.co/csukuangfj/spleeter-torch/resolve/main/2stems/vocals.pt
fi

if [ ! -f accompaniment.pt ]; then
  curl -SL -O https://huggingface.co/csukuangfj/spleeter-torch/resolve/main/2stems/accompaniment.pt
fi
