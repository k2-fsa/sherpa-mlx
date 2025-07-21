# Introduction

## Remove silences from audio files
```bash
# 1. Install mlx

# Linux
python3 -m pip install mlx[cpu]

# macOS
python3 -m pip install mlx

2. Build sherpa-mlx

git clone https://github.com/k2-fsa/sherpa-mlx
cd sherpa-mlx
mkdir build
cd build
export MLX_CMAKE_DIR=$(python3 -m mlx --cmake-dir)
cmake ..
make

3. Download test files to the build directory

wget https://github.com/k2-fsa/sherpa-mlx/releases/download/vad-models/lei-jun-test.wav

# for macOS
wget https://github.com/k2-fsa/sherpa-mlx/releases/download/vad-models/silero-vad-v4.mlxfn

# for linux
# wget -O silero-vad-v4.mlxfn https://github.com/k2-fsa/sherpa-mlx/releases/download/vad-models/silero-vad-v4-linux.mlxfn

4. Run it

./bin/sherpa-mlx-vad \
  --silero-vad-model=./silero-vad-v4.mlxfn \
  ./lei-jun-test.wav \
  ./lei-jun-test-no-silence.wav
```
