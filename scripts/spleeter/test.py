#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import time

import kaldi_native_fbank as knf
import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--use-quant",
        type=int,
        default=0,
    )
    return parser.parse_args()


class MlxModel:
    def __init__(self, vocals: str, accompaniment: str):
        print(vocals, accompaniment)
        self.vocals = mx.import_function(vocals)
        self.accompaniment = mx.import_function(accompaniment)

    def run_vocals(self, x):
        out = self.vocals(x)
        return out[0]

    def run_accompaniment(self, x):
        out = self.accompaniment(x)
        return out[0]


def load_audio(filename):
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    if sample_rate != 44100:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=44100,
        )
        sample_rate = 44100

    # audio: (num_samples, num_channels)
    return np.ascontiguousarray(audio), sample_rate


def main():
    args = get_args()

    audio, sample_rate = load_audio("./audio_example.wav")

    print("audio.shape", audio.shape)

    # 512 frames is 1024/44100*512 = 11.88 seconds
    stft_config = knf.StftConfig(
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        center=False,
        window_type="hann",
    )
    knf_stft = knf.Stft(stft_config)
    knf_istft = knf.IStft(stft_config)

    start = time.time()

    stft_result_c0 = knf_stft(audio[:, 0].tolist())
    stft_result_c1 = knf_stft(audio[:, 1].tolist())
    print("c0 stft", stft_result_c0.num_frames)

    orig_real0 = np.array(stft_result_c0.real, dtype=np.float32).reshape(
        stft_result_c0.num_frames, -1
    )
    orig_imag0 = np.array(stft_result_c0.imag, dtype=np.float32).reshape(
        stft_result_c0.num_frames, -1
    )

    orig_real1 = np.array(stft_result_c1.real, dtype=np.float32).reshape(
        stft_result_c1.num_frames, -1
    )
    orig_imag1 = np.array(stft_result_c1.imag, dtype=np.float32).reshape(
        stft_result_c1.num_frames, -1
    )

    real0 = mx.array(orig_real0)
    imag0 = mx.array(orig_imag0)
    real1 = mx.array(orig_real1)
    imag1 = mx.array(orig_imag1)
    # (num_frames, n_fft/2_1)
    print("real0", real0.shape)

    # keep only the first 1024 bins
    real0 = real0[:, :1024]
    imag0 = imag0[:, :1024]
    real1 = real1[:, :1024]
    imag1 = imag1[:, :1024]
    print("real0", real0.shape)

    stft0 = (real0.square() + imag0.square()).sqrt()
    stft1 = (real1.square() + imag1.square()).sqrt()

    # pad it to multiple of 512
    padding = 512 - real0.shape[0] % 512
    print("padding", padding)
    if padding > 0:
        stft0 = mx.pad(
            stft0,
            pad_width=[(0, padding), (0, 0)],
            mode="constant",
            constant_values=0,
        )

        stft1 = mx.pad(
            stft1,
            pad_width=[(0, padding), (0, 0)],
            mode="constant",
            constant_values=0,
        )
    stft0 = stft0.reshape(1, -1, 512, 1024)
    stft1 = stft1.reshape(1, -1, 512, 1024)

    stft_01 = mx.concatenate([stft0, stft1], axis=0)

    print("stft_01", stft_01.shape, stft_01.dtype)

    suffix = args.dtype if args.use_quant == 0 else f"{args.dtype}-4bit"

    model = MlxModel(
        vocals=f"./vocals.{suffix}.mlxfn",
        accompaniment=f"./accompaniment.{suffix}.mlxfn",
    )

    vocals_spec = model.run_vocals(stft_01)
    accompaniment_spec = model.run_accompaniment(stft_01)
    mx.eval(vocals_spec)
    mx.eval(accompaniment_spec)
    # (num_audio_channels, num_splits, 512, 1024) # (C, N, H, W)

    sum_spec = (vocals_spec.square() + accompaniment_spec.square()) + 1e-10
    print("sum spec.shape", sum_spec.shape)

    vocals_spec = (vocals_spec**2 + 1e-10 / 2) / sum_spec
    accompaniment_spec = (accompaniment_spec**2 + 1e-10 / 2) / sum_spec

    for name, spec in zip(
        ["vocals", "accompaniment"], [vocals_spec, accompaniment_spec]
    ):
        spec_c0 = spec[0]
        spec_c1 = spec[1]

        spec_c0 = spec_c0.reshape(-1, 1024)
        spec_c1 = spec_c1.reshape(-1, 1024)

        spec_c0 = spec_c0[: stft_result_c0.num_frames, :]
        spec_c1 = spec_c1[: stft_result_c0.num_frames, :]

        spec_c0 = mx.pad(spec_c0, ((0, 0), (0, 2049 - 1024)))
        spec_c1 = mx.pad(spec_c1, ((0, 0), (0, 2049 - 1024)))

        spec_c0_real = spec_c0 * orig_real0
        spec_c0_imag = spec_c0 * orig_imag0

        spec_c1_real = spec_c1 * orig_real1
        spec_c1_imag = spec_c1 * orig_imag1

        result0 = knf.StftResult(
            real=spec_c0_real.reshape(-1).tolist(),
            imag=spec_c0_imag.reshape(-1).tolist(),
            num_frames=orig_real0.shape[0],
        )

        result1 = knf.StftResult(
            real=spec_c1_real.reshape(-1).tolist(),
            imag=spec_c1_imag.reshape(-1).tolist(),
            num_frames=orig_real1.shape[0],
        )

        wav0 = knf_istft(result0)
        wav1 = knf_istft(result1)

        wav = np.array([wav0, wav1], dtype=np.float32)
        wav = np.transpose(wav)
        # now wav is (num_samples, num_channels)
        sf.write(f"./{name}.wav", wav, 44100)

        print(f"Saved to ./{name}.wav")

    end = time.time()
    elapsed_seconds = end - start
    audio_duration = audio.shape[0] / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
