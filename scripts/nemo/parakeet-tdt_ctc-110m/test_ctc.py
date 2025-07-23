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


def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = 80

    opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(opts)
    return fbank


class MlxModel:
    def __init__(
        self,
        model: str,
    ):
        print(model)
        self.model = mx.import_function(model)

    def __call__(self, x: mx.array):
        """
        Args:
          x: (T, C)
        Returns:
          log_probs: (N, T, C)
        """
        x = mx.expand_dims(x, 0)
        log_probs = self.model(x)[0]
        return log_probs


def compute_features(audio, fbank):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans


def main():
    args = get_args()
    suffix = args.dtype if args.use_quant == 0 else f"{args.dtype}-4bit"
    model = MlxModel(
        model=f"./model.{suffix}.mlxfn",
    )
    id2token = dict()
    with open("./tokens.txt", encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    start = time.time()
    fbank = create_fbank()
    audio, sample_rate = sf.read("./0.wav", dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    tail_padding = np.zeros(sample_rate * 2, dtype=np.float32)
    audio = np.concatenate([audio, tail_padding], axis=0)

    features = compute_features(audio, fbank)
    target_length = ((features.shape[0] + 49) // 50) * 50

    if features.shape[0] < target_length:
        padding = np.zeros(
            (target_length - features.shape[0], 80),
            dtype=np.float32,
        )
        #  padding = features.mean(axis=0, keepdims=True).repeat(
        #      target_length - features.shape[0],
        #      axis=0,
        #  )
        features = np.concatenate([features, padding], axis=0)

    mean = features.mean(axis=0, keepdims=True)
    stddev = features.std(axis=0, keepdims=True) + 1e-5
    features = (features - mean) / stddev
    log_probs = model(mx.array(features))
    mx.eval(log_probs)

    log_probs = log_probs[0]  # remove batch dim
    # log_probs.shape: (119, 1025)

    blank = len(id2token) - 1
    ans = []
    prev = -1

    ids = log_probs.argmax(axis=1).tolist()
    for k in ids:
        if k != blank and k != prev:
            ans.append(k)
        prev = k

    end = time.time()

    elapsed_seconds = end - start
    audio_duration = audio.shape[0] / 16000
    real_time_factor = elapsed_seconds / audio_duration

    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()

    print(text)
    print(f"RTF: {real_time_factor}")


if __name__ == "__main__":
    main()
