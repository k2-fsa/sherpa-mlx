#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import argparse
from typing import Tuple

import kaldi_native_fbank as knf
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


def load_cmvn(filename) -> Tuple[str, str]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def compute_feat(
    samples,
    sample_rate,
    neg_mean: np.ndarray,
    inv_stddev: np.ndarray,
    window_size: int = 7,  # lfr_m
    window_shift: int = 6,  # lfr_n
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )

    num_frames = features.shape[0]
    if num_frames % 50 != 0:
        pad = 50 - num_frames % 50
        padding = np.zeros((pad, 80), dtype=np.float32)
        features = np.concatenate([features, padding])

    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )

    features = (features + neg_mean) * inv_stddev

    return features


def main():
    args = get_args()
    samples, sample_rate = load_audio("./zh.wav")
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    neg_mean, inv_stddev = load_cmvn("./am.mvn")
    neg_mean = np.array(
        list(map(lambda x: float(x), neg_mean.split(","))), dtype=np.float32
    )

    inv_stddev = np.array(
        list(map(lambda x: float(x), inv_stddev.split(","))), dtype=np.float32
    )

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        neg_mean=neg_mean,
        inv_stddev=inv_stddev,
        window_size=7,
        window_shift=6,
    )
    features = mx.array(features)[None]
    print("features.shape", features.shape)

    suffix = args.dtype if args.use_quant == 0 else f"{args.dtype}-4bit"
    model = mx.import_function(f"./model.{suffix}.mlxfn")

    language = mx.array([0], dtype=mx.int32)  # auto
    text_norm = mx.array([14], dtype=mx.int32)  # with_itn
    logits = model(features, language, text_norm)[0]
    print(logits.shape)
    idx = logits[0].argmax(axis=-1).tolist()

    last = -1
    blank = 0
    ans = []
    for i in idx:
        if i == last:
            continue
        elif i == blank:
            continue
        else:
            ans.append(i)
            last = i

    id2token = load_tokens("./tokens.txt")
    ans = [id2token[i] for i in ans]
    underline = "▁"
    text = "".join(ans).replace(underline, " ").strip()

    print(text)


if __name__ == "__main__":
    main()
