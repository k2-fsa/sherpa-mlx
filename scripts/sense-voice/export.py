#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import quantize_model

from model import SenseVoiceSmall as SenseVoiceSmallMlx


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


def main():
    args = get_args()

    model = SenseVoiceSmallMlx()
    model.eval()
    mx.eval(model.parameters())

    weights = mx.load("./sense-voice-small.safetensors")
    model.update(tree_unflatten(list(weights.items())))

    curr_weights = dict(tree_flatten(model.parameters()))
    if args.dtype == "float32":
        dtype = mx.float32
    elif args.dtype == "float16":
        dtype = mx.float16
    elif args.dtype == "bfloat16":
        dtype = mx.bfloat16
    else:
        assert False, f"Unsupported dtype {args.dtype}"
    suffix = args.dtype if args.use_quant == 0 else f"{args.dtype}-4bit"

    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    if args.use_quant:
        model, config = quantize_model(model, {}, group_size=64, bits=4)
        print("config", config)

    mx.eval(model.parameters())

    def my_export(x, language, text_norm):
        x_len = mx.array([x.shape[1]], dtype=mx.int32)
        logits, _ = model(x, x_len, language, text_norm)
        return logits

    language = mx.array([0], dtype=mx.int32)
    text_norm = mx.array([14], dtype=mx.int32)
    with mx.exporter(f"model.{suffix}.mlxfn", my_export) as exporter:
        for i in range(1, 30):
            # 1 second interval, max 30 seconds
            num_frames = int(i * 100)
            n = (num_frames - 7) // 6 + 1
            shape = (1, n, 560)
            x = mx.zeros(shape, dtype=mx.float32)
            exporter(x, language, text_norm)


if __name__ == "__main__":
    main()
