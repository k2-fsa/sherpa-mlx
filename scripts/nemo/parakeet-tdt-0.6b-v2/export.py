#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json

import mlx.core as mx
from dacite import from_dict
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import quantize_model
from parakeet_mlx.parakeet import ParakeetTDT, ParakeetTDTArgs


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


def export_encoder(encoder, suffix):
    mx.eval(encoder.parameters())

    def my_export(x):
        out, _ = encoder(x)
        return out

    with mx.exporter(f"encoder.{suffix}.mlxfn", my_export) as exporter:
        for i in range(1, 100):
            # 0.5 seconds interval
            exporter(mx.zeros((1, i * 50, 128)))


def export_decoder(decoder, suffix):
    mx.eval(decoder.parameters())
    y = mx.array([[1]], dtype=mx.int32)
    h = mx.zeros((2, 1, 640), dtype=mx.float32)
    c = mx.zeros((2, 1, 640), dtype=mx.float32)

    def my_export(y, h, c):
        out_y, (out_h, out_c) = decoder(y, (h, c))
        return out_y, out_h, out_c

    mx.export_function(f"decoder.{suffix}.mlxfn", my_export, y, h, c)


def export_joiner(joiner, suffix):
    mx.eval(joiner.parameters())
    enc = mx.zeros((1, 1, 1024), dtype=mx.float32)
    dec = mx.zeros((1, 1, 640), dtype=mx.float32)

    def my_export(enc, dec):
        return joiner(enc, dec)

    mx.export_function(f"joiner.{suffix}.mlxfn", my_export, enc, dec)


def main():
    args = get_args()

    with open("./config.json", "r") as f:
        config = json.load(f)

    weight = "model.safetensors"
    cfg = from_dict(ParakeetTDTArgs, config)
    model = ParakeetTDT(cfg)
    model.load_weights(weight)

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
    model.eval()
    mx.eval(model.parameters())

    if args.use_quant:
        model, config = quantize_model(model, {}, q_group_size=64, q_bits=4)
        print("config", config)
    mx.eval(model.parameters())

    export_encoder(model.encoder, suffix=suffix)
    export_decoder(model.decoder, suffix=suffix)
    export_joiner(model.joint, suffix=suffix)


if __name__ == "__main__":
    main()
