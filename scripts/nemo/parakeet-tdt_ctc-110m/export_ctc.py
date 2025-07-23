#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json

import mlx.core as mx
from dacite import from_dict
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import quantize_model
from parakeet_mlx.parakeet import ParakeetTDTCTC, ParakeetTDTCTCArgs


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


def export(model, suffix):
    mx.eval(model.parameters())

    def my_export(x):
        encoder_out, _ = model.encoder(x)
        log_probs = model.ctc_decoder(encoder_out)
        return log_probs

    with mx.exporter(f"model.{suffix}.mlxfn", my_export) as exporter:
        for i in range(1, 100):
            # 0.5 seconds interval
            exporter(mx.zeros((1, i * 50, 80)))


def main():
    args = get_args()

    with open("./config.json", "r") as f:
        config = json.load(f)

    weight = "model.safetensors"
    cfg = from_dict(ParakeetTDTCTCArgs, config)
    model = ParakeetTDTCTC(cfg)
    model.load_weights(weight)
    #  print(model)

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

    export(model, suffix=suffix)


if __name__ == "__main__":
    main()
