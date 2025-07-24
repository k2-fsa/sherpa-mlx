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


def export_encoder(encoder, feat_dim, suffix):
    mx.eval(encoder.parameters())

    def my_export(x):
        out, _ = encoder(x)
        return out

    with mx.exporter(f"encoder.{suffix}.mlxfn", my_export) as exporter:
        for i in range(1, 100):
            # 0.5 seconds interval
            exporter(mx.zeros((1, i * 50, feat_dim)))


def export_decoder(decoder, rnn_layers, hidden_dim, suffix):
    mx.eval(decoder.parameters())
    y = mx.array([[1]], dtype=mx.int32)
    h = mx.zeros((rnn_layers, 1, hidden_dim), dtype=mx.float32)
    c = mx.zeros((rnn_layers, 1, hidden_dim), dtype=mx.float32)

    def my_export(y, h, c):
        out_y, (out_h, out_c) = decoder(y, (h, c))
        return out_y, out_h, out_c

    mx.export_function(f"decoder.{suffix}.mlxfn", my_export, y, h, c)


def export_joiner(joiner, enc_dim, dec_dim, suffix):
    mx.eval(joiner.parameters())
    enc = mx.zeros((1, 1, enc_dim), dtype=mx.float32)
    dec = mx.zeros((1, 1, dec_dim), dtype=mx.float32)

    def my_export(enc, dec):
        return joiner(enc, dec)

    mx.export_function(f"joiner.{suffix}.mlxfn", my_export, enc, dec)


def main():
    args = get_args()

    with open("./config.json", "r") as f:
        config = json.load(f)

    weight = "model.safetensors"
    cfg = from_dict(ParakeetTDTCTCArgs, config)

    feat_dim = cfg.preprocessor.features
    encoder_out_dim = cfg.encoder.d_model
    decoder_out_dim = cfg.decoder.prednet.pred_hidden
    decoder_rnn_layers = cfg.decoder.prednet.pred_rnn_layers

    print("feat_dim", feat_dim)
    print("encoder_out_dim", encoder_out_dim)
    print("decoder_out_dim", decoder_out_dim)
    print("decoder_rnn_layers", decoder_rnn_layers)

    model = ParakeetTDTCTC(cfg)
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

    export_encoder(model.encoder, feat_dim=feat_dim, suffix=suffix)

    export_decoder(
        model.decoder,
        rnn_layers=decoder_rnn_layers,
        hidden_dim=decoder_out_dim,
        suffix=suffix,
    )

    export_joiner(
        model.joint,
        enc_dim=encoder_out_dim,
        dec_dim=decoder_out_dim,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()
