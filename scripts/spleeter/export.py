#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import mlx.core as mx
import numpy as np
from mlx_lm.utils import quantize_model
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from torch_unet import UNet as TorchUNet
from unet import UNet


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


def replace(state_dict):
    ans = dict()
    for k, v in state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        elif k == "up7.weight":
            v = v.permute(0, 2, 3, 1)
        elif k.startswith("conv") and k.endswith(".weight") and len(v.shape) == 4:
            v = v.permute(0, 2, 3, 1)
        elif k.startswith("up") and k.endswith(".weight") and len(v.shape) == 4:
            v = v.permute(1, 2, 3, 0)

        ans[k] = v

    return ans


@torch.no_grad()
def test(torch_model, mlx_model):
    torch.manual_seed(20250729)
    torch_model.eval()
    mlx_model.eval()
    num_audio_channels = 2
    num_splits = 3
    H = 512
    W = 1024
    x0 = torch.rand(num_audio_channels, num_splits, H, W)
    x1 = mx.array(x0.numpy())

    y0 = torch_model(x0)
    y1 = mlx_model(x1)
    y1 = torch.from_numpy(np.array(y1))
    print("y0-y1", (y0 - y1).abs().max())
    assert torch.allclose(y0, y1, atol=1e-5), (y0 - y1).abs().max()


def export(model, name, suffix):
    mx.eval(model.parameters())

    def my_export(x):
        out = model(x)
        return out

    with mx.exporter(f"{name}.{suffix}.mlxfn", my_export) as exporter:
        # 100 is 1180 seconds, or  19.667 minutes
        for num_splits in range(1, 100):
            exporter(mx.zeros((2, num_splits, 512, 1024)))


def process(args, name):
    state_dict = torch.load(f"./{name}.pt", weights_only=True, map_location="cpu")
    torch_model = TorchUNet()
    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.eval()

    mlx_model = UNet()
    mx.eval(mlx_model.parameters())

    state_dict = replace(state_dict)

    state_dict = tree_map(mx.array, state_dict)
    mlx_model.update(tree_unflatten(list(state_dict.items())))

    mlx_model.eval()
    test(torch_model, mlx_model)

    curr_weights = dict(tree_flatten(mlx_model.parameters()))
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
    mlx_model.update(tree_unflatten(curr_weights))
    mlx_model.eval()
    mx.eval(mlx_model.parameters())

    if args.use_quant:
        model, config = quantize_model(mlx_model, {}, q_group_size=64, q_bits=4)
        print("config", config)
    mx.eval(mlx_model.parameters())

    export(mlx_model, name=name, suffix=suffix)


@torch.no_grad()
def main():
    args = get_args()
    process(args, "accompaniment")
    process(args, "vocals")


if __name__ == "__main__":
    main()
