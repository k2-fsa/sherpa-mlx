#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import mlx.core as mx
import numpy as np
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


def export(args, name):
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


@torch.no_grad()
def main():
    args = get_args()
    export(args, "accompaniment")
    export(args, "vocals")
    return


if __name__ == "__main__":
    main()
