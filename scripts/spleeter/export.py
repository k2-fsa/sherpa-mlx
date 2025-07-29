#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import mlx.core as mx
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
        elif k.startswith("conv") and k.endswith(".weight") and len(v.shape) == 4:
            v = v.permute(0, 2, 3, 1)
        elif k.startswith("up") and k.endswith(".weight") and len(v.shape) == 4:
            v = v.permute(1, 2, 3, 0)

        ans[k] = v

    return ans


def test(torch_model, mlx_model):
    torch.manual_seed(20250729)
    x0 = torch.rand(2, 3, 512, 1024)  # (N, C, H, W)
    x1 = x0.permute(0, 2, 3, 1)  # (N, H, W, C)
    x1 = mx.array(x1.numpy())
    print(x0.shape, x1.shape, x0.sum(), x1.sum(), x0.mean(), x1.mean())


def main():
    args = get_args()

    accompaniment_state_dict = torch.load(
        "./accompaniment.pt", weights_only=True, map_location="cpu"
    )
    vocals_state_dict = torch.load("./vocals.pt", weights_only=True, map_location="cpu")

    accompaniment = TorchUNet()
    vocals = TorchUNet()

    accompaniment.load_state_dict(accompaniment_state_dict, strict=True)
    vocals.load_state_dict(vocals_state_dict, strict=True)

    accompaniment_mlx = UNet()
    vocals_mlx = UNet()

    mx.eval(accompaniment_mlx.parameters())
    mx.eval(vocals_mlx.parameters())

    accompaniment_mlx_parameters = tree_flatten(accompaniment_mlx.parameters())
    vocals_mlx_parameters = tree_flatten(vocals_mlx.parameters())

    # transpose the conv.weight from (16, 2, 5, 5) to (16, 5, 5, 2)
    # transpose the conv1.weight from (32, 16, 5, 5) to (32, 5, 5, 16)
    # transpose the conv2.weight from (64, 32, 5, 5) to (64, 5, 5, 32)
    # transpose the conv3.weight from (128, 64, 5, 5) to (128, 5, 5, 64)
    # transpose the conv4.weight from (256, 128, 5, 5) to (256, 5, 5, 128)
    # transpose the conv5.weight from (512, 256, 5, 5) to (512, 5, 5, 256)

    # transpose the up1.weight from (512, 256, 5, 5) to (256, 5, 5, 512)
    # transpose the up2.weight from (512, 128, 5, 5) to (128, 5, 5, 512)
    # transpose the up3.weight from (256, 64, 5, 5) to (64, 5, 5, 256)
    # transpose the up4.weight from (128, 32, 5, 5) to (32, 5, 5, 128)
    # transpose the up5.weight from (64, 16, 5, 5) to (16, 5, 5, 64)
    # transpose the up6.weight from (32, 1, 5, 5) to (1, 5, 5, 32)
    # transpose the up7.weight from (2, 1, 4, 4) to (2, 4, 4, 1)
    if False:
        for k, v in accompaniment_mlx_parameters:
            print(k, v.shape, k in accompaniment_state_dict)
            if k in accompaniment_state_dict:
                print(" ", accompaniment_state_dict[k].shape)

    accompaniment_state_dict = replace(accompaniment_state_dict)
    vocals_state_dict = replace(vocals_state_dict)

    accompaniment_state_dict = tree_map(mx.array, accompaniment_state_dict)
    vocals_state_dict = tree_map(mx.array, vocals_state_dict)

    accompaniment_mlx.update(tree_unflatten(list(accompaniment_state_dict.items())))
    vocals_mlx.update(tree_unflatten(list(vocals_state_dict.items())))

    accompaniment_mlx.eval()
    vocals_mlx.eval()

    test(accompaniment, accompaniment_mlx)


if __name__ == "__main__":
    main()
