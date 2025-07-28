#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import mlx.core as mx
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from torch_unet import UNet as TorchUNet
from unet import UNet


def main():
    accompaniment_state_dict = torch.load(
        "./accompaniment.pt", weights_only=True, map_location="cpu"
    )
    vocals_state_dict = torch.load("./vocals.pt", weights_only=True, map_location="cpu")

    accompaniment = TorchUNet()
    vocals = TorchUNet()

    accompaniment.load_state_dict(accompaniment_state_dict, strict=True)
    vocals.load_state_dict(vocals_state_dict, strict=True)

    accompaniment_mlx = UNet()
    mx.eval(accompaniment_mlx.parameters())
    d = tree_flatten(accompaniment_mlx.parameters())

    # transpose the conv.weight from (16, 2, 5, 5) to (16, 5, 5, 2)
    # transpose the conv1.weight from (32, 16, 5, 5) to (32, 5, 5, 16)
    for k, v in d:
        print(k, v.shape, k in accompaniment_state_dict)
        if k in accompaniment_state_dict:
            print(" ", accompaniment_state_dict[k].shape)


if __name__ == "__main__":
    main()
