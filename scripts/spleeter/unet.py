# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import mlx.nn as nn
import mlx.core as mx
import torch


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 16, kernel_size=5, stride=(2, 2), padding=0)
        self.bn = nn.BatchNorm(16, track_running_stats=True, eps=1e-3, momentum=0.01)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=5, stride=(2, 2), padding=0)
        self.bn1 = nn.BatchNorm(32, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=(2, 2), padding=0)
        self.bn2 = nn.BatchNorm(64, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=(2, 2), padding=0)
        self.bn3 = nn.BatchNorm(128, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=(2, 2), padding=0)
        self.bn4 = nn.BatchNorm(256, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=(2, 2), padding=0)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm(256, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.up2 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2)
        self.bn6 = nn.BatchNorm(128, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2)
        self.bn7 = nn.BatchNorm(64, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.up4 = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2)
        self.bn8 = nn.BatchNorm(32, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.up5 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2)
        self.bn9 = nn.BatchNorm(16, track_running_stats=True, eps=1e-3, momentum=0.01)

        self.up6 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)
        self.bn10 = nn.BatchNorm(1, track_running_stats=True, eps=1e-3, momentum=0.01)

        # output logit is False, so we need self.up7
        self.up7 = nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding=3)

    def forward(self, x):
        """
        Args:
          x: (num_audio_channels, num_splits, 512, 1024)
        Returns:
          y: (num_audio_channels, num_splits, 512, 1024)
        """
        x = mx.transpose(x, (1, 2, 3, 0))  # CNHW -> NHWC
        in_x = x

        # in_x is (3, 512, 1024, 2) = (T, 512, 1024, 2)
        x = mx.pad(
            x,
            pad_width=[(0, 0), (1, 2), (1, 2), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        # x: (3, 515, 1027, 2)

        conv1 = self.conv(x)
        batch1 = self.bn(conv1)
        rel1 = nn.leaky_relu(batch1, negative_slope=0.2)

        x = mx.pad(
            rel1,
            pad_width=[(0, 0), (1, 2), (1, 2), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        conv2 = self.conv1(x)  # (3, 128, 256, 32)
        batch2 = self.bn1(conv2)
        rel2 = nn.leaky_relu(batch2, negative_slope=0.2)  # (3, 128, 256, 32)

        x = mx.pad(
            rel2,
            pad_width=[(0, 0), (1, 2), (1, 2), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        conv3 = self.conv2(x)  # (3, 64, 128, 64)
        batch3 = self.bn2(conv3)
        rel3 = nn.leaky_relu(batch3, negative_slope=0.2)  # (3, 64, 128, 64)

        x = mx.pad(
            rel3,
            pad_width=[(0, 0), (1, 2), (1, 2), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        conv4 = self.conv3(x)  # (3, 128, 32, 64)
        batch4 = self.bn3(conv4)
        rel4 = nn.leaky_relu(batch4, negative_slope=0.2)  # (3, 32, 64, 128)

        x = mx.pad(
            rel4,
            pad_width=[(0, 0), (1, 2), (1, 2), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        conv5 = self.conv4(x)  # (3, 16, 32, 256)
        batch5 = self.bn4(conv5)
        rel6 = nn.leaky_relu(batch5, negative_slope=0.2)  # (3, 16, 32, 256)

        x = mx.pad(
            rel6,
            pad_width=[(0, 0), (1, 2), (1, 2), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        conv6 = self.conv5(x)  # (3, 8, 16, 512)

        up1 = self.up1(conv6)  # (3, 19, 35, 256)
        up1 = up1[:, 1:-2, 1:-2, :]  # (3, 16, 32, 256)
        up1 = nn.relu(up1)
        batch7 = self.bn5(up1)
        merge1 = mx.concatenate([conv5, batch7], axis=3)  # (3, 16, 32, 512)

        up2 = self.up2(merge1)
        up2 = up2[:, 1:-2, 1:-2, :]
        up2 = nn.relu(up2)
        batch8 = self.bn6(up2)

        merge2 = mx.concatenate([conv4, batch8], axis=3)  # (3, 32, 64, 256)

        up3 = self.up3(merge2)
        up3 = up3[:, 1:-2, 1:-2, :]
        up3 = nn.relu(up3)
        batch9 = self.bn7(up3)

        merge3 = mx.concatenate([conv3, batch9], axis=3)  # (3, 128, 64, 128)

        up4 = self.up4(merge3)
        up4 = up4[:, 1:-2, 1:-2, :]
        up4 = nn.relu(up4)
        batch10 = self.bn8(up4)

        merge4 = mx.concatenate([conv2, batch10], axis=3)  # (3, 128, 256, 64)

        up5 = self.up5(merge4)
        up5 = up5[:, 1:-2, 1:-2, :]
        up5 = nn.relu(up5)
        batch11 = self.bn9(up5)

        merge5 = mx.concatenate([conv1, batch11], axis=3)  # (3, 256, 512, 32)

        up6 = self.up6(merge5)
        up6 = up6[:, 1:-2, 1:-2, :]
        up6 = nn.relu(up6)
        batch12 = self.bn10(up6)  # (3, 512, 1024, 1)  = (T, 512, 1024, 1)

        up7 = self.up7(batch12)
        up7 = mx.sigmoid(up7)  # (3, 2, 512, 1024)

        ans = up7 * in_x
        return mx.transpose(ans, (3, 0, 1, 2))  # NHWC -> CNHW

    __call__ = forward
