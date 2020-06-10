import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import numpy as np

_MODEL_URL_DOMAIN = "http://zifuture.com:1000/fs/public_models"
_MODEL_URL_LARGE = "mbv3large-76f5a50e.pth"
_MODEL_URL_SMALL = "mbv3small-09ace125.pth"


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(self.pool(x))


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class Mbv3SmallFast(nn.Module):
    def __init__(self):
        super(Mbv3SmallFast, self).__init__()

        self.keep = [0, 2, 7]
        self.uplayer_shape = [16, 24, 48]
        self.output_channels = 96

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 2),  # 0 *
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),  # 1
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),  # 2 *
            Block(5, 24, 96, 40, nn.ReLU(inplace=True), SeModule(40), 2),  # 3
            Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 4
            Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 5
            Block(5, 40, 120, 48, nn.ReLU(inplace=True), SeModule(48), 1),  # 6
            Block(5, 48, 144, 48, nn.ReLU(inplace=True), SeModule(48), 1),  # 7 *
            Block(5, 48, 288, 96, nn.ReLU(inplace=True), SeModule(96), 2),  # 8
        )

    def load_pretrain(self):
        checkpoint = model_zoo.load_url(f"{_MODEL_URL_DOMAIN}/{_MODEL_URL_SMALL}")
        self.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        return self.hs1(self.bn1(self.conv1(x)))


# Conv BatchNorm Activation
class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

