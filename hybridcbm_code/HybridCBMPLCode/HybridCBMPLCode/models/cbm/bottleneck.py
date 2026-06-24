import torch.nn as nn


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(
            in_planes, width, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False
        )

        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
