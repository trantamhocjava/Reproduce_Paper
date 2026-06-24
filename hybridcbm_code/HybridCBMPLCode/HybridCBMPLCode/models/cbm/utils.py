import torch.nn as nn


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def get_conv_layer_substring(name):
    # This logic is probably more complex than it needs to be but it works.
    if name[:5] == "layer":
        sublayer_substring = ".".join(name.split(".")[:3])
        if "conv" in sublayer_substring:
            return sublayer_substring

    return None
