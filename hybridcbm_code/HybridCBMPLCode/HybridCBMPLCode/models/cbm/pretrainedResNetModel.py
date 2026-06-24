import math

import torch.nn as nn
from kltn_utils import kltn_utils

from .basicBlock import BasicBlock
from .bottleneck import Bottleneck


class PretrainedResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.inplanes = 64
        self.config = config

        BlockClass = (
            BasicBlock
            if config.model.pretrained_model_name in ["resnet18", "resnet34"]
            else Bottleneck
        )
        layers = {
            "resnet18": [2, 2, 2, 2],
            "resnet34": [3, 4, 6, 3],
            "resnet50": [3, 4, 6, 3],
            "resnet101": [3, 4, 23, 3],
            "resnet152": [3, 8, 36, 3],
        }[config.model.pretrained_model_name]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BlockClass, 64, layers[0])
        self.layer2 = self._make_layer(BlockClass, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BlockClass, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BlockClass, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(config.model.dropout, inplace=False)
        self.conv_layer_dims = {"conv1": 64, "conv2": 128, "conv3": 256, "conv4": 512}
        previous_layer_dims = 512 * BlockClass.expansion

        for i, layer in enumerate(config.model.fc_layers):
            setattr(self, "fc" + str(i + 1), nn.Linear(previous_layer_dims, layer))
            previous_layer_dims = layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def compute_cnn_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.compute_cnn_features(x)
        N_layers = len(self.config.model.fc_layers)

        for i, layer in enumerate(self.config.model.fc_layers):
            fn = getattr(self, "fc" + str(i + 1))
            x = fn(x)
            # No ReLu for last layer
            if i != N_layers - 1:
                x = self.relu(x)

            # Cache results to get intermediate outputs
            setattr(self, "fc%s_out" % str(i + 1), x)

        return x

    def _make_layer(self, BlockClass, planes, blocks, stride=1):
        downsample = None

        # TODO: DEBUG
        kltn_utils.rank_zero_info_newline(f"stride: {stride}")
        kltn_utils.rank_zero_info_newline(f"self.inplanes: {self.inplanes}")
        kltn_utils.rank_zero_info_newline(
            f"planes * BlockClass.expansion: {planes * BlockClass.expansion}"
        )
        # END DEBUG

        if stride != 1 or self.inplanes != planes * BlockClass.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * BlockClass.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BlockClass.expansion),
            )

        layers = []
        layers.append(BlockClass(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * BlockClass.expansion

        for i in range(1, blocks):
            layers.append(BlockClass(self.inplanes, planes))

        return nn.Sequential(*layers)
