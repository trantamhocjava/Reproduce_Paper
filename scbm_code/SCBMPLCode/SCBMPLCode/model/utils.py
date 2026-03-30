import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet101_Weights, resnet18, resnet101
from torchvision.transforms import v2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True


def get_backbone(backbone_name):
    if backbone_name == "resnet101_imagenet":
        weights = ResNet101_Weights.IMAGENET1K_V2
        model = resnet101(weights=weights)
        preprocess = weights.transforms()
    elif backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        preprocess = weights.transforms()

    return model, preprocess


def get_v2_list_from_v1_preprocess(preprocess):
    resize = v2.Resize(
        size=[preprocess.resize_size], interpolation=preprocess.interpolation
    )
    center_crop = v2.CenterCrop(size=(preprocess.crop_size, preprocess.crop_size))
    normalize = v2.Normalize(mean=preprocess.mean, std=preprocess.std)

    return [resize, center_crop, v2.ToDtype(torch.float32, scale=True), normalize]
