import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet101_Weights, resnet18, resnet101
from torchvision.transforms import v2

from .FCNNEncoder import FCNNEncoder


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True


def get_pretrained_model(backbone_name):
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


def get_encoder(config):
    encoder = None
    preprocess = None
    n_features = 256

    if config.encoder_arch == "FCNN":
        n_features = 256
        encoder = FCNNEncoder(
            num_inputs=config.num_covariates, num_hidden=n_features, num_deep=2
        )
    elif config.encoder_arch == "resnet18":
        encoder, preprocess = get_pretrained_model(config.encoder_arch)

        n_features = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif config.encoder_arch == "simple_CNN":
        n_features = 256
        encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, n_features),
            nn.ReLU(),
        )

    return encoder, preprocess, n_features


def get_head_layer(head_arch, num_concepts, pred_dim):
    if head_arch == "linear":
        head = nn.Linear(num_concepts, pred_dim)
    else:
        fc1_y = nn.Linear(num_concepts, 256)
        fc2_y = nn.Linear(256, pred_dim)
        head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    return head


def get_pred_dim(num_classes):
    if num_classes == 2:
        pred_dim = 1
    elif num_classes > 2:
        pred_dim = num_classes

    return pred_dim
