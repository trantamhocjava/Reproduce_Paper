import torch.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet101_Weights, resnet18, resnet101


def get_cnn_model(backbone_name):
    if backbone_name == "resnet101_imagenet":
        weights = ResNet101_Weights.IMAGENET1K_V2
        model = resnet101(weights=weights)
        preprocess = weights.transforms()
    elif backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        preprocess = weights.transforms()

    return model, preprocess


def get_encoder(config):
    encoder = None
    preprocess = None
    num_feature = 256

    if config.model.encoder_arch == "FCNN":
        num_feature = 256
        encoder = FCNNEncoder(
            num_inputs=config.model.num_covariates, num_hidden=num_feature, num_deep=2
        )
    elif config.model.encoder_arch == "resnet18":
        encoder, preprocess = get_cnn_model(config.model.encoder_arch)

        num_feature = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif config.model.encoder_arch == "simple_CNN":
        num_feature = 256
        encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, num_feature),
            nn.ReLU(),
        )

    return encoder, preprocess, num_feature


def get_head_layer(head_arch, num_concepts, pred_dim):
    if head_arch == "linear":
        head = nn.Linear(num_concepts, pred_dim)
    else:
        fc1_y = nn.Linear(num_concepts, 256)
        fc2_y = nn.Linear(256, pred_dim)
        head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    return head


class FCNNEncoder(nn.Module):
    def __init__(self, num_inputs: int, num_hidden: int, num_deep: int):
        super(FCNNEncoder, self).__init__()

        self.fc0 = nn.Linear(num_inputs, num_hidden)
        self.bn0 = nn.BatchNorm1d(num_hidden)
        self.fcs = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
        self.dp = nn.Dropout(0.05)

    def forward(self, x):
        z = self.bn0(self.dp(F.relu(self.fc0(x))))
        for bn, fc in zip(self.bns, self.fcs):
            z = bn(self.dp(F.relu(fc(z))))
        return z
