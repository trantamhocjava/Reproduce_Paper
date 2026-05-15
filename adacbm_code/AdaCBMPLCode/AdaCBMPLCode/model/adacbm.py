import torch
from kltn_utils import kltn_utils
from torch import nn
from torch.nn import functional as F

from . import utils as model_utils


class AdaptiveModule(nn.Module):
    def __init__(self, dim, num_layers=1):
        super().__init__()

        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.LeakyReLU())

        self.linear1 = nn.Sequential(*layers)

    def forward(self, org_img_feat):
        img_feat = self.linear1(org_img_feat)
        img_feat = img_feat + org_img_feat  # residual handling
        img_feat = F.normalize(img_feat, dim=-1)  # normalize

        return img_feat


class AdaCBM(nn.Module):
    def __init__(
        self,
        select_concepts_data,
        config,
    ):
        super().__init__()
        self.config = config

        self.clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

        self.concept_feat = select_concepts_data["select_concept_feat"]
        self.concept2cls = select_concepts_data["select_concept2cls"]

        num_concept, embedding = self.concept_feat.shape

        self.adaptive_layer = AdaptiveModule(
            dim=embedding,
            num_layers=config.num_layers,
        )

        self.mask = model_utils.get_class2concept(
            len(config.class_names),
            self.concept2cls,
        )

        self.class_concept_weight = nn.Parameter(self.mask.clone())
        self.dot_product_bias = nn.Parameter(torch.zeros(num_concept))
        self.class_concept_bias = nn.Parameter(torch.zeros(len(config.class_names)))

        # Grad
        kltn_utils.freeze_module(self.clip_model)

    def to_device(self, device):
        self.mask = self.mask.to(device)
        self.concept_feat = self.concept_feat.to(device)

    def forward(self, imgs):
        self.to_device(imgs.device)

        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.clip_model, imgs
        ).to(torch.float32)
        self.concept_feat = self.concept_feat.to(torch.float32)

        mat = self.class_concept_weight * self.mask
        img_feat = self.adaptive_layer(img_feat)

        dot_product = img_feat @ self.concept_feat.t() + self.dot_product_bias
        class_logit = dot_product @ mat.t() + self.class_concept_bias
        return class_logit
