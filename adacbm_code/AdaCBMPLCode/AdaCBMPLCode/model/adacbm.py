import torch
from torch import nn
from torch.nn import functional as F

from .. import utils
from . import utils as model_utils


def get_init_mask(num_class, concept2cls, init_val=1):
    num_concept = concept2cls.shape[1]
    concept2cls = torch.from_numpy(concept2cls).long().view(1, -1)
    init_mask = torch.zeros((num_class, num_concept))
    init_mask.scatter_(0, concept2cls, init_val)

    return init_mask


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
        img_feat = F.normalize(img_feat, dim=1)  # normalize

        return img_feat


class AdaCBM(nn.Module):
    def __init__(
        self,
        select_concepts_data,
        config,
    ):
        super().__init__()
        self.config = config

        self.clip_model, tokenizer = utils.build_clip_model(config.clip_model)

        self.concept_feat = select_concepts_data["select_concept_feat"]
        self.concept2cls = select_concepts_data["select_concept2cls"]

        num_concept, embedding = self.concept_feat.shape

        self.adaptive_layer = AdaptiveModule(
            dim=embedding,
            num_layers=config.num_layers,
        )

        self.mask = get_init_mask(
            len(config.class_names),
            self.concept2cls,
        )

        self.class_concept_weight = nn.Parameter(self.mask.clone())
        self.dot_product_bias = nn.Parameter(torch.zeros(num_concept))
        self.class_concept_bias = nn.Parameter(torch.zeros(len(config.class_names)))

        # Grad
        model_utils.freeze_module(self.clip_model)

    def forward(self, imgs):
        self.clip_model.eval()
        with torch.no_grad():
            img_feat = self.clip_model(imgs, None)[0]

        mat = self.class_concept_weight * self.mask
        image_embed = self.adaptive_layer(img_feat)

        dot_product = image_embed @ self.concept_feat.t() + self.dot_product_bias
        class_logit = dot_product @ mat.t() + self.class_concept_bias
        return class_logit, dot_product
