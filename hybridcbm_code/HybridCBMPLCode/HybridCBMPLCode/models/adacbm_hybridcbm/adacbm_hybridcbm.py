import torch
import torch.nn.functional as F
from kltn_utils import kltn_const, kltn_utils
from torch import nn

from ..hybridcbm import hybridcbm


class AdaHybridCBM(hybridcbm.HybridCBM):
    def __init__(self, config, select_concept_data):
        super().__init__(config, select_concept_data)

        ## Get cls_head
        self.cls_head = get_cls_head(
            num_class=config.num_class,
            num_dynamic_concept=config.model.num_dynamic_concept,
            concept2class=select_concept_data["concept2class"],
        )

        clip_model_config = kltn_const.CLIP_MODELS[config.model.clip_model]
        img_feat_dim = clip_model_config["embedding_dim"]

        ## Get adaptive_module
        self.adaptive_module = AdaptiveModule(
            dim=img_feat_dim, num_layers=config.model.num_ada_layer
        )

    def setup_grad(self):
        kltn_utils.freeze_module(self.clip_model)

    def forward(self, img):
        # Get concept feat
        concept_feat = torch.cat(
            [self.static_concept_feat, self.dynamic_concept_feat],
            dim=0,
        )
        concept_feat = F.normalize(concept_feat, dim=1)

        # Get img_feat
        self.clip_model.eval()
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.model.clip_model, img
        )
        img_feat = self.adaptive_module(img_feat)
        img_feat = F.normalize(img_feat, dim=-1)

        # Get hybrid_concept_logits
        hybrid_concept_logits = self.scale * (img_feat @ concept_feat.T)

        # Get label logits
        label_logits = self.cls_head(hybrid_concept_logits)

        # Get concept_logits
        concept_logits = hybrid_concept_logits[:, : self.num_static_concept]

        return label_logits, concept_logits, img_feat


def get_cls_head(num_class, num_dynamic_concept, concept2class):
    concept2class = [[item] for item in concept2class]
    class_indices = list(range(num_class))
    hybrid_concept2class = concept2class + [class_indices] * num_dynamic_concept

    mask = kltn_utils.build_class_concept_matrix(
        concept2class=hybrid_concept2class,
        num_class=num_class,
    )
    cls_head = MaskClsHead(mask, num_class)

    return cls_head


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


class MaskClsHead(nn.Module):
    def __init__(self, mask, num_class):
        super().__init__()
        self.weight = nn.Parameter(mask.clone().to(dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_class))

        self.register_buffer("mask", mask)

    def forward(
        self,
        concept_logits,
    ):
        class_logit = concept_logits @ (self.weight * self.mask).T + self.bias

        return class_logit
