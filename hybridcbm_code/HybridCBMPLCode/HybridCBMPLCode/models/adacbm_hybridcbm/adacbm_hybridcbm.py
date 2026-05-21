import torch
import torch.nn.functional as F
from kltn_utils import kltn_const, kltn_utils
from torch import nn


class AdaHybridCBM(nn.Module):
    def __init__(self, config, concept_feat, concept2class):
        super().__init__()
        self.config = config

        # var
        self.register_buffer(
            "scale",
            torch.tensor(
                kltn_const.CLIP_MODELS[config.hybridcbm.clip_model]["logit_scale"]
            ),
        )
        self.register_buffer("concept_feat", F.normalize(concept_feat, dim=-1))

        # module
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.hybridcbm.clip_model
        )

        num_concept = concept_feat.shape[0]
        concept2class = get_hybrid_concept2class(
            concept2class=concept2class,
            num_dynamic_concept=config.num_dynamic_concept,
            num_class=config.num_class,
        )
        self.cls_head = get_cls_head(
            cls_head_type=config.hybridcbm.cls_head_type,
            num_concept=num_concept,
            num_class=config.num_class,
            concept2class=concept2class,
        )

        img_feat_dim = kltn_const.CLIP_MODELS[config.hybridcbm.clip_model][
            "embedding_dim"
        ]
        self.adaptive_module = AdaptiveModule(
            dim=img_feat_dim, num_layers=config.hybridcbm.ada_num_layer
        )

        # grad
        kltn_utils.freeze_module(self.clip_model)

    def forward(self, img):
        self.clip_model.eval()
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.hybridcbm.clip_model, img
        )
        img_feat = F.normalize(self.adaptive_module(img_feat), dim=-1)
        concept_logits = self.scale * (img_feat @ self.concept_feat.T)
        label_logits = self.cls_head(concept_logits)

        return label_logits, img_feat, concept_logits


def get_cls_head(cls_head_type, num_concept, num_class, concept2class):
    if cls_head_type == "linear":
        result = nn.Linear(num_concept, num_class)
    elif cls_head_type == "mask":
        mask = kltn_utils.build_class_concept_matrix(
            concept2class=concept2class,
            num_class=num_class,
        )
        result = MaskClsHead(mask, num_class)

    return result


def get_hybrid_concept2class(concept2class, num_dynamic_concept, num_class):
    class_indices = list(range(num_class))
    result = concept2class + [class_indices] * num_dynamic_concept

    return result


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
        self.weight = nn.Parameter(mask.clone())
        self.bias = nn.Parameter(torch.zeros(num_class))

        self.register_buffer("mask", mask)

    def forward(
        self,
        concept_logits,
    ):
        class_logit = concept_logits @ (self.weight * self.mask).T + self.bias

        return class_logit
