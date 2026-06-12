import torch
from kltn_utils import kltn_utils
from torch import nn
from torch.nn import functional as F


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

        # module
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.model.clip_model
        )
        kltn_utils.rank_zero_info_newline(
            f"Load clip_model {config.model.clip_model} ok"
        )

        num_concept, embedding_dim = select_concepts_data["concept_feat"].shape
        self.adaptive_layer = AdaptiveModule(
            dim=embedding_dim,
            num_layers=config.model.num_adamodule_layer,
        )

        mask = kltn_utils.build_class_concept_matrix(
            concept2class=select_concepts_data["concept2class"],
            num_class=config.num_class,
        )
        self.class_concept_weight = nn.Parameter(mask.clone())
        self.concept_bias = nn.Parameter(torch.zeros(num_concept))
        self.class_bias = nn.Parameter(torch.zeros(config.num_class))

        # var
        self.register_buffer(
            "concept_feat", F.normalize(select_concepts_data["concept_feat"], dim=-1)
        )
        self.register_buffer("mask", mask)

    def setup_grad(self):
        # Grad
        kltn_utils.freeze_module(self.clip_model)

    def forward(self, img):
        self.clip_model.eval()
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.model.clip_model, img
        )

        img_feat = F.normalize(self.adaptive_layer(img_feat), dim=-1)
        concept_logits = img_feat @ self.concept_feat.T + self.concept_bias
        class_logit = (
            concept_logits @ (self.class_concept_weight * self.mask).T + self.class_bias
        )

        return class_logit, concept_logits
