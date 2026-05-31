import torch
import torch.nn.functional as F
from kltn_utils import kltn_utils
from torch import nn


class HybridCBM(nn.Module):
    def __init__(self, config, select_concept_data):
        super().__init__()
        self.config = config

        self.num_static_concept, embedding_dim = select_concept_data[
            "concept_feat"
        ].shape
        self.num_concept = self.num_static_concept + config.model.num_dynamic_concept

        ## Get clip_model
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.model.clip_model
        )

        ## Get cls_head
        self.cls_head = torch.nn.Linear(self.num_concept, self.config.num_class)

        ## Get dynamic_concept_feat
        self.dynamic_concept_feat = nn.Parameter(
            torch.randn(config.model.num_dynamic_concept, embedding_dim)
        )

        # var
        self.register_buffer(
            "scale", torch.tensor(config.model.scale, dtype=torch.float32)
        )
        self.register_buffer(
            "static_concept_feat",
            select_concept_data["concept_feat"],
        )
        self.register_buffer(
            "class_feat",
            select_concept_data["class_feat"],
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
        img_feat = F.normalize(img_feat, dim=-1)

        # Get hybrid_concept_logits
        hybrid_concept_logits = self.scale * (img_feat @ concept_feat.T)

        # Get label logits
        label_logits = self.cls_head(hybrid_concept_logits)

        # Get concept_logits
        concept_logits = hybrid_concept_logits[:, : self.num_static_concept]

        return label_logits, concept_logits, img_feat
