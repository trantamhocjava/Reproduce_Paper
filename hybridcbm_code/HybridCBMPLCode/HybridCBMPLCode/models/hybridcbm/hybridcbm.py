import torch
import torch.nn.functional as F
from kltn_utils import kltn_utils
from torch import nn


class HybridCBM(nn.Module):
    def __init__(self, config, concept_feat):
        super().__init__()
        self.config = config

        # var
        self.register_buffer(
            "scale", torch.tensor(config.hybridcbm.hybridcbm_scale, dtype=torch.float32)
        )
        self.register_buffer("concept_feat", F.normalize(concept_feat, dim=-1))

        # module
        num_concept = concept_feat.shape[0]
        self.classifier = torch.nn.Linear(num_concept, self.config.num_class)
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.hybridcbm.clip_model
        )

        # grad
        kltn_utils.freeze_module(self.clip_model)

    def forward(self, img):
        self.clip_model.eval()
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.clip_model, img
        )
        img_feat = F.normalize(img_feat, dim=-1)

        concept_logits = self.scale * (img_feat @ self.concept_feat.T)
        label_logits = self.classifier(concept_logits)

        return label_logits, img_feat, concept_logits
