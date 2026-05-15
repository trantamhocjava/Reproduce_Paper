import torch
import torch.nn.functional as F
from kltn_utils import kltn_utils
from torch import nn


class ExplicdHybridCBM(nn.Module):
    def __init__(self, config, concept_feat):
        super().__init__()
        self.config = config
        self.concept_feat = concept_feat

        self.num_concept = concept_feat.shape[0]

        # var
        self.register_buffer(
            "scale", torch.tensor(config.hybridcbm.hybridcbm_scale, dtype=torch.float32)
        )

        # module
        self.classifier = torch.nn.Linear(self.num_concept, self.config.num_class)
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.hybridcbm.clip_model
        )

        # grad
        kltn_utils.freeze_module(self.clip_model)

    def forward(self, img):
        self.clip_model.eval()
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.hybridcbm.clip_model, img
        )
        norm_img_feat = F.normalize(img_feat, dim=-1)

        adapted_img_feat = F.normalize(self.adaptive_module(img_feat), dim=-1)
        norm_concept_feat = F.normalize(self.concept_feat, dim=-1)

        sim_score = self.scale * (adapted_img_feat @ norm_concept_feat.T)
        concept_logits = sim_score
        label_logits = self.classifier(sim_score)

        return label_logits, norm_img_feat, concept_logits
