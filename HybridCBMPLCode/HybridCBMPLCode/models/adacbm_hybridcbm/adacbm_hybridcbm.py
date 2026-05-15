import torch
import torch.nn.functional as F
from kltn_utils import kltn_const, kltn_utils
from torch import nn


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


class AdaHybridCBM(nn.Module):
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
        img_feat_dim = kltn_const.CLIP_MODELS[config.hybridcbm.clip_model][
            "embedding_dim"
        ]
        self.adaptive_module = AdaptiveModule(
            dim=img_feat_dim, num_layers=config.adaptive_module.num_layers
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
