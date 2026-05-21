import torch
import torch.nn.functional as F

from models.conceptBank.dynamic_bank import DynamicConceptBank
from models.conceptBank.static_bank import StaticConceptBank


class HybridConceptBank(torch.nn.Module):
    def __init__(self, config, select_concept_data):
        super().__init__()

        # banks
        self.static_bank = StaticConceptBank(select_concept_data)
        concept_feat_dim = self.static_bank.concept_feat.shape[1]
        self.dynamic_bank = DynamicConceptBank(
            config, concept_feat_dim=concept_feat_dim
        )

        # attr
        self.register_buffer("concept_feat", self.get_concept_feat())

    def get_concept_feat(self):
        features = torch.cat(
            [self.static_bank.concept_feat, self.dynamic_bank.concept_feat],
            dim=0,
        )
        features = F.normalize(features, dim=-1)
        return features
