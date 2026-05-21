import torch
from torch import nn


class DynamicConceptBank(nn.Module):
    def __init__(self, config, concept_feat_dim):
        super().__init__()

        # selected concept indices
        self.concept_feat = nn.Parameter(
            torch.randn(config.num_dynamic_concepts, concept_feat_dim)
        )
