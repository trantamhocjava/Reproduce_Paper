import torch.nn as nn
import torch.nn.functional as F


class StaticConceptBank(nn.Module):
    def __init__(self, select_concept_data):
        super().__init__()

        # concept feat
        self.register_buffer("concept_feat", select_concept_data["concept_feat"])
        self.register_buffer("concept2class", select_concept_data["concept2class"])
        self.register_buffer(
            "class_feat", F.normalize(select_concept_data["class_feat"], dim=-1)
        )
        self.register_buffer("concepts", select_concept_data["concepts"])
