import torch.nn as nn
import torch.nn.functional as F


class ExplicdLoss(nn.Module):
    def __init__(
        self,
        concept_criteria,
    ):
        super().__init__()
        self.concept_criteria = concept_criteria

    def forward(
        self,
        concepts_pred_logits,
        concepts_true,
        target_pred_logits,
        target_true,
    ):
        loss_cls = F.cross_entropy(target_pred_logits, target_true)

        loss_concept = self.compute_concept_loss(concepts_pred_logits, concepts_true)

        loss = loss_cls + loss_concept

        return loss, loss_cls, loss_concept

    def compute_concept_loss(self, concepts_pred_logits, concepts_true):
        sum_loss_concept = 0
        for idx, key in enumerate(self.concept_criteria):
            loss_concept = F.cross_entropy(
                concepts_pred_logits[key], concepts_true[:, idx]
            )
            sum_loss_concept += loss_concept

        loss_concept = sum_loss_concept / len(self.concept_criteria)

        return loss_concept
