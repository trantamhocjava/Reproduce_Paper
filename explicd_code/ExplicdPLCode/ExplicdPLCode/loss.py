import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ExplicdLoss(nn.Module):
    def forward(
        self,
        concept_logits_dict,
        concept,
        y_logits,
        y,
    ):
        cls_loss = F.cross_entropy(y_logits, y)
        concept_loss = self.compute_concept_loss(concept_logits_dict, concept)

        loss = cls_loss + concept_loss

        return loss, cls_loss, concept_loss

    def compute_concept_loss(self, concept_logits_dict, concept):
        concept_loss = []

        criteria = concept_logits_dict.keys()

        for idx, criterion in enumerate(criteria):
            loss_concept_item = F.cross_entropy(
                concept_logits_dict[criterion], concept[:, idx]
            )
            concept_loss.append(loss_concept_item)

        concept_loss = np.array(concept_loss).mean()

        return concept_loss
