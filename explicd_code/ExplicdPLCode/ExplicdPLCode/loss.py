import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptLoss(nn.Module):
    def forward(
        self,
        concept_logits_dict,
        concept,
    ):
        concept_loss = []

        criteria = concept_logits_dict.keys()

        for idx, criterion in enumerate(criteria):
            loss_concept_item = F.cross_entropy(
                concept_logits_dict[criterion], concept[:, idx]
            )
            concept_loss.append(loss_concept_item)

        concept_loss = torch.stack(concept_loss).mean()

        return concept_loss
