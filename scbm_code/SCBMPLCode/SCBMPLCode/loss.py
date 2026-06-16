import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBLoss(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config

    def forward(
        self,
        concept_logits,
        concept,
        label_logits,
        label,
    ):
        concept_loss = self.compute_concept_loss(concept, concept_logits)

        target_loss = F.cross_entropy(label_logits, label.long())

        return target_loss, concept_loss

    def compute_concept_loss(self, concept_true, concept_logits):
        concept_true = concept_true.float()  # [B, C]
        concepts_loss = F.binary_cross_entropy_with_logits(
            concept_logits, concept_true, reduction="none"
        )  #  [B, C]

        concepts_loss = concepts_loss.mean(dim=0).sum()

        return concepts_loss


class SCBLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(
        self,
        concept_mcmc_logit,
        concept_true,
        label_logits,
        label,
        c_triang_cov,
        cov_not_triang=False,
    ):
        concept_loss = self.compute_concept_loss(concept_mcmc_logit, concept_true)

        class_loss = F.cross_entropy(label_logits, label)

        # Get precision loss
        if self.config.loss.reg_precision == "l1":
            if cov_not_triang:
                prec_matrix = torch.inverse(c_triang_cov.float())
            else:
                c_triang_inv = torch.inverse(c_triang_cov.float())
                prec_matrix = torch.matmul(
                    torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
                )

            prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
                offset=0, dim1=1, dim2=2
            ).abs().sum(-1)

            if prec_matrix.size(1) > 1:
                prec_loss = prec_loss / (
                    prec_matrix.size(1) * (prec_matrix.size(1) - 1)
                )

            prec_loss = self.config.loss.reg_weight * prec_loss.mean(-1)
        else:
            prec_loss = torch.zeros_like(concept_loss)

        return class_loss, concept_loss, prec_loss

    def compute_concept_loss(self, concept_mcmc_logit, concept_true):
        concepts_true_expanded = concept_true.unsqueeze(-1).expand_as(
            concept_mcmc_logit
        )

        bce_loss = F.binary_cross_entropy_with_logits(
            concept_mcmc_logit, concepts_true_expanded.float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        # The concept loss computation is bounded by - log_num_mc adding log_num_mc moves
        # bound to 0. Preventing negative losses.

        return torch.mean(mcmc_loss) + math.log(self.config.num_monte_carlo)
