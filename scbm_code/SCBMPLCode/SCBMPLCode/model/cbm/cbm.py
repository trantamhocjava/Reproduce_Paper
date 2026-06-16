import math

import torch
from kltn_utils import kltn_utils
from torch import nn
from torch.distributions import RelaxedBernoulli

from ..scbm import utils as scbm_utils


class CBM(nn.Module):
    def __init__(self, config, num_concept):
        super().__init__()
        self.config = config

        # Configuration arguments
        self.num_concept = num_concept

        self.curr_temp = 1.0
        self.num_epochs = config.end_epoch - config.start_epoch + 1

        # Architectures
        # Encoder h(.)
        self.encoder, preprocess, num_feature = scbm_utils.get_encoder(config)

        # Concept predictor
        self.concept_predictor = nn.Linear(num_feature, self.num_concept, bias=True)

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        self.head = scbm_utils.get_head_layer(
            config.model.head_arch, self.num_concept, config.num_class
        )

    def setup_grad(self):
        kltn_utils.freeze_module(self.encoder)

    def forward(
        self,
        img,
        epoch,
        validation=False,
    ):
        # Get intermediate representations
        img_embed = self.encoder(img)

        # Get concept predictions
        concept_logtis = self.concept_predictor(img_embed)
        concept_probs = self.act_c(concept_logtis)

        if self.config.model.concept_learning in ("hard"):
            # Hard CBM
            if validation:
                # Sample from Bernoulli M times, as we don't need to backprop
                c_prob_mcmc = concept_probs.unsqueeze(-1).expand(
                    -1, -1, self.config.num_monte_carlo
                )
                concept_hard = torch.bernoulli(c_prob_mcmc)
            else:
                # Relax bernoulli sampling with Gumbel Softmax to allow for backpropagation
                curr_temp = self.compute_temperature(epoch, device=concept_probs.device)
                dist = RelaxedBernoulli(temperature=curr_temp, probs=concept_probs)
                c_relaxed = dist.rsample([self.config.num_monte_carlo]).movedim(0, -1)

                if self.config.model.straight_through:
                    # Straight-Through Gumbel Softmax
                    c_hard = (c_relaxed > 0.5) * 1
                    concept_hard = c_hard - c_relaxed.detach() + c_relaxed
                else:
                    # Reparametrization trick.
                    concept_hard = c_relaxed

        # Get predicted targets
        if self.config.model.concept_learning == "hard":
            # Hard CBM or validation of AR. Takes MCMC samples.
            # MCMC loop for predicting label
            y_pred_probs_i = 0

            for i in range(self.config.num_monte_carlo):
                c_i = concept_hard[:, :, i]
                y_pred_logits_i = self.head(c_i)
                y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
            y_pred_probs = y_pred_probs_i / self.config.num_monte_carlo

            label_logits = torch.log(y_pred_probs + 1e-6)
        elif self.config.model.concept_learning == "soft":
            # Soft CBM
            label_logits = self.head(
                concept_logtis
            )  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.
            concept_hard = torch.empty_like(concept_probs)

        return concept_probs, concept_logtis, label_logits, concept_hard

    def intervene(
        self,
        concepts_interv_probs,
        concepts_mask,
    ):
        if self.config.model.concept_learning == "soft":
            # Soft CBM
            c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
            y_pred_logits = self.head(c_logit)

        elif self.config.model.concept_learning in ("hard"):
            # Hard CBM or AR
            y_pred_probs_i = 0

            c_prob_mcmc = concepts_interv_probs.unsqueeze(-1).expand(
                -1, -1, self.num_monte_carlo
            )
            c = torch.bernoulli(c_prob_mcmc)

            # Fix intervened-on concepts to ground truth
            c[concepts_mask == 1] = (
                concepts_interv_probs[concepts_mask == 1]
                .unsqueeze(-1)
                .expand(-1, self.num_monte_carlo)
            )
            weight = torch.ones(
                (c.shape[0], self.config.num_monte_carlo), device=c.device
            )

            for i in range(self.config.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.softmax(
                    y_pred_logits_i, dim=1
                )

            y_pred_probs = y_pred_probs_i / torch.sum(weight, dim=1).unsqueeze(1)
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp
