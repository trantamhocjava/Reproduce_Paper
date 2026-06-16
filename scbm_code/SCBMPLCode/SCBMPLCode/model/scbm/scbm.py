import math

import torch
import torch.nn.functional as F
from kltn_utils import kltn_utils
from torch import nn
from torch.distributions import MultivariateNormal, RelaxedBernoulli

from . import utils


class SCBM(nn.Module):
    def __init__(self, config, num_concept):
        super().__init__()

        self.config = config
        self.num_concept = num_concept

        self.curr_temp = 1.0
        self.init_temp = 1.0
        self.final_temp = 0.5
        epochs = config.end_epoch - config.start_epoch + 1
        self.temp_decay_rate = (
            math.log(self.final_temp) - math.log(self.init_temp)
        ) / float(epochs)

        # Encoder
        self.encoder, preprocess, num_feature = utils.get_encoder(config)

        self.mu_concepts = nn.Linear(num_feature, num_concept, bias=True)

        self.sigma_concepts = nn.Linear(
            num_feature,
            int(num_concept * (num_concept + 1) / 2),
            bias=True,
        )
        self.sigma_concepts.weight.data *= (
            0.01  # To prevent exploding precision matrix at initialization
        )

        self.act_c = nn.Sigmoid()

        self.head = utils.get_head_layer(
            config.model.head_arch, num_concept, self.config.num_class
        )

    def setup_grad(self):
        kltn_utils.freeze_module(self.encoder)

    def forward(self, img, epoch, validation=False):
        device = img.device

        # Get intermediate representations
        img_embed = self.encoder(img)

        # Get mu and cholesky decomposition of covariance
        c_mu = self.mu_concepts(img_embed)

        c_sigma = self.sigma_concepts(img_embed)

        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        c_triang_cov = torch.zeros(
            (c_sigma.shape[0], self.num_concept, self.num_concept),
            device=device,
        )

        rows, cols = torch.tril_indices(
            row=self.num_concept, col=self.num_concept, offset=0
        )
        diag_idx = rows == cols

        c_sigma = c_sigma.to(
            dtype=c_triang_cov.dtype,
        )
        c_triang_cov[:, rows, cols] = c_sigma
        c_triang_cov[:, range(self.num_concept), range(self.num_concept)] = (
            F.softplus(c_sigma[:, diag_idx]) + 1e-6
        ).to(device=device)

        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.config.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,num_concepts,mcmc_size]
        c_mcmc_prob = self.act_c(c_mcmc_logit)
        concept_logits = c_mcmc_logit.mean(-1)

        # For all MCMC samples simultaneously sample from Bernoulli
        if validation:
            # No backward
            c_mcmc = torch.bernoulli(c_mcmc_prob)
        else:
            # Need backward
            curr_temp = self.compute_temperature(epoch)
            dist = RelaxedBernoulli(
                temperature=torch.tensor(curr_temp, device=c_mcmc_prob.device),
                probs=c_mcmc_prob,
            )

            # Bernoulli relaxation
            mcmc_relaxed = dist.rsample()

            if self.config.model.straight_through:
                # Straight-Through Gumbel Softmax
                mcmc_hard = (mcmc_relaxed > 0.5) * 1
                c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
            else:
                c_mcmc = mcmc_relaxed

        label_logits = self.compute_label_logits(c_mcmc, c_mcmc_logit)

        # Return concept mu for interventions
        return (
            c_mcmc_prob,
            c_mu,
            c_triang_cov,
            label_logits,
            c_mcmc_logit,
            concept_logits,
        )

    def compute_label_logits(self, c_mcmc_probs, c_mcmc_logits):
        # Pick the concept tensor: [B, C, M]
        x = (
            c_mcmc_probs
            if self.config.model.concept_learning == "hard"
            else c_mcmc_logits
        )
        B, C, M = x.shape

        # Run the head over all M samples at once: reshape to [B*M, C]
        x_flat = x.permute(0, 2, 1).reshape(B * M, C)  # [B*M, C]
        y_logits_flat = self.head(x_flat)  # [B*M, K] or [B*M, 1]

        # Multiclass: compute log(mean softmax) in a numerically stable way:
        # log(mean_i p_i) == logsumexp(log p_i) - log(M), where log p_i = log_softmax(logits_i)
        y_log_probs = F.log_softmax(y_logits_flat, dim=-1).view(
            B, M, self.config.num_class
        )  # [B, M, K]
        label_logits = torch.logsumexp(y_log_probs, dim=1) - math.log(M)  # [B, K]

        return label_logits

    def intervene(self, c_mcmc_probs, c_mcmc_logits):
        y_pred_probs_i = 0
        c_hard = torch.bernoulli(c_mcmc_probs)
        for i in range(self.config.num_monte_carlo):
            if self.config.model.concept_learning == "soft":
                c_i = c_mcmc_logits[:, :, i]
            else:
                c_i = c_hard[:, :, i]

            y_pred_logits_i = self.head(c_i)
            y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        y_pred_probs = y_pred_probs_i / self.config.num_monte_carlo
        y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits

    def compute_temperature(self, epoch):
        curr_temp = max(
            self.init_temp * math.exp(self.temp_decay_rate * epoch), self.final_temp
        )
        self.curr_temp = curr_temp

        return curr_temp
