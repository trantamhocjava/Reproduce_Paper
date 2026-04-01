import math

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli

from . import utils


class CBM(nn.Module):
    """
    Model class encompassing all baselines: Hard & Soft Concept Bottleneck Model (CBM),
                                            Concept Embedding Model (CEM), and Autoregressive CBM (AR).

    This class implements the baselines. Depending on the choice of model, only a small part of the full code is used.
    Check the if statements in the forward method to see which part of the code is used for which model.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        concept_learning (str): The concept learning method ("hard", "soft", "embedding", or "autoregressive").
                                This determines the type of method to use
        num_monte_carlo (int): The number of Monte Carlo samples for sampling Gumbel Softmax in AR.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
    """

    def __init__(self, config):
        super(CBM, self).__init__()

        # Configuration arguments
        self.num_concepts = config.num_concepts
        self.num_classes = len(config.class_names)
        self.encoder_arch = config.encoder_arch
        self.head_arch = config.head_arch
        self.concept_learning = config.concept_learning

        self.num_monte_carlo = config.num_monte_carlo
        self.straight_through = config.straight_through
        self.curr_temp = 1.0
        self.num_epochs = config.end_epoch - config.start_epoch + 1

        # Architectures
        # Encoder h(.)
        self.encoder, preprocess, n_features = utils.get_encoder(config)
        if config.freezebb:
            self.encoder.apply(utils.freeze_module)

        # Concept predictor
        self.concept_predictor = nn.Linear(n_features, self.num_concepts, bias=True)
        self.concept_dim = self.num_concepts

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        self.pred_dim = utils.get_pred_dim(self.num_classes)
        self.head = utils.get_head_layer(
            config.head_arch, self.num_concepts, self.pred_dim
        )

        self.preprocess_list = utils.get_v2_list_from_v1_preprocess(preprocess)

    def forward(
        self,
        x,
        epoch,
        validation=False,
    ):
        """
        Perform a forward pass through one of the baselines.

        This method performs a forward pass predicting concept probabilities and logits for the target variable.
        It handles different concept learning strategies and training modes, including hard, soft, autoregressive, and embedding-based concepts.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            concepts_train_ar (torch.Tensor, optional): Ground-truth concept values for autoregressive training. Default is False.

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, label_dim)
                - c (torch.Tensor): Predicted hard concept values (if method permits, otherwise the concept representation). Shape: (batch_size, num_concepts, num_monte_carlo) for MCMC sampling or (batch_size, num_concepts) otherwise.
        """
        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get concept predictions
        c_logit = self.concept_predictor(intermediate)
        c_prob = self.act_c(c_logit)

        if self.concept_learning in ("hard"):
            # Hard CBM
            if validation:
                # Sample from Bernoulli M times, as we don't need to backprop
                c_prob_mcmc = c_prob.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)
                c = torch.bernoulli(c_prob_mcmc)
            else:
                # Relax bernoulli sampling with Gumbel Softmax to allow for backpropagation
                curr_temp = self.compute_temperature(epoch, device=c_prob.device)
                dist = RelaxedBernoulli(temperature=curr_temp, probs=c_prob)
                c_relaxed = dist.rsample([self.num_monte_carlo]).movedim(0, -1)

                if self.straight_through:
                    # Straight-Through Gumbel Softmax
                    c_hard = (c_relaxed > 0.5) * 1
                    c = c_hard - c_relaxed.detach() + c_relaxed
                else:
                    # Reparametrization trick.
                    c = c_relaxed

        # Get predicted targets
        if self.concept_learning == "hard":
            # Hard CBM or validation of AR. Takes MCMC samples.
            # MCMC loop for predicting label
            y_pred_probs_i = 0
            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
            y_pred_probs = y_pred_probs_i / self.num_monte_carlo

            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)
        elif self.concept_learning == "soft":
            # Soft CBM
            y_pred_logits = self.head(
                c_logit
            )  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.
            c = torch.empty_like(c_prob)

        return c_prob, c_logit, y_pred_logits, c

    def intervene(
        self,
        concepts_interv_probs,
        concepts_mask,
    ):
        if self.concept_learning == "soft":
            # Soft CBM
            c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
            y_pred_logits = self.head(c_logit)

        elif self.concept_learning in ("hard"):
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
            weight = torch.ones((c.shape[0], self.num_monte_carlo), device=c.device)

            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.sigmoid(
                        y_pred_logits_i
                    )
                else:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.softmax(
                        y_pred_logits_i, dim=1
                    )
            y_pred_probs = y_pred_probs_i / torch.sum(weight, dim=1).unsqueeze(1)
            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp
