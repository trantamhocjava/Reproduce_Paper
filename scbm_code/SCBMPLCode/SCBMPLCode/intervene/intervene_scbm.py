import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info
from scipy.stats import chi2
from torch import nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from torchmin import minimize

from .. import utils
from ..loss import SCBLoss
from ..model.scbm import SCBM
from ..train import MetricCalculator
from . import utils as intervene_utils


def intervene_scbm(test_loader, config):
    policy = config.inter_policy
    strategy = config.inter_strategy
    num_interventions = min(config.min_num_interventions, config.num_concepts)

    # Intervening with different strategies
    intervention_dataset_base = []
    intervention_dataset_fixed = []
    intervention_policy = intervene_utils.define_policy(policy)
    intervention_strategy = SCBM_Strategy(strategy, config)

    ## One full model pass without interventions to set up the dataset required at each intervention step
    rank_zero_info("0 concept")

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        inference_mode=False,
    )

    model = SCBMIntervene0Concept(
        config=config,
        intervention_strategy=intervention_strategy,
        intervention_policy=intervention_policy,
        intervention_dataset_base=intervention_dataset_base,
        intervention_dataset_fixed=intervention_dataset_fixed,
    )
    tester.test(
        model=model,
        ckpt_path=config.best_model,
        dataloaders=test_loader,
    )

    ## Computing intervention curves using stored concept predictions
    # Preparing dataset
    intervention_dataset_base = [
        (torch.cat([sublist[i] for sublist in intervention_dataset_base], dim=0).cpu())
        for i in range(len(intervention_dataset_base[0]))
    ]
    intervention_dataset_fixed = [
        (torch.cat([sublist[i] for sublist in intervention_dataset_fixed], dim=0).cpu())
        for i in range(len(intervention_dataset_fixed[0]))
    ]
    # Initializing concepts with 0's
    intervention_dataset = TensorDataset(
        *intervention_dataset_base,
        torch.zeros_like(intervention_dataset_fixed[-2]),
        *intervention_dataset_fixed,
    )

    # Performing interventions
    for num_intervened in range(1, num_interventions + 1):
        # Update intervened-on concept mask in dataloader
        rank_zero_info(f"{num_intervened} concept")

        updated_intervention_dataset = []
        intervention_loader = DataLoader(
            intervention_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            shuffle=False,
        )

        tester = Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
            inference_mode=False,
        )

        model = SCBMIntervene(
            config=config,
            num_concepts=num_intervened,
            intervention_strategy=intervention_strategy,
            intervention_policy=intervention_policy,
            updated_intervention_dataset=updated_intervention_dataset,
        )
        tester.test(
            model=model,
            ckpt_path=config.best_model,
            dataloaders=intervention_loader,
        )

        # Updating dataset
        intervention_dataset = TensorDataset(
            *[
                (
                    torch.cat(
                        [sublist[i] for sublist in updated_intervention_dataset],
                        dim=0,
                    ).cpu()
                )
                for i in range(len(updated_intervention_dataset[0]))
            ],
            *intervention_dataset_fixed,
        )


class SCBMIntervene0Concept(pl.LightningModule):
    def __init__(
        self,
        config,
        intervention_strategy,
        intervention_policy,
        intervention_dataset_base: list = [],
        intervention_dataset_fixed: list = [],
    ):
        super().__init__()

        self.config = config
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.intervention_dataset_base = intervention_dataset_base
        self.intervention_dataset_fixed = intervention_dataset_fixed

        self.test_metric = MetricCalculator(self.config.num_concepts)
        self.model = SCBM(config)
        self.loss_fn = SCBLoss(config=config)

    def get_loss(self, batch):
        return_result = None

        data, label, concept = batch

        # Forward pass
        (concepts_mcmc_probs, c_mu, triang_cov, target_pred_logits, c_mcmc_logit) = (
            self.model(data, self.current_epoch, validation=True, return_full=True)
        )

        # Compute the loss
        target_loss, concepts_loss, prec_loss, total_loss = self.loss_fn(
            c_mcmc_logit,
            concept,
            target_pred_logits,
            label,
            triang_cov,
        )

        concepts_pred_probs = concepts_mcmc_probs.mean(-1)

        return_result = {
            "concepts_pred_probs": concepts_pred_probs,
            "target_pred_logits": target_pred_logits,
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
            "c_mu": c_mu,
            "triang_cov": triang_cov,
            "data": data,
            "label": label,
            "concept": concept,
        }

        return return_result

    def on_test_epoch_start(self):
        self.test_metric.reset()
        self.test_time = time.time()

    def on_test_epoch_end(self):
        self.test_time = time.time() - self.test_time
        test_result = {
            f"test_{key}": value
            for key, value in self.test_metric.return_metrics().items()
        }
        test_result["test_time"] = self.test_time
        test_result["n_concepts"] = 0

        intervene_utils.log_in_csv(test_result)

    def test_step(self, batch, batch_idx):
        return_result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(
            y_logit=return_result["target_pred_logits"],
            c_prob=return_result["concepts_pred_probs"],
            y=return_result["label"],
            c=return_result["concept"],
            target_loss=return_result["target_loss"],
            concepts_loss=return_result["concepts_loss"],
            total_loss=return_result["total_loss"],
        )

        # Update rest
        c_cov = torch.matmul(
            return_result["triang_cov"],
            torch.transpose(return_result["triang_cov"], dim0=1, dim1=2),
        )
        c_cov = utils.numerical_stability_check(c_cov)

        (
            _,
            _,
            c_mcmc_probs,
            _,
        ) = self.intervention_strategy.compute_intervention_0concept(
            return_result["c_mu"],
            c_cov,
            return_result["concept"],
            torch.zeros_like(
                return_result["concept"], device=return_result["concept"].device
            ),
        )
        concepts_pred_probs = c_mcmc_probs.mean(-1)

        device = "cpu"
        self.intervention_dataset_base.append(
            [
                return_result["c_mu"].to(device),
                c_cov.to(device),
                concepts_pred_probs.to(device),
            ]
        )
        self.intervention_dataset_fixed.append(
            [
                return_result["c_mu"].to(device),
                c_cov.to(device),
                return_result["concept"].to(device),
                return_result["label"].to(device),
            ]
        )


class SCBMIntervene(pl.LightningModule):
    def __init__(
        self,
        config,
        num_concepts,
        intervention_strategy,
        intervention_policy,
        updated_intervention_dataset: list = [],
    ):
        super().__init__()

        self.config = config
        self.num_concepts = num_concepts
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.updated_intervention_dataset = updated_intervention_dataset

        self.test_metric = MetricCalculator(self.config.num_concepts)
        self.model = SCBM(config)
        self.loss_fn = SCBLoss(config=config)

    def get_loss(self, batch):
        return_result = None

        (
            c_mu,
            c_cov,
            concepts_pred_probs,
            concepts_mask,
            c_mu_original,
            c_cov_original,
            concept,
            label,
        ) = batch

        # Determining new concept to intervene on
        concepts_mask_new = self.intervention_policy.compute_intervention_mask(
            concepts_mask,
            concepts_pred_probs=concepts_pred_probs,
            mu=c_mu,
            cov=c_cov,
        )

        # Intervening including new concept
        (
            c_interv_mu,
            c_interv_cov,
            c_mcmc_probs,
            c_mcmc_logits,
        ) = self.intervention_strategy.compute_intervention(
            c_mu_original,
            c_cov_original,
            concept,
            concepts_mask_new,
        )

        target_pred_logits = self.model.intervene(c_mcmc_probs, c_mcmc_logits)

        target_loss, concepts_loss, prec_loss, total_loss = self.loss_fn(
            c_mcmc_logits,
            concept,
            target_pred_logits,
            label,
            c_interv_cov,
            cov_not_triang=True,
        )

        concepts_pred_probs = c_mcmc_probs.mean(-1)

        return_result = {
            "concepts_pred_probs": concepts_pred_probs,
            "target_pred_logits": target_pred_logits,
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
            "c_interv_mu": c_interv_mu,
            "c_interv_cov": c_interv_cov,
            "concepts_mask_new": concepts_mask_new,
            "label": label,
            "concept": concept,
        }

        return return_result

    def on_test_epoch_start(self):
        self.test_metric.reset()
        self.test_time = time.time()

    def on_test_epoch_end(self):
        self.test_time = time.time() - self.test_time
        test_result = {
            f"test_{key}": value
            for key, value in self.test_metric.return_metrics().items()
        }
        test_result["test_time"] = self.test_time
        test_result["n_concepts"] = self.num_concepts

        intervene_utils.log_in_csv(test_result)

    def test_step(self, batch, batch_idx):
        return_result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(
            y_logit=return_result["target_pred_logits"],
            c_prob=return_result["concepts_pred_probs"],
            y=return_result["label"],
            c=return_result["concept"],
            target_loss=return_result["target_loss"],
            concepts_loss=return_result["concepts_loss"],
            total_loss=return_result["total_loss"],
        )

        # Update rest
        self.updated_intervention_dataset.append(
            [
                return_result["c_interv_mu"],
                return_result["c_interv_cov"],
                return_result["concepts_pred_probs"],
                return_result["concepts_mask_new"],
            ]
        )


class SCBM_Strategy:
    def __init__(self, inter_strategy, config):
        self.num_monte_carlo = config.num_monte_carlo
        self.num_concepts = config.num_concepts
        self.act_c = nn.Sigmoid()

        if inter_strategy == "simple_perc":
            self.interv_strat = PercentileStrategy()
        elif inter_strategy == "conf_interval_optimal":
            self.interv_strat = ConfIntervalOptimalStrategy(level=config.level)

    def compute_intervention_0concept(self, c_mu, c_cov, c_true, c_mask):
        # No intervention
        interv_mu = c_mu
        interv_cov = c_cov
        # Sample from normal distribution
        dist = MultivariateNormal(interv_mu, covariance_matrix=interv_cov)
        mcmc_logits = dist.rsample([self.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,bottleneck_size,mcmc_size]

        # Compute probabilities and set intervened-on probs to 0/1
        mcmc_probs = self.act_c(mcmc_logits)

        # Set intervened-on hard concepts to 0/1
        mcmc_probs = (c_true * c_mask).unsqueeze(2).repeat(
            1, 1, self.num_monte_carlo
        ) + mcmc_probs * (1 - c_mask).unsqueeze(2).repeat(1, 1, self.num_monte_carlo)

        return interv_mu, interv_cov, mcmc_probs, mcmc_logits

    def compute_intervention(self, c_mu, c_cov, c_true, c_mask):
        num_intervened = c_mask.sum(1)[0]

        # Compute logits of intervened-on concepts
        c_intervened_logits = self.interv_strat.compute_intervened_logits(
            c_mu, c_cov, c_true, c_mask
        )

        ## Compute conditional normal distribution sample-wise
        # Permute covariance s.t. intervened-on concepts are a block at start
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
        perm_cov = c_cov.gather(1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2)))
        perm_cov = perm_cov.gather(
            2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
        )
        perm_mu = c_mu.gather(1, indices)
        perm_c_intervened_logits = c_intervened_logits.gather(1, indices)

        # Compute mu and covariance conditioned on intervened-on concepts
        # Intermediate steps
        perm_intermediate_cov = torch.matmul(
            perm_cov[:, num_intervened:, :num_intervened],
            torch.inverse(perm_cov[:, :num_intervened, :num_intervened]),
        )
        perm_intermediate_mu = (
            perm_c_intervened_logits[:, :num_intervened] - perm_mu[:, :num_intervened]
        )
        # Mu and Cov
        perm_interv_mu = perm_mu[:, num_intervened:] + torch.matmul(
            perm_intermediate_cov, perm_intermediate_mu.unsqueeze(-1)
        ).squeeze(-1)
        perm_interv_cov = perm_cov[:, num_intervened:, num_intervened:] - torch.matmul(
            perm_intermediate_cov, perm_cov[:, :num_intervened, num_intervened:]
        )

        # Adjust for floating point errors in the covariance computation to keep it symmetric
        perm_interv_cov = utils.numerical_stability_check(
            perm_interv_cov
        )  # Uncomment if Normal throws an error. Takes some time so maybe code it more smartly

        # Sample from conditional normal
        perm_dist = MultivariateNormal(
            perm_interv_mu, covariance_matrix=perm_interv_cov
        )
        perm_mcmc_logits = (
            perm_dist.rsample([self.num_monte_carlo]).movedim(0, -1).to(torch.float32)
        )  # [bottleneck_size-num_intervened,mcmc_size]

        # Concat logits of intervened-on concepts
        perm_mcmc_logits = torch.cat(
            (
                perm_c_intervened_logits[:, :num_intervened]
                .unsqueeze(-1)
                .repeat(1, 1, self.num_monte_carlo),
                perm_mcmc_logits,
            ),
            dim=1,
        )

        # Permute back into original form and store
        indices_reversed = torch.argsort(indices)
        mcmc_logits = perm_mcmc_logits.gather(
            1,
            indices_reversed.unsqueeze(2).expand(-1, -1, perm_mcmc_logits.size(2)),
        )

        # Return conditional mu&cov
        interv_mu = perm_interv_mu
        interv_cov = perm_interv_cov

        # Compute probabilities and set intervened-on probs to 0/1
        mcmc_probs = self.act_c(mcmc_logits)

        # Set intervened-on hard concepts to 0/1
        mcmc_probs = (c_true * c_mask).unsqueeze(2).repeat(
            1, 1, self.num_monte_carlo
        ) + mcmc_probs * (1 - c_mask).unsqueeze(2).repeat(1, 1, self.num_monte_carlo)

        return interv_mu, interv_cov, mcmc_probs, mcmc_logits


class PercentileStrategy:
    # Set intervened concepts to 0.05 & 0.95 probabilities
    def __init__(self):
        pass

    def _compute_intervened_probs(self, c_true, c_mask):
        return (0.05 + 0.9 * c_true) * c_mask

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_probs(c_true, c_mask)
        c_intervened_logits = torch.logit(c_intervened_probs, eps=1e-6)
        return c_intervened_logits

    def compute_intervention_cbm(self, c_pred, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_probs(c_true, c_mask)
        c_intervened = c_intervened_probs + c_pred * (1 - c_mask)
        return c_intervened


class ConfIntervalOptimalStrategy:
    # Set intervened concept logits to bounds of 90% confidence interval
    def __init__(self, level=0.9):
        self.level = level

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        # Find values that lie on confidence region ball
        # Approach: Find theta s.t.  Λn(θ)= −2(ℓ(θ)−ℓ(θ^))=χ^2_{1-α,n} and minimize concept loss of intervened concepts.
        # Note, theta^ is = mu, evaluated for the N(mu,Sigma) distribution, while theta is point on the boundary of the confidence region
        # Then, we make theta by arg min Concept BCE(θ) s.t. Λn(θ) <= holds with 1-α = self.level for theta~N(0,Sigma) (not fully correct explanation, but intuition).
        device = c_mu.device

        n_intervened = int(c_mask.sum(1)[0].item())
        # Separate intervened-on concepts from others
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)

        perm_cov = c_cov.gather(1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2)))
        perm_cov = perm_cov.gather(
            2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
        )
        marginal_interv_cov = perm_cov[:, :n_intervened, :n_intervened]
        marginal_interv_cov = utils.numerical_stability_check(
            marginal_interv_cov.float()
        ).to(device)
        target = (
            (c_true * c_mask).gather(1, indices)[:, :n_intervened].float().to(device)
        )
        marginal_c_mu = c_mu.gather(1, indices)[:, :n_intervened].float().to(device)
        interv_direction = (
            ((2 * c_true - 1) * c_mask)
            .gather(1, indices)[:, :n_intervened]
            .float()
            .to(device)
        )  # direction
        quantile_cutoff = chi2.ppf(q=self.level, df=n_intervened)

        # Finding good init point on confidence region boundary (each dim with equal magnitude)
        zeros = torch.zeros(
            n_intervened,
            device=device,
        )

        dist = MultivariateNormal(zeros, marginal_interv_cov)
        loglikeli_theta_hat = dist.log_prob(zeros)

        def conf_region(scale):
            loglikeli_theta_star = dist.log_prob(scale * interv_direction)
            log_likelihood_ratio = -2 * (loglikeli_theta_star - loglikeli_theta_hat)
            return ((quantile_cutoff - log_likelihood_ratio) ** 2).sum(-1)

        with torch.enable_grad():
            scale = minimize(
                conf_region,
                x0=torch.ones(
                    c_mu.shape[0], 1, device=c_mu.device, dtype=c_mu.dtype
                ).requires_grad_(True),
                method="bfgs",
                max_iter=50,
                tol=1e-5,
            ).x

        scale = (
            scale.abs()
        )  # in case negative root was found (note that both give same log-likelihood as its point-symmetric around 0)
        x0 = marginal_c_mu + (interv_direction * scale)

        # Define bounds on logits
        lb_interv = torch.where(
            interv_direction > 0,
            marginal_c_mu + 1e-4,
            torch.tensor(float("-inf"), device=device),
        )
        ub_interv = torch.where(
            interv_direction < 0,
            marginal_c_mu - 1e-4,
            torch.tensor(float("inf"), device=device),
        )

        # Define confidence region
        dist_logits = MultivariateNormal(marginal_c_mu, marginal_interv_cov)
        loglikeli_theta_hat = dist_logits.log_prob(marginal_c_mu)
        loglikeli_goal = -quantile_cutoff / 2 + loglikeli_theta_hat

        # Initialize variables
        cov_inverse = torch.linalg.inv(marginal_interv_cov)
        interv_vector = torch.empty_like(marginal_c_mu)

        #### Sample-wise constrained optimization (as there are no batched functions available out-of-the-box). Can surely be optimized
        for i in range(marginal_c_mu.shape[0]):
            # Define variables required for optimization
            dist_logits_uni = MultivariateNormal(
                marginal_c_mu[i].detach().cpu().double(),
                marginal_interv_cov[i].detach().cpu().double(),
            )
            loglikeli_goal_uni = loglikeli_goal[i].detach().cpu().double()
            target_uni = target[i].detach().cpu().double()
            inverse = cov_inverse[i].detach().cpu().double()
            marginal = marginal_c_mu[i].detach().cpu().double()

            # Define minimization objective and jacobian
            def loglikeli_bern_uni(marginal_interv_vector):
                return F.binary_cross_entropy_with_logits(
                    input=marginal_interv_vector, target=target_uni, reduction="sum"
                )

            def jac_min_fct(x):
                return torch.sigmoid(x) - target_uni

            # Define confidence region constraint and its jacobian
            def conf_region_uni(marginal_interv_vector):
                loglikeli_theta_star = dist_logits_uni.log_prob(marginal_interv_vector)
                return loglikeli_theta_star - loglikeli_goal_uni

            def jac_constraint(x):
                return -(inverse @ (x - marginal).unsqueeze(-1)).squeeze(-1)

            # Wrapper for scipy "minimize" function
            # Find intervention logits by minimizing the concept BCE s.t. they still lie on the boundary of the confidence region
            minimum = intervene_utils.minimize_constr(
                f=loglikeli_bern_uni,
                x0=x0[i].detach().cpu(),
                jac=jac_min_fct,
                method="SLSQP",
                constr={
                    "fun": conf_region_uni,
                    "lb": 0,
                    "ub": float("inf"),
                    "jac": jac_constraint,
                },
                bounds={"lb": lb_interv[i], "ub": ub_interv[i]},
                max_iter=50,
                tol=1e-4 * n_intervened,
            )
            interv_vector[i] = minimum.x.to(device)

        # Permute intervened concept logits back into original order
        indices_reversed = torch.argsort(indices)
        interv_vector_unordered = torch.full_like(
            c_mu, float("nan"), device=c_mu.device, dtype=torch.float32
        )
        interv_vector_unordered[:, :n_intervened] = interv_vector
        c_intervened_logits = interv_vector_unordered.gather(1, indices_reversed)

        return c_intervened_logits
