import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader, TensorDataset

from ..loss import CBLoss
from ..model.cbm.cbm import CBM
from ..train import MetricCalculator
from . import utils as intervene_utils


def intervene_cbm(
    test_loader,
    config,
):
    """
    Compute the efficacy of intervening on a model using different intervention strategies and policies for baselines.

    This function evaluates the efficacy of intervening on a model using various intervention strategies and policies.
    It performs interventions on the model's predicted concepts and computes the change in performance after intervention.
    The function logs the metrics at each step of the intervention process into wandb.
    Note that multiple comma-separated strategies and policies can be passed in the config file, and the function will
    iterate over all combinations.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data. Used for computing empirical percentiles.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The model to be evaluated.
        metrics (object): An object to track and compute metrics.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and data settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None
    """
    policy = config.inter_policy
    strategy = config.inter_strategy
    num_interventions = min(config.min_num_interventions, config.num_concepts)

    # Intervening with different strategies
    intervention_dataset_base = []
    intervention_policy = intervene_utils.define_policy(policy)
    intervention_strategy = HardCBMStrategy()

    # One full model pass without interventions
    rank_zero_info("0 concept")

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    model = CBMIntervene0Concept(
        config=config,
        intervention_strategy=intervention_strategy,
        intervention_policy=intervention_policy,
        intervention_dataset_base=intervention_dataset_base,
    )
    tester.test(
        model=model,
        ckpt_path=config.best_model,
        dataloaders=test_loader,
    )

    # Computing intervention curves using stored concept predictions
    # Preparing dataset
    intervention_dataset_base = [
        (torch.cat([sublist[i] for sublist in intervention_dataset_base], dim=0).cpu())
        for i in range(len(intervention_dataset_base[0]))
    ]

    concepts_dataset_mask = torch.zeros_like(intervention_dataset_base[1], device="cpu")

    for num_intervened in range(1, num_interventions + 1):
        rank_zero_info(f"{num_intervened} concept")

        intervention_dataset = TensorDataset(
            *intervention_dataset_base, concepts_dataset_mask
        )
        intervention_loader = DataLoader(
            intervention_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            shuffle=False,
        )
        concepts_dataset_mask_new = []

        tester = Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
        )

        model = CBMIntervene(
            config=config,
            num_concepts=num_intervened,
            intervention_strategy=intervention_strategy,
            intervention_policy=intervention_policy,
            concepts_dataset_mask_new=concepts_dataset_mask_new,
        )
        tester.test(
            model=model,
            ckpt_path=config.best_model,
            dataloaders=intervention_loader,
        )

        # Updating mask
        concepts_dataset_mask = torch.cat(concepts_dataset_mask_new, dim=0).cpu()


class CBMIntervene0Concept(pl.LightningModule):
    def __init__(
        self,
        config,
        intervention_strategy,
        intervention_policy,
        intervention_dataset_base: list = [],
    ):
        super().__init__()

        self.config = config
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.intervention_dataset_base = intervention_dataset_base

        self.test_metric = MetricCalculator(self.config.num_concepts)
        self.model = CBM(config)
        self.loss_fn = CBLoss(config)

    def get_loss(self, batch):
        return_result = None

        data, label, concept = batch

        # Forward pass
        (
            concepts_pred_probs,
            concepts_pred_logits,
            target_pred_logits,
            concepts_hard,
        ) = self.model(data, self.current_epoch, validation=True)

        # Compute the loss
        target_loss, concepts_loss, total_loss = self.loss_fn(
            concepts_pred_logits,
            concept,
            target_pred_logits,
            label,
        )

        return_result = {
            "concepts_pred_probs": concepts_pred_probs,
            "target_pred_logits": target_pred_logits,
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
            "concepts_hard": concepts_hard,
            "concepts_pred_logits": concepts_pred_logits,
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
        data, label, concept = batch

        return_result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(
            y_logit=return_result["target_pred_logits"],
            c_prob=return_result["concepts_pred_probs"],
            y=label,
            c=concept,
            target_loss=return_result["target_loss"],
            concepts_loss=return_result["concepts_loss"],
            total_loss=return_result["total_loss"],
        )

        # Update rest
        self.intervention_dataset_base.append(
            [
                return_result["concepts_pred_probs"],
                concept,
                label,
                data,
                return_result["concepts_hard"],
                return_result["concepts_pred_logits"],
            ]
        )


class CBMIntervene(pl.LightningModule):
    def __init__(
        self,
        config,
        num_concepts,
        intervention_strategy,
        intervention_policy,
        concepts_dataset_mask_new: list = [],
    ):
        super().__init__()

        self.config = config
        self.num_concepts = num_concepts
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.concepts_dataset_mask_new = concepts_dataset_mask_new

        self.test_metric = MetricCalculator(self.config.num_concepts)
        self.model = CBM(config)
        self.loss_fn = CBLoss(config)

    def get_loss(
        self,
        concepts_pred_probs,
        concept,
        label,
        data,
        concepts_hard,
        concepts_mask,
        concepts_pred_logits,
    ):
        return_result = None

        # Forward pass
        concepts_mask_new = self.intervention_policy.compute_intervention_mask(
            concepts_mask,
            concepts_pred_probs=concepts_pred_probs,
        )
        concepts_interv_probs = self.intervention_strategy.compute_intervention_cbm(
            concepts_pred_probs,
            concept,
            concepts_mask_new,
        )
        target_pred_logits = self.model.intervene(
            concepts_interv_probs,
            concepts_mask_new,
        )

        # Compute the loss
        target_loss, concepts_loss, total_loss = self.loss_fn(
            concepts_pred_logits,
            concept,
            target_pred_logits,
            label,
        )

        return_result = {
            "concepts_pred_probs": concepts_pred_probs,
            "target_pred_logits": target_pred_logits,
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
            "concepts_mask_new": concepts_mask_new,
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
        (
            concepts_pred_probs,
            concept,
            label,
            data,
            concepts_hard,
            concepts_pred_logits,
            concepts_mask,
        ) = batch

        return_result = self.get_loss(
            concepts_pred_probs,
            concept,
            label,
            data,
            concepts_hard,
            concepts_mask,
            concepts_pred_logits,
        )

        # Update loss and metric
        self.test_metric.update(
            y_logit=return_result["target_pred_logits"],
            c_prob=return_result["concepts_pred_probs"],
            y=label,
            c=concept,
            target_loss=return_result["target_loss"],
            concepts_loss=return_result["concepts_loss"],
            total_loss=return_result["total_loss"],
        )

        # Update rest
        self.concepts_dataset_mask_new.append(return_result["concepts_mask_new"])


class HardCBMStrategy:
    # Set intervened concepts to 0 & 1
    def __init__(self):
        pass

    def compute_intervention_cbm(self, c_pred, c_true, c_mask):
        c_intervened = c_true * c_mask + c_pred * (1 - c_mask)
        return c_intervened
