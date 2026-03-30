import time

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn import metrics

from . import utils
from .loss import CBLoss, SCBLoss
from .model.cbm import CBM
from .model.scbm import SCBM


class MetricCalculator:
    def __init__(self, n_concepts):
        self.n_concepts = n_concepts

        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
        self.c_pred = torch.empty((0, self.n_concepts))
        self.c_true = torch.empty((0, self.n_concepts))

        self.target_loss = 0
        self.concepts_loss = 0
        self.total_loss = 0

        self.n_batchs = 0

    def update(self, y_logit, c_logit, y, c, target_loss, concepts_loss, total_loss):
        self.n_batchs += 1

        y_logit = y_logit.clone().detach()
        c_logit = c_logit.clone().detach()

        _, y_pred = torch.max(y_logit, 1)

        c_pred = c_logit > 0.5

        self.c_pred = torch.cat((self.c_pred, c_pred.cpu()), 0)
        self.c_true = torch.cat((self.c_true, c.cpu()), 0)
        self.y_pred = torch.cat((self.y_pred, y_pred.cpu()), 0)
        self.y_true = torch.cat((self.y_true, y.cpu()), 0)

        self.target_loss += target_loss.item()
        self.concepts_loss += concepts_loss.item()
        self.total_loss += total_loss.item()

    def return_metrics(self):
        c_acc_overall = self.get_overall_concept_accuracy(
            self.c_true.cpu().numpy(), self.c_pred.cpu().numpy()
        )
        c_acc = self.get_concept_accuracy(
            self.c_true.cpu().numpy(), self.c_pred.cpu().numpy()
        )

        y_acc = (
            metrics.accuracy_score(self.y_true.cpu().numpy(), self.y_pred.cpu().numpy())
            * 100
        )
        y_bmac = (
            metrics.balanced_accuracy_score(
                self.y_true.cpu().numpy(), self.y_pred.cpu().numpy()
            )
            * 100
        )

        target_loss = self.target_loss / self.n_batchs
        concepts_loss = self.concepts_loss / self.n_batchs
        total_loss = self.total_loss / self.n_batchs

        return {
            "c_acc_overall": c_acc_overall,
            "c_acc": c_acc,
            "y_acc": y_acc,
            "y_bmac": y_bmac,
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
        }

    def reset(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
        self.c_pred = torch.empty((0, self.n_concepts))
        self.c_true = torch.empty((0, self.n_concepts))

        self.target_loss = 0
        self.concepts_loss = 0
        self.total_loss = 0

        self.n_batchs = 0

    def get_concept_accuracy(self, c_true, c_pred):
        return (c_true == c_pred).mean() * 100

    def get_overall_concept_accuracy(self, c_true, c_pred):
        return np.mean(np.all(c_true == c_pred, axis=1)) * 100


# class CBMTrain:
#     def __init__(
#         self,
#         trainLoader,
#         valLoader,
#         model,
#         optimizer,
#         scaler,
#         mode,
#         config,
#         loss_fn,
#         epoch,
#     ):
#         self.trainLoader = trainLoader
#         self.valLoader = valLoader
#         self.model = model
#         self.optimizer = optimizer
#         self.scaler = scaler
#         self.mode = mode
#         self.config = config
#         self.loss_fn = loss_fn
#         self.epoch = epoch

#     def train_one_epoch(self):
#         metric = MetricCalculator(self.config.num_concepts)
#         loss_return = LossCalculator(len(self.trainLoader))

#         self.model.train()

#         if self.config.training_mode in ("sequential", "independent"):
#             if self.mode == "c":
#                 self.model.head.eval()
#             elif self.mode == "t":
#                 self.model.encoder.eval()

#         for data, label, concept in self.trainLoader:
#             data = data.float().cuda()
#             label = label.long().cuda()
#             concept = concept.long().cuda()

#             self.optimizer.zero_grad(set_to_none=True)

#             if self.config.amp:
#                 with torch.autocast(device_type=const.DEVICE, dtype=torch.float16):
#                     (
#                         loss,
#                         concepts_pred_probs,
#                         target_pred_logits,
#                         target_loss,
#                         concepts_loss,
#                         total_loss,
#                     ) = self.get_loss(data, concept, label)

#                 self.scaler.scale(loss).backward()
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()

#             else:
#                 (
#                     loss,
#                     concepts_pred_probs,
#                     target_pred_logits,
#                     target_loss,
#                     concepts_loss,
#                     total_loss,
#                 ) = self.get_loss(data, concept, label)

#                 loss.backward()
#                 self.optimizer.step()

#             # Update loss and metric
#             loss_return.update(target_loss, concepts_loss, total_loss)
#             metric.update(target_pred_logits, concepts_pred_probs, label, concept)

#         return loss_return.return_metrics(), metric.return_metrics()

#     def validate_one_epoch(self):
#         metric = MetricCalculator(self.config.num_concepts)
#         loss_return = LossCalculator(len(self.valLoader))

#         self.model.eval()
#         with torch.no_grad():
#             for data, label, concept in self.valLoader:
#                 data = data.float().cuda()
#                 label = label.long().cuda()
#                 concept = concept.long().cuda()

#                 (
#                     concepts_pred_probs,
#                     concepts_pred_logits,
#                     target_pred_logits,
#                     concepts_hard,
#                 ) = self.model(data, self.epoch, validation=True)

#                 if self.config.concept_learning == "autoregressive":
#                     concepts_pred_probs = torch.mean(concepts_pred_probs, dim=-1)

#                 target_loss, concepts_loss, total_loss = self.loss_fn(
#                     concepts_pred_logits, concept, target_pred_logits, label
#                 )

#                 # Update loss and metric
#                 loss_return.update(target_loss, concepts_loss, total_loss)
#                 metric.update(target_pred_logits, concepts_pred_probs, label, concept)

#         return loss_return.return_metrics(), metric.return_metrics()

#     def get_loss(self, data, concept, label):
#         # Forward pass
#         concepts_pred_probs, concepts_pred_logits, target_pred_logits, concepts_hard = (
#             self.forward_cbm(data, concept)
#         )

#         # Compute the loss
#         target_loss, concepts_loss, total_loss = self.loss_fn(
#             concepts_pred_logits, concept, target_pred_logits, label
#         )

#         loss = None
#         if self.mode == "j":
#             loss = total_loss
#         elif self.mode == "c":
#             loss = concepts_loss
#         else:
#             loss = target_loss

#         return (
#             loss,
#             concepts_pred_probs,
#             target_pred_logits,
#             target_loss,
#             concepts_loss,
#             total_loss,
#         )

#     def forward_cbm(self, data, concept):
#         if self.config.training_mode == "independent" and self.mode == "t":
#             (
#                 concepts_pred_probs,
#                 concepts_pred_logits,
#                 target_pred_logits,
#                 concepts_hard,
#             ) = self.model(data, self.epoch, concept)
#         elif self.config.concept_learning == "autoregressive" and self.mode == "c":
#             (
#                 concepts_pred_probs,
#                 concepts_pred_logits,
#                 target_pred_logits,
#                 concepts_hard,
#             ) = self.model(data, self.epoch, concepts_train_ar=concept)
#         else:
#             (
#                 concepts_pred_probs,
#                 concepts_pred_logits,
#                 target_pred_logits,
#                 concepts_hard,
#             ) = self.model(data, self.epoch)

#         return (
#             concepts_pred_probs,
#             concepts_pred_logits,
#             target_pred_logits,
#             concepts_hard,
#         )


class SCBMTrainPL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.train_metric = MetricCalculator(self.config.num_concepts)
        self.val_metric = MetricCalculator(self.config.num_concepts)
        self.test_metric = MetricCalculator(self.config.num_concepts)

        self.model = SCBM(config)

        self.loss_fn = SCBLoss(config=config, reduction="mean")

    def init_covariance(self, trainLoader):
        # Initialize covariance with empirical covariance
        if self.config.cov_type == "empirical":
            self.model.sigma_concepts = utils.get_empirical_covariance(
                trainLoader
            ).cuda()
        elif self.config.cov_type == "global":
            lower_triangle = utils.get_empirical_covariance(trainLoader).cuda()
            rows, cols = torch.tril_indices(
                row=self.config.num_concepts, col=self.config.num_concepts, offset=0
            )
            self.model.sigma_concepts = torch.nn.Parameter(lower_triangle[rows, cols])
            # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
            diag_idx = rows == cols
            with torch.no_grad():
                self.model.sigma_concepts[diag_idx] = (
                    lower_triangle[rows, cols][diag_idx].expm1().clamp_min(1e-6).log()
                )  # softplus inverse of diag

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = utils.build_optimizer(self.model, self.config)
        lr_scheduler, monitor = utils.build_scheduler(optimizer, self.config)
        res = {
            "optimizer": optimizer,
        }

        if lr_scheduler is not None:
            res["lr_scheduler"] = lr_scheduler

        if monitor is not None:
            res["monitor"] = monitor

        return res

    def get_loss(self, data, concept, label):
        # Forward pass
        concepts_mcmc_probs, concepts_mcmc_logits, triang_cov, target_pred_logits = (
            self.model(data, self.current_epoch, c_true=concept)
        )

        # Compute the loss
        target_loss, concepts_loss, prec_loss, total_loss = self.loss_fn(
            concepts_mcmc_logits,
            concept,
            target_pred_logits,
            label,
            triang_cov,
        )

        concepts_pred_probs = concepts_mcmc_probs.mean(-1)

        loss = total_loss

        return (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.epoch_time = time.time()

    def training_step(self, batch, batch_idx):
        data, label, concept = batch

        (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        ) = self.get_loss(data, concept, label)

        # Update loss and metric
        self.train_metric.update(
            target_pred_logits,
            concepts_pred_probs,
            label,
            concept,
            target_loss,
            concepts_loss,
            total_loss,
        )

        return loss

    def on_validation_epoch_start(self):
        self.val_metric.reset()

    def on_validation_epoch_end(self):
        self.epoch_time = time.time() - self.epoch_time

        self.log_result(self.train_metric.return_metrics(), "train")
        self.log_result(self.val_metric.return_metrics(), "val")
        self.log(
            "epoch_time", self.epoch_time, on_step=False, on_epoch=True, sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        data, label, concept = batch

        (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        ) = self.get_loss(data, concept, label)

        # Update loss and metric
        self.val_metric.update(
            target_pred_logits,
            concepts_pred_probs,
            label,
            concept,
            target_loss,
            concepts_loss,
            total_loss,
        )

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
        utils.print_dict(test_result)

    def test_step(self, batch, batch_idx):
        data, label, concept = batch

        (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        ) = self.get_loss(data, concept, label)

        # Update loss and metric
        self.test_metric.update(
            target_pred_logits,
            concepts_pred_probs,
            label,
            concept,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def log_result(self, metric, mode):
        for key, value in metric.items():
            self.log(
                f"{mode}_{key}", value, on_step=False, on_epoch=True, sync_dist=True
            )


class JointCBMTrainPL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.train_metric = MetricCalculator(self.config.num_concepts)
        self.val_metric = MetricCalculator(self.config.num_concepts)
        self.test_metric = MetricCalculator(self.config.num_concepts)

        self.model = CBM(config)

        self.loss_fn = CBLoss(config=config, reduction="mean")

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = utils.build_optimizer(self.model, self.config)
        lr_scheduler, monitor = utils.build_scheduler(optimizer, self.config)
        res = {
            "optimizer": optimizer,
        }

        if lr_scheduler is not None:
            res["lr_scheduler"] = lr_scheduler

        if monitor is not None:
            res["monitor"] = monitor

        return res

    def get_loss(self, data, concept, label):
        # Forward pass
        concepts_pred_probs, concepts_pred_logits, target_pred_logits, concepts_hard = (
            self.model(data, self.current_epoch)
        )

        # Compute the loss
        target_loss, concepts_loss, total_loss = self.loss_fn(
            concepts_pred_logits, concept, target_pred_logits, label
        )

        loss = total_loss

        return (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.epoch_time = time.time()

    def training_step(self, batch, batch_idx):
        data, label, concept = batch

        (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        ) = self.get_loss(data, concept, label)

        # Update loss and metric
        self.train_metric.update(
            target_pred_logits,
            concepts_pred_probs,
            label,
            concept,
            target_loss,
            concepts_loss,
            total_loss,
        )

        return loss

    def get_loss_for_val(self, data, concept, label):
        (
            concepts_pred_probs,
            concepts_pred_logits,
            target_pred_logits,
            concepts_hard,
        ) = self.model(data, self.current_epoch, validation=True)

        if self.config.concept_learning == "autoregressive":
            concepts_pred_probs = torch.mean(concepts_pred_probs, dim=-1)

        target_loss, concepts_loss, total_loss = self.loss_fn(
            concepts_pred_logits, concept, target_pred_logits, label
        )

        return (
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def on_validation_epoch_start(self):
        self.val_metric.reset()

    def on_validation_epoch_end(self):
        self.epoch_time = time.time() - self.epoch_time

        self.log_result(self.train_metric.return_metrics(), "train")
        self.log_result(self.val_metric.return_metrics(), "val")
        self.log(
            "epoch_time", self.epoch_time, on_step=False, on_epoch=True, sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        data, label, concept = batch

        (
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        ) = self.get_loss_for_val(data, concept, label)

        # Update loss and metric
        self.val_metric.update(
            target_pred_logits,
            concepts_pred_probs,
            label,
            concept,
            target_loss,
            concepts_loss,
            total_loss,
        )

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
        utils.print_dict(test_result)

    def test_step(self, batch, batch_idx):
        data, label, concept = batch

        (
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        ) = self.get_loss_for_val(data, concept, label)

        # Update loss and metric
        self.test_metric.update(
            target_pred_logits,
            concepts_pred_probs,
            label,
            concept,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def log_result(self, metric, mode):
        for key, value in metric.items():
            self.log(
                f"{mode}_{key}", value, on_step=False, on_epoch=True, sync_dist=True
            )
