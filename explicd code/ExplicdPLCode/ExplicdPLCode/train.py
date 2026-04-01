import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn import metrics

from . import const, utils
from .loss import ExplicdLoss
from .model.explicd import ExpLICD


class MetricCalculator:
    def __init__(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])

        self.loss = 0
        self.loss_cls = 0
        self.loss_concept = 0

        self.n_batchs = 0

    def update(self, y_logit, y_true, loss, loss_cls, loss_concept):
        self.n_batchs += 1

        y_pred = torch.argmax(y_logit.detach(), dim=1)

        self.y_pred = torch.cat((self.y_pred, y_pred.cpu()), 0)
        self.y_true = torch.cat((self.y_true, y_true.cpu()), 0)

        self.loss += loss.item()
        self.loss_cls += loss_cls.item()
        self.loss_concept += loss_concept.item()

    def return_metrics(self):
        y_acc = metrics.accuracy_score(self.y_true.numpy(), self.y_pred.numpy()) * 100
        y_bmac = (
            metrics.balanced_accuracy_score(self.y_true.numpy(), self.y_pred.numpy())
            * 100
        )

        loss = self.loss / self.n_batchs
        loss_cls = self.loss_cls / self.n_batchs
        loss_concept = self.loss_concept / self.n_batchs

        return {
            "y_acc": y_acc,
            "y_bmac": y_bmac,
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_concept": loss_concept,
        }

    def reset(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])

        self.loss = 0
        self.loss_cls = 0
        self.loss_concept = 0

        self.n_batchs = 0

    def get_concept_accuracy(self, c_true, c_pred):
        return (c_true == c_pred).mean() * 100

    def get_overall_concept_accuracy(self, c_true, c_pred):
        return np.mean(np.all(c_true == c_pred, axis=1)) * 100


class MetricCalculatorBlackBox:
    def __init__(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])

        self.loss = 0

        self.n_batchs = 0

    def update(self, y_logit, y_true, loss):
        self.n_batchs += 1

        y_pred = torch.argmax(y_logit.detach(), dim=1)

        self.y_pred = torch.cat((self.y_pred, y_pred.cpu()), 0)
        self.y_true = torch.cat((self.y_true, y_true.cpu()), 0)

        self.loss += loss.item()

    def return_metrics(self):
        y_acc = metrics.accuracy_score(self.y_true.numpy(), self.y_pred.numpy()) * 100
        y_bmac = (
            metrics.balanced_accuracy_score(self.y_true.numpy(), self.y_pred.numpy())
            * 100
        )

        loss = self.loss / self.n_batchs

        return {
            "y_acc": y_acc,
            "y_bmac": y_bmac,
            "loss": loss,
        }

    def reset(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])

        self.loss = 0

        self.n_batchs = 0


class ExplicdTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.train_metric = MetricCalculator()
        self.val_metric = MetricCalculator()
        self.test_metric = MetricCalculator()

        self.model = ExpLICD(
            concept_dict=const.CONCEPT_DATASET_DICT[config.dataset_name],
            config=config,
        )

        self.loss_fn = ExplicdLoss(
            concept_criteria=self.model.concept_token_dict.keys()
        )

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
        cls_logits, concept_logits_dict = self.model(data)

        # Compute the loss
        loss, loss_cls, loss_concept = self.loss_fn(
            concept_logits_dict, concept, cls_logits, label
        )

        return loss, loss_cls, loss_concept, cls_logits

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.epoch_time = time.time()

    def training_step(self, batch, batch_idx):
        data, label, concept = batch

        loss, loss_cls, loss_concept, label_logit = self.get_loss(data, concept, label)

        # Update loss and metric
        self.train_metric.update(
            label_logit,
            label,
            loss,
            loss_cls,
            loss_concept,
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
        data, y_true, concept = batch

        loss, loss_cls, loss_concept, y_logit = self.get_loss(data, concept, y_true)

        # Update loss and metric
        self.val_metric.update(
            y_logit,
            y_true,
            loss,
            loss_cls,
            loss_concept,
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

        utils.save_dict_to_json(test_result, f"{const.CP_PATH}/test_result.json")

    def test_step(self, batch, batch_idx):
        data, y_true, concept = batch

        loss, loss_cls, loss_concept, y_logit = self.get_loss(data, concept, y_true)

        # Update loss and metric
        self.test_metric.update(
            y_logit,
            y_true,
            loss,
            loss_cls,
            loss_concept,
        )

    def log_result(self, metric, mode):
        for key, value in metric.items():
            self.log(
                f"{mode}_{key}", value, on_step=False, on_epoch=True, sync_dist=True
            )


class BlackBoxTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.train_metric = MetricCalculatorBlackBox()
        self.val_metric = MetricCalculatorBlackBox()
        self.test_metric = MetricCalculatorBlackBox()

        self.model = utils.build_blackbox_model(config)

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

    def get_loss(self, data, label):
        # Forward pass
        label_logit = self.model(data)

        # Compute the loss
        loss = F.cross_entropy(label_logit, label)

        return loss

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.epoch_time = time.time()

    def training_step(self, batch, batch_idx):
        data, label = batch

        loss, label_logit = self.get_loss(data, label)

        # Update loss and metric
        self.train_metric.update(
            label_logit,
            label,
            loss,
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
        data, label = batch

        loss, label_logit = self.get_loss(data, label)

        # Update loss and metric
        self.val_metric.update(
            label_logit,
            label,
            loss,
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

        utils.save_dict_to_json(test_result, f"{const.CP_PATH}/test_result.json")

    def test_step(self, batch, batch_idx):
        data, label = batch

        loss, label_logit = self.get_loss(data, label)

        # Update loss and metric
        self.test_metric.update(
            label_logit,
            label,
            loss,
        )

    def log_result(self, metric, mode):
        for key, value in metric.items():
            self.log(
                f"{mode}_{key}", value, on_step=False, on_epoch=True, sync_dist=True
            )
