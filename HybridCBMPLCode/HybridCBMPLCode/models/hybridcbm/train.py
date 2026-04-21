import time

import numpy as np
import pytorch_lightning as pl
import torch
from kltn_utils import kltn_utils
from sklearn import metrics
from torch.nn import functional as F

from . import const


class MetricCalculator:
    def __init__(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])

        self.loss = 0
        self.loss_cls = 0

        self.n_batchs = 0

    def update(self, y_logit, y_true, loss, loss_cls):
        self.n_batchs += 1

        y_pred = torch.argmax(y_logit.detach(), dim=1)

        self.y_pred = torch.cat((self.y_pred, y_pred.cpu()), 0)
        self.y_true = torch.cat((self.y_true, y_true.cpu()), 0)

        self.loss += loss.item()
        self.loss_cls += loss_cls.item()

    def return_metrics(self):
        y_acc = metrics.accuracy_score(self.y_true.numpy(), self.y_pred.numpy()) * 100
        y_bmac = (
            metrics.balanced_accuracy_score(self.y_true.numpy(), self.y_pred.numpy())
            * 100
        )

        loss = self.loss / self.n_batchs
        loss_cls = self.loss_cls / self.n_batchs

        return {
            "y_acc": y_acc,
            "y_bmac": y_bmac,
            "loss": loss,
            "loss_cls": loss_cls,
        }

    def reset(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])

        self.loss = 0
        self.loss_cls = 0

        self.n_batchs = 0

    def get_concept_accuracy(self, c_true, c_pred):
        return (c_true == c_pred).mean() * 100

    def get_overall_concept_accuracy(self, c_true, c_pred):
        return np.mean(np.all(c_true == c_pred, axis=1)) * 100


class HybridCBMTrain(pl.LightningModule):
    def __init__(self, config, concept_bank):
        super().__init__()

        self.config = config
        self.concept_bank = concept_bank

        # Model
        self.model = HybridCBM(config)

        #

        # Metric
        self.train_metric = MetricCalculator()
        self.val_metric = MetricCalculator()
        self.test_metric = MetricCalculator()

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = kltn_utils.build_optimizer(self.model, self.config)
        lr_scheduler, monitor = kltn_utils.build_scheduler(optimizer, self.config)
        res = {
            "optimizer": optimizer,
        }

        if lr_scheduler is not None:
            res["lr_scheduler"] = lr_scheduler

        if monitor is not None:
            res["monitor"] = monitor

        return res

    def get_loss(self, batch):
        data, label = batch

        # Forward pass
        label_logits = self.model(data)

        # Compute the loss
        loss_cls = F.cross_entropy(label_logits, label)
        loss = loss_cls

        return loss, loss_cls, label_logits, label

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.epoch_time = time.time()

    def training_step(self, batch, batch_idx):
        loss, loss_cls, label_logit, label = self.get_loss(batch)

        # Update loss and metric
        self.train_metric.update(
            label_logit,
            label,
            loss,
            loss_cls,
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
        loss, loss_cls, label_logit, label = self.get_loss(batch)

        # Update loss and metric
        self.val_metric.update(
            label_logit,
            label,
            loss,
            loss_cls,
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

        kltn_utils.save_dict_to_json(test_result, f"{const.CP_PATH}/test_result.json")

    def test_step(self, batch, batch_idx):
        loss, loss_cls, label_logit, label = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(
            label_logit,
            label,
            loss,
            loss_cls,
        )

    def log_result(self, metric, mode):
        for key, value in metric.items():
            self.log(
                f"{mode}_{key}", value, on_step=False, on_epoch=True, sync_dist=True
            )
