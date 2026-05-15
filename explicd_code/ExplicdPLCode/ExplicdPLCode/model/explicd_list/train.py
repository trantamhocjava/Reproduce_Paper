import time

import numpy as np
import pytorch_lightning as pl
import torch
from kltn_utils import kltn_utils

from ... import const
from ...loss import ExplicdLoss
from .explicd import Explicd


class MetricCalculator:
    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.c_pred = []
        self.c_true = []

        self.loss_dict = {
            "loss": [],
            "cls_loss": [],
            "concept_loss": [],
        }

    def update(self, result):
        y_pred = torch.argmax(result["y_logits"].detach().cpu(), dim=1)
        self.y_pred.append(y_pred)
        self.y_true.append(result["y"].cpu())

        c_logits_dict = kltn_utils.detach_dict(result["c_logits_dict"])
        c_pred = []
        for value in c_logits_dict.values():
            criterion_pred = torch.argmax(value, dim=1)
            c_pred.append(criterion_pred)
        c_pred = torch.stack(c_pred, dim=1)
        self.c_pred.append(c_pred)

        self.c_true.append(result["c"])

        self.update_loss_dict(result)

    def return_metrics(self):
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        concept_true = torch.cat(self.c_true, dim=0).numpy()
        concept_pred = torch.cat(self.c_pred, dim=0).numpy()

        y_acc = kltn_utils.cal_label_accuracy(y_true, y_pred, "acc")
        y_bmac = kltn_utils.cal_label_accuracy(y_true, y_pred, "bmac")
        c_acc = kltn_utils.cal_concept_accuracy(concept_true, concept_pred, "acc")
        c_overall_acc = kltn_utils.cal_concept_accuracy(
            concept_true, concept_pred, "overall_acc"
        )

        return {
            "y_acc": y_acc,
            "y_bmac": y_bmac,
            "c_acc": c_acc,
            "c_overall_acc": c_overall_acc,
            **self.return_loss_dict(),
        }

    def update_loss_dict(self, result):
        for key, value in result.items():
            self.loss_dict[key].append(value.item())

    def return_loss_dict(self):
        result = {}

        for key, value in self.loss_dict.items():
            result[key] = np.array(value).mean()

        return result


class ExplicdTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.train_metric = MetricCalculator()
        self.val_metric = MetricCalculator()
        self.test_metric = MetricCalculator()

        self.model = Explicd(
            config=config,
        )

        self.loss_fn = ExplicdLoss()

        # auto off
        self.automatic_optimization = False

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer_clip_model = kltn_utils.build_optimizer(
            self.model.clip_model.parameters(), self.config.optimizer_clip_model
        )
        optimizer_bridge = kltn_utils.build_optimizer(
            self.model.get_bridge_param(), self.config.optimizer_bridge
        )

        return [optimizer_clip_model, optimizer_bridge]

    def get_loss(self, batch):
        img, label, concept = batch

        # Forward pass
        cls_logits, concept_logits_dict = self.model(img)

        # Compute the loss
        loss, cls_loss, concept_loss = self.loss_fn(
            concept_logits_dict, concept, cls_logits, label
        )

        return {
            "y_logits": cls_logits,
            "y": label,
            "c_logits_dict": concept_logits_dict,
            "c": concept,
            "loss": loss,
            "cls_loss": cls_loss,
            "concept_loss": concept_loss,
        }

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.val_metric.reset()
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update optimizer
        self.manual_backward(result["loss"])

        optimizer_clip_model, optimizer_bridge = self.optimizers()

        kltn_utils.update_optimizer(optimizer_clip_model)
        kltn_utils.update_optimizer(optimizer_bridge)

        # Update loss and metric
        self.train_metric.update(result)

    def on_validation_epoch_end(self):
        metric = {
            **kltn_utils.add_prefix_in_dict(
                self.train_metric.return_metrics(), "train"
            ),
            **kltn_utils.add_prefix_in_dict(self.test_metric.return_metrics(), "val"),
            "epoch_time": time.time() - self.start_time,
        }

        self.log_result(metric)

    def validation_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.val_metric.update(result)

    def on_test_epoch_start(self):
        self.test_metric.reset()
        self.start_time = time.time()

    def on_test_epoch_end(self):
        test_result = {
            **kltn_utils.add_prefix_in_dict(self.test_metric.return_metrics(), "test"),
            "test_time": time.time() - self.start_time,
        }

        kltn_utils.save_dict_to_json(test_result, f"{const.CP_PATH}/test_result.json")

    def test_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(result)

    def log_result(self, metric):
        for key, value in metric.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)
