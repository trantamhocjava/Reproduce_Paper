import time

import pytorch_lightning as pl
import torch.nn.functional as F
from kltn_utils import kltn_utils

from ... import const


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


class BlackBoxTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.train_metric = MetricCalculatorBlackBox()
        self.val_metric = MetricCalculatorBlackBox()
        self.test_metric = MetricCalculatorBlackBox()

        self.model = kltn_utils.build_blackbox_model(config)

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

        kltn_utils.save_dict_to_json(test_result, f"{const.CP_PATH}/test_result.json")

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
