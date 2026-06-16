import time

import numpy as np
import pytorch_lightning as pl
import torch
from kltn_utils import kltn_utils

from ... import const
from .translator import ConceptTranslator


class MetricCalculator:
    def reset(self):
        self.loss = []
        self.loss_token = []

        self.tokens = []
        self.pred_tokens = []

    def update(self, result):
        pred_tokens = torch.argmax(result["token_logits"].detach(), dim=1)
        self.pred_tokens.append(pred_tokens.cpu())
        self.tokens.append(result["token"].cpu())

        self.loss_token.append(result["loss_token"].item())
        self.loss.append(result["loss"].item())

    def return_metrics(self):
        pred_tokens = torch.cat(self.pred_tokens, dim=0).numpy()
        tokens = torch.cat(self.tokens, dim=0).numpy()

        token_acc = self.cal_token_level_acc(pred_tokens, tokens)

        loss_token = np.array(self.loss_token).mean()
        loss = np.array(self.loss).mean()

        return {
            "token_acc": token_acc,
            "loss_token": loss_token,
            "loss": loss,
        }

    def cal_token_level_acc(self, pred_tokens, tokens):
        non_padding_mask = tokens > 0
        token_acc = (
            ((pred_tokens == tokens) * non_padding_mask).sum()
            / non_padding_mask.sum()
            * 100
        )

        return token_acc


class ConceptTranslatorTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_metric = MetricCalculator()
        self.val_metric = MetricCalculator()
        self.test_metric = MetricCalculator()

        self.translator = ConceptTranslator(clip_model_name=config.clip_model)
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = kltn_utils.build_optimizer(
            self.translator,
            self.config.optimizer,
        )
        lr_scheduler, monitor = kltn_utils.build_scheduler(
            optimizer, self.config.scheduler
        )
        res = {
            "optimizer": optimizer,
        }

        if lr_scheduler is not None:
            res["lr_scheduler"] = lr_scheduler

        if monitor is not None:
            res["monitor"] = monitor

        return res

    def get_loss(self, batch):
        embedding, token = batch

        token_logits, _ = self.translator(embedding, token)
        token = token.flatten()
        loss_token = self.loss_ce(token_logits, token)

        loss = loss_token

        return {
            "token_logits": token_logits,
            "token": token,
            "loss_token": loss_token,
            "loss": loss,
        }

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.val_metric.reset()
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.train_metric.update(result)

        return result["loss"]

    def on_validation_epoch_end(self):
        epoch_time = time.time() - self.start_time

        self.log_result(self.train_metric.return_metrics(), "train")
        self.log_result(self.val_metric.return_metrics(), "val")
        self.log("epoch_time", epoch_time, on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.val_metric.update(result)

    def on_test_epoch_start(self):
        self.test_metric.reset()
        self.start_time = time.time()

    def on_test_epoch_end(self):
        test_time = time.time() - self.start_time
        test_result = kltn_utils.add_prefix_in_dict(
            self.test_metric.return_metrics(), "test"
        )
        test_result["test_time"] = test_time

        kltn_utils.save_dict_to_json(test_result, f"{const.CP_PATH}/test_result.json")

    def test_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(result)

    def log_result(self, metric, mode):
        metric = kltn_utils.add_prefix_in_dict(metric, mode)

        for key, value in metric.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)
