import time

import pytorch_lightning as pl
import torch
from kltn_utils import kltn_utils

from .. import const
from .translator import ConceptTranslator


class MetricCalculator:
    def reset(self):
        self.loss = 0
        self.loss_token = 0
        self.token_acc = 0

        self.n_batchs = 0

    def update(self, logits, tokens, loss_token, loss):
        self.n_batchs += 1

        token_acc = ((logits.argmax(1) == tokens) * (tokens > 0)).sum() / (
            tokens > 0
        ).sum()

        self.token_acc += token_acc.item()
        self.loss_token += loss_token.item()
        self.loss += loss.item()

    def return_metrics(self):
        token_acc = self.token_acc / self.n_batchs
        loss_token = self.loss_token / self.n_batchs
        loss = self.loss / self.n_batchs

        return {
            "token_acc": token_acc,
            "loss_token": loss_token,
            "loss": loss,
        }


class ConceptTranslatorTrain(pl.LightningModule):
    def __init__(self, config, n_batchs):
        super().__init__()
        config.n_batchs = n_batchs
        config.epochs = config.end_epoch - config.start_epoch + 1

        self.config = config

        self.train_metric = MetricCalculator()
        self.test_metric = MetricCalculator()

        self.translator = ConceptTranslator(clip_model_name=config.clip_model)
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = kltn_utils.build_optimizer(self.translator, self.config)
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
        feature, tokens = batch

        logits, _ = self.translator(feature.to(torch.float32), tokens)
        logits = logits.logits
        logits = logits[:, :-1]
        logits = logits.reshape(-1, logits.shape[-1])

        tokens = tokens.flatten()
        loss_token = self.loss_ce(logits, tokens)

        loss = loss_token

        return logits, tokens, loss_token, loss

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.epoch_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.log_result(self.train_metric.return_metrics(), "train")
        self.log(
            "epoch_time", self.epoch_time, on_step=False, on_epoch=True, sync_dist=True
        )

    def training_step(self, batch, batch_idx):
        logits, tokens, loss_token, loss = self.get_loss(batch)

        # Update loss and metric
        self.train_metric.update(logits, tokens, loss_token, loss)

        return loss

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
        logits, tokens, loss_token, loss = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(logits, tokens, loss_token, loss)

    def log_result(self, metric, mode):
        for key, value in metric.items():
            self.log(
                f"{mode}_{key}", value, on_step=False, on_epoch=True, sync_dist=True
            )
