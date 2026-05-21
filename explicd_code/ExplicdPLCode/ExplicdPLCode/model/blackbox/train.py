import numpy as np
import torch
import torch.nn as nn
from kltn_utils import kltn_class, kltn_utils

from ... import const


class CustomMetric(kltn_class.MetricCalculator):
    def reset(self):
        self.y_pred = []
        self.y_true = []

        self.loss_dict = kltn_utils.deepcopy_obj(const.LOSS_DICT)

    def update(self, result):
        y_pred = torch.argmax(result["y_logits"].detach().cpu(), dim=1)
        self.y_pred.append(y_pred)
        self.y_true.append(result["y"].cpu())

        self.update_loss_dict(result)

    def return_metrics(self):
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_pred = torch.cat(self.y_pred, dim=0).numpy()

        y_acc = kltn_utils.cal_label_accuracy(y_true, y_pred, "acc")
        y_bmac = kltn_utils.cal_label_accuracy(y_true, y_pred, "bmac")

        return {
            "y_acc": y_acc,
            "y_bmac": y_bmac,
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


class BlackboxTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config):
        super().__init__(CustomMetric, cp_path)
        self.config = config

        self.blackbox = kltn_utils.build_blackbox_model(
            config.blackbox, config.num_class
        )

        self.cls_loss = nn.CrossEntropyLoss()

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = kltn_utils.build_optimizer(
            self.blackbox.parameters(), self.config.optimizer
        )

        return {"optimizer": optimizer}

    def get_loss(self, batch):
        img, label = batch

        # Forward pass
        cls_logits = self.blackbox(img)

        # Compute the loss
        cls_loss = self.cls_loss(cls_logits, label)

        loss = cls_loss

        return {
            "y_logits": cls_logits,
            "y": label,
            "loss": loss,
            "cls_loss": cls_loss,
        }
