import numpy as np
import torch
import torch.nn as nn
from kltn_utils import kltn_class, kltn_utils

from ... import const
from ...loss import ConceptLoss
from .explicd import Explicd


class MetricCalculator:
    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.c_pred = []
        self.c_true = []

        self.loss_dict = kltn_utils.deepcopy_obj(obj=const.LOSS_DICT)

    def update(self, result):
        y_pred = torch.argmax(result["y_logits"].detach(), dim=1)
        self.y_pred.append(y_pred.cpu())
        self.y_true.append(result["y"].cpu())

        c_pred = []
        for key, logits in result["c_logits_dict"].items():
            c_pred_cri = torch.argmax(logits.detach(), dim=1).unsqueeze(1)
            c_pred.append(c_pred_cri)
        c_pred = torch.cat(c_pred, dim=1)

        self.c_pred.append(c_pred.cpu())
        self.c_true.append(result["c"].cpu())

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

    def get_loss_dict(self):
        pass

    def update_loss_dict(self, result):
        for key, value in self.loss_dict.items():
            value.append(result[key].item())

    def return_loss_dict(self):
        result = {}

        for key, value in self.loss_dict.items():
            result[key] = np.array(value).mean()

        return result


class ExplicdTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config):
        super().__init__(CustomMetric, cp_path)
        self.config = config

        # Module
        self.model = Explicd(
            config=config,
        )

        # Loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.concept_loss = ConceptLoss()

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
        cls_loss = self.cls_loss(cls_logits, label)
        concept_loss = self.concept_loss(concept_logits_dict, concept)

        # TODO: DEBUG
        kltn_utils.rank_zero_info_newline(f"cls_loss: {cls_loss.item()}")
        kltn_utils.rank_zero_info_newline(f"concept_loss: {concept_loss.item()}")

        # END DEBUG

        loss = cls_loss + concept_loss

        return {
            "y_logits": cls_logits,
            "y": label,
            "c_logits_dict": concept_logits_dict,
            "c": concept,
            "loss": loss,
            "cls_loss": cls_loss,
            "concept_loss": concept_loss,
        }

    def update_optimizer_manually(self, result):
        # Update optimizer
        self.manual_backward(result["loss"])

        optimizer_clip_model, optimizer_bridge = self.optimizers()

        kltn_utils.update_optimizer(optimizer_clip_model)
        kltn_utils.update_optimizer(optimizer_bridge)
