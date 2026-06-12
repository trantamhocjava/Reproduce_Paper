import numpy as np
import torch
from kltn_utils import kltn_class, kltn_utils

from ... import const, loss
from .ecbm import ECBM


# CONTINUE HERE
class MetricCalculator:
    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.c_pred = []
        self.c_true = []

        self.loss_dict = kltn_utils.deepcopy_obj(obj=const.LOSS_DICT)

    def update(self, result):
        _, y_pred = torch.min(result["xy_energy"].detach(), 1)
        _, c_pred = torch.min(result["xc_energy"].detach(), 2)

        self.y_pred.append(y_pred.cpu())
        self.c_pred.append(c_pred.cpu())

        self.y_true.append(result["y"].cpu())
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

    def update_loss_dict(self, result):
        for key, value in self.loss_dict.items():
            value.append(result[key].item())

    def return_loss_dict(self):
        result = {}

        for key, value in self.loss_dict.items():
            result[key] = np.array(value).mean()

        return result


class ECBMTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config, num_concept):
        super().__init__(CustomMetric, cp_path)

        self.config = config

        # Model
        self.model = ECBM(config=config, num_concept=num_concept)

        # Loss
        self.class_loss = loss.EBMLabelLoss()
        self.concept_loss = loss.EBMConceptLoss()

    def setup_grad(self):
        # Grad
        self.model.setup_grad()

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer = kltn_utils.build_optimizer(
            self.model.parameters(),
            self.config.optimizer,
        )

        return {"optimizer": optimizer}

    def get_loss(self, batch):
        img, label, concept = batch

        xy_energy, cy_energy, xc_energy = self.model(
            img, concept, is_train=True, use_cy=True
        )

        class_loss = self.class_loss(xy_energy, label)
        cy_loss = self.class_loss(cy_energy, label)
        concept_loss = self.concept_loss(xc_energy, concept)

        loss = (
            self.config.loss.cpt_lambda * concept_loss
            + self.config.loss.cls_lambda * class_loss
            + self.config.loss.cy_lambda * cy_loss
        )

        return {
            "y": label,
            "c": concept,
            "xy_energy": xy_energy,
            "xc_energy": xc_energy,
            "concept_loss": concept_loss,
            "class_loss": class_loss,
            "cy_loss": cy_loss,
            "loss": loss,
        }
