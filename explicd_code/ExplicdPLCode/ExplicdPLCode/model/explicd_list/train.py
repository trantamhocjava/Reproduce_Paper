import torch.nn as nn
from kltn_utils import kltn_class, kltn_utils

from ... import const
from .explicd import ExplicdList


class MetricCalculator(kltn_class.MetricCalculator):
    def get_loss_dict(self):
        return kltn_utils.deepcopy_obj(const.LOSS_DICT)


class ExplicdListTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config):
        super().__init__(CustomMetric, cp_path)
        self.config = config

        # Module
        self.model = ExplicdList(
            config=config,
        )

        # Loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.concept_loss = nn.BCEWithLogitsLoss()

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
        cls_logits, concept_logits = self.model(img)

        # Compute the loss
        cls_loss = self.cls_loss(cls_logits, label)
        concept_loss = self.concept_loss(concept_logits, concept)
        loss = cls_loss + concept_loss

        return {
            "y_logits": cls_logits,
            "y": label,
            "c_logits": concept_logits,
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
