import torch.nn as nn
from kltn_utils import kltn_class, kltn_utils

from .cbm import ModelXtoCtoY


class MetricCalculator(kltn_class.MetricCalculator):
    def get_loss_dict(self):
        return kltn_utils.deepcopy_obj(
            {
                "class_loss": [],
                "concept_loss": [],
                "loss": [],
            }
        )


class CBMTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config):
        super().__init__(CustomMetric, cp_path)

        self.config = config

        # Model
        self.model = ModelXtoCtoY(config=config)

        # Loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.concept_loss = nn.BCEWithLogitsLoss()

    def setup_grad(self):
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

        label_logits, concept_logits = self.model(img)

        # Compute the loss
        cls_loss = self.cls_loss(label_logits, label)
        concept_loss = self.concept_loss(concept_logits, concept)

        loss = cls_loss + self.config.loss.lambda_concept * concept_loss

        return {
            "y": label,
            "y_logits": label_logits,
            "c": concept,
            "c_logits": concept_logits,
            "loss": loss,
            "class_loss": cls_loss,
            "concept_loss": concept_loss,
        }
