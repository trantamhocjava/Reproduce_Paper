import torch
import torch.nn as nn
from kltn_utils import kltn_class, kltn_utils

from ... import const
from ...loss.alignment import SinkhornDistanceLoss
from ...loss.regularization import (
    DiscriminabilityLoss,
    OrthogonalityLoss,
)
from .hybridcbm import HybridCBM


class MetricCalculator(kltn_class.MetricCalculator):
    def get_loss_dict(self):
        return kltn_utils.deepcopy_obj(const.LOSS_DICT)


class HybridCBMTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config, select_concept_data):
        super().__init__(CustomMetric, cp_path)

        self.config = config

        # Model
        self.model = HybridCBM(config=config, select_concept_data=select_concept_data)

        # Loss
        self.discri_loss = DiscriminabilityLoss()
        self.ortho_loss = OrthogonalityLoss(num_class=config.num_class)
        self.align_loss = SinkhornDistanceLoss()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.concept_loss = nn.BCEWithLogitsLoss(reduction="mean")

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

        label_logits, concept_logits, img_feat = self.model(img)

        cls_loss = self.cls_loss(label_logits, label)
        concept_loss = self.concept_loss(concept_logits, concept)
        classifier_weight_loss = torch.linalg.vector_norm(
            self.model.cls_head.weight, ord=1, dim=-1
        ).mean()

        discri_loss = self.discri_loss(
            img_feat,
            self.model.dynamic_concept_feat,
            label,
            self.model.class_feat,
        )

        ort_loss = self.ortho_loss(
            self.model.dynamic_concept_feat,
            self.model.static_concept_feat,
        )

        # align loss, should not normalize the concept feature
        align_loss = self.align_loss(
            self.model.dynamic_concept_feat,
            self.model.static_concept_feat,
        )

        # final loss
        loss = (
            discri_loss * self.config.loss.lambda_discri
            + ort_loss * self.config.loss.lambda_ort
            + align_loss * self.config.loss.lambda_align
            + cls_loss * self.config.loss.lambda_cls
            + classifier_weight_loss * self.config.loss.lambda_classifier_weight
        )

        return {
            "y": label,
            "y_logits": label_logits,
            "c": concept,
            "c_logits": concept_logits,
            "loss": loss,
            "discri_loss": discri_loss,
            "ort_loss": ort_loss,
            "align_loss": align_loss,
            "class_loss": cls_loss,
            "concept_loss": concept_loss,
        }
