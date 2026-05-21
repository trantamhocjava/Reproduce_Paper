import torch
import torch.nn as nn
from kltn_utils import kltn_class, kltn_utils

from ... import const
from ...loss.alignment import SinkhornDistanceLoss
from ...loss.regularization import (
    DiscriminabilityLoss,
    OrthogonalityLoss,
)
from ..conceptBank.hybrid_bank import HybridConceptBank
from .hybridcbm import HybridCBM


class MetricCalculator(kltn_class.MetricCalculator):
    def get_loss_dict(self):
        return kltn_utils.deepcopy_obj(const.LOSS_DICT)


class HybridCBMTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config, select_concept_data):
        super().__init__(CustomMetric, cp_path)

        self.config = config

        # Model
        self.hybrid_bank = HybridConceptBank(config, select_concept_data)
        self.hybridcbm = HybridCBM(
            config=config, concept_feat=self.hybrid_bank.concept_feat
        )

        # Loss
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.discri_loss = DiscriminabilityLoss()
        self.ortho_loss = OrthogonalityLoss(num_class=config.num_class)
        self.align_loss = SinkhornDistanceLoss()
        self.concept_loss = nn.BCEWithLogitsLoss()

        # off auto optimization
        self.automatic_optimization = False

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer_dynamic_concept = kltn_utils.build_optimizer(
            self.hybrid_bank.dynamic_bank,
            self.config.optimizer_dynamic_concept,
        )
        optimizer_hybridcbm = kltn_utils.build_optimizer(
            self.hybridcbm,
            self.config.optimizer_hybridcbm,
        )

        return optimizer_dynamic_concept, optimizer_hybridcbm

    def train_concept(self, img_feat, label):
        discri_loss = self.discri_loss(
            img_feat,
            self.hybrid_bank.concept_feat,
            label,
            self.hybrid_bank.static_bank.class_feat,
        )

        ort_loss = self.ortho_loss(
            self.hybrid_bank.dynamic_bank.concept_feat,
            self.hybrid_bank.static_bank.concept_feat,
        )

        # align loss, should not normalize the concept feature
        align_loss = self.align_loss(
            self.hybrid_bank.dynamic_bank.concept_feat,
            self.hybrid_bank.static_bank.concept_feat,
        )

        return discri_loss, ort_loss, align_loss

    def train_hybridcbm(self, image, label, concept):
        label_logits, img_feat, concept_logits = self.hybridcbm(image)

        cls_loss = self.cls_loss(label_logits, label)
        concept_loss = self.concept_loss(concept_logits, concept)

        # loss regularize weight of self.hybridcbm.classifier
        classifier_weight_loss = torch.linalg.vector_norm(
            self.hybridcbm.classifier.weight, ord=1, dim=-1
        ).mean()

        return (
            cls_loss,
            classifier_weight_loss,
            concept_loss,
            label_logits,
            img_feat,
            concept_logits,
        )

    def get_loss(self, batch):
        image, label, concept = batch

        # train classifier
        (
            cls_loss,
            classifier_weight_loss,
            concept_loss,
            label_logits,
            img_feat,
            concept_logits,
        ) = self.train_hybridcbm(image, label, concept)

        # train concept
        (
            discri_loss,
            ort_loss,
            align_loss,
        ) = self.train_concept(img_feat, label)

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

    def update_optimizer_manually(self, result):
        # Update optimizer
        self.manual_backward(result["loss"])

        opt_dynamic_concept, opt_classifier = self.optimizers()

        kltn_utils.update_optimizer(opt_dynamic_concept)
        kltn_utils.update_optimizer(opt_classifier)
