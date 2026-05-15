import time

import pytorch_lightning as pl
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
from .explicd_hybridcbm import AdaHybridCBM


class AdaHybridCBMTrain(pl.LightningModule):
    def __init__(self, config, select_concept_data):
        super().__init__()

        self.config = config

        # Model
        self.hybrid_bank = HybridConceptBank(config, select_concept_data)
        self.hybridcbm = AdaHybridCBM(
            config=config, concept_feat=self.hybrid_bank.concept_feat
        )

        # Loss
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.discri_loss = DiscriminabilityLoss()
        self.ortho_loss = OrthogonalityLoss(num_class=config.num_class)
        self.align_loss = SinkhornDistanceLoss()
        self.concept_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # Metric
        self.train_metric = kltn_class.MetricCalculator()
        self.val_metric = kltn_class.MetricCalculator()
        self.test_metric = kltn_class.MetricCalculator()

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

        return [optimizer_dynamic_concept, optimizer_hybridcbm]

    def train_concept(self, norm_img_feat, label):
        discri_loss = self.discri_loss(
            norm_img_feat,
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
        label_logits, norm_img_feat, concept_logits = self.hybridcbm(image)

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
            norm_img_feat,
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
            norm_img_feat,
            concept_logits,
        ) = self.train_hybridcbm(image, label, concept)

        # train concept
        (
            discri_loss,
            ort_loss,
            align_loss,
        ) = self.train_concept(norm_img_feat, label)

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

    def on_train_epoch_start(self):
        self.train_metric.reset(kltn_utils.deepcopy_obj(const.LOSS_DICT))
        self.val_metric.reset(kltn_utils.deepcopy_obj(const.LOSS_DICT))
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update optimizer
        self.manual_backward(result["loss"])

        opt_dynamic_concept, opt_classifier = self.optimizers()

        kltn_utils.update_optimizer(opt_dynamic_concept)
        kltn_utils.update_optimizer(opt_classifier)

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
        self.test_metric.reset(kltn_utils.deepcopy_obj(const.LOSS_DICT))
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
