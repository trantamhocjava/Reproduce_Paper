from kltn_utils import kltn_utils

from ..adacbm_hybridcbm import train as adacbm_hybridcbm_train
from .explicd_hybridcbm import ExplicdHybridCBM


class ExplicdHybridCBMTrain(adacbm_hybridcbm_train.AdaHybridCBMTrain):
    def __init__(self, CustomMetric, cp_path, config, select_concept_data):
        super().__init__(CustomMetric, cp_path, config, select_concept_data)

        # Model
        self.model = ExplicdHybridCBM(
            config=config, select_concept_data=select_concept_data
        )

        ## Get clip_model
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.model.clip_model
        )

    def setup_grad(self):
        # grad
        self.model.setup_grad()
        kltn_utils.freeze_module(self.clip_model)

    def get_loss(self, batch):
        img, label, concept = batch

        # get img feat
        self.clip_model.eval()
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.clip_model, self.config.model.clip_model, img
        )

        # forward
        label_logits, concept_logits = self.model(img)

        # get loss
        cls_loss = self.cls_loss(label_logits, label)
        concept_loss = self.concept_loss(concept_logits, concept)

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

        align_loss = self.align_loss(
            self.model.dynamic_concept_feat,
            self.model.static_concept_feat,
        )

        loss = (
            discri_loss * self.config.loss.lambda_discri
            + ort_loss * self.config.loss.lambda_ort
            + align_loss * self.config.loss.lambda_align
            + cls_loss * self.config.loss.lambda_cls
            + concept_loss * self.config.loss.lambda_concept
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
