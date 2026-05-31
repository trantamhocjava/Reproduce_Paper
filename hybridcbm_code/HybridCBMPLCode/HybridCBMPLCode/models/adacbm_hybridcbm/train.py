import torch

from ... import const
from ..hybridcbm import train as hybridcbm_train
from .adacbm_hybridcbm import AdaHybridCBM


class AdaHybridCBMTrain(hybridcbm_train.HybridCBMTrain):
    def __init__(self, CustomMetric, cp_path, config, select_concept_data):
        super().__init__(CustomMetric, cp_path, config, select_concept_data)

        # Model
        self.model = AdaHybridCBM(
            config=config, select_concept_data=select_concept_data
        )

        # Loss
        weight = get_CEL_weight(config.dataset_name, config.class_names)
        self.cls_loss = torch.nn.CrossEntropyLoss(weight=weight)

        # Grad
        self.model.setup_grad()

    def get_loss(self, batch):
        img, label, concept = batch

        label_logits, concept_logits, img_feat = self.model(img)

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


def get_CEL_weight(dataset_name, class_names):
    weight_dict = const.CEL_WEIGHT[dataset_name]
    weight = []
    for class_name in class_names:
        weight.append(weight_dict[class_name])

    weight = torch.tensor(weight)

    return weight
