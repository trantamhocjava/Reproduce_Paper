from kltn_utils import kltn_class, kltn_utils

from ... import const
from ...loss import SCBLoss
from .scbm import SCBM


class MetricCalculator(kltn_class.MetricCalculator):
    def get_loss_dict(self):
        return kltn_utils.deepcopy_obj(const.LOSS_DICT)


class SCBMTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config, num_concept):
        super().__init__(CustomMetric, cp_path)

        self.config = config

        # Model
        self.model = SCBM(config, num_concept=num_concept)

        # Loss
        self.scb_loss = SCBLoss(config=config)

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

        # Forward pass
        (
            _,
            _,
            triang_cov,
            label_logits,
            concept_mcmc_logit,
            concept_logits,
        ) = self.model(img, self.current_epoch)

        # Compute the loss
        class_loss, concept_loss, prec_loss = self.scb_loss(
            concept_mcmc_logit,
            concept,
            label_logits,
            label,
            triang_cov,
        )

        loss = class_loss + self.config.loss.concept_lambda * concept_loss + prec_loss

        return {
            "y": label,
            "y_logits": label_logits,
            "c": concept,
            "c_logits": concept_logits,
            "loss": loss,
            "class_loss": class_loss,
            "concept_loss": concept_loss,
        }
