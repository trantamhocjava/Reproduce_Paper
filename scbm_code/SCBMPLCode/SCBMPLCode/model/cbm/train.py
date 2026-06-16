from kltn_utils import kltn_class, kltn_utils

from ...loss import CBLoss
from .cbm import CBM


class CBMTrain(kltn_class.BaseTrain):
    def __init__(self, CustomMetric, cp_path, config, num_concept):
        super().__init__(CustomMetric, cp_path)

        self.config = config

        self.model = CBM(config, num_concept=num_concept)

        self.cb_loss = CBLoss(config=config)

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
        concept_probs, concept_logits, label_logits, _ = self.model(
            img, self.current_epoch
        )

        # Compute the loss
        class_loss, concept_loss = self.cb_loss(
            concept_logits, concept, label_logits, label
        )

        loss = class_loss + self.config.loss.concept_lambda * concept_loss

        return {
            "y": label,
            "y_logits": label_logits,
            "c": concept,
            "c_logits": concept_logits,
            "loss": loss,
            "class_loss": class_loss,
            "concept_loss": concept_loss,
        }
