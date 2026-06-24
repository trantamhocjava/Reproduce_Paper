from kltn_utils import kltn_utils

from .pretrainedResNetModel import PretrainedResNetModel


class ModelXtoCtoY(PretrainedResNetModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def setup_grad(self):
        pass

    def forward(self, img):
        x = img
        x = self.compute_cnn_features(x)
        x = self.dropout(x)

        outputs = {}  # { 'pool': x }

        for i, layer in enumerate(self.config.model.fc_layers):
            fc_name = "fc" + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)

            if fc_name == self.config.model.C_fc_name:
                # No ReLu for concept layer
                outputs["C"] = x
                continue

            elif fc_name == self.config.model.y_fc_name:
                # No ReLu for y layer
                outputs["y"] = x
                continue

            x = self.relu(x)

        label_logits = outputs["y"]
        concept_logits = outputs["C"]

        # TODO: DEBUG
        kltn_utils.rank_zero_info_newline("ModelXtoCtoY forward")
        kltn_utils.rank_zero_info_newline(f"label_logits.shape: {label_logits.shape}")
        kltn_utils.rank_zero_info_newline(
            f"concept_logits.shape: {concept_logits.shape}"
        )
        # END DEBUG

        return label_logits, concept_logits
