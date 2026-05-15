import torch

from models.module.adaptive import AdaptiveModule

from .linearCBM import LinearCBM


class AdaptiveCBM(LinearCBM):
    def __init__(self, config, conceptbank):
        super().__init__(config, conceptbank)

        clip_dim = self.conceptbank.clip_encoder.embedding_dim

        self.adaptive_module = AdaptiveModule(
            dim=clip_dim,
            num_layers=config.adaptive_num_layers,
            residual=config.adaptive_residual,
            use_img_norm=config.adaptive_use_img_norm,
        )

    def forward(self, img_feat, concept_features=None):
        if concept_features is None:
            concept_features = self.concept_features

        adapted_img_feat = self.adaptive_module(img_feat)
        sim_score = self.scale * adapted_img_feat @ concept_features.T  # B, C
        logits = self.classifier(sim_score)
        return logits

    def configure_optimizers(self):
        if self.is_train_concept:
            optimizer_dynamic = torch.optim.Adam(
                [
                    {
                        "params": self.conceptbank.dynamic_bank.parameters(),
                        "lr": self.config.concept_lr,
                    },
                ]
            )

            optimizer_classifier = torch.optim.Adam(
                [
                    {"params": self.classifier.parameters(), "lr": self.config.lr},
                    {"params": self.scale, "lr": self.config.lr},
                    {"params": self.adaptive_module.parameters(), "lr": self.config.lr},
                ]
            )
            return [optimizer_dynamic, optimizer_classifier], []

        else:
            # Khi không train concept riêng, gộp chung tất cả vào 1 Optimizer
            optimizer_classifier = torch.optim.Adam(
                [
                    {"params": self.classifier.parameters(), "lr": self.config.lr},
                    {
                        "params": self.conceptbank.dynamic_bank.parameters(),
                        "lr": self.config.lr,
                    },
                    {"params": self.scale, "lr": self.config.lr},
                    # 🚀 ĐĂNG KÝ ADAPTIVE MODULE VÀO ĐÂY
                    {"params": self.adaptive_module.parameters(), "lr": self.config.lr},
                ]
            )
            return optimizer_classifier
