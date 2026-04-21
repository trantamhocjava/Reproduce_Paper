import pytorch_lightning as pl
import torch


class ConceptFeatGetter(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.concept_feat = []

    def on_test_epoch_end(self):
        self.concept_feat = torch.cat(self.concept_feat, dim=0)

    def test_step(self, batch, batch_idx):
        concepts = batch
        concept_token = self.tokenizer(concepts).to(self.device)

        concept_feat = self.model(None, concept_token)[1]

        self.concept_feat.append(concept_feat.detach())
