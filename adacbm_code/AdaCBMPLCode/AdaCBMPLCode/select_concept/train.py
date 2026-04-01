import pytorch_lightning as pl
import torch


class ImgFeatGetter(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.img_feat = []
        self.label = []

    def on_test_epoch_end(self):
        self.img_feat = torch.cat(self.img_feat, dim=0)
        self.label = torch.cat(self.label, dim=0)

    def test_step(self, batch, batch_idx):
        img, label = batch

        img_feat = self.model(img, None)[0]

        self.img_feat.append(img_feat)
        self.label.append(label)


class ConceptFeatGetter(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.concept_feat = []

    def on_test_epoch_end(self):
        self.concept_feat = torch.cat(self.concept_feat, dim=0)

    def test_step(self, batch, batch_idx):
        concept_token = batch

        concept_feat = self.model(None, concept_token)[1]

        self.concept_feat.append(concept_feat)
