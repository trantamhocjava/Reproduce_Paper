import pytorch_lightning as pl
import torch
from kltn_utils import kltn_utils


class ImgFeatGetter(pl.LightningModule):
    def __init__(self, model, model_name):
        super().__init__()

        self.model = model
        self.model_name = model_name
        self.img_feat = []
        self.label = []

    def on_test_epoch_end(self):
        self.img_feat = torch.cat(self.img_feat, dim=0)
        self.label = torch.cat(self.label, dim=0)

    def test_step(self, batch, batch_idx):
        img, label = batch

        img_feat = kltn_utils.get_img_feat_from_clip_model(
            self.model, self.model_name, img
        )

        self.img_feat.append(img_feat)
        self.label.append(label)


class ConceptFeatGetter(pl.LightningModule):
    def __init__(self, model, model_name, tokenizer):
        super().__init__()

        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.concept_feat = []

    def on_test_epoch_end(self):
        self.concept_feat = torch.cat(self.concept_feat, dim=0)

    def test_step(self, batch, batch_idx):
        concepts = batch
        concept_token = self.tokenizer(concepts).to(self.device)
        concept_feat = kltn_utils.get_concept_feat_from_clip_model(
            self.model, self.model_name, concept_token
        )

        self.concept_feat.append(concept_feat)
