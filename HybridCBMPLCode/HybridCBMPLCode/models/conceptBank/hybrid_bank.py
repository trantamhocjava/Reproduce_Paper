import pandas as pd
import torch
import torch.nn.functional as F
from kltn_utils import kltn_utils

from models.conceptBank.dynamic_bank import DynamicConceptBank
from models.conceptBank.static_bank import StaticConceptBank


class HybridConceptBank(torch.nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.static_bank = StaticConceptBank(config)
        self.dynamic_bank = DynamicConceptBank(config)

        # freeze
        kltn_utils.freeze_module(self.static_bank)

    @property
    def clip_encoder(self):
        return self.static_bank.clip_encoder

    @property
    def translator(self):
        return self.dynamic_bank.translator

    def translate(self, features, entry_length=30, temperature=1, batch_size=64):
        return self.dynamic_bank.translator.decode(
            features,
            entry_length=entry_length,
            temperature=temperature,
            batch_size=batch_size,
        )

    @property
    def concept_features(self):
        features = torch.cat(
            (self.static_bank.concept_features, self.dynamic_bank.concept_features),
            dim=0,
        )

        features = F.normalize(features, dim=-1)
        return features

    @property
    def dynamic_features(self):
        concept_features = self.dynamic_bank.concept_features

        concept_features = F.normalize(concept_features, dim=-1)
        return concept_features

    @property
    def static_features(self):
        concept_features = self.static_bank.concept_features
        concept_features = F.normalize(concept_features, dim=-1)
        return concept_features

    @property
    def classes_embeddings(self):
        class_embeddings = self.static_bank.classes_embeddings
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        return class_embeddings

    @property
    def concepts_label(self):
        static = self.static_bank.concepts_label
        dynamic = self.dynamic_bank.concepts_label
        label = torch.cat((static, dynamic), dim=0)

        return label

    def initialize(self, img_features, num_images_per_class, captions=None):
        self.static_bank.initialize(img_features, num_images_per_class)

        if captions is not None:
            captions_embeddings = self.clip_encoder.encode_text(
                captions, batch_size=128
            )
            self.dynamic_bank.concept_features = captions_embeddings

    def get_static_concepts(self, class_idx, idx, dataframe=False):
        return self.static_bank.get_concepts(class_idx, idx, dataframe)

    def get_dynamic_concepts(self, class_idx, idx, dataframe=False):
        return self.dynamic_bank.get_concepts(class_idx, idx, dataframe)

    def get_concepts(self, class_idx, idx, dataframe=False):
        if isinstance(idx, int):
            idx = [idx]

        if isinstance(idx, list):
            sid = [i for i in idx if i < self.static_bank.num_concept]
            did = [
                i - self.static_bank.num_concept
                for i in idx
                if i >= self.static_bank.num_concept
            ]

        if isinstance(idx, torch.Tensor):
            if idx.dim() == 1:
                idx = idx.cpu().numpy().tolist()
                sid = [i for i in idx if i < self.static_bank.num_concept]
                did = [
                    i - self.static_bank.num_concept
                    for i in idx
                    if i >= self.static_bank.num_concept
                ]
            elif idx.dim() == 2:
                res = []
                for i, cid in enumerate(idx):
                    concepts = self.get_concepts(class_idx, cid, True)
                    concepts["idx"] = i
                    res.append(concepts)
                res = pd.concat(res, axis=0)
                return res

        static_concepts = self.get_static_concepts(class_idx, sid, dataframe)
        dynamic_concepts = self.get_dynamic_concepts(class_idx, did, dataframe)

        if dataframe:
            static_concepts["source"] = "static"
            dynamic_concepts["source"] = "dynamic"
            return pd.concat([static_concepts, dynamic_concepts], axis=0)

        return static_concepts + dynamic_concepts

    def save_concepts(self):
        self.static_bank.save_concepts()
        self.dynamic_bank.save_concepts()

    def get_init_weight_from_cls(self, method="cosine"):
        classes_embeddings = self.classes_embeddings
        concept_features = self.concept_features
        concept_features = concept_features / concept_features.norm(
            dim=-1, keepdim=True
        )
        num_cls = classes_embeddings.shape[0]
        num_concept = concept_features.shape[0]

        if method == "topk":
            num_concept_per_cls = num_concept // num_cls
            dis = torch.cdist(classes_embeddings, concept_features)
            # select top k concept with smallest distanct to the class name
            _, idx = torch.topk(dis, num_concept_per_cls, largest=False)
            init_weight = torch.zeros(num_cls, num_concept)
            init_weight.scatter_(1, idx, 1)
        elif method == "cosine":
            init_weight = (
                classes_embeddings.to(concept_features.device) @ concept_features.T
            )
        elif method == "label":
            label = self.concepts_label.long().view(1, -1)
            init_weight = torch.zeros(num_cls, num_concept)
            init_weight.scatter_(0, label, 1)
        elif method == "caption":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return init_weight
