import os

import numpy as np
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from .dataset import CustomConceptDataset, CustomImageDataset
from .train import ConceptFeatGetter, ImgFeatGetter


def get_num_images_per_class(config):
    num_images_per_class = [
        len(os.listdir(f"{config.dataset_dir}/{class_name}"))
        for class_name in config.class_names
    ]

    return num_images_per_class


def prepare_img_feat(model, transform, config):
    dataset = CustomImageDataset(
        dataset_dir=f"{config.dataset_dir}/train", transform=transform, config=config
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    img_feat_getter = ImgFeatGetter(model=model)

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )
    tester.test(model=img_feat_getter, dataloaders=dataloader)

    img_feat = F.normalize(img_feat_getter.img_feat, dim=1)
    label = img_feat_getter.label

    return img_feat, label


def get_all_concepts_and_concept2cls(config):
    num_concept = sum([len(concepts) for concepts in config.class2concepts.values()])
    concept2cls = np.zeros(num_concept)
    all_concepts = []

    for i, (class_name, concepts) in enumerate(config.class2concepts.items()):
        class_idx = config.class_names.index(class_name)

        for concept in concepts:
            all_concepts.append(concept)
            concept2cls[i] = class_idx

    all_concepts = np.array(all_concepts)

    return all_concepts, concept2cls, num_concept


def has_pattern(concepts, pattern):
    """
    Return a boolean array where it is true if one concept contains the pattern
    """
    return np.char.find(concepts, pattern) != -1


def check_no_cls_names(concepts, cls_names):
    res = np.ones(len(concepts), dtype=bool)

    for cls_name in cls_names:
        no_cls_name = not has_pattern(concepts, cls_name)
        res = res & no_cls_name

    return res


def preprocess_concept(concepts, config, concept2cls):
    """
    concepts: numpy array of strings of concepts

    This function checks all input concepts, remove duplication, and
    remove class names if necessary
    """
    _, left_idx = np.unique(concepts, return_index=True)

    is_good = check_no_cls_names(concepts, config.class_names)
    left_idx = left_idx[is_good]

    concepts = concepts[left_idx]
    concept2cls = concept2cls[left_idx]

    return concepts, concept2cls


def prepare_txt_feat(model, all_concepts, config, tokenizer):
    dataset = CustomConceptDataset(concepts=all_concepts, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    concept_feat_getter = ConceptFeatGetter(model=model)

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )
    tester.test(model=concept_feat_getter, dataloaders=dataloader)

    concept_feat = F.normalize(concept_feat_getter.concept_feat, dim=1)

    return concept_feat
