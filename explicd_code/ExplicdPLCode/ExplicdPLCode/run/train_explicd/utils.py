import os

import torch
from kltn_utils import kltn_utils
from torch.utils.data import DataLoader

from ... import const
from .dataset import ImageConceptDataset


def load_dataset(config, transform, mode):
    dataset = ImageConceptDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
        transform=transform,
        class2concept=config.class2concept,
        config=config,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


def setup_train(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(config.cp_path, exist_ok=True)

    config.class_concept = const.CLASS_AND_CONCEPT[config.dataset_name]
    config.concept_dict = config.class_concept["concept_dict"]
    config.class_names = config.class_concept["class_names"]
    config.class2concept = torch.tensor(config.class_concept["class2concept"])
    config.num_class = len(config.class_names)
    config.concepts = config.class_concept["concepts"]
    config.concept2class = config.class_concept["concept2class"]
    config.num_concept = len(config.concepts)
