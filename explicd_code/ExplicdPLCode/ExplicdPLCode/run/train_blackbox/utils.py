import torch
from torch.utils.data import DataLoader

from .dataset import ImageConceptDataset


def load_dataset(config, class2concept, transform, mode):
    dataset = ImageConceptDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
        transform=transform,
        class2concept=class2concept,
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


def update_config(config):
    config.class_names = const.CLASS_NAMES[config.dataset_name]
    config.class2concept = torch.tensor(
        const.CLASS2CONCEPT[config.dataset_name], dtype=torch.long
    )

    return config
