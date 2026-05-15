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


def update_config(config):
    config.class_concept = const.CLASS_AND_CONCEPT[config.dataset_name]
    config.class_names = config.class_concept["class_names"]
    config.num_class = len(config.class_names)
    config.class2concept = config.class_concept["class2concept"]
    config.concept_dict = config.class_concept["concept_dict"]
    config.num_concept = 

    return config
